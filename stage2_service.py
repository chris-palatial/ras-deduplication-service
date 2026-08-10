"""
Core runner for the RAS Deduplication Service.

This module does NOT reimplement VGGT/SAM/dedup. It calls the public
ReplicateAnyScene Stage 2 path (same sequence as their main.py Stage 2 block).

Upstream: https://github.com/xiac20/ReplicateAnyScene
"""

from __future__ import annotations

import os
import base64
import binascii
import fcntl
import importlib
import importlib.util
import json
import math
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from artifact_contract import GLB_MAX_BYTES
from object_catalog import (
    OBJECT_CATALOG_JSON_NAME,
    OBJECT_CATALOG_VERSION,
    OBJECT_CROPS_ATLAS_NAME,
    build_object_catalog,
)

# Resolve the pinned upstream checkout. Standalone wrapper layout:
#   <service-root>/
#     stage2_service.py
#     vendor/ReplicateAnyScene/   (paper repo)
RAS_ROOT = Path(os.environ.get("RAS_ROOT", Path(__file__).resolve().parent / "vendor" / "ReplicateAnyScene")).resolve()
RAS_REVISION = os.environ.get("RAS_REVISION", "671191457e7244d9337ef3faf558ee92bbf9bf73")
DEFAULT_VGGT_REVISION = "9e4fa662a8893ed348d048e8b57816c12593448b"
VGGT_REVISION = os.environ.get("VGGT_REVISION", DEFAULT_VGGT_REVISION)
DEFAULT_SAM3_REVISION = "bfbed072a07a6a52c8d5fdc75a7a186251a835b1"
SAM3_REVISION = os.environ.get("SAM3_REVISION", DEFAULT_SAM3_REVISION)
MAX_INLINE_VIDEO_BYTES = 6 * 1024 * 1024
MAX_REMOTE_VIDEO_BYTES = 64 * 1024 * 1024
DEFAULT_VGGT_MODEL_ID = "facebook/VGGT-1B"
COMMERCIAL_VGGT_MODEL_ID = "facebook/VGGT-1B-Commercial"
SAM3_MODEL_ID = "facebook/sam3"
VGGT_MODEL_MARKER = ".stage2_model_id"
VGGT_OMEGA_ANALYSIS_TYPE = "geometry_vggt_omega_1b"
VGGT_OMEGA_MODEL_ID = "facebook/VGGT-Omega"
VGGT_OMEGA_SPACE_ID = "facebook/vggt-omega"
VGGT_OMEGA_SPACE_REVISION = "2597ec6a276ea34d26206087a511f517e2a0024f"
VGGT_OMEGA_GITHUB_REVISION = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
VGGT_OMEGA_MODEL_REVISION = "05654241adc2f218dfb089c373a011f8a7040576"
VGGT_OMEGA_CHECKPOINT = "vggt_omega_1b_512.pt"
VGGT_OMEGA_SPACE_ORIGIN = "https://facebook-vggt-omega.hf.space"
VGGT_OMEGA_SPACE_HOST = "facebook-vggt-omega.hf.space"
VGGT_OMEGA_METADATA_URL = "https://huggingface.co/api/spaces/facebook/vggt-omega"
VGGT_OMEGA_MAX_FRAMES = 24
VGGT_OMEGA_MAX_POINTS_K = 500
VGGT_OMEGA_CONFIDENCE_PERCENTILE = 50.0
VGGT_OMEGA_PROVENANCE_LEVEL = "hosted_unattested"
_VGGT_OMEGA_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_GLB_JSON_CHUNK = 0x4E4F534A
ARTIFACT_MEDIA_TYPES = {
    "point_cloud.glb": "model/gltf-binary",
    "point_cloud.ply": "application/octet-stream",
    "instance_masks.mp4": "video/mp4",
    "camera_intrinsics.json": "application/json",
    OBJECT_CATALOG_JSON_NAME: "application/json",
    OBJECT_CROPS_ATLAS_NAME: "image/jpeg",
}
REQUIRED_ARTIFACTS = {
    "geometry": ("point_cloud.glb",),
    "full": ("point_cloud.glb", "instance_masks.mp4"),
}
OBJECT_CATALOG_ARTIFACTS = (
    OBJECT_CATALOG_JSON_NAME,
    OBJECT_CROPS_ATLAS_NAME,
)
OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE = "validation_object_catalog_transport_v1"
OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY = "synthetic_transport_canary"
RUNPOD_ANALYSIS_TYPE_MODES = {
    "validation_v1": "dry_run",
    OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE: "dry_run",
    "geometry_vggt_1b": "geometry",
    VGGT_OMEGA_ANALYSIS_TYPE: "geometry",
    "dedup_ras_vggt_sam3": "full",
}
NON_RUNPOD_ANALYSIS_TYPES = {
    "mask_sam3": "route it to the fal SAM 3 adapter",
    "mask_sam31": "route it to the fal SAM 3.1 adapter",
}
VGGT_1B_ANALYSIS_TYPES = frozenset({"geometry_vggt_1b", "dedup_ras_vggt_sam3"})
_PROCESS_INITIALIZATION_LOCK = threading.Lock()


def _source_frame_plan(
    video_path: Path,
    sampled_count: int,
) -> tuple[list[int], list[float] | None] | None:
    """Build indices and presentation times from one decoded-frame listing."""
    import numpy as np

    frames = _ffprobe_json(
        video_path,
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time,pkt_dts_time",
    ) or {}
    entries = frames.get("frames") or []
    if not isinstance(entries, list) or not entries or sampled_count <= 0:
        return None
    count = min(sampled_count, len(entries))
    indices = np.linspace(0, len(entries) - 1, count).astype(int).tolist()

    sampled_timestamps: list[float] = []
    for index in indices:
        entry = entries[index]
        if not isinstance(entry, dict):
            return indices, None
        timestamp = next(
            (
                parsed
                for field in ("best_effort_timestamp_time", "pts_time", "pkt_dts_time")
                if (parsed := _finite_number(entry.get(field))) is not None
            ),
            None,
        )
        if timestamp is None:
            return indices, None
        sampled_timestamps.append(float(timestamp))
    origin = sampled_timestamps[0]
    normalized = [max(0.0, timestamp - origin) for timestamp in sampled_timestamps]
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        return indices, None
    return indices, normalized


def _sampled_source_frame_indices(video_path: Path, sampled_count: int) -> list[int] | None:
    """Mirror upstream np.linspace sampling over FFmpeg-decoded frames."""
    plan = _source_frame_plan(video_path, sampled_count)
    return plan[0] if plan else None


def _canonical_source_frame_timestamps(values: Any) -> list[float] | None:
    """Use the catalog's six-decimal timeline representation everywhere."""
    if not isinstance(values, (list, tuple)):
        return None
    canonical: list[float] = []
    for raw_value in values:
        value = _finite_number(raw_value)
        if value is None or value < 0:
            return None
        canonical.append(round(value, 6))
    if any(right <= left for left, right in zip(canonical, canonical[1:])):
        return None
    return canonical



def _run(cmd: list[str], *, check: bool = True) -> int:
    import subprocess

    print(f"[stage2] $ {' '.join(cmd)}", flush=True)
    return subprocess.check_call(cmd) if check else subprocess.call(cmd)


def _pip_install(*args: str) -> None:
    _run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *args])


def _ras_root() -> Path:
    return Path(os.environ.get("RAS_ROOT", Path(__file__).resolve().parent / "vendor" / "ReplicateAnyScene")).resolve()


def _models_dir(ras: Path | None = None) -> Path:
    ras = ras or _ras_root()
    return Path(os.environ.get("STAGE2_MODELS_DIR", ras / "models")).resolve()


def _checkout_state(dest: Path, *, ignore_submodules: bool = False) -> tuple[str, bool]:
    """Return checkout HEAD and tracked-file cleanliness without trusting stderr."""
    import subprocess

    if not dest.is_dir() or not (dest / ".git").exists():
        return "", False
    try:
        head = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff_cmd = ["git", "-C", str(dest), "diff-index", "--quiet"]
        if ignore_submodules:
            diff_cmd.append("--ignore-submodules=all")
        diff_cmd.extend(["HEAD", "--"])
        clean = subprocess.call(
            diff_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) == 0
    except (OSError, subprocess.SubprocessError):
        return "", False
    return head, clean


def _fetch_pinned_revision(repo: Path, revision: str, *, attempts: int = 3) -> None:
    """Fetch one immutable public revision with bounded transient retries."""
    import subprocess

    command = [
        "git",
        "-C",
        str(repo),
        "fetch",
        "--depth",
        "1",
        "origin",
        revision,
    ]
    last_detail = ""
    for attempt in range(1, attempts + 1):
        print(f"[stage2] $ {' '.join(command)} (attempt {attempt}/{attempts})", flush=True)
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                check=False,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            if result.returncode == 0:
                return
            detail_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
            last_detail = " | ".join(detail_lines[-4:])[-800:]
        except subprocess.TimeoutExpired:
            last_detail = "fetch timed out after 120 seconds"
        except OSError as exc:
            last_detail = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(2 ** (attempt - 1))
    suffix = f" Last error: {last_detail}" if last_detail else ""
    raise RuntimeError(
        f"git fetch of pinned revision failed after {attempts} attempts.{suffix}"
    )


def _build_pinned_checkout(url: str, dest: Path, revision: str) -> None:
    """Build and verify a checkout in an otherwise empty staging directory."""
    _run(["git", "init", "-q", str(dest)])
    _run(["git", "-C", str(dest), "remote", "add", "origin", url])
    _fetch_pinned_revision(dest, revision)
    _run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"])
    head, clean = _checkout_state(dest)
    if head != revision or not clean:
        raise RuntimeError(f"staged source checkout did not verify pinned revision {revision}")


def _remove_runtime_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _runtime_path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _recover_interrupted_checkout(dest: Path, backup: Path) -> None:
    """Recover the old tree if a prior worker stopped between the two renames."""
    if not _runtime_path_exists(backup):
        return
    if not _runtime_path_exists(dest):
        os.replace(backup, dest)
        return
    try:
        _remove_runtime_path(backup)
    except OSError as exc:
        print(
            f"[stage2] warning: could not remove stale checkout backup ({type(exc).__name__})",
            flush=True,
        )


def _clone_if_missing(
    url: str,
    dest: Path,
    revision: str,
    *,
    ignore_submodules: bool = False,
) -> None:
    """Install an exact checkout without exposing the final path to partial clones."""
    backup = dest.parent / f".{dest.name}.previous"
    _recover_interrupted_checkout(dest, backup)
    head, clean = _checkout_state(dest, ignore_submodules=ignore_submodules)
    if head == revision and clean:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{dest.name}.checkout-", dir=str(dest.parent))
    )
    try:
        _build_pinned_checkout(url, staged, revision)

        # Another initializer may have completed while this checkout was being
        # staged. Preserve its verified result rather than replacing live work.
        head, clean = _checkout_state(dest, ignore_submodules=ignore_submodules)
        if head == revision and clean:
            return

        # Network failures happen before this point. Publish through a
        # recoverable sibling rename so even a local filesystem error restores
        # the previous runtime, and a killed worker can recover it next time.
        had_previous = _runtime_path_exists(dest)
        if _runtime_path_exists(backup):
            _remove_runtime_path(backup)
        if had_previous:
            os.replace(dest, backup)
        try:
            os.replace(staged, dest)
        except BaseException:
            if had_previous and not _runtime_path_exists(dest) and _runtime_path_exists(backup):
                os.replace(backup, dest)
            raise
        if _runtime_path_exists(backup):
            try:
                _remove_runtime_path(backup)
            except OSError as exc:
                print(
                    f"[stage2] warning: could not remove replaced checkout backup ({type(exc).__name__})",
                    flush=True,
                )
    finally:
        _remove_runtime_path(staged)


def _verified_checkout_revision(
    dest: Path,
    expected_revision: str,
    *,
    ignore_submodules: bool = False,
) -> str:
    """Return the actual clean checkout revision or fail before claiming provenance."""
    head, clean = _checkout_state(dest, ignore_submodules=ignore_submodules)
    if head != expected_revision or not clean:
        raise RuntimeError(
            f"source checkout at {dest} does not match clean revision {expected_revision}"
        )
    return head


def _prefer_source_checkouts(ras: Path) -> None:
    """Make the verified runtime-owned source trees win over stale site packages."""
    for source_root in (ras / "sam3", ras / "vggt", ras):
        if not source_root.is_dir():
            continue
        value = str(source_root)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
        # These revision-scoped paths may not exist when the worker process
        # starts. Python caches a failed path lookup, so invalidate that entry
        # after the checkout is created on the network volume.
        sys.path_importer_cache.pop(value, None)
    importlib.invalidate_caches()

    vggt_package = ras / "vggt" / "vggt"
    if vggt_package.is_dir():
        _activate_source_namespace("vggt", vggt_package)


def _activate_source_namespace(package_name: str, package_dir: Path) -> None:
    """Pin a PEP 420 package to one reviewed source checkout.

    VGGT intentionally has no ``__init__.py``. A regular package with the same
    name in the base image otherwise wins over that namespace even when the
    checkout is first on ``sys.path``. Build the namespace explicitly so both
    first-job imports and provenance checks resolve the revision-scoped tree.
    """
    if not package_dir.is_dir() or (package_dir / "__init__.py").exists():
        raise RuntimeError(
            f"{package_name} source namespace has an unexpected layout at {package_dir}"
        )
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    if spec is None:
        raise RuntimeError(f"could not create source namespace for {package_name}")
    spec.submodule_search_locations = [str(package_dir.resolve())]
    sys.modules[package_name] = importlib.util.module_from_spec(spec)


def _verify_import_from_checkout(module_name: str, checkout: Path) -> None:
    module = importlib.import_module(module_name)
    module_file_value = getattr(module, "__file__", None)
    if not isinstance(module_file_value, str) or not module_file_value:
        raise RuntimeError(
            f"{module_name} did not resolve to a concrete file inside {checkout}"
        )
    module_file = Path(module_file_value).resolve()
    try:
        module_file.relative_to(checkout.resolve())
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            f"{module_name} resolved outside the verified source checkout at {checkout}"
        ) from exc


@contextmanager
def _stage2_initialization_lock(models_dir: Path):
    """Serialize source/package/weight initialization across worker processes."""
    models_dir.mkdir(parents=True, exist_ok=True)
    lock_path = models_dir / ".stage2_initialization.lock"
    with _PROCESS_INITIALIZATION_LOCK:
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _find_sam3_pt(sam_dir: Path) -> Path | None:
    direct = sam_dir / "sam3.pt"
    if direct.is_file() and direct.stat().st_size > 1_000_000:
        return direct
    if not sam_dir.is_dir():
        return None
    for p in sorted(sam_dir.rglob("sam3.pt")):
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
    return None


def _vggt_model_id() -> str:
    """Research checkpoint by default; production must explicitly select Commercial."""
    return os.environ.get("VGGT_MODEL_ID", DEFAULT_VGGT_MODEL_ID).strip() or DEFAULT_VGGT_MODEL_ID


def _vggt_license_scope(model_id: str | None = None) -> str:
    return "commercial" if (model_id or _vggt_model_id()) == COMMERCIAL_VGGT_MODEL_ID else "research_noncommercial"


def _hugging_face_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _vggt_weights_ok(vggt_dir: Path, expected_model_id: str | None = None) -> bool:
    if not vggt_dir.is_dir():
        return False
    if expected_model_id:
        marker = vggt_dir / VGGT_MODEL_MARKER
        # Existing endpoint volumes predate the marker and contain the original
        # VGGT-1B checkpoint. Reuse those for the internal demo; an explicit
        # switch to Commercial still requires a matching marker/download.
        if marker.is_file() and marker.read_text().strip() != expected_model_id:
            return False
        if not marker.is_file() and expected_model_id != DEFAULT_VGGT_MODEL_ID:
            return False
    for name in ("model.safetensors", "model.pt", "config.json"):
        if (vggt_dir / name).is_file():
            # need at least one weight file + preferably config
            if name.startswith("model"):
                return True
    # nested layout
    for p in vggt_dir.rglob("model.safetensors"):
        if p.is_file() and p.stat().st_size > 1_000_000:
            return True
    for p in vggt_dir.rglob("model.pt"):
        if p.is_file() and p.stat().st_size > 1_000_000:
            return True
    return False


def _weights_ready(models_dir: Path) -> bool:
    vggt_ok = _vggt_weights_ok(models_dir / "VGGT", _vggt_model_id())
    sam_pt = _find_sam3_pt(models_dir / "SAM3")
    return bool(vggt_ok and sam_pt)


def _ensure_sam3_pt_layout(models_dir: Path) -> Path:
    """RAS models.py expects ./models/SAM3/sam3.pt."""
    sam_dir = models_dir / "SAM3"
    sam_dir.mkdir(parents=True, exist_ok=True)
    found = _find_sam3_pt(sam_dir)
    if found is None:
        raise RuntimeError(
            "SAM3 weights missing: expected sam3.pt under STAGE2_MODELS_DIR/SAM3. "
            "Accept the facebook/sam3 license on Hugging Face with the endpoint HF_TOKEN, "
            "then re-run full mode so weights can download."
        )
    target = sam_dir / "sam3.pt"
    if found.resolve() != target.resolve():
        if target.is_symlink() or target.exists():
            target.unlink()
        try:
            target.symlink_to(found.resolve())
        except OSError:
            shutil.copy2(found, target)
    return target


def _download_vggt_weights(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    vggt_dir = models_dir / "VGGT"
    model_id = _vggt_model_id()
    if not _vggt_weights_ok(vggt_dir, model_id):
        # The default research checkpoint is public. Keep geometry independent
        # from SAM3 approval and from any stale/revoked endpoint credential by
        # always downloading it anonymously. Explicit alternate checkpoints
        # (including the gated Commercial model) require real credentials.
        token = False if model_id == DEFAULT_VGGT_MODEL_ID else _hugging_face_token()
        if model_id != DEFAULT_VGGT_MODEL_ID and token is None:
            raise RuntimeError(
                f"{model_id} requires an approved Hugging Face token in HF_TOKEN"
            )
        from huggingface_hub import snapshot_download

        print(f"[stage2] downloading {model_id} ...", flush=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(vggt_dir),
            token=token,
        )
        (vggt_dir / VGGT_MODEL_MARKER).write_text(model_id + "\n")


def _download_sam3_weights(models_dir: Path) -> None:
    vggt_dir = models_dir / "VGGT"
    sam_dir = models_dir / "SAM3"
    if not _find_sam3_pt(sam_dir):
        token = _hugging_face_token()
        if token is None:
            raise RuntimeError(
                "SAM3 weights are gated and require an approved Hugging Face token in HF_TOKEN"
            )
        from huggingface_hub import snapshot_download

        print("[stage2] downloading facebook/sam3 ...", flush=True)
        try:
            snapshot_download(
                repo_id="facebook/sam3",
                local_dir=str(sam_dir),
                token=token,
            )
        except Exception as e:
            # clear false ready markers from older builds
            for marker in (vggt_dir / ".stage2_ready", sam_dir / ".stage2_ready"):
                if marker.exists():
                    marker.unlink()
            raise RuntimeError(
                "Failed to download facebook/sam3 (gated). "
                "Log into Hugging Face with the same token as HF_TOKEN on the endpoint, "
                f"accept the SAM3 license at https://huggingface.co/facebook/sam3, then retry. Detail: {e}"
            ) from e

    if not _weights_ready(models_dir):
        for marker in (vggt_dir / ".stage2_ready", sam_dir / ".stage2_ready"):
            if marker.exists():
                marker.unlink()
        raise RuntimeError(
            f"Required model weights are incomplete under {models_dir}. "
            f"VGGT ok={_vggt_weights_ok(vggt_dir, _vggt_model_id())} "
            f"model={_vggt_model_id()} SAM3 pt={_find_sam3_pt(sam_dir)}"
        )

    _ensure_sam3_pt_layout(models_dir)
    (vggt_dir / ".stage2_ready").touch()
    (sam_dir / ".stage2_ready").touch()
    print("[stage2] weights ready", flush=True)


def _ensure_python_packages(ras: Path, *, require_sam3: bool) -> None:
    """Clone is not enough: RAS imports require pip-installed vggt + sam3 packages."""
    import importlib.util

    # Host deps used by RAS Stage-2 modules (not Stage-3 sam-3d-objects).
    common_modules = ("einops", "safetensors", "scipy", "trimesh", "colorcet", "matplotlib", "omegaconf", "hydra", "transformers", "timm", "ftfy", "regex", "iopath", "huggingface_hub", "PIL")
    if require_sam3:
        common_modules += ("open3d",)
    # Use find_spec instead of importing optional packages before installation.
    # huggingface_hub caches optional dependency availability when imported;
    # importing it before safetensors exists makes first-job from_pretrained fail.
    missing_common = [module for module in common_modules if importlib.util.find_spec(module) is None]
    if missing_common:
        _pip_install(
            "numpy<2", "einops", "safetensors", "scipy", "trimesh", "colorcet",
            "matplotlib", "omegaconf", "hydra-core", "transformers", "timm>=1.0.17",
            "ftfy==6.1.1", "regex", "iopath>=0.1.10", "huggingface_hub>=0.23",
            "Pillow", *(("open3d",) if require_sam3 else ()),
        )

    # Install source packages so `import vggt` / `import sam3` resolve.
    vggt_py = ras / "vggt" / "pyproject.toml"
    sam_py = ras / "sam3" / "pyproject.toml"
    if not vggt_py.is_file():
        raise RuntimeError(f"vggt tree missing pyproject at {vggt_py}")
    if require_sam3 and not sam_py.is_file():
        raise RuntimeError(f"sam3 tree missing pyproject at {sam_py}")

    def _import_error(mod: str) -> str | None:
        try:
            __import__(mod)
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    if _import_error("vggt.models.vggt"):
        print("[stage2] pip install vggt ...", flush=True)
        # A runtime editable install relies on a .pth file that Python only
        # reads at process startup. Install a normal wheel so this same job can
        # import VGGT immediately after first-time bootstrap.
        _pip_install(str(ras / "vggt"))
        _activate_source_namespace("vggt", ras / "vggt" / "vggt")
    if require_sam3 and _import_error("sam3.model_builder"):
        print("[stage2] pip install sam3 ...", flush=True)
        # non-editable install is more reliable on network volumes
        _pip_install(str(ras / "sam3"))

    vggt_error = _import_error("vggt.models.vggt")
    if vggt_error:
        raise RuntimeError(
            "vggt package still not importable after pip install. "
            f"Check {ras / 'vggt'} layout (expected package dir vggt/vggt). Detail: {vggt_error}"
        )
    sam3_error = _import_error("sam3.model_builder") if require_sam3 else None
    if sam3_error:
        raise RuntimeError(
            "sam3 package still not importable after pip install. "
            f"Check {ras / 'sam3'} layout. Detail: {sam3_error}"
        )
    packages = "vggt + sam3" if require_sam3 else "vggt"
    print(f"[stage2] python packages importable ({packages})", flush=True)


def _ensure_ras_installed(
    *,
    require_sam3: bool = True,
    ensure_weights: bool = True,
) -> None:
    """Install paper repo + packages + weights (first full call, then cached on volume)."""
    ras = _ras_root()
    models_dir = _models_dir(ras)

    with _stage2_initialization_lock(models_dir):
        # VGGT and SAM3 are managed as separately verified pinned checkouts.
        # Their intentional revisions must not make the parent RAS gitlinks
        # look dirty and trigger a destructive parent re-clone on every job.
        _clone_if_missing(
            "https://github.com/xiac20/ReplicateAnyScene.git",
            ras,
            RAS_REVISION,
            ignore_submodules=True,
        )
        # Keep the exact upstream gitlinks used by the reviewed RAS revision.
        _clone_if_missing("https://github.com/facebookresearch/vggt.git", ras / "vggt", VGGT_REVISION)
        if require_sam3:
            _clone_if_missing("https://github.com/facebookresearch/sam3.git", ras / "sam3", SAM3_REVISION)

        _prefer_source_checkouts(ras)

        # Install dependencies before importing huggingface_hub to download weights.
        _ensure_python_packages(ras, require_sam3=require_sam3)
        # VGGT's top-level `vggt` is a PEP 420 namespace package and has no
        # __file__. Verify the concrete model modules that inference imports.
        _verify_import_from_checkout("vggt.models.vggt", ras / "vggt")
        if require_sam3:
            _verify_import_from_checkout("sam3.model_builder", ras / "sam3")

        if ensure_weights and not _vggt_weights_ok(models_dir / "VGGT", _vggt_model_id()):
            _download_vggt_weights(models_dir)

        # Older builds wrote .stage2_ready even when SAM3 download failed — ignore markers alone.
        if ensure_weights and require_sam3 and not _weights_ready(models_dir):
            for marker in (models_dir / "VGGT" / ".stage2_ready", models_dir / "SAM3" / ".stage2_ready"):
                if marker.exists():
                    marker.unlink()
            _download_sam3_weights(models_dir)
        elif ensure_weights and require_sam3:
            _ensure_sam3_pt_layout(models_dir)


def _ensure_ras_on_path() -> Path:
    ras = _ras_root()
    if not ras.is_dir() or not (ras / "main.py").is_file():
        raise RuntimeError(
            f"ReplicateAnyScene checkout not found at {ras}. "
            "Clone https://github.com/xiac20/ReplicateAnyScene and install vggt+sam3 packages."
        )
    _prefer_source_checkouts(ras)
    # Models loader expects cwd-relative ./models and ./sam3 paths.
    os.chdir(ras)
    return ras


def _download_video(video_url: str, dest_dir: Path, timeout_s: int = 180) -> Path:
    import requests

    parsed_url = urlparse(video_url)
    if (
        parsed_url.scheme.lower() != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise RuntimeError("video_url must be a credential-free HTTPS URL")

    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed_url.path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        suffix = ".mp4"
    dest = dest_dir / f"input{suffix}"
    # This path is also callable outside Agent Lab, so enforce the worker's own
    # disk boundary instead of trusting the signed-URL issuer or response header.
    dest.unlink(missing_ok=True)
    try:
        # Signed input URLs are exact-object capabilities. Following a redirect
        # would silently hand that capability (and any future custom headers)
        # to a different origin, so every 3xx is terminal.
        with requests.get(
            video_url,
            stream=True,
            timeout=timeout_s,
            allow_redirects=False,
        ) as r:
            if 300 <= r.status_code < 400:
                raise RuntimeError("video URL redirects are not allowed")
            if r.status_code in (401, 403):
                raise RuntimeError(
                    f"video URL blocked (HTTP {r.status_code}). "
                    "Use an exact-object signed HTTPS URL accessible to the worker."
                )
            r.raise_for_status()
            declared_raw = r.headers.get("content-length")
            if declared_raw is not None:
                try:
                    declared = int(declared_raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("video download returned an invalid Content-Length") from exc
                if declared <= 0:
                    raise RuntimeError("video download returned an empty file")
                if declared > MAX_REMOTE_VIDEO_BYTES:
                    raise RuntimeError(
                        f"video download exceeds the {MAX_REMOTE_VIDEO_BYTES // 1024 // 1024} MiB service limit"
                    )

            written = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > MAX_REMOTE_VIDEO_BYTES:
                        raise RuntimeError(
                            f"video download exceeds the {MAX_REMOTE_VIDEO_BYTES // 1024 // 1024} MiB service limit"
                        )
                    f.write(chunk)
            if written == 0:
                raise RuntimeError("video download returned an empty file")
    except requests.RequestException as exc:
        dest.unlink(missing_ok=True)
        # Requests exceptions may contain the complete URL, including its
        # signed query. Return only the exception class to the endpoint caller.
        raise RuntimeError(f"video download failed ({type(exc).__name__})") from exc
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest


def _materialize_video(payload: dict[str, Any], work: Path) -> Path:
    """Accept a bounded remote video or the legacy bounded inline payload."""
    b64 = payload.get("video_b64")
    video_url = str(payload.get("video_url") or "").strip()

    # Prefer inline bytes. Never pass fake schemes (inline://) to requests.
    if isinstance(b64, str) and b64.strip():
        try:
            raw = base64.b64decode(b64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RuntimeError("video_b64 is not valid base64") from exc
        if not raw:
            raise RuntimeError("video_b64 decoded to an empty file")
        if len(raw) > MAX_INLINE_VIDEO_BYTES:
            raise RuntimeError("inline video exceeds the 6 MiB service limit")
        # sniff extension from media_type if provided
        mt = str(payload.get("media_type") or "video/mp4").lower()
        ext = ".webm" if "webm" in mt else ".mov" if "quicktime" in mt or "mov" in mt else ".mp4"
        dest = work / f"input{ext}"
        dest.write_bytes(raw)
        return dest

    if not video_url:
        raise RuntimeError("video_url or video_b64 is required")
    if video_url.startswith("inline:") or video_url.startswith("data:"):
        raise RuntimeError(
            "received inline video_url without video_b64 "
            "(worker may be stale, or base64 was stripped as too large). "
            "Redeploy/restart the endpoint worker, send up to 6 MiB through legacy video_b64, "
            "or provide an HTTPS video_url up to 64 MiB."
        )
    return _download_video(video_url, work)


class VggtOmegaSpaceError(RuntimeError):
    """Safe, structured failure from the official VGGT-Omega Space adapter."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _omega_space_headers(*, accept: str = "application/json") -> dict[str, str]:
    headers = {"Accept": accept}
    token = _hugging_face_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _close_http_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _verify_vggt_omega_space_revision() -> dict[str, str]:
    """Fail closed unless the public Space and its running replica are reviewed."""
    import requests

    response = None
    try:
        response = requests.get(
            VGGT_OMEGA_METADATA_URL,
            headers=_omega_space_headers(),
            timeout=_env_int("STAGE2_OMEGA_METADATA_TIMEOUT_SECONDS", 30, 5, 120),
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise VggtOmegaSpaceError(
                "omega_space_revision_unverified",
                "Official VGGT-Omega Space metadata redirected; revision was not verified.",
            )
        if response.status_code != 200:
            raise VggtOmegaSpaceError(
                "omega_space_revision_unverified",
                f"Official VGGT-Omega Space metadata returned HTTP {response.status_code}; revision was not verified.",
            )
        try:
            metadata = response.json()
        except (TypeError, ValueError) as exc:
            raise VggtOmegaSpaceError(
                "omega_space_revision_unverified",
                "Official VGGT-Omega Space metadata was not valid JSON; revision was not verified.",
            ) from exc
    except VggtOmegaSpaceError:
        raise
    except requests.RequestException as exc:
        raise VggtOmegaSpaceError(
            "omega_space_revision_unverified",
            f"Official VGGT-Omega Space metadata request failed ({type(exc).__name__}); revision was not verified.",
        ) from exc
    finally:
        if response is not None:
            _close_http_response(response)

    runtime = metadata.get("runtime") if isinstance(metadata, dict) else None
    repo_revision = metadata.get("sha") if isinstance(metadata, dict) else None
    runtime_revision = runtime.get("sha") if isinstance(runtime, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("id") != VGGT_OMEGA_SPACE_ID
        or repo_revision != VGGT_OMEGA_SPACE_REVISION
        or runtime_revision != VGGT_OMEGA_SPACE_REVISION
        or runtime.get("stage") != "RUNNING"
    ):
        raise VggtOmegaSpaceError(
            "omega_space_revision_mismatch",
            "Official VGGT-Omega Space revision changed; this adapter refuses unreviewed model execution.",
        )
    return {
        "space_id": VGGT_OMEGA_SPACE_ID,
        "space_revision": VGGT_OMEGA_SPACE_REVISION,
    }


def _sample_vggt_omega_frames(
    video_path: Path,
    target_dir: Path,
    requested_max_frames: int,
) -> tuple[list[Path], list[int], list[float] | None]:
    """Decode one exact, uniform frame plan and cap the hosted call at 24 images."""
    import cv2

    sample_count = min(max(2, int(requested_max_frames)), VGGT_OMEGA_MAX_FRAMES)
    plan = _source_frame_plan(video_path, sample_count)
    if plan is None:
        raise VggtOmegaSpaceError(
            "omega_frame_sampling_failed",
            "Could not enumerate decoded source frames for exact VGGT-Omega sampling.",
        )
    source_indices, source_timestamps = plan
    if (
        len(source_indices) < 2
        or len(source_indices) > VGGT_OMEGA_MAX_FRAMES
        or source_indices != sorted(set(source_indices))
    ):
        raise VggtOmegaSpaceError(
            "omega_frame_sampling_failed",
            "Source video did not provide a valid exact VGGT-Omega frame plan.",
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VggtOmegaSpaceError(
            "omega_frame_sampling_failed",
            "Could not decode the source video for VGGT-Omega.",
        )

    paths: list[Path] = []
    target_lookup = {frame_index: ordinal for ordinal, frame_index in enumerate(source_indices)}
    last_index = source_indices[-1]
    decoded_index = 0
    try:
        while decoded_index <= last_index:
            ok, frame = capture.read()
            if not ok:
                break
            ordinal = target_lookup.get(decoded_index)
            if ordinal is not None:
                path = target_dir / f"{ordinal:06d}.jpg"
                saved = cv2.imwrite(
                    str(path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
                if not saved or not path.is_file() or path.stat().st_size <= 0:
                    raise VggtOmegaSpaceError(
                        "omega_frame_sampling_failed",
                        "Could not encode an exact sampled frame for VGGT-Omega.",
                    )
                paths.append(path)
            decoded_index += 1
    finally:
        capture.release()

    if len(paths) != len(source_indices):
        raise VggtOmegaSpaceError(
            "omega_frame_sampling_failed",
            "Source video ended before all exact VGGT-Omega sample frames were decoded.",
        )
    return paths, source_indices, source_timestamps


def _upload_vggt_omega_frames(frame_paths: list[Path]) -> list[dict[str, Any]]:
    """Upload only the exact sampled frames to the reviewed Gradio origin."""
    import requests

    if len(frame_paths) < 2 or len(frame_paths) > VGGT_OMEGA_MAX_FRAMES:
        raise VggtOmegaSpaceError(
            "omega_frame_upload_failed",
            "VGGT-Omega frame upload requires between 2 and 24 sampled images.",
        )

    response = None
    try:
        with ExitStack() as stack:
            files = [
                (
                    "files",
                    (
                        path.name,
                        stack.enter_context(path.open("rb")),
                        "image/jpeg",
                    ),
                )
                for path in frame_paths
            ]
            response = requests.post(
                f"{VGGT_OMEGA_SPACE_ORIGIN}/gradio_api/upload",
                headers=_omega_space_headers(),
                files=files,
                timeout=_env_int("STAGE2_OMEGA_UPLOAD_TIMEOUT_SECONDS", 180, 30, 600),
                allow_redirects=False,
            )
        if 300 <= response.status_code < 400:
            raise VggtOmegaSpaceError(
                "omega_frame_upload_failed",
                "Official VGGT-Omega Space frame upload redirected and was rejected.",
            )
        if response.status_code != 200:
            raise VggtOmegaSpaceError(
                "omega_frame_upload_failed",
                f"Official VGGT-Omega Space frame upload returned HTTP {response.status_code}.",
            )
        try:
            remote_paths = response.json()
        except (TypeError, ValueError) as exc:
            raise VggtOmegaSpaceError(
                "omega_frame_upload_failed",
                "Official VGGT-Omega Space frame upload returned invalid JSON.",
            ) from exc
    except VggtOmegaSpaceError:
        raise
    except requests.RequestException as exc:
        raise VggtOmegaSpaceError(
            "omega_frame_upload_failed",
            f"Official VGGT-Omega Space frame upload failed ({type(exc).__name__}).",
        ) from exc
    finally:
        if response is not None:
            _close_http_response(response)

    if not isinstance(remote_paths, list) or len(remote_paths) != len(frame_paths):
        raise VggtOmegaSpaceError(
            "omega_frame_upload_failed",
            "Official VGGT-Omega Space did not acknowledge every sampled frame.",
        )

    uploaded: list[dict[str, Any]] = []
    for local_path, remote_path in zip(frame_paths, remote_paths):
        if (
            not isinstance(remote_path, str)
            or not remote_path
            or len(remote_path) > 4096
            or "\n" in remote_path
            or "\r" in remote_path
        ):
            raise VggtOmegaSpaceError(
                "omega_frame_upload_failed",
                "Official VGGT-Omega Space returned an invalid sampled-frame handle.",
            )
        uploaded.append(
            {
                "path": remote_path,
                "orig_name": local_path.name,
                "mime_type": "image/jpeg",
                "meta": {"_type": "gradio.FileData"},
            }
        )
    return uploaded


def _submit_vggt_omega_gradio(api_name: str, data: list[Any]) -> str:
    import requests

    if api_name not in {"update_gallery_on_upload", "gradio_demo"}:
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            "Unsupported official VGGT-Omega Space API operation.",
        )
    response = None
    try:
        response = requests.post(
            f"{VGGT_OMEGA_SPACE_ORIGIN}/gradio_api/call/{api_name}",
            headers={**_omega_space_headers(), "Content-Type": "application/json"},
            json={"data": data},
            timeout=_env_int("STAGE2_OMEGA_SUBMIT_TIMEOUT_SECONDS", 60, 10, 180),
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise VggtOmegaSpaceError(
                "omega_space_protocol_failed",
                f"Official VGGT-Omega Space {api_name} submission redirected and was rejected.",
            )
        if response.status_code != 200:
            raise VggtOmegaSpaceError(
                "omega_space_protocol_failed",
                f"Official VGGT-Omega Space {api_name} submission returned HTTP {response.status_code}.",
            )
        try:
            value = response.json()
        except (TypeError, ValueError) as exc:
            raise VggtOmegaSpaceError(
                "omega_space_protocol_failed",
                f"Official VGGT-Omega Space {api_name} submission returned invalid JSON.",
            ) from exc
    except VggtOmegaSpaceError:
        raise
    except requests.RequestException as exc:
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            f"Official VGGT-Omega Space {api_name} submission failed ({type(exc).__name__}).",
        ) from exc
    finally:
        if response is not None:
            _close_http_response(response)

    event_id = value.get("event_id") if isinstance(value, dict) else None
    if not isinstance(event_id, str) or not _VGGT_OMEGA_EVENT_ID_RE.fullmatch(event_id):
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            f"Official VGGT-Omega Space {api_name} submission returned an invalid event id.",
        )
    return event_id


def _wait_vggt_omega_gradio(
    api_name: str,
    event_id: str,
    *,
    deadline: float | None = None,
) -> list[Any]:
    import requests

    if api_name not in {"update_gallery_on_upload", "gradio_demo"}:
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            "Unsupported official VGGT-Omega Space API operation.",
        )
    if not _VGGT_OMEGA_EVENT_ID_RE.fullmatch(event_id):
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            "Official VGGT-Omega Space event id was invalid.",
        )

    timeout_seconds = _env_int("STAGE2_OMEGA_RESULT_TIMEOUT_SECONDS", 900, 60, 1200)
    if deadline is None:
        deadline = time.monotonic() + timeout_seconds

    def timeout_error() -> VggtOmegaSpaceError:
        return VggtOmegaSpaceError(
            "omega_space_result_timeout",
            f"Official VGGT-Omega Space {api_name} result exceeded its total wait deadline.",
        )

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise timeout_error()

    response = None
    deadline_timer: threading.Timer | None = None
    deadline_expired = threading.Event()
    try:
        response = requests.get(
            f"{VGGT_OMEGA_SPACE_ORIGIN}/gradio_api/call/{api_name}/{event_id}",
            headers=_omega_space_headers(accept="text/event-stream"),
            stream=True,
            timeout=max(0.1, min(float(timeout_seconds), remaining)),
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise VggtOmegaSpaceError(
                "omega_space_protocol_failed",
                f"Official VGGT-Omega Space {api_name} result redirected and was rejected.",
            )
        if response.status_code != 200:
            raise VggtOmegaSpaceError(
                "omega_space_protocol_failed",
                f"Official VGGT-Omega Space {api_name} result returned HTTP {response.status_code}.",
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise timeout_error()

        # `requests` timeouts limit one idle socket read, not the complete SSE
        # lifetime. Gradio emits heartbeat lines while a job is queued, so a
        # read timeout alone can keep a paid RunPod worker alive indefinitely.
        # Close the stream at the monotonic deadline and also check that
        # deadline on every provider line.
        def expire_response() -> None:
            deadline_expired.set()
            if response is not None:
                _close_http_response(response)

        deadline_timer = threading.Timer(remaining, expire_response)
        deadline_timer.daemon = True
        deadline_timer.start()

        event_name = ""
        for raw_line in response.iter_lines(decode_unicode=True):
            if deadline_expired.is_set() or time.monotonic() >= deadline:
                raise timeout_error()
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip().lower()
                continue
            if not line.startswith("data:"):
                continue
            if event_name == "error":
                raise VggtOmegaSpaceError(
                    "omega_space_execution_failed",
                    f"Official VGGT-Omega Space {api_name} job failed.",
                )
            if event_name != "complete":
                continue
            try:
                result = json.loads(line.split(":", 1)[1].strip())
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise VggtOmegaSpaceError(
                    "omega_space_protocol_failed",
                    f"Official VGGT-Omega Space {api_name} result was invalid JSON.",
                ) from exc
            if not isinstance(result, list):
                raise VggtOmegaSpaceError(
                    "omega_space_protocol_failed",
                    f"Official VGGT-Omega Space {api_name} result had an invalid shape.",
                )
            if deadline_expired.is_set() or time.monotonic() >= deadline:
                raise timeout_error()
            return result
        if deadline_expired.is_set() or time.monotonic() >= deadline:
            raise timeout_error()
    except VggtOmegaSpaceError:
        raise
    except requests.RequestException as exc:
        if deadline_expired.is_set() or time.monotonic() >= deadline:
            raise timeout_error() from exc
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            f"Official VGGT-Omega Space {api_name} result failed ({type(exc).__name__}).",
        ) from exc
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
        if response is not None:
            _close_http_response(response)

    raise VggtOmegaSpaceError(
        "omega_space_protocol_failed",
        f"Official VGGT-Omega Space {api_name} result stream ended before completion.",
    )


def _vggt_omega_glb_url(file_data: Any) -> str:
    value = file_data.get("url") if isinstance(file_data, dict) else None
    metadata = file_data.get("meta") if isinstance(file_data, dict) else None
    if (
        not isinstance(value, str)
        or not value
        or not isinstance(metadata, dict)
        or metadata.get("_type") != "gradio.FileData"
    ):
        raise VggtOmegaSpaceError(
            "omega_glb_invalid",
            "Official VGGT-Omega Space did not return a downloadable GLB artifact.",
        )
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise VggtOmegaSpaceError(
            "omega_glb_host_rejected",
            "Official VGGT-Omega Space returned a GLB URL with an invalid host.",
        ) from exc
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != VGGT_OMEGA_SPACE_HOST
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/gradio_api/file=")
        or parsed.fragment
    ):
        raise VggtOmegaSpaceError(
            "omega_glb_host_rejected",
            "Official VGGT-Omega Space returned a GLB URL outside the reviewed artifact host.",
        )
    return value


def _validate_vggt_omega_glb(path: Path) -> int:
    size = path.stat().st_size if path.is_file() else 0
    if size < 12:
        raise VggtOmegaSpaceError(
            "omega_glb_invalid",
            "Official VGGT-Omega Space returned an empty or truncated GLB artifact.",
        )
    if size > GLB_MAX_BYTES:
        raise VggtOmegaSpaceError(
            "omega_glb_too_large",
            f"Official VGGT-Omega GLB exceeds the {GLB_MAX_BYTES}-byte artifact contract.",
        )
    with path.open("rb") as handle:
        header = handle.read(12)
        magic, version, declared_size = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or declared_size != size:
            raise VggtOmegaSpaceError(
                "omega_glb_invalid",
                "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
            )

        chunk_index = 0
        while handle.tell() < size:
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                raise VggtOmegaSpaceError(
                    "omega_glb_invalid",
                    "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
                )
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            if chunk_length % 4 != 0 or chunk_length > size - handle.tell():
                raise VggtOmegaSpaceError(
                    "omega_glb_invalid",
                    "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
                )
            chunk = handle.read(chunk_length)
            if len(chunk) != chunk_length:
                raise VggtOmegaSpaceError(
                    "omega_glb_invalid",
                    "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
                )
            if chunk_index == 0:
                if chunk_type != _GLB_JSON_CHUNK or not chunk:
                    raise VggtOmegaSpaceError(
                        "omega_glb_invalid",
                        "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
                    )
                try:
                    document = json.loads(chunk.decode("utf-8"))
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                    raise VggtOmegaSpaceError(
                        "omega_glb_invalid",
                        "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
                    ) from exc
                asset = document.get("asset") if isinstance(document, dict) else None
                if not isinstance(asset, dict) or asset.get("version") != "2.0":
                    raise VggtOmegaSpaceError(
                        "omega_glb_invalid",
                        "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
                    )
            chunk_index += 1

    if chunk_index == 0:
        raise VggtOmegaSpaceError(
            "omega_glb_invalid",
            "Official VGGT-Omega Space returned an invalid glTF 2.0 GLB artifact.",
        )
    return size


def _download_vggt_omega_glb(url: str, destination: Path) -> int:
    import requests

    # Validate before the request so the worker can never be turned into a
    # generic fetcher by a compromised or changed Space response.
    _vggt_omega_glb_url(
        {"url": url, "meta": {"_type": "gradio.FileData"}}
    )
    response = None
    destination.unlink(missing_ok=True)
    try:
        response = requests.get(
            url,
            headers=_omega_space_headers(accept="model/gltf-binary"),
            stream=True,
            timeout=_env_int("STAGE2_OMEGA_GLB_TIMEOUT_SECONDS", 180, 30, 600),
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise VggtOmegaSpaceError(
                "omega_glb_download_failed",
                "Official VGGT-Omega GLB download redirected and was rejected.",
            )
        if response.status_code != 200:
            raise VggtOmegaSpaceError(
                "omega_glb_download_failed",
                f"Official VGGT-Omega GLB download returned HTTP {response.status_code}.",
            )
        declared_raw = response.headers.get("content-length")
        if declared_raw is not None:
            try:
                declared = int(declared_raw)
            except (TypeError, ValueError) as exc:
                raise VggtOmegaSpaceError(
                    "omega_glb_invalid",
                    "Official VGGT-Omega GLB returned an invalid Content-Length.",
                ) from exc
            if declared <= 0:
                raise VggtOmegaSpaceError(
                    "omega_glb_invalid",
                    "Official VGGT-Omega Space returned an empty GLB artifact.",
                )
            if declared > GLB_MAX_BYTES:
                raise VggtOmegaSpaceError(
                    "omega_glb_too_large",
                    f"Official VGGT-Omega GLB exceeds the {GLB_MAX_BYTES}-byte artifact contract.",
                )

        written = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                written += len(chunk)
                if written > GLB_MAX_BYTES:
                    raise VggtOmegaSpaceError(
                        "omega_glb_too_large",
                        f"Official VGGT-Omega GLB exceeds the {GLB_MAX_BYTES}-byte artifact contract.",
                    )
                handle.write(chunk)
        return _validate_vggt_omega_glb(destination)
    except VggtOmegaSpaceError:
        destination.unlink(missing_ok=True)
        raise
    except requests.RequestException as exc:
        destination.unlink(missing_ok=True)
        raise VggtOmegaSpaceError(
            "omega_glb_download_failed",
            f"Official VGGT-Omega GLB download failed ({type(exc).__name__}).",
        ) from exc
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            _close_http_response(response)


def _run_vggt_omega_space(
    frame_paths: list[Path],
    destination: Path,
) -> dict[str, Any]:
    revision = _verify_vggt_omega_space_revision()
    uploaded_frames = _upload_vggt_omega_frames(frame_paths)
    result_deadline = time.monotonic() + _env_int(
        "STAGE2_OMEGA_RESULT_TIMEOUT_SECONDS",
        900,
        60,
        1200,
    )

    prepare_event = _submit_vggt_omega_gradio(
        "update_gallery_on_upload",
        [None, uploaded_frames, 1.0],
    )
    prepare_result = _wait_vggt_omega_gradio(
        "update_gallery_on_upload",
        prepare_event,
        deadline=result_deadline,
    )
    if len(prepare_result) != 4:
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            "Official VGGT-Omega Space returned an unexpected preparation result.",
        )
    target_dir = prepare_result[1]
    if not isinstance(target_dir, str) or not target_dir or len(target_dir) > 4096:
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            "Official VGGT-Omega Space did not return a valid reconstruction handle.",
        )

    demo_event = _submit_vggt_omega_gradio(
        "gradio_demo",
        [
            target_dir,
            VGGT_OMEGA_CONFIDENCE_PERCENTILE,
            False,
            False,
            True,
            False,
            VGGT_OMEGA_MAX_POINTS_K,
        ],
    )
    demo_result = _wait_vggt_omega_gradio(
        "gradio_demo",
        demo_event,
        deadline=result_deadline,
    )
    if len(demo_result) != 2:
        raise VggtOmegaSpaceError(
            "omega_space_protocol_failed",
            "Official VGGT-Omega Space returned an unexpected reconstruction result.",
        )
    glb_file_data = demo_result[0]
    glb_url = _vggt_omega_glb_url(glb_file_data)
    temporary_destination = destination.with_name(f".{destination.name}.omega-download")
    temporary_destination.unlink(missing_ok=True)
    try:
        glb_bytes = _download_vggt_omega_glb(glb_url, temporary_destination)
        # The two Gradio calls share opaque replica-local state. Re-check the
        # reviewed Space revision after that stateful operation and before the
        # artifact can enter the normal durable-delivery path.
        final_revision = _verify_vggt_omega_space_revision()
        if final_revision != revision:
            raise VggtOmegaSpaceError(
                "omega_space_revision_mismatch",
                "Official VGGT-Omega Space revision changed during execution; the artifact was rejected.",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_destination, destination)
    finally:
        temporary_destination.unlink(missing_ok=True)
    return {
        **revision,
        "glb_bytes": glb_bytes,
        "max_points": VGGT_OMEGA_MAX_POINTS_K * 1000,
        "confidence_percentile": VGGT_OMEGA_CONFIDENCE_PERCENTILE,
    }


def _masks_to_instances(all_masks: dict[str, list]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for category, tracks in (all_masks or {}).items():
        for i, track in enumerate(tracks):
            frame_ids = sorted({int(fr["frame_id"]) for fr in track})
            instances.append(
                {
                    "category": category,
                    "instance_id": f"{category}_{i}",
                    "frame_ids": frame_ids,
                    "frame_count": len(frame_ids),
                }
            )
    return instances


def _positive_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp_span(entries: list[dict[str, Any]], timestamp_fields: tuple[str, ...]) -> float | None:
    """Return the decoded timeline span represented by packets or frames."""
    samples: list[tuple[float, float | None]] = []
    for entry in entries:
        timestamp = next(
            (
                parsed
                for field in timestamp_fields
                if (parsed := _finite_number(entry.get(field))) is not None
            ),
            None,
        )
        if timestamp is None:
            continue
        packet_duration = next(
            (
                parsed
                for field in ("duration_time", "pkt_duration_time")
                if (parsed := _positive_duration(entry.get(field))) is not None
            ),
            None,
        )
        samples.append((timestamp, packet_duration))
    if not samples:
        return None

    ordered_starts = sorted({timestamp for timestamp, _duration in samples})
    deltas = [
        right - left
        for left, right in zip(ordered_starts, ordered_starts[1:])
        if right > left
    ]
    inferred_duration = sorted(deltas)[len(deltas) // 2] if deltas else None
    ends = [
        timestamp + (duration or inferred_duration or 0.0)
        for timestamp, duration in samples
    ]
    span = max(ends) - min(timestamp for timestamp, _duration in samples)
    return _positive_duration(span)


def _ffprobe_json(path: Path, *args: str) -> dict[str, Any] | None:
    import subprocess

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                *args,
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        value = json.loads(completed.stdout) if completed.returncode == 0 else None
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError):
        return None


def _probe_video_duration(path: Path) -> float | None:
    """Determine the video timeline through progressively deeper ffprobe data."""
    container = _ffprobe_json(path, "-show_entries", "format=duration") or {}
    duration = _positive_duration((container.get("format") or {}).get("duration"))
    if duration is not None:
        return duration

    streams = _ffprobe_json(
        path,
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
    ) or {}
    for stream in streams.get("streams") or []:
        duration = _positive_duration(stream.get("duration"))
        if duration is not None:
            return duration

    packets = _ffprobe_json(
        path,
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,dts_time,duration_time",
    ) or {}
    duration = _timestamp_span(packets.get("packets") or [], ("pts_time", "dts_time"))
    if duration is not None:
        return duration

    frames = _ffprobe_json(
        path,
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time,pkt_dts_time,pkt_duration_time,duration_time",
    ) or {}
    return _timestamp_span(
        frames.get("frames") or [],
        ("best_effort_timestamp_time", "pts_time", "pkt_dts_time"),
    )


def _probe_video_frame_count(path: Path) -> int | None:
    import subprocess

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        streams = json.loads(completed.stdout).get("streams", []) if completed.returncode == 0 else []
        stream = streams[0] if streams else {}
        for field in ("nb_frames", "nb_read_frames"):
            count = int(stream.get(field) or 0)
            if count > 0:
                return count
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError):
        pass
    return None


def _sampled_source_frame_timestamps(
    video_path: Path,
    source_frame_indices: list[int],
) -> list[float] | None:
    """Resolve decoded source-frame indices to their real presentation times."""
    plan = _source_frame_plan(video_path, max(source_frame_indices, default=-1) + 1)
    if not plan:
        return None
    all_indices, all_timestamps = plan
    if not all_timestamps or all_indices != list(range(len(all_indices))):
        # Asking for every decoded frame above produces a direct index -> PTS map.
        return None
    if max(source_frame_indices, default=-1) >= len(all_timestamps):
        return None
    sampled = [all_timestamps[index] for index in source_frame_indices]
    origin = sampled[0]
    normalized = [max(0.0, timestamp - origin) for timestamp in sampled]
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        return None
    return normalized


def _normalize_mask_video(
    mask_video: Path,
    source_video: Path,
    source_frame_indices: list[int] | None = None,
    source_frame_timestamps: list[float] | None = None,
) -> dict[str, Any]:
    """Produce browser-safe H.264 and preserve the source playback timeline."""
    import subprocess

    if not mask_video.is_file() or mask_video.stat().st_size <= 0:
        raise RuntimeError("SAM3 mask visualization did not produce a video")
    source_duration = _probe_video_duration(source_video)
    if source_duration is None:
        raise RuntimeError(
            "source video timeline could not be established; refusing to publish an unsynchronized mask video"
        )
    normalized = mask_video.with_name(mask_video.stem + ".browser.mp4")
    normalized.unlink(missing_ok=True)
    timeline_dir = Path(tempfile.mkdtemp(prefix="stage2-mask-timeline-", dir=str(mask_video.parent)))
    try:
        extracted = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(mask_video),
                "-map",
                "0:v:0",
                "-an",
                "-vsync",
                "0",
                str(timeline_dir / "frame-%06d.png"),
            ],
            check=False,
            capture_output=True,
            timeout=300,
        )
        frame_files = sorted(timeline_dir.glob("frame-*.png"))
        if extracted.returncode != 0 or not frame_files:
            raise RuntimeError("SAM3 mask visualization frames could not be decoded")
        generated_frames = len(frame_files)
        plan = None if source_frame_indices and source_frame_timestamps else _source_frame_plan(
            source_video,
            generated_frames,
        )
        indices = source_frame_indices or (plan[0] if plan else None)
        if not indices or len(indices) != generated_frames:
            raise RuntimeError("SAM3 mask frames do not match the source sampling plan")
        timestamps = source_frame_timestamps or (plan[1] if plan else None)
        if not timestamps or len(timestamps) != generated_frames:
            raise RuntimeError("sampled source frame timestamps could not be established")
        tail_duration = source_duration - timestamps[-1]
        if tail_duration <= 0:
            raise RuntimeError("sampled source timestamps exceed the source duration")
        durations = [
            right - left
            for left, right in zip(timestamps, timestamps[1:])
        ] + [tail_duration]

        concat_file = timeline_dir / "timeline.ffconcat"
        concat_lines = ["ffconcat version 1.0"]
        for frame, duration in zip(frame_files, durations):
            concat_lines.extend([f"file '{frame.name}'", f"duration {duration:.12g}"])
        # ffconcat only applies the last duration when another file follows it.
        # The output `-t` boundary removes this duplicate at source_duration.
        concat_lines.append(f"file '{frame_files[-1].name}'")
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "vfr",
                "-t",
                f"{source_duration:.12g}",
                "-movflags",
                "+faststart",
                "-video_track_timescale",
                "90000",
                str(normalized),
            ],
            check=False,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        normalized.unlink(missing_ok=True)
        raise RuntimeError("failed to normalize SAM3 mask video for browser playback") from exc
    finally:
        shutil.rmtree(timeline_dir, ignore_errors=True)
    if completed.returncode != 0 or not normalized.is_file() or normalized.stat().st_size <= 0:
        normalized.unlink(missing_ok=True)
        raise RuntimeError("failed to normalize SAM3 mask video for browser playback")
    normalized.replace(mask_video)

    output_duration = _probe_video_duration(mask_video)
    tolerance = max(0.15, min(1.0, source_duration * 0.005))
    aligned = bool(
        output_duration is not None
        and abs(output_duration - source_duration) <= tolerance
    )
    if not aligned:
        raise RuntimeError("normalized SAM3 mask video duration does not match the source clip")
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "faststart": True,
        "frame_rate": generated_frames / source_duration,
        "frame_count": generated_frames,
        "timeline_mode": "source_frame_pts",
        "source_duration_seconds": source_duration,
        "output_duration_seconds": output_duration,
        "duration_aligned": aligned,
    }


def run_stage2_dry(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate video I/O and, for typed checks, all pinned source runtimes."""
    t0 = time.time()
    video_url = str(payload.get("video_url") or "").strip()
    categories = payload.get("categories") or []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    categories = [str(c).strip() for c in categories if str(c).strip()]
    max_frames = max(2, min(int(payload.get("max_frames") or 24), 160))
    has_video = bool(video_url) or bool(payload.get("video_b64"))
    if not has_video or not categories:
        return {"status": "error", "error": "video_url/video_b64 and categories are required", "mode": "dry_run"}

    work = Path(tempfile.mkdtemp(prefix="ras-stage2-dry-"))
    try:
        video_path = _materialize_video(payload, work)
        # Lightweight sample with OpenCV only (no RAS models).
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"status": "error", "error": "could not decode downloaded video", "mode": "dry_run"}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = min(max_frames, total) if total > 0 else max_frames
        idxs = np.linspace(0, max(total - 1, 0), n).astype(int).tolist() if total > 0 else list(range(n))
        frames_ok = 0
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, _ = cap.read()
            if ok:
                frames_ok += 1
        cap.release()
        source_validation = None
        source_bootstrap_ms = None
        if payload.get("analysis_type") == "validation_v1":
            source_started = time.time()
            _ensure_ras_installed(require_sam3=True, ensure_weights=False)
            ras = _ras_root()
            source_validation = {
                "ras_revision": _verified_checkout_revision(
                    ras,
                    RAS_REVISION,
                    ignore_submodules=True,
                ),
                "vggt_revision": _verified_checkout_revision(ras / "vggt", VGGT_REVISION),
                "sam3_revision": _verified_checkout_revision(ras / "sam3", SAM3_REVISION),
                "sam3_required": True,
                "weights_required": False,
            }
            source_bootstrap_ms = int((time.time() - source_started) * 1000)
        pipeline = [
            {"id": "intake", "name": "Video intake", "status": "ok"},
            {"id": "sample_frames", "name": "Frame sampling", "status": "ok", "detail": {"frames_used": frames_ok}},
            {"id": "vggt", "name": "VGGT", "status": "skipped_dry_run"},
            {"id": "sam", "name": "SAM3", "status": "skipped_dry_run"},
            {"id": "dedup", "name": "Spatial dedup", "status": "skipped_dry_run"},
        ]
        if source_validation:
            pipeline.insert(2, {
                "id": "source_bootstrap",
                "name": "Pinned RAS + VGGT + SAM3 source bootstrap",
                "status": "ok",
                "ms": source_bootstrap_ms,
            })
        response = {
            "status": "ok",
            "mode": "dry_run",
            "implementation": (
                "ReplicateAnyScene input and pinned-source validation "
                "(no model weights or inference)"
                if source_validation
                else "ReplicateAnyScene input validation (video download and frame sampling only)"
            ),
            "upstream": "https://github.com/xiac20/ReplicateAnyScene",
            "frames_used": frames_ok,
            "source_frame_indices": idxs,
            "video_meta": {"total_frames": total, "fps": fps, "width": w, "height": h, "sampled": frames_ok},
            "categories": categories,
            "raw_track_count": 0,
            "instance_count": 0,
            "instances": [],
            "pipeline": pipeline,
            "timings_ms": {"total": int((time.time() - t0) * 1000)},
        }
        if source_validation:
            response["source"] = source_validation
        return response
    except Exception as e:
        return {
            "status": "error",
            "mode": "dry_run",
            "error": str(e),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _link_models_dir(ras_root: Path) -> None:
    models_dir = _models_dir(ras_root)
    models_link = ras_root / "models"
    if models_dir == models_link:
        return
    # First jobs can reach this boundary concurrently after lazy bootstrap.
    # Use the same cross-worker lock as source/weight initialization so two
    # workers never race between the existence check and symlink creation.
    with _stage2_initialization_lock(models_dir):
        if models_link.is_symlink():
            if models_link.resolve() == models_dir:
                return
            models_link.unlink()
        elif models_link.exists():
            if models_link.is_dir() and not any(models_link.iterdir()):
                models_link.rmdir()
            else:
                raise RuntimeError(f"cannot link models: non-empty path exists at {models_link}")
        models_link.symlink_to(models_dir, target_is_directory=True)


def _upload_ticket(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    upload = payload.get("upload") if isinstance(payload, dict) else None
    return upload if isinstance(upload, dict) else None


def _resolve_object_catalog_version(
    payload: dict[str, Any],
    mode: str,
) -> tuple[int | None, str | None]:
    """Validate explicit catalog negotiation without accepting JSON coercions."""
    if "object_catalog_version" not in payload:
        return None, None
    version = payload.get("object_catalog_version")
    if type(version) is not int or version != OBJECT_CATALOG_VERSION:
        return None, "object_catalog_version must be the integer 1"
    if mode != "full":
        return None, "object_catalog_version 1 is supported only for full mode"
    if payload.get("analysis_type") != "dedup_ras_vggt_sam3":
        return None, (
            "object_catalog_version 1 requires analysis_type "
            "dedup_ras_vggt_sam3"
        )
    return version, None


def _object_catalog_negotiated(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        str(payload.get("mode") or "").lower() == "full"
        and type(payload.get("object_catalog_version")) is int
        and payload.get("object_catalog_version") == OBJECT_CATALOG_VERSION
        and payload.get("analysis_type") == "dedup_ras_vggt_sam3"
    )


def _object_catalog_transport_canary(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        str(payload.get("mode") or "").lower() == "dry_run"
        and payload.get("analysis_type") == OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE
        and type(payload.get("object_catalog_transport_canary")) is bool
        and payload.get("object_catalog_transport_canary") is True
        and payload.get("categories") == [OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY]
        and "object_catalog_version" not in payload
        and not payload.get("video_url")
        and not payload.get("video_b64")
        and bool(_upload_ticket(payload))
    )


def _resolve_object_catalog_transport_canary(
    payload: dict[str, Any],
    mode: str,
    analysis_type: str | None,
) -> tuple[bool, str | None]:
    flag_present = "object_catalog_transport_canary" in payload
    typed_canary = analysis_type == OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE
    if not flag_present and not typed_canary:
        return False, None
    if not typed_canary:
        return False, (
            "object_catalog_transport_canary is reserved for analysis_type "
            f"{OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE}"
        )
    if mode != "dry_run":
        return False, (
            f"analysis_type {OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE} "
            "requires mode dry_run"
        )
    if type(payload.get("object_catalog_transport_canary")) is not bool or not payload.get(
        "object_catalog_transport_canary"
    ):
        return False, "object_catalog_transport_canary must be the boolean true"
    if "object_catalog_version" in payload:
        return False, "transport canaries must not set object_catalog_version"
    if payload.get("categories") != [OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY]:
        return False, (
            "transport canary categories must be exactly "
            f"['{OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY}']"
        )
    if payload.get("video_url") or payload.get("video_b64"):
        return False, "transport canaries do not accept video input"
    if not _upload_ticket(payload):
        return False, "transport canaries require a privileged artifact upload ticket"
    return True, None


def _required_artifacts(payload: dict[str, Any] | None) -> tuple[str, ...]:
    if _object_catalog_transport_canary(payload):
        return OBJECT_CATALOG_ARTIFACTS
    mode = str((payload or {}).get("mode") or "")
    required = REQUIRED_ARTIFACTS.get(mode, ())
    if _object_catalog_negotiated(payload):
        return (*required, *OBJECT_CATALOG_ARTIFACTS)
    return required


def _artifact_manifest(out_dir: Path, work: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deliver artifacts durably and return receipts, never temporary paths."""
    artifact_root = os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()
    allowed_artifacts = set(ARTIFACT_MEDIA_TYPES)
    if not (
        _object_catalog_negotiated(payload)
        or _object_catalog_transport_canary(payload)
    ):
        allowed_artifacts.difference_update(OBJECT_CATALOG_ARTIFACTS)
    files = sorted(
        p.name
        for p in out_dir.iterdir()
        if p.is_file() and p.name in allowed_artifacts
    ) if out_dir.is_dir() else []
    upload = _upload_ticket(payload)
    if upload:
        from artifact_upload import artifact_failure_record, upload_artifact_file

        required_files = list(_required_artifacts(payload))
        required_set = set(required_files)
        # Deliver the product contract before optional debug artifacts. This
        # matters when a short-lived ticket or proxy upload fails partway.
        upload_order = [name for name in required_files if name in files]
        upload_order.extend(name for name in files if name not in required_set)
        receipts: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for name in required_files:
            if name not in files:
                errors.append(
                    {
                        "name": name,
                        "code": "artifact_generation_missing",
                        "detail": "required artifact was not generated",
                    }
                )
        for name in upload_order:
            try:
                if _object_catalog_transport_canary(payload):
                    receipt = upload_artifact_file(
                        upload,
                        out_dir / name,
                        ARTIFACT_MEDIA_TYPES[name],
                        require_put_acknowledgement=True,
                    )
                else:
                    receipt = upload_artifact_file(
                        upload,
                        out_dir / name,
                        ARTIFACT_MEDIA_TYPES[name],
                    )
                receipts.append(receipt)
            except Exception as exc:
                # Signed PUT URLs are credentials. Keep only the uploader's
                # bounded phase/status record in both logs and provider JSON.
                failure = artifact_failure_record(name, exc)
                print(
                    json.dumps(
                        {"event": "artifact_upload_failed", **failure},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                errors.append(failure)
        receipt_names = {str(receipt.get("name") or "") for receipt in receipts}
        missing_required = [name for name in required_files if name not in receipt_names]
        complete = not missing_required
        durable = bool(receipts)
        return {
            "durable": durable,
            "complete": complete,
            "delivery": "agent-lab-r2",
            "files": files,
            "required_files": required_files,
            "missing_required": missing_required,
            "receipts": receipts,
            "errors": errors,
            "note": (
                "Artifacts uploaded directly to Agent Lab storage; receipts require edge verification."
                if complete and not errors
                else "Required result files could not all be delivered; the processing run must be treated as failed."
                if not complete
                else "Some artifacts could not be uploaded; only verified receipts are usable."
                if durable
                else "Artifact upload failed; no worker-local path is exposed."
            ),
        }
    if not artifact_root:
        return {
            "durable": False,
            "files": files,
            "note": (
                "Artifacts were generated for this job but no durable download store is configured."
                if files
                else "Artifact export was skipped because no durable download store is configured."
            ),
        }
    dest = Path(artifact_root) / work.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(out_dir, dest)
    return {
        "durable": True,
        "files": files,
        "storage_key": work.name,
        "note": "Artifacts were copied to the worker's configured persistent store.",
    }


def _artifact_delivery_error(
    payload: dict[str, Any] | None,
    artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    """Fail closed when an upload-ticket job misses its required deliverables."""
    if not _upload_ticket(payload) or artifacts.get("complete") is True:
        return None
    missing = artifacts.get("missing_required")
    if not isinstance(missing, list):
        missing = list(_required_artifacts(payload))
    return {
        "error_code": "artifact_delivery_failed",
        "error": "Required result files could not be delivered to durable storage.",
        "artifact_delivery": {
            "required_files": artifacts.get("required_files", []),
            "missing_required": missing,
            "errors": artifacts.get("errors", []),
        },
    }


def run_object_catalog_transport_canary(payload: dict[str, Any]) -> dict[str, Any]:
    """Exercise catalog generation and durable delivery without video or models."""
    is_canary, canary_error = _resolve_object_catalog_transport_canary(
        payload,
        str(payload.get("mode") or "").lower(),
        payload.get("analysis_type") if isinstance(payload.get("analysis_type"), str) else None,
    )
    if canary_error or not is_canary:
        return {
            "status": "error",
            "mode": "dry_run",
            "error": canary_error or "invalid object catalog transport canary request",
            "synthetic_transport_canary": True,
            "instance_count": 0,
            "instances": [],
        }
    import numpy as np

    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="ras-object-catalog-canary-"))
    out_dir = work / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, int] = {}
    try:
        generation_started = time.time()
        catalog = build_object_catalog(
            all_masks={OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY: []},
            colors=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            world_points=np.zeros((1, 8, 8, 3), dtype=np.float32),
            requested_categories=[OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY],
            source_frame_indices=None,
            source_frame_timestamps=None,
            surface_area_fn=lambda _pointmap, _mask: 0.0,
            out_dir=out_dir,
        )
        timings["object_catalog"] = int((time.time() - generation_started) * 1000)

        delivery_started = time.time()
        artifacts = _artifact_manifest(out_dir, work, payload)
        timings["artifact_delivery"] = int((time.time() - delivery_started) * 1000)
        timings["total"] = int((time.time() - started) * 1000)
        delivery_error = _artifact_delivery_error(payload, artifacts)
        response = {
            "status": "ok",
            "mode": "dry_run",
            "implementation": "Synthetic object catalog transport canary (no video or models)",
            "synthetic_transport_canary": True,
            "frames_used": 1,
            "categories": [OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY],
            "raw_track_count": 0,
            "instance_count": 0,
            "instances": [],
            "object_catalog": {
                "schema_version": catalog["schema_version"],
                "total_count": 0,
                "returned_count": 0,
                "truncated": False,
                "artifact_name": OBJECT_CATALOG_JSON_NAME,
                "atlas_artifact_name": OBJECT_CROPS_ATLAS_NAME,
            },
            "artifacts": artifacts,
            "timings_ms": timings,
            "pipeline": [
                {
                    "id": "synthetic_object_catalog",
                    "status": "ok",
                    "ms": timings.get("object_catalog"),
                },
                {
                    "id": "artifact_delivery",
                    "status": "error" if delivery_error else "ok",
                    "ms": timings.get("artifact_delivery"),
                },
            ],
        }
        if delivery_error:
            response.update({"status": "error", **delivery_error})
        return response
    except Exception as exc:
        timings["total"] = int((time.time() - started) * 1000)
        return {
            "status": "error",
            "mode": "dry_run",
            "error": f"Object catalog transport canary failed ({type(exc).__name__}).",
            "synthetic_transport_canary": True,
            "instance_count": 0,
            "instances": [],
            "timings_ms": timings,
        }
    finally:
        if os.environ.get("STAGE2_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


def _artifact_exports_enabled(payload: dict[str, Any] | None = None) -> bool:
    """Only spend time exporting geometry when the files will remain inspectable."""
    return bool(_upload_ticket(payload)) or bool(os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()) or os.environ.get("STAGE2_KEEP_WORK") == "1"


def _durable_artifact_delivery_enabled(payload: dict[str, Any] | None = None) -> bool:
    """Return whether a completed artifact can outlive the current worker."""
    return bool(_upload_ticket(payload)) or bool(
        os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()
    )


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def _debug_artifacts_enabled() -> bool:
    return os.environ.get("STAGE2_EXPORT_DEBUG_ARTIFACTS", "").strip() == "1"


def _room_alignment_was_applied(rotation: Any, translation: Any) -> bool:
    """Distinguish RAS's identity/zero fallback from a real room alignment."""
    import numpy as np

    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        return False
    return not (
        np.allclose(R, np.eye(3), atol=1e-6)
        and np.allclose(t, np.zeros(3), atol=1e-6)
    )


def _upstream_preview_point_cloud(pred: dict[str, Any]) -> tuple[Any, Any] | None:
    """Return RAS's depth-confidence-filtered preview points when available."""
    import numpy as np

    cloud = pred.get("point_cloud_data")
    if cloud is None:
        return None
    vertices = np.asarray(getattr(cloud, "vertices", None))
    colors = np.asarray(getattr(cloud, "colors", None))
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) == 0
        or colors.ndim != 2
        or colors.shape[0] != len(vertices)
        or colors.shape[1] < 3
    ):
        return None
    return vertices, colors[:, :3]


def _export_vggt_artifacts(
    pred: dict[str, Any],
    out_dir: Path,
    *,
    coordinate_system: str = "vggt_first_camera",
) -> dict[str, Any]:
    """Export the compact viewer GLB; large compatibility files are opt-in."""
    from point_cloud_glb import GLB_HARD_MAX_POINTS, write_point_cloud_glb

    exported: dict[str, Any] = {}
    warnings: list[dict[str, str]] = []
    try:
        upstream_cloud = _upstream_preview_point_cloud(pred)
        if upstream_cloud is not None:
            preview_points, preview_colors = upstream_cloud
            preview_confidence = None
            prefiltered_percentile = 50.0
        else:
            # A raw fallback is safe only when its matching depth-confidence
            # branch is explicitly present. Never silently downgrade preview
            # quality or pair depth points with point-map confidence.
            preview_points = pred["world_points"]
            preview_colors = pred["colors"]
            preview_confidence = pred.get("depth_conf")
            if preview_confidence is None:
                raise ValueError(
                    "RAS preview point cloud and matching depth confidence are unavailable"
                )
            prefiltered_percentile = None
        exported["point_cloud_glb"] = write_point_cloud_glb(
            out_dir / "point_cloud.glb",
            preview_points,
            preview_colors,
            confidence=preview_confidence,
            extrinsics=pred.get("extrinsics"),
            max_points=_env_int(
                "STAGE2_POINT_CLOUD_MAX_POINTS",
                300_000,
                10_000,
                GLB_HARD_MAX_POINTS,
            ),
            confidence_percentile=50.0,
            confidence_prefiltered_percentile=prefiltered_percentile,
            coordinate_system=coordinate_system,
        )
    except Exception as exc:
        warnings.append({"name": "point_cloud.glb", "error": f"{type(exc).__name__}: {exc}"})

    if _debug_artifacts_enabled():
        try:
            if pred.get("point_cloud_data") is not None:
                pred["point_cloud_data"].export(str(out_dir / "point_cloud.ply"))
                exported["point_cloud_ply"] = True
        except Exception as exc:
            warnings.append({"name": "point_cloud.ply", "error": f"{type(exc).__name__}: {exc}"})

        try:
            intrinsic = pred["intrinsic"]
            values = intrinsic.tolist() if hasattr(intrinsic, "tolist") else intrinsic
            with (out_dir / "camera_intrinsics.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {"schema": "vggt-camera-intrinsics-v1", "intrinsics": values},
                    handle,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            exported["camera_intrinsics"] = True
        except Exception as exc:
            warnings.append(
                {"name": "camera_intrinsics.json", "error": f"{type(exc).__name__}: {exc}"}
            )
    if warnings:
        exported["warnings"] = warnings
    return exported


def run_stage2_geometry(payload: dict[str, Any]) -> dict[str, Any]:
    """Real VGGT geometry path that intentionally does not import or load SAM3."""
    t_all = time.time()
    work = Path(tempfile.mkdtemp(prefix="ras-stage2-geometry-"))
    out_dir = work / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, int] = {}
    try:
        t0 = time.time()
        video_path = _materialize_video(payload, work)
        timings["download"] = int((time.time() - t0) * 1000)

        _ensure_ras_installed(require_sam3=False)
        ras_root = _ensure_ras_on_path()
        vggt_source_revision = _verified_checkout_revision(ras_root / "vggt", VGGT_REVISION)
        _link_models_dir(ras_root)

        import gc
        import torch
        from vggt.models.vggt import VGGT
        from src.utils import load_video_frames
        from src.vggt_predict import vggt_predict

        if not torch.cuda.is_available():
            raise RuntimeError("geometry mode requires a CUDA GPU")
        device = "cuda"
        max_frames = int(payload["max_frames"])

        t0 = time.time()
        frames = load_video_frames(str(video_path), max_frames).to(device)
        n_frames = int(frames.shape[0])
        source_frame_plan = _source_frame_plan(video_path, n_frames)
        source_frame_indices = source_frame_plan[0] if source_frame_plan else None
        timings["sample"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        model = VGGT.from_pretrained("./models/VGGT").to(device)
        pred = vggt_predict(frames, model)
        model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        timings["vggt"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        export_meta = _export_vggt_artifacts(
            pred,
            out_dir,
            coordinate_system="vggt_first_camera",
        ) if _artifact_exports_enabled(payload) else {}
        timings["artifact_export"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        artifacts = _artifact_manifest(out_dir, work, payload)
        timings["artifact_delivery"] = int((time.time() - t0) * 1000)
        timings["total"] = int((time.time() - t_all) * 1000)
        delivery_error = _artifact_delivery_error(payload, artifacts)
        response = {
            "status": "ok",
            "mode": "geometry",
            "implementation": "ReplicateAnyScene VGGT geometry preview",
            "upstream_revision": RAS_REVISION,
            "frames_used": n_frames,
            "source_frame_indices": source_frame_indices,
            "categories": payload["categories"],
            "raw_track_count": 0,
            "instance_count": 0,
            "instances": [],
            "geometry": {
                "backend": "vggt",
                "device": device,
                "model_id": _vggt_model_id(),
                "source_revision": vggt_source_revision,
                "license_scope": _vggt_license_scope(),
                "world_points_shape": list(pred["world_points"].shape),
                "sam3_required": False,
                "artifact_export": export_meta,
            },
            "sam": {"backend": "not_run", "reason": "geometry mode intentionally skips SAM3"},
            "artifacts": artifacts,
            "timings_ms": timings,
            "pipeline": [
                {"id": "intake", "name": "Video intake", "status": "ok", "ms": timings.get("download")},
                {"id": "sample_frames", "name": "Frame sampling", "status": "ok", "ms": timings.get("sample")},
                {"id": "vggt", "name": "VGGT geometry", "status": "ok", "ms": timings.get("vggt")},
                {"id": "sam", "name": "SAM3", "status": "skipped_geometry_mode"},
                {"id": "dedup", "name": "Spatial dedup", "status": "skipped_geometry_mode"},
                {
                    "id": "artifact_delivery",
                    "name": "Artifact delivery",
                    "status": "error" if delivery_error else "ok",
                    "ms": timings.get("artifact_delivery"),
                },
            ],
        }
        if delivery_error:
            response.update({"status": "error", **delivery_error})
        return response
    except Exception as exc:
        return {
            "status": "error",
            "mode": "geometry",
            "error": str(exc),
            "timings_ms": timings,
        }
    finally:
        if os.environ.get("STAGE2_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


def run_stage2_omega_geometry(payload: dict[str, Any]) -> dict[str, Any]:
    """Run genuine VGGT-Omega geometry through Meta's reviewed public Space.

    RunPod remains the job orchestrator and the only artifact-delivery owner.
    The hosted Space is used only for Omega geometry; this path never imports
    local VGGT, CUDA, SAM3, or ReplicateAnyScene deduplication code.
    """
    t_all = time.time()
    timings: dict[str, int] = {}
    stage = "artifact_contract"
    if not _durable_artifact_delivery_enabled(payload):
        return {
            "status": "error",
            "mode": "geometry",
            "error_code": "artifact_delivery_unavailable",
            "error": (
                "VGGT-Omega geometry requires durable artifact delivery; "
                "provide an upload ticket or configure persistent artifact storage."
            ),
            "timings_ms": {"total": int((time.time() - t_all) * 1000)},
        }

    work = Path(tempfile.mkdtemp(prefix="ras-stage2-omega-"))
    out_dir = work / "output"
    frame_dir = work / "omega-frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        stage = "intake"
        t0 = time.time()
        video_path = _materialize_video(payload, work)
        timings["download"] = int((time.time() - t0) * 1000)

        stage = "sample_frames"
        t0 = time.time()
        frame_paths, source_frame_indices, source_frame_timestamps = _sample_vggt_omega_frames(
            video_path,
            frame_dir,
            int(payload["max_frames"]),
        )
        timings["sample"] = int((time.time() - t0) * 1000)

        stage = "vggt_omega"
        t0 = time.time()
        omega_result = _run_vggt_omega_space(frame_paths, out_dir / "point_cloud.glb")
        timings["vggt_omega"] = int((time.time() - t0) * 1000)

        stage = "artifact_delivery"
        t0 = time.time()
        artifacts = _artifact_manifest(out_dir, work, payload)
        timings["artifact_delivery"] = int((time.time() - t0) * 1000)
        timings["total"] = int((time.time() - t_all) * 1000)
        delivery_error = _artifact_delivery_error(payload, artifacts)
        response = {
            "status": "ok",
            "mode": "geometry",
            "implementation": "VGGT-Omega geometry via Meta's official Hugging Face Space",
            "frames_used": len(frame_paths),
            "source_frame_indices": source_frame_indices,
            "source_frame_timestamps": source_frame_timestamps,
            "categories": payload["categories"],
            "raw_track_count": 0,
            "instance_count": 0,
            "instances": [],
            "geometry": {
                "backend": "huggingface_space",
                "model_id": VGGT_OMEGA_MODEL_ID,
                "source_revision": VGGT_OMEGA_SPACE_REVISION,
                "space_id": VGGT_OMEGA_SPACE_ID,
                "space_revision": VGGT_OMEGA_SPACE_REVISION,
                "github_source_revision": VGGT_OMEGA_GITHUB_REVISION,
                "model_repository_revision": VGGT_OMEGA_MODEL_REVISION,
                "checkpoint_filename": VGGT_OMEGA_CHECKPOINT,
                "provenance_level": VGGT_OMEGA_PROVENANCE_LEVEL,
                "license_scope": "research_noncommercial",
                "sam3_required": False,
                "execution_orchestrator": "runpod",
                "artifact_export": {
                    "point_cloud_glb": {
                        "bytes": omega_result["glb_bytes"],
                        "requested_max_points": omega_result["max_points"],
                        "confidence_percentile": omega_result["confidence_percentile"],
                    }
                },
            },
            "sam": {
                "backend": "not_run",
                "reason": "VGGT-Omega analysis is geometry-only and intentionally skips SAM3",
            },
            "artifacts": artifacts,
            "timings_ms": timings,
            "pipeline": [
                {"id": "intake", "name": "Video intake", "status": "ok", "ms": timings.get("download")},
                {
                    "id": "sample_frames",
                    "name": "Exact uniform frame sampling",
                    "status": "ok",
                    "ms": timings.get("sample"),
                    "detail": {"frames_used": len(frame_paths), "frame_cap": VGGT_OMEGA_MAX_FRAMES},
                },
                {
                    "id": "vggt_omega",
                    "name": "VGGT-Omega hosted geometry",
                    "status": "ok",
                    "ms": timings.get("vggt_omega"),
                },
                {"id": "sam", "name": "SAM3", "status": "skipped_geometry_mode"},
                {"id": "dedup", "name": "Spatial dedup", "status": "skipped_geometry_mode"},
                {
                    "id": "artifact_delivery",
                    "name": "Artifact delivery",
                    "status": "error" if delivery_error else "ok",
                    "ms": timings.get("artifact_delivery"),
                },
            ],
        }
        if delivery_error:
            response.update({"status": "error", **delivery_error})
        return response
    except VggtOmegaSpaceError as exc:
        timings["total"] = int((time.time() - t_all) * 1000)
        return {
            "status": "error",
            "mode": "geometry",
            "error_code": exc.code,
            "error": str(exc),
            "failed_stage": stage,
            "timings_ms": timings,
        }
    except Exception as exc:
        timings["total"] = int((time.time() - t_all) * 1000)
        return {
            "status": "error",
            "mode": "geometry",
            "error_code": "omega_internal_failure",
            "error": f"VGGT-Omega geometry execution failed ({type(exc).__name__}).",
            "failed_stage": stage,
            "timings_ms": timings,
        }
    finally:
        if os.environ.get("STAGE2_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


def run_stage2_full(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Full Stage 2 using ReplicateAnyScene sources only.

    Mirrors main.py Stage 2 (VGGT → room align → SAM3 video track →
    self/cross category dedup → mask video). Does not run Stage 3+.
    """
    payload_mode = str(payload.get("mode") or "full").lower()
    _catalog_version, catalog_error = _resolve_object_catalog_version(payload, payload_mode)
    if catalog_error:
        return {"status": "error", "error": catalog_error, "mode": "full"}
    payload = {**payload, "mode": "full"}
    t_all = time.time()
    video_url = str(payload.get("video_url") or "").strip()
    categories = payload.get("categories") or []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    categories = [str(c).strip() for c in categories if str(c).strip()]
    max_frames = max(2, min(int(payload.get("max_frames") or 48), 160))
    room_align = str(payload.get("room_align", os.environ.get("STAGE2_ROOM_ALIGN", "1"))).lower() not in {
        "0",
        "false",
        "no",
    }
    has_video = bool(video_url) or bool(payload.get("video_b64"))
    if not has_video or not categories:
        return {"status": "error", "error": "video_url/video_b64 and categories are required", "mode": "full"}

    work = Path(tempfile.mkdtemp(prefix="ras-stage2-full-"))
    out_dir = work / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, int] = {}

    try:
        t0 = time.time()
        video_path = _materialize_video(payload, work)
        timings["download"] = int((time.time() - t0) * 1000)

        _ensure_ras_installed()
        ras_root = _ensure_ras_on_path()
        vggt_source_revision = _verified_checkout_revision(ras_root / "vggt", VGGT_REVISION)
        sam3_source_revision = _verified_checkout_revision(ras_root / "sam3", SAM3_REVISION)
        _link_models_dir(ras_root)

        import torch
        import cv2

        from src.models import (
            load_sam3_image_model,
            load_sam3_video_model,
            load_vggt_model,
            unload_model,
        )
        from src.utils import load_video_frames, vis_instance_masks
        from src.geometry_utils import (
            align_to_room_coordinate_system,
            align_vggt_predictions,
            compute_surface_area_from_pointmap,
        )
        from src.vggt_predict import vggt_predict
        from src.object_segmentation import segment_and_track, segment_wall_and_floor
        from src.sg_deduplication import cross_category_deduplicate, self_category_deduplicate

        device = "cuda" if torch.cuda.is_available() else "cpu"

        t0 = time.time()
        frames = load_video_frames(str(video_path), max_frames).to(device)
        n_frames = int(frames.shape[0])
        source_frame_plan = _source_frame_plan(video_path, n_frames)
        source_frame_indices = source_frame_plan[0] if source_frame_plan else None
        source_frame_timestamps = _canonical_source_frame_timestamps(
            source_frame_plan[1] if source_frame_plan else None
        )
        timings["sample"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        vggt_model = load_vggt_model().to(device)
        pred = vggt_predict(frames, vggt_model)
        unload_model(vggt_model)
        timings["vggt"] = int((time.time() - t0) * 1000)

        wall_masks, floor_masks = [], []
        room_alignment_applied = False
        if room_align:
            t0 = time.time()
            sam3_image = load_sam3_image_model()
            wall_masks, floor_masks = segment_wall_and_floor(pred["colors"], sam3_image)
            R, t = align_to_room_coordinate_system(pred["world_points"], wall_masks, floor_masks)
            room_alignment_applied = _room_alignment_was_applied(R, t)
            if room_alignment_applied:
                pred = align_vggt_predictions(pred, R, t)
            unload_model(sam3_image)
            timings["room_align"] = int((time.time() - t0) * 1000)
        else:
            timings["room_align"] = 0

        # Persist color frames for SAM3 video session (same as RAS main.py).
        color_dir = out_dir / "color"
        color_dir.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(pred["colors"]):
            cv2.imwrite(str(color_dir / f"{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        export_meta = _export_vggt_artifacts(
            pred,
            out_dir,
            coordinate_system="room_z_up" if room_alignment_applied else "vggt_first_camera",
        ) if _artifact_exports_enabled(payload) else {}

        t0 = time.time()
        sam3_video = load_sam3_video_model()
        response = sam3_video.handle_request(
            request=dict(type="start_session", resource_path=str(color_dir)),
        )
        session_id = response["session_id"]
        all_masks: dict[str, list] = {}
        for category in categories:
            category_masks = segment_and_track(category, sam3_video, session_id)
            all_masks[category] = self_category_deduplicate(
                category_masks,
                pred["world_points"],
                pred["world_points_conf"],
            )
        deduped = cross_category_deduplicate(
            all_masks,
            pred["world_points"],
            pred["world_points_conf"],
        )
        mask_video = out_dir / "instance_masks.mp4"
        vis_instance_masks(pred["colors"], deduped, str(mask_video))
        unload_model(sam3_video)
        timings["sam_dedup_vis"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        mask_video_meta = _normalize_mask_video(
            mask_video,
            video_path,
            source_frame_indices,
            source_frame_timestamps,
        )
        timings["mask_video_normalize"] = int((time.time() - t0) * 1000)

        object_catalog_summary = None
        if _object_catalog_negotiated(payload):
            t0 = time.time()
            catalog = build_object_catalog(
                all_masks=deduped,
                colors=pred["colors"],
                world_points=pred["world_points"],
                requested_categories=categories,
                source_frame_indices=source_frame_indices,
                source_frame_timestamps=source_frame_timestamps,
                surface_area_fn=compute_surface_area_from_pointmap,
                out_dir=out_dir,
            )
            object_catalog_summary = {
                "schema_version": catalog["schema_version"],
                "total_count": catalog["total_count"],
                "returned_count": catalog["returned_count"],
                "truncated": catalog["truncated"],
                "artifact_name": OBJECT_CATALOG_JSON_NAME,
                "atlas_artifact_name": OBJECT_CROPS_ATLAS_NAME,
            }
            timings["object_catalog"] = int((time.time() - t0) * 1000)

        instances = _masks_to_instances(deduped)
        raw_count = sum(len(v) for v in all_masks.values())
        t0 = time.time()
        artifacts = _artifact_manifest(out_dir, work, payload)
        timings["artifact_delivery"] = int((time.time() - t0) * 1000)
        timings["total"] = int((time.time() - t_all) * 1000)
        delivery_error = _artifact_delivery_error(payload, artifacts)

        pipeline = [
            {"id": "intake", "status": "ok", "ms": timings.get("download")},
            {"id": "sample_frames", "status": "ok", "ms": timings.get("sample")},
            {"id": "vggt", "status": "ok", "ms": timings.get("vggt")},
            {
                "id": "room_align",
                "status": "ok" if room_alignment_applied else "not_applied" if room_align else "skipped",
                "ms": timings.get("room_align"),
            },
            {"id": "sam_dedup", "status": "ok", "ms": timings.get("sam_dedup_vis")},
            {"id": "mask_video", "status": "ok", "ms": timings.get("mask_video_normalize")},
        ]
        if object_catalog_summary is not None:
            pipeline.append(
                {"id": "object_catalog", "status": "ok", "ms": timings.get("object_catalog")}
            )
        pipeline.extend([
            {
                "id": "artifact_delivery",
                "status": "error" if delivery_error else "ok",
                "ms": timings.get("artifact_delivery"),
            },
            {"id": "emit", "status": "ok", "detail": {"instance_ids": [x["instance_id"] for x in instances]}},
        ])

        response = {
            "status": "ok",
            "mode": "full",
            "implementation": "ReplicateAnyScene spatial-guided visual deduplication (vendor pipeline)",
            "upstream": "https://github.com/xiac20/ReplicateAnyScene",
            "upstream_revision": RAS_REVISION,
            "frames_used": n_frames,
            "source_frame_indices": source_frame_indices,
            "source_frame_timestamps": source_frame_timestamps,
            "categories": categories,
            "raw_track_count": raw_count,
            "instance_count": len(instances),
            "instances": instances,
            "geometry": {
                "backend": "vggt",
                "device": device,
                "model_id": _vggt_model_id(),
                "source_revision": vggt_source_revision,
                "license_scope": _vggt_license_scope(),
                "room_align": room_align,
                "room_align_requested": room_align,
                "room_alignment_applied": room_alignment_applied,
                "wall_mask_frames": len(wall_masks),
                "floor_mask_frames": len(floor_masks),
                "artifact_export": export_meta,
            },
            "sam": {
                "backend": "sam3_video",
                "model_id": SAM3_MODEL_ID,
                "source_revision": sam3_source_revision,
                "raw_tracks": raw_count,
                "mask_video": mask_video_meta,
            },
            "artifacts": artifacts,
            "timings_ms": timings,
            "pipeline": pipeline,
            "paper_mapping": {
                "paper": "ReplicateAnyScene (arXiv:2604.10789)",
                "stage": 2,
                "title": "Spatial-Guided Visual Deduplication",
            },
        }
        if object_catalog_summary is not None:
            response["object_catalog"] = object_catalog_summary
        if delivery_error:
            response.update({"status": "error", **delivery_error})
        return response
    except Exception as e:
        return {
            "status": "error",
            "mode": "full",
            "error": str(e),
            "timings_ms": timings,
            "hint": (
                "Ensure vendor/ReplicateAnyScene is present with sam3+vggt installed, "
                "and weights under STAGE2_MODELS_DIR (VGGT/ and SAM3/) via scripts/download_weights.sh."
            ),
        }
    finally:
        if os.environ.get("STAGE2_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


def _resolve_runpod_analysis_type(payload: dict[str, Any], mode: str) -> tuple[str | None, str | None]:
    raw_analysis_type = payload.get("analysis_type")
    if raw_analysis_type is None:
        return None, None
    if not isinstance(raw_analysis_type, str) or not raw_analysis_type:
        supported = ", ".join(RUNPOD_ANALYSIS_TYPE_MODES)
        return None, f"analysis_type must be one of: {supported}"

    if raw_analysis_type in NON_RUNPOD_ANALYSIS_TYPES:
        return raw_analysis_type, (
            f"analysis_type {raw_analysis_type} is not owned by this RunPod worker; "
            f"{NON_RUNPOD_ANALYSIS_TYPES[raw_analysis_type]}"
        )

    expected_mode = RUNPOD_ANALYSIS_TYPE_MODES.get(raw_analysis_type)
    if expected_mode is None:
        supported = ", ".join(RUNPOD_ANALYSIS_TYPE_MODES)
        return raw_analysis_type, f"analysis_type must be one of: {supported}"
    if expected_mode != mode:
        return raw_analysis_type, (
            f"analysis_type {raw_analysis_type} requires mode {expected_mode}, not {mode}"
        )
    if raw_analysis_type in VGGT_1B_ANALYSIS_TYPES:
        configured_model_id = _vggt_model_id()
        if configured_model_id != DEFAULT_VGGT_MODEL_ID:
            return raw_analysis_type, (
                f"analysis_type {raw_analysis_type} requires VGGT model {DEFAULT_VGGT_MODEL_ID}; "
                f"this worker is configured for {configured_model_id}"
            )
        if VGGT_REVISION != DEFAULT_VGGT_REVISION:
            return raw_analysis_type, (
                f"analysis_type {raw_analysis_type} requires VGGT source {DEFAULT_VGGT_REVISION}; "
                f"this worker is configured for {VGGT_REVISION}"
            )
        expected_model_id = payload.get("expected_geometry_model_id")
        if expected_model_id is not None and expected_model_id != DEFAULT_VGGT_MODEL_ID:
            return raw_analysis_type, (
                f"expected_geometry_model_id must be {DEFAULT_VGGT_MODEL_ID} "
                f"for analysis_type {raw_analysis_type}"
            )
        expected_source_revision = payload.get("expected_geometry_source_revision")
        if expected_source_revision is not None and expected_source_revision != DEFAULT_VGGT_REVISION:
            return raw_analysis_type, (
                f"expected_geometry_source_revision must be {DEFAULT_VGGT_REVISION} "
                f"for analysis_type {raw_analysis_type}"
            )
    if raw_analysis_type == VGGT_OMEGA_ANALYSIS_TYPE:
        expected_model_id = payload.get("expected_geometry_model_id")
        if expected_model_id is not None and expected_model_id != VGGT_OMEGA_MODEL_ID:
            return raw_analysis_type, (
                f"expected_geometry_model_id must be {VGGT_OMEGA_MODEL_ID} "
                f"for analysis_type {raw_analysis_type}"
            )
        expected_source_revision = payload.get("expected_geometry_source_revision")
        if (
            expected_source_revision is not None
            and expected_source_revision != VGGT_OMEGA_SPACE_REVISION
        ):
            return raw_analysis_type, (
                f"expected_geometry_source_revision must be {VGGT_OMEGA_SPACE_REVISION} "
                f"for analysis_type {raw_analysis_type}"
            )
    if raw_analysis_type == "dedup_ras_vggt_sam3":
        if SAM3_REVISION != DEFAULT_SAM3_REVISION:
            return raw_analysis_type, (
                f"analysis_type {raw_analysis_type} requires SAM 3 source {DEFAULT_SAM3_REVISION}; "
                f"this worker is configured for {SAM3_REVISION}"
            )
        expected_model_id = payload.get("expected_sam_model_id")
        if expected_model_id is not None and expected_model_id != SAM3_MODEL_ID:
            return raw_analysis_type, (
                f"expected_sam_model_id must be {SAM3_MODEL_ID} "
                f"for analysis_type {raw_analysis_type}"
            )
        expected_source_revision = payload.get("expected_sam_source_revision")
        if expected_source_revision is not None and expected_source_revision != DEFAULT_SAM3_REVISION:
            return raw_analysis_type, (
                f"expected_sam_source_revision must be {DEFAULT_SAM3_REVISION} "
                f"for analysis_type {raw_analysis_type}"
            )
    return raw_analysis_type, None


def _attach_success_provenance(
    result: dict[str, Any],
    *,
    mode: str,
    analysis_type: str | None,
    object_catalog_version: int | None = None,
) -> dict[str, Any]:
    # Legacy callers predate versioned analysis types. Preserve their response
    # shape, but never manufacture provenance for the typed contract: the
    # model runner must report what it actually loaded and we verify it here.
    if analysis_type is None:
        return result
    enriched = {**result, "analysis_type": analysis_type}
    if result.get("status") != "ok":
        return enriched

    def provenance_error(component: str) -> dict[str, Any]:
        return {
            "status": "error",
            "mode": mode,
            "analysis_type": analysis_type,
            "error": (
                f"{component} execution provenance did not match the immutable "
                f"analysis_type {analysis_type} contract"
            ),
        }

    if analysis_type == "validation_v1":
        source = result.get("source")
        if not isinstance(source, dict) or (
            source.get("ras_revision") != RAS_REVISION
            or source.get("vggt_revision") != DEFAULT_VGGT_REVISION
            or source.get("sam3_revision") != DEFAULT_SAM3_REVISION
            or source.get("sam3_required") is not True
            or source.get("weights_required") is not False
        ):
            return provenance_error("source bootstrap")
    if analysis_type == OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE:
        object_catalog = result.get("object_catalog")
        artifacts = result.get("artifacts")
        if (
            result.get("synthetic_transport_canary") is not True
            or result.get("frames_used") != 1
            or result.get("categories") != [OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY]
            or result.get("raw_track_count") != 0
            or result.get("instance_count") != 0
            or result.get("instances") != []
            or not isinstance(object_catalog, dict)
            or object_catalog.get("schema_version") != "palatial.object_catalog.v1"
            or object_catalog.get("total_count") != 0
            or object_catalog.get("returned_count") != 0
            or object_catalog.get("truncated") is not False
            or not isinstance(artifacts, dict)
            or artifacts.get("complete") is not True
            or artifacts.get("required_files") != list(OBJECT_CATALOG_ARTIFACTS)
            or artifacts.get("files") != list(OBJECT_CATALOG_ARTIFACTS)
        ):
            return provenance_error("object catalog transport canary")
    if analysis_type in VGGT_1B_ANALYSIS_TYPES:
        geometry = result.get("geometry")
        if not isinstance(geometry, dict) or (
            geometry.get("model_id") != DEFAULT_VGGT_MODEL_ID
            or geometry.get("source_revision") != DEFAULT_VGGT_REVISION
        ):
            return provenance_error("VGGT")
    if analysis_type == VGGT_OMEGA_ANALYSIS_TYPE:
        geometry = result.get("geometry")
        if not isinstance(geometry, dict) or (
            geometry.get("backend") != "huggingface_space"
            or geometry.get("model_id") != VGGT_OMEGA_MODEL_ID
            or geometry.get("source_revision") != VGGT_OMEGA_SPACE_REVISION
            or geometry.get("space_id") != VGGT_OMEGA_SPACE_ID
            or geometry.get("space_revision") != VGGT_OMEGA_SPACE_REVISION
            or geometry.get("github_source_revision") != VGGT_OMEGA_GITHUB_REVISION
            or geometry.get("model_repository_revision") != VGGT_OMEGA_MODEL_REVISION
            or geometry.get("checkpoint_filename") != VGGT_OMEGA_CHECKPOINT
            or geometry.get("provenance_level") != VGGT_OMEGA_PROVENANCE_LEVEL
            or geometry.get("license_scope") != "research_noncommercial"
            or geometry.get("sam3_required") is not False
            or geometry.get("execution_orchestrator") != "runpod"
        ):
            return provenance_error("VGGT-Omega")
    if analysis_type == "dedup_ras_vggt_sam3":
        sam = result.get("sam")
        if not isinstance(sam, dict) or (
            sam.get("model_id") != SAM3_MODEL_ID
            or sam.get("source_revision") != DEFAULT_SAM3_REVISION
        ):
            return provenance_error("SAM 3")
    if object_catalog_version == OBJECT_CATALOG_VERSION:
        object_catalog = result.get("object_catalog")
        artifacts = result.get("artifacts")
        frames_used = result.get("frames_used")
        source_frame_indices = result.get("source_frame_indices")
        source_frame_timestamps = result.get("source_frame_timestamps")
        indices_valid = (
            source_frame_indices is None
            or isinstance(source_frame_indices, list)
            and len(source_frame_indices) == frames_used
            and all(type(value) is int and value >= 0 for value in source_frame_indices)
            and all(
                right > left
                for left, right in zip(source_frame_indices, source_frame_indices[1:])
            )
        )
        timestamps_valid = (
            source_frame_timestamps is None
            or isinstance(source_frame_timestamps, list)
            and len(source_frame_timestamps) == frames_used
            and all(
                type(value) in {int, float} and math.isfinite(value) and value >= 0
                for value in source_frame_timestamps
            )
            and all(
                right > left
                for left, right in zip(source_frame_timestamps, source_frame_timestamps[1:])
            )
        )
        timeline_valid = (
            isinstance(frames_used, int)
            and not isinstance(frames_used, bool)
            and 1 <= frames_used <= 160
            and indices_valid
            and timestamps_valid
        )
        total_count = object_catalog.get("total_count") if isinstance(object_catalog, dict) else None
        returned_count = (
            object_catalog.get("returned_count") if isinstance(object_catalog, dict) else None
        )
        truncated = object_catalog.get("truncated") if isinstance(object_catalog, dict) else None
        counts_valid = (
            type(total_count) is int
            and type(returned_count) is int
            and total_count >= 0
            and 0 <= returned_count <= min(total_count, 128)
            and type(truncated) is bool
            and truncated is (total_count > returned_count)
        )
        if (
            not timeline_valid
            or not counts_valid
            or result.get("synthetic_transport_canary") is not None
            or not isinstance(object_catalog, dict)
            or object_catalog.get("schema_version") != "palatial.object_catalog.v1"
            or object_catalog.get("artifact_name") != OBJECT_CATALOG_JSON_NAME
            or object_catalog.get("atlas_artifact_name") != OBJECT_CROPS_ATLAS_NAME
            or not isinstance(artifacts, dict)
            or artifacts.get("complete") is not True
            or artifacts.get("required_files") != [
                "point_cloud.glb",
                "instance_masks.mp4",
                *OBJECT_CATALOG_ARTIFACTS,
            ]
        ):
            return provenance_error("object catalog")

    return enriched


def run_stage2(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode") or os.environ.get("STAGE2_MODE_DEFAULT") or "full").lower()
    if mode not in {"dry_run", "geometry", "full"}:
        return {"status": "error", "mode": mode, "error": "mode must be dry_run, geometry, or full"}
    analysis_type, analysis_error = _resolve_runpod_analysis_type(payload, mode)
    if analysis_error:
        return {
            "status": "error",
            "mode": mode,
            "analysis_type": analysis_type,
            "error": analysis_error,
        }

    def input_error(message: str) -> dict[str, Any]:
        result = {"status": "error", "mode": mode, "error": message}
        if analysis_type is not None:
            result["analysis_type"] = analysis_type
        return result

    _catalog_version, catalog_error = _resolve_object_catalog_version(payload, mode)
    if catalog_error:
        return input_error(catalog_error)
    is_transport_canary, canary_error = _resolve_object_catalog_transport_canary(
        payload,
        mode,
        analysis_type,
    )
    if canary_error:
        return input_error(canary_error)
    if is_transport_canary:
        result = run_object_catalog_transport_canary({**payload, "mode": mode})
        return _attach_success_provenance(
            result,
            mode=mode,
            analysis_type=analysis_type,
        )

    try:
        max_frames = int(payload.get("max_frames") or (24 if mode == "dry_run" else 48))
    except (TypeError, ValueError):
        return input_error("max_frames must be an integer from 2 to 160")
    categories = payload.get("categories") or []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    categories = list(dict.fromkeys(str(c).strip() for c in categories if str(c).strip()))
    if not categories or len(categories) > 8 or any(len(c) > 64 for c in categories):
        return input_error("use 1-8 categories, each at most 64 characters")
    if max_frames < 2 or max_frames > 160:
        return input_error("max_frames must be an integer from 2 to 160")
    payload = {**payload, "mode": mode, "max_frames": max_frames, "categories": categories}
    if mode == "dry_run":
        result = run_stage2_dry(payload)
    elif mode == "geometry":
        result = (
            run_stage2_omega_geometry(payload)
            if analysis_type == VGGT_OMEGA_ANALYSIS_TYPE
            else run_stage2_geometry(payload)
        )
    else:
        result = run_stage2_full(payload)
    return _attach_success_provenance(
        result,
        mode=mode,
        analysis_type=analysis_type,
        object_catalog_version=_catalog_version,
    )
