"""
Stage 2 only: Spatial-Guided Visual Deduplication.

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
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Resolve vendor checkout. Layout:
#   services/replicate-any-scene-stage2/
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
ARTIFACT_MEDIA_TYPES = {
    "point_cloud.glb": "model/gltf-binary",
    "point_cloud.ply": "application/octet-stream",
    "instance_masks.mp4": "video/mp4",
    "camera_intrinsics.json": "application/json",
}
REQUIRED_ARTIFACTS = {
    "geometry": ("point_cloud.glb",),
    "full": ("point_cloud.glb", "instance_masks.mp4"),
}
RUNPOD_ANALYSIS_TYPE_MODES = {
    "validation_v1": "dry_run",
    "geometry_vggt_1b": "geometry",
    "dedup_ras_vggt_sam3": "full",
}
NON_RUNPOD_ANALYSIS_TYPES = {
    "geometry_vggt_omega_1b": "route it to the dedicated VGGT-Omega endpoint",
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


def _artifact_manifest(out_dir: Path, work: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deliver artifacts durably and return receipts, never temporary paths."""
    artifact_root = os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()
    files = sorted(
        p.name
        for p in out_dir.iterdir()
        if p.is_file() and p.name in ARTIFACT_MEDIA_TYPES
    ) if out_dir.is_dir() else []
    upload = _upload_ticket(payload)
    if upload:
        from artifact_upload import artifact_failure_record, upload_artifact_file

        mode = str((payload or {}).get("mode") or "")
        required_files = list(REQUIRED_ARTIFACTS.get(mode, ()))
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
                receipts.append(
                    upload_artifact_file(upload, out_dir / name, ARTIFACT_MEDIA_TYPES[name])
                )
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
        missing = list(REQUIRED_ARTIFACTS.get(str((payload or {}).get("mode") or ""), ()))
    return {
        "error_code": "artifact_delivery_failed",
        "error": "Required result files could not be delivered to durable storage.",
        "artifact_delivery": {
            "required_files": artifacts.get("required_files", []),
            "missing_required": missing,
            "errors": artifacts.get("errors", []),
        },
    }


def _artifact_exports_enabled(payload: dict[str, Any] | None = None) -> bool:
    """Only spend time exporting geometry when the files will remain inspectable."""
    return bool(_upload_ticket(payload)) or bool(os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()) or os.environ.get("STAGE2_KEEP_WORK") == "1"


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


def run_stage2_full(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Full Stage 2 using ReplicateAnyScene sources only.

    Mirrors main.py Stage 2 (VGGT → room align → SAM3 video track →
    self/cross category dedup → mask video). Does not run Stage 3+.
    """
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
        from src.geometry_utils import align_to_room_coordinate_system, align_vggt_predictions
        from src.vggt_predict import vggt_predict
        from src.object_segmentation import segment_and_track, segment_wall_and_floor
        from src.sg_deduplication import cross_category_deduplicate, self_category_deduplicate

        device = "cuda" if torch.cuda.is_available() else "cpu"

        t0 = time.time()
        frames = load_video_frames(str(video_path), max_frames).to(device)
        n_frames = int(frames.shape[0])
        source_frame_plan = _source_frame_plan(video_path, n_frames)
        source_frame_indices = source_frame_plan[0] if source_frame_plan else None
        source_frame_timestamps = source_frame_plan[1] if source_frame_plan else None
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

        instances = _masks_to_instances(deduped)
        raw_count = sum(len(v) for v in all_masks.values())
        t0 = time.time()
        artifacts = _artifact_manifest(out_dir, work, payload)
        timings["artifact_delivery"] = int((time.time() - t0) * 1000)
        timings["total"] = int((time.time() - t_all) * 1000)
        delivery_error = _artifact_delivery_error(payload, artifacts)

        response = {
            "status": "ok",
            "mode": "full",
            "implementation": "ReplicateAnyScene spatial-guided visual deduplication (vendor pipeline)",
            "upstream": "https://github.com/xiac20/ReplicateAnyScene",
            "upstream_revision": RAS_REVISION,
            "frames_used": n_frames,
            "source_frame_indices": source_frame_indices,
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
            "pipeline": [
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
                {
                    "id": "artifact_delivery",
                    "status": "error" if delivery_error else "ok",
                    "ms": timings.get("artifact_delivery"),
                },
                {"id": "emit", "status": "ok", "detail": {"instance_ids": [x["instance_id"] for x in instances]}},
            ],
            "paper_mapping": {
                "paper": "ReplicateAnyScene (arXiv:2604.10789)",
                "stage": 2,
                "title": "Spatial-Guided Visual Deduplication",
            },
        }
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
    if analysis_type in VGGT_1B_ANALYSIS_TYPES:
        geometry = result.get("geometry")
        if not isinstance(geometry, dict) or (
            geometry.get("model_id") != DEFAULT_VGGT_MODEL_ID
            or geometry.get("source_revision") != DEFAULT_VGGT_REVISION
        ):
            return provenance_error("VGGT")
    if analysis_type == "dedup_ras_vggt_sam3":
        sam = result.get("sam")
        if not isinstance(sam, dict) or (
            sam.get("model_id") != SAM3_MODEL_ID
            or sam.get("source_revision") != DEFAULT_SAM3_REVISION
        ):
            return provenance_error("SAM 3")

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
        result = run_stage2_geometry(payload)
    else:
        result = run_stage2_full(payload)
    return _attach_success_provenance(result, mode=mode, analysis_type=analysis_type)
