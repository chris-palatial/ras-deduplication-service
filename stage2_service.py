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
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Resolve vendor checkout. Layout:
#   services/replicate-any-scene-stage2/
#     stage2_service.py
#     vendor/ReplicateAnyScene/   (paper repo)
RAS_ROOT = Path(os.environ.get("RAS_ROOT", Path(__file__).resolve().parent / "vendor" / "ReplicateAnyScene")).resolve()
RAS_REVISION = os.environ.get("RAS_REVISION", "671191457e7244d9337ef3faf558ee92bbf9bf73")
VGGT_REVISION = os.environ.get("VGGT_REVISION", "44b3afbd1869d8bde4894dd8ea1e293112dd5eba")
SAM3_REVISION = os.environ.get("SAM3_REVISION", "bfbed072a07a6a52c8d5fdc75a7a186251a835b1")
MAX_INLINE_VIDEO_BYTES = 6 * 1024 * 1024



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


def _clone_if_missing(url: str, dest: Path, revision: str) -> None:
    if dest.is_dir() and (dest / ".git").exists():
        import subprocess

        head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if head == revision:
            return
        _run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", revision])
        _run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"])
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)])
    _run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", revision])
    _run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"])


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


def _vggt_weights_ok(vggt_dir: Path) -> bool:
    if not vggt_dir.is_dir():
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
    vggt_ok = _vggt_weights_ok(models_dir / "VGGT")
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
    from huggingface_hub import snapshot_download

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or True
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    vggt_dir = models_dir / "VGGT"
    if not _vggt_weights_ok(vggt_dir):
        print("[stage2] downloading facebook/VGGT-1B ...", flush=True)
        snapshot_download(
            repo_id="facebook/VGGT-1B",
            local_dir=str(vggt_dir),
            token=token,
        )


def _download_sam3_weights(models_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or True
    )
    vggt_dir = models_dir / "VGGT"
    sam_dir = models_dir / "SAM3"
    if not _find_sam3_pt(sam_dir):
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
            f"Stage2 weights incomplete under {models_dir}. "
            f"VGGT ok={_vggt_weights_ok(vggt_dir)} SAM3 pt={_find_sam3_pt(sam_dir)}"
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


def _ensure_ras_installed(*, require_sam3: bool = True) -> None:
    """Install paper repo + packages + weights (first full call, then cached on volume)."""
    ras = _ras_root()
    models_dir = _models_dir(ras)

    _clone_if_missing("https://github.com/xiac20/ReplicateAnyScene.git", ras, RAS_REVISION)
    # Keep the exact upstream gitlinks used by the reviewed RAS revision.
    _clone_if_missing("https://github.com/facebookresearch/vggt.git", ras / "vggt", VGGT_REVISION)
    if require_sam3:
        _clone_if_missing("https://github.com/facebookresearch/sam3.git", ras / "sam3", SAM3_REVISION)

    # Install dependencies before importing huggingface_hub to download weights.
    _ensure_python_packages(ras, require_sam3=require_sam3)

    if not _vggt_weights_ok(models_dir / "VGGT"):
        _download_vggt_weights(models_dir)

    # Older builds wrote .stage2_ready even when SAM3 download failed — ignore markers alone.
    if require_sam3 and not _weights_ready(models_dir):
        for marker in (models_dir / "VGGT" / ".stage2_ready", models_dir / "SAM3" / ".stage2_ready"):
            if marker.exists():
                marker.unlink()
        _download_sam3_weights(models_dir)
    elif require_sam3:
        _ensure_sam3_pt_layout(models_dir)


def _ensure_ras_on_path() -> Path:
    ras = _ras_root()
    if not ras.is_dir() or not (ras / "main.py").is_file():
        raise RuntimeError(
            f"ReplicateAnyScene checkout not found at {ras}. "
            "Clone https://github.com/xiac20/ReplicateAnyScene and install vggt+sam3 packages."
        )
    root = str(ras)
    # Prefer RAS root for `import src.*`; keep vendor package installs for vggt/sam3.
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    # Models loader expects cwd-relative ./models and ./sam3 paths.
    os.chdir(root)
    return ras


def _download_video(video_url: str, dest_dir: Path, timeout_s: int = 180) -> Path:
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(video_url).path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        suffix = ".mp4"
    dest = dest_dir / f"input{suffix}"
    headers = {}
    # Cloudflare Access service tokens (optional): allow RunPod to GET Access-protected media.
    cid = os.environ.get("CF_ACCESS_CLIENT_ID") or os.environ.get("CF_Access_Client_Id")
    csec = os.environ.get("CF_ACCESS_CLIENT_SECRET") or os.environ.get("CF_Access_Client_Secret")
    if cid and csec:
        headers["CF-Access-Client-Id"] = cid
        headers["CF-Access-Client-Secret"] = csec
    if not (video_url.startswith("http://") or video_url.startswith("https://")):
        raise RuntimeError(f"refusing non-HTTP video_url: {video_url[:64]}")
    with requests.get(video_url, stream=True, timeout=timeout_s, headers=headers) as r:
        if r.status_code in (401, 403, 302) or "cloudflareaccess" in (r.headers.get("location") or "").lower():
            raise RuntimeError(
                f"video URL blocked (HTTP {r.status_code}). "
                "If the file is on agents.palatial.cloud, Cloudflare Access is rejecting the GPU worker. "
                "Use a public HTTPS URL, STAGE2_PUBLIC_MEDIA_BASE, or set CF_ACCESS_CLIENT_ID/SECRET on the endpoint."
            )
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest


def _materialize_video(payload: dict[str, Any], work: Path) -> Path:
    """Accept video_url and/or video_b64 (Lab may inline small uploads)."""
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
            raise RuntimeError("inline video exceeds the 6 MB service limit")
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
            "Redeploy/restart the endpoint worker and use a clip under ~6MB, "
            "or configure STAGE2_PUBLIC_MEDIA_BASE / CF Access service tokens."
        )
    if not (video_url.startswith("http://") or video_url.startswith("https://")):
        raise RuntimeError(f"unsupported video_url scheme: {video_url[:40]}")
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


def run_stage2_dry(payload: dict[str, Any]) -> dict[str, Any]:
    """Wire-test path: download + frame sample only, no VGGT/SAM weights."""
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
            return {"status": "error", "error": f"could not open video: {video_url}", "mode": "dry_run"}
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
        return {
            "status": "ok",
            "mode": "dry_run",
            "implementation": "replicate-any-scene-stage2 dry_run (no VGGT/SAM)",
            "upstream": "https://github.com/xiac20/ReplicateAnyScene",
            "frames_used": frames_ok,
            "source_frame_indices": idxs,
            "video_meta": {"total_frames": total, "fps": fps, "width": w, "height": h, "sampled": frames_ok},
            "categories": categories,
            "raw_track_count": 0,
            "instance_count": 0,
            "instances": [],
            "pipeline": [
                {"id": "intake", "name": "Video intake", "status": "ok"},
                {"id": "sample_frames", "name": "Frame sampling", "status": "ok", "detail": {"frames_used": frames_ok}},
                {"id": "vggt", "name": "VGGT", "status": "skipped_dry_run"},
                {"id": "sam", "name": "SAM3", "status": "skipped_dry_run"},
                {"id": "dedup", "name": "Spatial dedup", "status": "skipped_dry_run"},
            ],
            "timings_ms": {"total": int((time.time() - t0) * 1000)},
        }
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


def _artifact_manifest(out_dir: Path, work: Path) -> dict[str, Any]:
    """Return only durable artifact claims; temporary worker paths are never API URLs."""
    artifact_root = os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()
    files = [p.name for p in out_dir.iterdir() if p.is_file()] if out_dir.is_dir() else []
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


def _artifact_exports_enabled() -> bool:
    """Only spend time exporting geometry when the files will remain inspectable."""
    return bool(os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()) or os.environ.get("STAGE2_KEEP_WORK") == "1"


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
        if _artifact_exports_enabled():
            try:
                import numpy as np

                if pred.get("point_cloud_data") is not None:
                    pred["point_cloud_data"].export(str(out_dir / "point_cloud.ply"))
                np.savetxt(str(out_dir / "intrinsic.txt"), pred["intrinsic"])
            except Exception as exc:
                pred["export_warning"] = str(exc)
        timings["artifact_export"] = int((time.time() - t0) * 1000)

        timings["total"] = int((time.time() - t_all) * 1000)
        return {
            "status": "ok",
            "mode": "geometry",
            "implementation": "ReplicateAnyScene Stage 2 VGGT geometry preflight",
            "upstream_revision": RAS_REVISION,
            "frames_used": n_frames,
            "source_frame_indices": list(range(n_frames)),
            "categories": payload["categories"],
            "raw_track_count": 0,
            "instance_count": 0,
            "instances": [],
            "geometry": {
                "backend": "vggt",
                "device": device,
                "world_points_shape": list(pred["world_points"].shape),
                "sam3_required": False,
            },
            "sam": {"backend": "not_run", "reason": "geometry mode intentionally skips SAM3"},
            "artifacts": _artifact_manifest(out_dir, work),
            "timings_ms": timings,
            "pipeline": [
                {"id": "intake", "name": "Video intake", "status": "ok", "ms": timings.get("download")},
                {"id": "sample_frames", "name": "Frame sampling", "status": "ok", "ms": timings.get("sample")},
                {"id": "vggt", "name": "VGGT geometry", "status": "ok", "ms": timings.get("vggt")},
                {"id": "sam", "name": "SAM3", "status": "skipped_geometry_mode"},
                {"id": "dedup", "name": "Spatial dedup", "status": "skipped_geometry_mode"},
            ],
        }
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
        _link_models_dir(ras_root)

        import torch
        import cv2
        import numpy as np

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
        timings["sample"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        vggt_model = load_vggt_model().to(device)
        pred = vggt_predict(frames, vggt_model)
        unload_model(vggt_model)
        timings["vggt"] = int((time.time() - t0) * 1000)

        wall_masks, floor_masks = [], []
        if room_align:
            t0 = time.time()
            sam3_image = load_sam3_image_model()
            wall_masks, floor_masks = segment_wall_and_floor(pred["colors"], sam3_image)
            R, t = align_to_room_coordinate_system(pred["world_points"], wall_masks, floor_masks)
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
        try:
            if pred.get("point_cloud_data") is not None:
                pred["point_cloud_data"].export(str(out_dir / "point_cloud.ply"))
            np.savetxt(str(out_dir / "intrinsic.txt"), pred["intrinsic"])
        except Exception as e:
            pred["export_warning"] = str(e)

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

        instances = _masks_to_instances(deduped)
        raw_count = sum(len(v) for v in all_masks.values())
        timings["total"] = int((time.time() - t_all) * 1000)

        artifacts = _artifact_manifest(out_dir, work)

        return {
            "status": "ok",
            "mode": "full",
            "implementation": "ReplicateAnyScene main.py Stage 2 (vendor)",
            "upstream": "https://github.com/xiac20/ReplicateAnyScene",
            "upstream_revision": RAS_REVISION,
            "frames_used": n_frames,
            "source_frame_indices": list(range(n_frames)),
            "categories": categories,
            "raw_track_count": raw_count,
            "instance_count": len(instances),
            "instances": instances,
            "geometry": {
                "backend": "vggt",
                "device": device,
                "room_align": room_align,
                "wall_mask_frames": len(wall_masks),
                "floor_mask_frames": len(floor_masks),
            },
            "sam": {"backend": "sam3_video", "raw_tracks": raw_count},
            "artifacts": artifacts,
            "timings_ms": timings,
            "pipeline": [
                {"id": "intake", "status": "ok", "ms": timings.get("download")},
                {"id": "sample_frames", "status": "ok", "ms": timings.get("sample")},
                {"id": "vggt", "status": "ok", "ms": timings.get("vggt")},
                {"id": "room_align", "status": "ok" if room_align else "skipped", "ms": timings.get("room_align")},
                {"id": "sam_dedup", "status": "ok", "ms": timings.get("sam_dedup_vis")},
                {"id": "emit", "status": "ok", "detail": {"instance_ids": [x["instance_id"] for x in instances]}},
            ],
            "paper_mapping": {
                "paper": "ReplicateAnyScene (arXiv:2604.10789)",
                "stage": 2,
                "title": "Spatial-Guided Visual Deduplication",
            },
        }
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


def run_stage2(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode") or os.environ.get("STAGE2_MODE_DEFAULT") or "full").lower()
    if mode not in {"dry_run", "geometry", "full"}:
        return {"status": "error", "mode": mode, "error": "mode must be dry_run, geometry, or full"}
    try:
        max_frames = int(payload.get("max_frames") or (24 if mode == "dry_run" else 48))
    except (TypeError, ValueError):
        return {"status": "error", "mode": mode, "error": "max_frames must be an integer from 2 to 160"}
    categories = payload.get("categories") or []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    categories = list(dict.fromkeys(str(c).strip() for c in categories if str(c).strip()))
    if not categories or len(categories) > 8 or any(len(c) > 64 for c in categories):
        return {"status": "error", "mode": mode, "error": "use 1-8 categories, each at most 64 characters"}
    if max_frames < 2 or max_frames > 160:
        return {"status": "error", "mode": mode, "error": "max_frames must be an integer from 2 to 160"}
    payload = {**payload, "mode": mode, "max_frames": max_frames, "categories": categories}
    if mode == "dry_run":
        return run_stage2_dry(payload)
    if mode == "geometry":
        return run_stage2_geometry(payload)
    return run_stage2_full(payload)
