"""
Stage 2 only: Spatial-Guided Visual Deduplication.

This module does NOT reimplement VGGT/SAM/dedup. It calls the public
ReplicateAnyScene Stage 2 path (same sequence as their main.py Stage 2 block).

Upstream: https://github.com/xiac20/ReplicateAnyScene
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

# Resolve vendor checkout. Layout:
#   services/replicate-any-scene-stage2/
#     stage2_service.py
#     vendor/ReplicateAnyScene/   (paper repo)
RAS_ROOT = Path(os.environ.get("RAS_ROOT", Path(__file__).resolve().parent / "vendor" / "ReplicateAnyScene")).resolve()



def _ensure_ras_installed() -> None:
    """Install paper repo + submodules if missing (first full call only)."""
    import subprocess
    ras = Path(os.environ.get("RAS_ROOT", Path(__file__).resolve().parent / "vendor" / "ReplicateAnyScene"))
    if not (ras / "main.py").is_file():
        ras.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/xiac20/ReplicateAnyScene.git", str(ras)])
    # public facebook clones if submodules empty
    if not (ras / "vggt" / "setup.py").exists() and not (ras / "vggt" / "pyproject.toml").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/vggt.git", str(ras / "vggt")])
    if not (ras / "sam3" / "pyproject.toml").exists() and not (ras / "sam3" / "setup.py").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/sam3.git", str(ras / "sam3")])
    models_dir = Path(os.environ.get("STAGE2_MODELS_DIR", ras / "models")).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    if not (models_dir / "VGGT" / ".stage2_ready").exists():
        subprocess.check_call(["python", "-m", "pip", "install", "-q", "huggingface_hub[cli]>=0.23"])
        # best-effort; may require HF_TOKEN
        subprocess.call(["hf", "download", "facebook/VGGT-1B", "--local-dir", str(models_dir / "VGGT")])
        subprocess.call(["hf", "download", "facebook/sam3", "--local-dir", str(models_dir / "SAM3")])
        (models_dir / "VGGT" / ".stage2_ready").touch()
        (models_dir / "SAM3" / ".stage2_ready").touch()


def _ensure_ras_on_path() -> Path:
    if not RAS_ROOT.is_dir():
        raise RuntimeError(
            f"ReplicateAnyScene checkout not found at {RAS_ROOT}. "
            "Clone https://github.com/xiac20/ReplicateAnyScene into vendor/ReplicateAnyScene "
            "and init submodules sam3 + vggt (not sam-3d-objects)."
        )
    root = str(RAS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Models loader expects cwd-relative ./models and ./sam3 paths.
    os.chdir(root)
    return RAS_ROOT


def _download_video(video_url: str, dest_dir: Path, timeout_s: int = 180) -> Path:
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
    import base64

    b64 = payload.get("video_b64")
    if isinstance(b64, str) and b64.strip():
        raw = base64.b64decode(b64)
        dest = work / "input.mp4"
        dest.write_bytes(raw)
        return dest
    video_url = str(payload.get("video_url") or "").strip()
    if not video_url:
        raise RuntimeError("video_url or video_b64 is required")
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
        # Contract-shaped fake instances for Lab wiring (not real detections).
        instances = []
        for cat in categories:
            instances.append(
                {
                    "category": cat,
                    "instance_id": f"{cat}_0",
                    "frame_ids": idxs[:2],
                    "frame_count": min(2, len(idxs)),
                    "note": "dry_run synthetic instance — not RAS model output",
                }
            )
        return {
            "status": "ok",
            "mode": "dry_run",
            "implementation": "replicate-any-scene-stage2 dry_run (no VGGT/SAM)",
            "upstream": "https://github.com/xiac20/ReplicateAnyScene",
            "frames_used": frames_ok,
            "source_frame_indices": idxs,
            "video_meta": {"total_frames": total, "fps": fps, "width": w, "height": h, "sampled": frames_ok},
            "categories": categories,
            "raw_track_count": len(instances),
            "instance_count": len(instances),
            "instances": instances,
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
            "trace": traceback.format_exc()[-2000:],
        }
    finally:
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
        # Models live under RAS_ROOT/models or STAGE2_MODELS_DIR symlinked as ./models
        models_dir = Path(os.environ.get("STAGE2_MODELS_DIR", ras_root / "models")).resolve()
        models_link = ras_root / "models"
        if models_dir != models_link:
            if models_link.is_symlink() or models_link.exists():
                if models_link.is_symlink():
                    models_link.unlink()
                elif models_link.is_dir() and not any(models_link.iterdir()):
                    models_link.rmdir()
            if not models_link.exists():
                models_link.symlink_to(models_dir, target_is_directory=True)

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

        artifact_root = os.environ.get("STAGE2_ARTIFACT_DIR", "").strip()
        artifacts: dict[str, Any] = {
            "work_dir": str(out_dir),
            "mask_video": str(mask_video) if mask_video.exists() else None,
            "point_cloud_ply": str(out_dir / "point_cloud.ply") if (out_dir / "point_cloud.ply").exists() else None,
        }
        if artifact_root:
            dest = Path(artifact_root) / work.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(out_dir, dest)
            artifacts["persisted_dir"] = str(dest)

        return {
            "status": "ok",
            "mode": "full",
            "implementation": "ReplicateAnyScene main.py Stage 2 (vendor)",
            "upstream": "https://github.com/xiac20/ReplicateAnyScene",
            "ras_root": str(ras_root),
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
            "trace": traceback.format_exc()[-3000:],
            "timings_ms": timings,
            "hint": (
                "Ensure vendor/ReplicateAnyScene is present with sam3+vggt installed, "
                "and weights under STAGE2_MODELS_DIR (VGGT/ and SAM3/) via scripts/download_weights.sh."
            ),
        }
    finally:
        if os.environ.get("STAGE2_KEEP_WORK") != "1" and not os.environ.get("STAGE2_ARTIFACT_DIR", "").strip():
            shutil.rmtree(work, ignore_errors=True)


def run_stage2(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode") or os.environ.get("STAGE2_MODE_DEFAULT") or "full").lower()
    if mode == "dry_run":
        return run_stage2_dry(payload)
    return run_stage2_full(payload)
