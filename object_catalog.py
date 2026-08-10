"""Bounded representative-image catalog for RAS deduplicated instances.

The catalog reuses the public RAS Stage-3 surface-area score for a bounded
shortlist of sampled views.  It does not discover categories and it does not
claim ground-truth identities: every row is one Stage-2 spatial-deduplication
hypothesis for a caller-requested category.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Callable


OBJECT_CATALOG_VERSION = 1
OBJECT_CATALOG_SCHEMA_VERSION = "palatial.object_catalog.v1"
OBJECT_CATALOG_JSON_NAME = "object_catalog.json"
OBJECT_CROPS_ATLAS_NAME = "object_crops.jpg"
OBJECT_CATALOG_JSON_MAX_BYTES = 512 * 1024
OBJECT_CROPS_ATLAS_MAX_BYTES = 8 * 1024 * 1024
OBJECT_CATALOG_MAX_OBJECTS = 128
OBJECT_CATALOG_CELL_SIZE = 224
OBJECT_CATALOG_MAX_COLUMNS = 12
OBJECT_CATALOG_MAX_SELECTOR_FRAMES = 16
OBJECT_CATALOG_MAX_SURFACE_PIXELS = 25_000
OBJECT_CROPS_MAX_WIDTH = OBJECT_CATALOG_MAX_COLUMNS * OBJECT_CATALOG_CELL_SIZE
OBJECT_CROPS_MAX_HEIGHT = math.ceil(
    OBJECT_CATALOG_MAX_OBJECTS / OBJECT_CATALOG_MAX_COLUMNS
) * OBJECT_CATALOG_CELL_SIZE

_SELECTION_SURFACE = "bounded_ras_stage3_surface_area_v1"
_SELECTION_FALLBACK = "largest_mask_fallback_v1"
_JPEG_QUALITIES = (88, 82, 76, 70, 64, 58, 52, 46, 40)

SurfaceAreaFn = Callable[[Any, Any], Any]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_frame_id(value: Any, frame_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("object catalog track has a non-integer frame_id")
    frame_id = int(value)
    if frame_id < 0 or frame_id >= frame_count:
        raise ValueError("object catalog track frame_id is outside sampled frames")
    return frame_id


def _normalize_track(
    track: Any,
    *,
    frame_count: int,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    import numpy as np

    if not isinstance(track, (list, tuple)) or not track:
        raise ValueError("object catalog instance has no mask track")
    merged: dict[int, Any] = {}
    for evidence in track:
        if not isinstance(evidence, dict) or "frame_id" not in evidence or "mask" not in evidence:
            raise ValueError("object catalog track evidence is malformed")
        frame_id = _normalize_frame_id(evidence["frame_id"], frame_count)
        mask = np.asarray(evidence["mask"])
        if mask.shape != (height, width):
            raise ValueError("object catalog mask shape does not match model frame")
        binary = mask.astype(bool, copy=False)
        if not np.any(binary):
            continue
        if frame_id in merged:
            merged[frame_id] = np.logical_or(merged[frame_id], binary)
        else:
            merged[frame_id] = binary.copy()
    if not merged:
        raise ValueError("object catalog instance has no non-empty masks")
    return [
        {"frame_id": frame_id, "mask": merged[frame_id]}
        for frame_id in sorted(merged)
    ]


def _bounded_surface_inputs(pointmap: Any, mask: Any) -> tuple[Any, Any]:
    """Bound the public Stage-3 Delaunay scorer without changing its interface."""
    import numpy as np

    pixel_count = int(np.count_nonzero(mask))
    if pixel_count <= OBJECT_CATALOG_MAX_SURFACE_PIXELS:
        return pointmap, mask
    stride = max(2, int(math.ceil(math.sqrt(pixel_count / OBJECT_CATALOG_MAX_SURFACE_PIXELS))))
    bounded_points = pointmap[::stride, ::stride]
    bounded_mask = mask[::stride, ::stride]
    while int(np.count_nonzero(bounded_mask)) > OBJECT_CATALOG_MAX_SURFACE_PIXELS:
        stride += 1
        bounded_points = pointmap[::stride, ::stride]
        bounded_mask = mask[::stride, ::stride]
    return bounded_points, bounded_mask


def select_representative_frame(
    world_points: Any,
    track: Any,
    surface_area_fn: SurfaceAreaFn,
    *,
    model_frame_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Select a bounded RAS Stage-3 view with a deterministic mask fallback."""
    import numpy as np

    points = np.asarray(world_points)
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[0] <= 0:
        raise ValueError("object catalog world_points must have shape (S,H,W,3)")
    height, width = points.shape[1:3]
    if model_frame_shape is not None and tuple(model_frame_shape) != (height, width):
        raise ValueError("object catalog world points do not match model frame shape")
    normalized = _normalize_track(
        track,
        frame_count=points.shape[0],
        height=height,
        width=width,
    )
    by_largest_mask = sorted(
        normalized,
        key=lambda item: (-int(np.count_nonzero(item["mask"])), item["frame_id"]),
    )
    shortlist = by_largest_mask[:OBJECT_CATALOG_MAX_SELECTOR_FRAMES]

    best_frame: dict[str, Any] | None = None
    best_area = 0.0
    for evidence in sorted(shortlist, key=lambda item: item["frame_id"]):
        pointmap, bounded_mask = _bounded_surface_inputs(
            points[evidence["frame_id"]], evidence["mask"]
        )
        try:
            area = float(surface_area_fn(pointmap, bounded_mask))
        except Exception:
            continue
        if not math.isfinite(area) or area <= 0:
            continue
        if best_frame is None or area > best_area:
            best_frame = evidence
            best_area = area

    selected = best_frame or by_largest_mask[0]
    mask_pixels = int(np.count_nonzero(selected["mask"]))
    return {
        "frame_id": int(selected["frame_id"]),
        "mask": selected["mask"],
        "mask_pixel_count": mask_pixels,
        "selection_method": _SELECTION_SURFACE if best_frame is not None else _SELECTION_FALLBACK,
        "visible_sampled_frame_count": len(normalized),
        "first_sampled_frame_id": int(normalized[0]["frame_id"]),
        "last_sampled_frame_id": int(normalized[-1]["frame_id"]),
    }


def _crop_bounds(mask: Any) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("object catalog representative mask is empty")
    height, width = mask.shape
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    margin = max(4, int(round(max(right - left, bottom - top) * 0.125)))
    x0 = max(0, left - margin)
    y0 = max(0, top - margin)
    x1 = min(width, right + margin)
    y1 = min(height, bottom + margin)
    return x0, y0, x1 - x0, y1 - y0


def _render_tile(frame_rgb: Any, mask: Any, crop: tuple[int, int, int, int]) -> Any:
    import cv2
    import numpy as np

    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("object catalog model frame must be RGB")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    contextual = (frame.astype(np.float32) * 0.36).astype(np.uint8)
    contextual[mask] = frame[mask]
    outline = np.logical_and(
        cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0,
        np.logical_not(mask),
    )
    contextual[outline] = np.array([203, 255, 54], dtype=np.uint8)

    x, y, width, height = crop
    cropped = contextual[y : y + height, x : x + width]
    if cropped.size == 0:
        raise ValueError("object catalog representative crop is empty")
    content_limit = OBJECT_CATALOG_CELL_SIZE - 8
    scale = min(content_limit / cropped.shape[1], content_limit / cropped.shape[0])
    resized_width = max(1, min(content_limit, int(round(cropped.shape[1] * scale))))
    resized_height = max(1, min(content_limit, int(round(cropped.shape[0] * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (resized_width, resized_height), interpolation=interpolation)

    tile = np.full(
        (OBJECT_CATALOG_CELL_SIZE, OBJECT_CATALOG_CELL_SIZE, 3),
        20,
        dtype=np.uint8,
    )
    x_offset = (OBJECT_CATALOG_CELL_SIZE - resized_width) // 2
    y_offset = (OBJECT_CATALOG_CELL_SIZE - resized_height) // 2
    tile[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return tile


def _encode_atlas(atlas_rgb: Any) -> bytes:
    import cv2
    import numpy as np

    bgr = np.ascontiguousarray(np.asarray(atlas_rgb)[:, :, ::-1])
    for quality in _JPEG_QUALITIES:
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                quality,
                cv2.IMWRITE_JPEG_OPTIMIZE,
                1,
            ],
        )
        if not ok:
            continue
        payload = encoded.tobytes()
        if (
            0 < len(payload) <= OBJECT_CROPS_ATLAS_MAX_BYTES
            and payload.startswith(b"\xff\xd8")
            and payload.endswith(b"\xff\xd9")
        ):
            return payload
    raise ValueError("object crop atlas exceeds the 8 MiB artifact contract")


def _optional_source_index(source_frame_indices: Any, frame_id: int) -> int | None:
    if not isinstance(source_frame_indices, (list, tuple)) or frame_id >= len(source_frame_indices):
        return None
    value = source_frame_indices[frame_id]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        return None
    return int(value)


def _optional_source_timestamp(source_frame_timestamps: Any, frame_id: int) -> float | None:
    if not isinstance(source_frame_timestamps, (list, tuple)) or frame_id >= len(source_frame_timestamps):
        return None
    try:
        value = float(source_frame_timestamps[frame_id])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return round(value, 6)


def build_object_catalog(
    *,
    all_masks: dict[str, Any],
    colors: Any,
    world_points: Any,
    requested_categories: list[str],
    source_frame_indices: list[int] | None,
    source_frame_timestamps: list[float] | None,
    surface_area_fn: SurfaceAreaFn,
    out_dir: Path,
) -> dict[str, Any]:
    """Write the negotiated v1 catalog and fixed JPEG atlas, then return the JSON."""
    import numpy as np

    frames = np.asarray(colors)
    points = np.asarray(world_points)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] <= 0:
        raise ValueError("object catalog colors must have shape (S,H,W,3)")
    if points.shape != (*frames.shape[:3], 3):
        raise ValueError("object catalog world points do not match model frames")
    if (
        not isinstance(requested_categories, list)
        or not 1 <= len(requested_categories) <= 8
        or any(not isinstance(category, str) or not 1 <= len(category) <= 64 for category in requested_categories)
        or len(set(requested_categories)) != len(requested_categories)
    ):
        raise ValueError("object catalog requested categories are invalid")
    requested_set = set(requested_categories)
    if not isinstance(all_masks, dict):
        raise ValueError("object catalog masks must be grouped by category")

    for raw_category, raw_tracks in all_masks.items():
        if not isinstance(raw_category, str) or raw_category not in requested_set:
            raise ValueError("object catalog label is not a requested category")
        if not isinstance(raw_tracks, (list, tuple)):
            raise ValueError("object catalog category tracks are malformed")
    flattened: list[tuple[str, Any]] = []
    for category in requested_categories:
        flattened.extend((category, track) for track in all_masks.get(category, ()))

    total_count = len(flattened)
    selected_instances = flattened[:OBJECT_CATALOG_MAX_OBJECTS]
    returned_count = len(selected_instances)
    if returned_count == 0:
        columns = rows = 1
    else:
        columns = min(OBJECT_CATALOG_MAX_COLUMNS, int(math.ceil(math.sqrt(returned_count))))
        rows = int(math.ceil(returned_count / columns))
    atlas_width = columns * OBJECT_CATALOG_CELL_SIZE
    atlas_height = rows * OBJECT_CATALOG_CELL_SIZE
    if atlas_width > OBJECT_CROPS_MAX_WIDTH or atlas_height > OBJECT_CROPS_MAX_HEIGHT:
        raise AssertionError("object crop atlas layout exceeded its fixed dimensions")
    atlas = np.full((atlas_height, atlas_width, 3), 20, dtype=np.uint8)

    objects: list[dict[str, Any]] = []
    for object_index, (category, track) in enumerate(selected_instances, start=1):
        selected = select_representative_frame(
            points,
            track,
            surface_area_fn,
            model_frame_shape=frames.shape[1:3],
        )
        frame_id = selected["frame_id"]
        crop = _crop_bounds(selected["mask"])
        tile = _render_tile(frames[frame_id], selected["mask"], crop)
        cell_index = object_index - 1
        cell_x = (cell_index % columns) * OBJECT_CATALOG_CELL_SIZE
        cell_y = (cell_index // columns) * OBJECT_CATALOG_CELL_SIZE
        atlas[
            cell_y : cell_y + OBJECT_CATALOG_CELL_SIZE,
            cell_x : cell_x + OBJECT_CATALOG_CELL_SIZE,
        ] = tile
        objects.append(
            {
                "id": f"object_{object_index:04d}",
                "label": category,
                "label_source": "requested_prompt",
                "visible_sampled_frame_count": selected["visible_sampled_frame_count"],
                "first_sampled_frame_id": selected["first_sampled_frame_id"],
                "last_sampled_frame_id": selected["last_sampled_frame_id"],
                "representative": {
                    "selection_method": selected["selection_method"],
                    "sampled_frame_id": frame_id,
                    "source_frame_index": _optional_source_index(source_frame_indices, frame_id),
                    "source_timestamp_s": _optional_source_timestamp(source_frame_timestamps, frame_id),
                    "model_frame_width": int(frames.shape[2]),
                    "model_frame_height": int(frames.shape[1]),
                    "model_frame_crop_xywh": list(crop),
                    "mask_pixel_count": selected["mask_pixel_count"],
                    "atlas_tile_xywh": [
                        cell_x,
                        cell_y,
                        OBJECT_CATALOG_CELL_SIZE,
                        OBJECT_CATALOG_CELL_SIZE,
                    ],
                },
            }
        )

    atlas_bytes = _encode_atlas(atlas)
    atlas_sha256 = hashlib.sha256(atlas_bytes).hexdigest()
    catalog: dict[str, Any] = {
        "schema_version": OBJECT_CATALOG_SCHEMA_VERSION,
        "scope": "requested_categories",
        "exhaustive": False,
        "identity_semantics": "ras_spatial_deduplicated_instance_hypothesis",
        "requested_categories": list(requested_categories),
        "total_count": total_count,
        "returned_count": returned_count,
        "truncated": total_count > returned_count,
        "atlas": {
            "artifact_name": OBJECT_CROPS_ATLAS_NAME,
            "media_type": "image/jpeg",
            "sha256": atlas_sha256,
            "width": atlas_width,
            "height": atlas_height,
            "cell_width": OBJECT_CATALOG_CELL_SIZE,
            "cell_height": OBJECT_CATALOG_CELL_SIZE,
            "columns": columns,
            "rows": rows,
        },
        "objects": objects,
    }
    catalog_bytes = json.dumps(
        catalog,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(catalog_bytes) > OBJECT_CATALOG_JSON_MAX_BYTES:
        raise ValueError("object catalog exceeds the 512 KiB artifact contract")

    _atomic_write(out_dir / OBJECT_CROPS_ATLAS_NAME, atlas_bytes)
    _atomic_write(out_dir / OBJECT_CATALOG_JSON_NAME, catalog_bytes)
    return catalog
