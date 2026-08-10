"""Best-view candidate scoring for deduplicated instance tracks.

This implements the view selection described by ReplicateAnyScene
(arXiv:2604.10789), but it does not reimplement the paper's surface-area score.
It bounds and feeds the unpatched upstream
``compute_surface_area_from_pointmap``, records every candidate frame it
scored, and writes the negotiated ``palatial.best_view.v1`` report.  The naive
largest-mask (2D) pick is always reported next to the 3D pick so a caller can
see when the two disagree instead of assuming they agree.

Two contracts are pinned here on purpose:

* ``decode_rle`` accepts only two strict integer run-length encodings and
  rejects anything whose decoded canvas is not exactly ``height * width``.
* ``derive_max_triangle_size`` replaces upstream's absolute
  ``max_triangle_size = 2e-4`` default with a scale-relative threshold.  VGGT
  geometry has an arbitrary, non-metric scale, so an absolute area constant
  either rejects every triangle or none of them depending on how large the
  reconstruction happens to come out.
"""

from __future__ import annotations

import json
import math
import numbers
import re
from pathlib import Path
from typing import Any, Callable

from object_catalog import (
    _atomic_write,
    _bounded_surface_inputs,
    _normalize_track,
    _optional_source_index,
    _optional_source_timestamp,
)


BEST_VIEW_SCHEMA = "palatial.best_view.v1"
BEST_VIEW_JSON_NAME = "best_view.json"
BEST_VIEW_JSON_MAX_BYTES = 8 * 1024 * 1024
BEST_VIEW_MAX_INSTANCES = 32
BEST_VIEW_MAX_CANDIDATE_FRAMES = 24
BEST_VIEW_MAX_MASK_PIXELS = 4_000_000
BEST_VIEW_MAX_RLE_CHARS = 256 * 1024
BEST_VIEW_SCALE_POLICY = "relative_median_adjacent_spacing_v1"
# A uniformly sampled, locally planar patch with adjacent-pixel spacing d has
# triangles of area d^2/2.  Admitting up to 100*d^2 keeps every plausible
# surface triangle and rejects only the slivers that bridge a depth
# discontinuity, which are the outliers the upstream parameter exists for.
BEST_VIEW_TRIANGLE_SIZE_FACTOR = 100.0

RLE_SCOPE_ALL = "all_candidate_frames"
RLE_SCOPE_WINNERS = "winning_frames_only"
RLE_SCOPE_NONE = "none"

SELECTION_SURFACE_AREA = "surface_area_3d"
SELECTION_MASK_PIXELS = "mask_pixel_count_fallback"

UNSCORED_DEGENERATE_SPACING = "degenerate_adjacent_spacing"
UNSCORED_SURFACE_AREA_FAILED = "surface_area_failed"
UNSCORED_NON_POSITIVE_AREA = "non_positive_surface_area"

_RLE_TOKEN = re.compile(r"[0-9]+")

SurfaceAreaFn = Callable[..., Any]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    number = int(value)
    return number if number > 0 else None


def _rle_tokens(rle: Any) -> list[int]:
    if not isinstance(rle, str):
        raise ValueError("best view mask RLE must be a string")
    if not 0 < len(rle) <= BEST_VIEW_MAX_RLE_CHARS:
        raise ValueError("best view mask RLE length is outside the accepted bounds")
    tokens = rle.split()
    if not tokens:
        raise ValueError("best view mask RLE is empty")
    for token in tokens:
        if not _RLE_TOKEN.fullmatch(token):
            raise ValueError("best view mask RLE must contain only non-negative integers")
    return [int(token) for token in tokens]


def _decode_coco_counts(counts: list[int], total: int) -> Any:
    """Strict COCO alternating run counts, background run first."""
    import numpy as np

    if sum(counts) != total:
        return None
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, counts)
    return flat if flat.size == total else None


def _decode_kaggle_pairs(counts: list[int], total: int) -> Any:
    """Strict Kaggle ``start length`` pairs over a 1-based column-major canvas."""
    import numpy as np

    if len(counts) % 2 != 0:
        return None
    flat = np.zeros(total, dtype=bool)
    cursor = 0
    for start, length in zip(counts[0::2], counts[1::2]):
        if length <= 0 or start <= cursor or start < 1:
            return None
        end = start + length - 1
        if end > total:
            return None
        flat[start - 1 : end] = True
        cursor = end
    return flat


def decode_rle(rle: Any, height: Any, width: Any) -> Any:
    """Decode one fal SAM 3 mask string into a boolean (H, W) array.

    The format is pinned by two strict parsers rather than sniffed: COCO
    alternating run counts first, then Kaggle start/length pairs.  Both are
    column-major.  Anything that does not describe exactly ``height * width``
    pixels is rejected, which is what keeps a format surprise from silently
    scoring a misaligned mask.
    """
    rows = _positive_int(height)
    columns = _positive_int(width)
    if rows is None or columns is None:
        raise ValueError("best view mask dimensions must be positive integers")
    total = rows * columns
    if total > BEST_VIEW_MAX_MASK_PIXELS:
        raise ValueError("best view mask exceeds the supported pixel bound")

    import numpy as np

    counts = _rle_tokens(rle)
    flat = _decode_coco_counts(counts, total)
    if flat is None:
        flat = _decode_kaggle_pairs(counts, total)
    if flat is None:
        raise ValueError(
            "best view mask RLE is neither a COCO run-count nor a Kaggle "
            f"start/length encoding of {rows}x{columns} pixels"
        )
    return np.reshape(flat, (rows, columns), order="F")


def derive_max_triangle_size(pointmap: Any, mask: Any) -> tuple[float, float]:
    """Return ``(median_adjacent_spacing, max_triangle_size)`` for one candidate.

    ``max_triangle_size`` is expressed in the same arbitrary units as the point
    map, so the ranking it produces is invariant to a global rescale of the
    reconstruction.  ``(0.0, 0.0)`` means the candidate has no usable adjacent
    pixel spacing and must not be scored in 3D.
    """
    import numpy as np

    points = np.asarray(pointmap, dtype=np.float64)
    binary = np.asarray(mask, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3 or binary.shape != points.shape[:2]:
        raise ValueError("best view pointmap and mask shapes do not agree")

    spans: list[Any] = []
    if binary.shape[1] > 1:
        pairs = np.logical_and(binary[:, :-1], binary[:, 1:])
        if np.any(pairs):
            delta = points[:, 1:, :] - points[:, :-1, :]
            spans.append(np.linalg.norm(delta[pairs], axis=-1))
    if binary.shape[0] > 1:
        pairs = np.logical_and(binary[:-1, :], binary[1:, :])
        if np.any(pairs):
            delta = points[1:, :, :] - points[:-1, :, :]
            spans.append(np.linalg.norm(delta[pairs], axis=-1))
    if not spans:
        return 0.0, 0.0

    merged = np.concatenate(spans)
    merged = merged[np.isfinite(merged) & (merged > 0)]
    if merged.size == 0:
        return 0.0, 0.0
    spacing = float(np.median(merged))
    if not math.isfinite(spacing) or spacing <= 0:
        return 0.0, 0.0
    threshold = BEST_VIEW_TRIANGLE_SIZE_FACTOR * spacing * spacing
    if not math.isfinite(threshold) or threshold <= 0:
        return 0.0, 0.0
    return spacing, threshold


def _bbox_xywh(mask: Any) -> list[int]:
    import numpy as np

    rows, columns = np.nonzero(mask)
    if columns.size == 0:
        raise ValueError("best view candidate mask is empty")
    left = int(columns.min())
    top = int(rows.min())
    return [left, top, int(columns.max()) - left + 1, int(rows.max()) - top + 1]


def _rle_by_frame(track: Any) -> dict[int, str]:
    """Keep a candidate's RLE only while it still describes the whole mask."""
    seen: dict[int, str | None] = {}
    for evidence in track:
        if not isinstance(evidence, dict):
            continue
        frame_id = evidence.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, numbers.Integral):
            continue
        key = int(frame_id)
        value = evidence.get("rle")
        if key in seen:
            seen[key] = None
            continue
        seen[key] = value if isinstance(value, str) and value else None
    return {key: value for key, value in seen.items() if value is not None}


def _argmax_frame(candidates: list[dict[str, Any]], score_key: str) -> int:
    """Highest score wins; the earliest sampled frame breaks a tie."""
    best = min(
        candidates,
        key=lambda item: (-float(item[score_key]), int(item["sampled_frame_id"])),
    )
    return int(best["sampled_frame_id"])


def score_track_candidates(
    world_points: Any,
    track: Any,
    surface_area_fn: SurfaceAreaFn,
    *,
    model_frame_shape: tuple[int, int] | None = None,
    source_frame_indices: Any = None,
    source_frame_timestamps: Any = None,
) -> dict[str, Any]:
    """Score every frame in which one instance is visible, 3D and 2D alike."""
    import numpy as np

    points = np.asarray(world_points)
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[0] <= 0:
        raise ValueError("best view world_points must have shape (S,H,W,3)")
    height, width = points.shape[1:3]
    if model_frame_shape is not None and tuple(model_frame_shape) != (height, width):
        raise ValueError("best view world points do not match the model frame shape")

    try:
        normalized = _normalize_track(
            track,
            frame_count=points.shape[0],
            height=height,
            width=width,
        )
    except ValueError as exc:
        raise ValueError(f"best view instance track is invalid ({exc})") from exc
    if len(normalized) > BEST_VIEW_MAX_CANDIDATE_FRAMES:
        raise ValueError("best view instance exceeds the candidate frame bound")
    retained_rle = _rle_by_frame(track)

    candidates: list[dict[str, Any]] = []
    for evidence in normalized:
        frame_id = int(evidence["frame_id"])
        mask = evidence["mask"]
        bounded_points, bounded_mask = _bounded_surface_inputs(points[frame_id], mask)
        spacing, threshold = derive_max_triangle_size(bounded_points, bounded_mask)

        area: float | None = None
        unscored_reason: str | None = None
        if spacing <= 0 or threshold <= 0:
            unscored_reason = UNSCORED_DEGENERATE_SPACING
        else:
            try:
                measured = float(
                    surface_area_fn(
                        bounded_points,
                        bounded_mask,
                        max_triangle_size=threshold,
                    )
                )
            except Exception:
                unscored_reason = UNSCORED_SURFACE_AREA_FAILED
            else:
                if math.isfinite(measured) and measured > 0:
                    area = measured
                else:
                    unscored_reason = UNSCORED_NON_POSITIVE_AREA

        candidates.append(
            {
                "sampled_frame_id": frame_id,
                "source_frame_index": _optional_source_index(source_frame_indices, frame_id),
                "source_timestamp_s": _optional_source_timestamp(
                    source_frame_timestamps, frame_id
                ),
                "scored_3d": area is not None,
                "surface_area_3d": area,
                "median_adjacent_spacing": spacing if spacing > 0 else None,
                "max_triangle_size": threshold if threshold > 0 else None,
                "scale_policy": BEST_VIEW_SCALE_POLICY,
                "unscored_reason": unscored_reason,
                "mask_pixel_count": int(np.count_nonzero(mask)),
                "bbox_xywh": _bbox_xywh(mask),
                "rle": retained_rle.get(frame_id),
            }
        )

    scored = [candidate for candidate in candidates if candidate["scored_3d"]]
    best_frame_3d = _argmax_frame(scored, "surface_area_3d") if scored else None
    best_frame_2d = _argmax_frame(candidates, "mask_pixel_count")
    return {
        "candidate_count": len(candidates),
        "scored_candidate_count": len(scored),
        "first_sampled_frame_id": int(normalized[0]["frame_id"]),
        "last_sampled_frame_id": int(normalized[-1]["frame_id"]),
        "best_frame_3d": best_frame_3d,
        "best_frame_2d": best_frame_2d,
        "best_frame": best_frame_2d if best_frame_3d is None else best_frame_3d,
        "selection_method": (
            SELECTION_MASK_PIXELS if best_frame_3d is None else SELECTION_SURFACE_AREA
        ),
        # None means "no candidate could be scored in 3D", which is different
        # from "the two rules disagree".
        "agreement": None if best_frame_3d is None else bool(best_frame_3d == best_frame_2d),
        "candidates": candidates,
    }


def _instance_max_mask_pixels(instance: dict[str, Any]) -> int:
    import numpy as np

    declared = instance.get("max_mask_pixel_count")
    if not isinstance(declared, bool) and isinstance(declared, numbers.Integral):
        return int(declared)
    return max(
        (
            int(np.count_nonzero(evidence["mask"]))
            for evidence in instance.get("track") or ()
            if isinstance(evidence, dict) and evidence.get("mask") is not None
        ),
        default=0,
    )


def _report_bytes(report: dict[str, Any]) -> bytes:
    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strip_rle(report: dict[str, Any], *, keep_winners: bool) -> None:
    for instance in report["instances"]:
        winner = instance.get("best_frame")
        for candidate in instance["candidates"]:
            if keep_winners and candidate["sampled_frame_id"] == winner:
                continue
            candidate["rle"] = None


def _fit_report_to_contract(report: dict[str, Any]) -> bytes:
    """Degrade deterministically instead of emitting an oversized artifact."""
    payload = _report_bytes(report)
    if len(payload) <= BEST_VIEW_JSON_MAX_BYTES:
        return payload

    _strip_rle(report, keep_winners=True)
    report["rle_scope"] = RLE_SCOPE_WINNERS
    report["warnings"].append(
        {
            "code": "rle_dropped_for_size",
            "detail": "non-winning candidate masks were dropped to fit the 8 MiB report contract",
        }
    )
    payload = _report_bytes(report)
    if len(payload) <= BEST_VIEW_JSON_MAX_BYTES:
        return payload

    _strip_rle(report, keep_winners=False)
    report["rle_scope"] = RLE_SCOPE_NONE
    report["warnings"].append(
        {
            "code": "all_rle_dropped_for_size",
            "detail": "every candidate mask was dropped to fit the 8 MiB report contract",
        }
    )
    payload = _report_bytes(report)
    if len(payload) <= BEST_VIEW_JSON_MAX_BYTES:
        return payload

    report["warnings"].append(
        {
            "code": "instances_truncated_for_size",
            "detail": "instances were dropped to fit the 8 MiB report contract",
        }
    )
    while len(report["instances"]) > 1:
        report["instances"].pop()
        report["returned_instances"] = len(report["instances"])
        report["truncated"] = report["total_instances"] > report["returned_instances"]
        report["agreement_counts"] = _agreement_counts(report["instances"])
        payload = _report_bytes(report)
        if len(payload) <= BEST_VIEW_JSON_MAX_BYTES:
            return payload
    raise ValueError("best view report exceeds the 8 MiB artifact contract")


def _agreement_counts(instances: list[dict[str, Any]]) -> dict[str, int]:
    agree = sum(1 for instance in instances if instance["agreement"] is True)
    disagree = sum(1 for instance in instances if instance["agreement"] is False)
    return {
        "agree": agree,
        "disagree": disagree,
        "unscored": len(instances) - agree - disagree,
    }


def build_best_view_report(
    *,
    instances: list[dict[str, Any]],
    world_points: Any,
    surface_area_fn: SurfaceAreaFn,
    out_dir: Path,
    requested_categories: list[str],
    frames_used: int,
    model_frame_width: int,
    model_frame_height: int,
    source_frame_indices: Any = None,
    source_frame_timestamps: Any = None,
    total_instances: int | None = None,
    coordinate_system: str = "vggt_first_camera",
) -> dict[str, Any]:
    """Score the bounded instance set and write ``best_view.json`` atomically."""
    ordered = sorted(
        enumerate(instances),
        key=lambda item: (
            -_instance_max_mask_pixels(item[1]),
            str(item[1].get("category") or ""),
            int(item[1].get("track_id") or 0),
            item[0],
        ),
    )
    selected = [instance for _index, instance in ordered[:BEST_VIEW_MAX_INSTANCES]]
    discovered = len(instances) if total_instances is None else int(total_instances)
    if discovered < len(selected):
        raise ValueError("best view total_instances is smaller than the scored set")

    scored_instances: list[dict[str, Any]] = []
    for position, instance in enumerate(selected, start=1):
        ranking = score_track_candidates(
            world_points,
            instance["track"],
            surface_area_fn,
            model_frame_shape=(model_frame_height, model_frame_width),
            source_frame_indices=source_frame_indices,
            source_frame_timestamps=source_frame_timestamps,
        )
        scored_instances.append(
            {
                "instance_id": f"instance_{position:04d}",
                "category": str(instance["category"]),
                "track_id": int(instance["track_id"]),
                **ranking,
            }
        )

    report: dict[str, Any] = {
        "schema": BEST_VIEW_SCHEMA,
        "scope": "requested_categories",
        "exhaustive": False,
        "identity_semantics": "fal_sam3_video_track",
        "coordinate_system": coordinate_system,
        "scale_policy": BEST_VIEW_SCALE_POLICY,
        "triangle_size_factor": BEST_VIEW_TRIANGLE_SIZE_FACTOR,
        "requested_categories": list(requested_categories),
        "frames_used": int(frames_used),
        "model_frame_width": int(model_frame_width),
        "model_frame_height": int(model_frame_height),
        "total_instances": discovered,
        "returned_instances": len(scored_instances),
        "truncated": discovered > len(scored_instances),
        "rle_scope": RLE_SCOPE_ALL,
        "agreement_counts": _agreement_counts(scored_instances),
        "warnings": [],
        "instances": scored_instances,
    }
    payload = _fit_report_to_contract(report)
    _atomic_write(Path(out_dir) / BEST_VIEW_JSON_NAME, payload)
    return report
