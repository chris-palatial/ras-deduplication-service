"""Bounded inline object catalog for the Scene Parse production profile.

The public RAS implementation drops object hypotheses visible in fewer than
three sampled frames during cross-category deduplication.  Scene Parse needs to
show those brief observations instead of silently losing them, so this module
also owns a conservative short-track recovery pass.  Existing Agent Lab modes
do not import or use this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from typing import Any, Callable


SCENE_PARSE_CATALOG_SCHEMA = "palatial.ras.scene_parse_catalog.v1"
SCENE_PARSE_CATALOG_MAX_OBJECTS = 128
SCENE_PARSE_CATALOG_MAX_EVIDENCE_PER_OBJECT = 96
SCENE_PARSE_CATALOG_MAX_INLINE_BYTES = 4 * 1024 * 1024
SCENE_PARSE_UPSTREAM_MIN_EVIDENCE = 3
SCENE_PARSE_DEDUP_OVERLAP_THRESHOLD = 0.5

OverlapFn = Callable[[Any, Any], float]


def _frame_id(value: Any, frame_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("scene parse catalog track has a non-integer frame_id")
    frame_id = int(value)
    if frame_id < 0 or frame_id >= frame_count:
        raise ValueError("scene parse catalog track frame_id is outside sampled frames")
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
        raise ValueError("scene parse catalog object has no mask evidence")
    merged: dict[int, Any] = {}
    for raw in track:
        if not isinstance(raw, dict) or "frame_id" not in raw or "mask" not in raw:
            raise ValueError("scene parse catalog track evidence is malformed")
        frame_id = _frame_id(raw["frame_id"], frame_count)
        mask = np.asarray(raw["mask"])
        if mask.shape != (height, width):
            raise ValueError("scene parse catalog mask shape does not match model frames")
        binary = mask.astype(bool, copy=False)
        if not np.any(binary):
            continue
        merged[frame_id] = (
            np.logical_or(merged[frame_id], binary)
            if frame_id in merged
            else binary.copy()
        )
    if not merged:
        raise ValueError("scene parse catalog object has no non-empty mask evidence")
    normalized = [
        {"frame_id": frame_id, "mask": merged[frame_id]}
        for frame_id in sorted(merged)
    ]
    if len(normalized) > SCENE_PARSE_CATALOG_MAX_EVIDENCE_PER_OBJECT:
        raise ValueError("scene parse catalog track exceeds the bounded sampling policy")
    return normalized


def _mask_bbox(mask: Any) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("scene parse catalog mask is empty")
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    return left, top, right - left, bottom - top


def _normalized_bbox(mask: Any) -> tuple[dict[str, float], tuple[int, int, int, int]]:
    height, width = mask.shape
    x, y, box_width, box_height = _mask_bbox(mask)
    return (
        {
            "x": round(x / width, 8),
            "y": round(y / height, 8),
            "width": round(box_width / width, 8),
            "height": round(box_height / height, 8),
        },
        (x, y, box_width, box_height),
    )


def _track_signature(category_prompt: str, evidence: list[dict[str, Any]]) -> str:
    canonical = [
        [
            item["sampled_frame_index"],
            item["source_frame_index"],
            item["timestamp_ms"],
            item["pixel_bbox"],
            item["mask_pixel_count"],
        ]
        for item in evidence
    ]
    return json.dumps(
        [category_prompt, canonical],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _evidence_strength(count: int) -> str:
    if count == 1:
        return "single_frame"
    if count < SCENE_PARSE_UPSTREAM_MIN_EVIDENCE:
        return "limited"
    return "supported"


def build_scene_parse_objects(
    *,
    all_masks: dict[str, Any],
    frame_shape: tuple[int, int, int],
    source_sha256: str,
    category_prompts: list[str],
    source_frame_indices: list[int],
    source_frame_timestamps_s: list[float],
) -> dict[str, Any]:
    """Build deterministic result-scoped objects with every sampled mask view."""
    import numpy as np

    frame_count, height, width = frame_shape
    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError("scene parse catalog frame shape is invalid")
    if (
        len(source_frame_indices) != frame_count
        or len(source_frame_timestamps_s) != frame_count
    ):
        raise ValueError("scene parse catalog source timeline does not match sampled frames")
    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0
        for value in source_frame_indices
    ):
        raise ValueError("scene parse catalog source frame indices are invalid")
    if any(
        not isinstance(value, numbers.Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in source_frame_timestamps_s
    ):
        raise ValueError("scene parse catalog source timestamps are invalid")
    if not isinstance(all_masks, dict):
        raise ValueError("scene parse catalog masks must be grouped by prompt")
    prompt_order = {prompt: index for index, prompt in enumerate(category_prompts)}
    if any(prompt not in prompt_order for prompt in all_masks):
        raise ValueError("scene parse catalog contains an unrequested category prompt")

    normalized_objects: list[dict[str, Any]] = []
    for prompt in category_prompts:
        tracks = all_masks.get(prompt, ())
        if not isinstance(tracks, (list, tuple)):
            raise ValueError("scene parse catalog category tracks are malformed")
        for track in tracks:
            normalized = _normalize_track(
                track,
                frame_count=frame_count,
                height=height,
                width=width,
            )
            pixel_counts = [int(np.count_nonzero(item["mask"])) for item in normalized]
            largest = max(pixel_counts)
            evidence: list[dict[str, Any]] = []
            for item, pixel_count in zip(normalized, pixel_counts):
                sampled_frame_index = int(item["frame_id"])
                bbox, pixel_bbox = _normalized_bbox(item["mask"])
                evidence.append(
                    {
                        "sampled_frame_index": sampled_frame_index,
                        "source_frame_index": int(source_frame_indices[sampled_frame_index]),
                        "timestamp_ms": int(
                            round(float(source_frame_timestamps_s[sampled_frame_index]) * 1000)
                        ),
                        "bbox": bbox,
                        "pixel_bbox": list(pixel_bbox),
                        "mask_pixel_count": pixel_count,
                        "mask_area_ratio": round(pixel_count / (height * width), 8),
                        "quality_score": round(pixel_count / largest, 6),
                    }
                )
            signature = _track_signature(prompt, evidence)
            first_bbox = evidence[0]["pixel_bbox"]
            normalized_objects.append(
                {
                    "category_prompt": prompt,
                    "evidence": evidence,
                    "best_evidence_index": max(
                        range(len(evidence)),
                        key=lambda index: (
                            evidence[index]["quality_score"],
                            evidence[index]["mask_pixel_count"],
                            -evidence[index]["sampled_frame_index"],
                        ),
                    ),
                    "evidence_strength": _evidence_strength(len(evidence)),
                    "_signature": signature,
                    "_sort": (
                        prompt_order[prompt],
                        evidence[0]["sampled_frame_index"],
                        first_bbox[1],
                        first_bbox[0],
                        -sum(pixel_counts),
                        signature,
                    ),
                }
            )

    normalized_objects.sort(key=lambda item: item["_sort"])
    total_count = len(normalized_objects)
    selected = normalized_objects[:SCENE_PARSE_CATALOG_MAX_OBJECTS]
    signature_occurrences: dict[tuple[str, str], int] = {}
    objects: list[dict[str, Any]] = []
    for item in selected:
        signature_key = (item["category_prompt"], item["_signature"])
        occurrence = signature_occurrences.get(signature_key, 0)
        signature_occurrences[signature_key] = occurrence + 1
        digest = hashlib.sha256(
            (
                source_sha256
                + "\0"
                + item["category_prompt"]
                + "\0"
                + item["_signature"]
                + "\0"
                + str(occurrence)
            ).encode("utf-8")
        ).hexdigest()[:24]
        evidence = []
        for raw in item["evidence"]:
            evidence.append({key: value for key, value in raw.items() if key != "pixel_bbox"})
        objects.append(
            {
                "object_key": f"obj_{digest}",
                "category_prompt": item["category_prompt"],
                "evidence_strength": item["evidence_strength"],
                "evidence": evidence,
                "best_evidence_index": item["best_evidence_index"],
            }
        )

    return {
        "objects": objects,
        "total_count": total_count,
        "returned_count": len(objects),
        "truncated": total_count > len(objects),
        "evidence_count": sum(len(item["evidence"]) for item in objects),
        "limited_evidence_count": sum(
            item["evidence_strength"] != "supported" for item in objects
        ),
    }


def _track_frame_count(track: Any) -> int:
    if not isinstance(track, (list, tuple)):
        return 0
    return len(
        {
            int(item["frame_id"])
            for item in track
            if isinstance(item, dict)
            and isinstance(item.get("frame_id"), numbers.Integral)
            and not isinstance(item.get("frame_id"), bool)
        }
    )


def _track_mask_signature(track: Any) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for item in sorted(track, key=lambda raw: int(raw["frame_id"])):
        mask = np.asarray(item["mask"]).astype(bool, copy=False)
        digest.update(str(int(item["frame_id"])).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(mask.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.packbits(mask, bitorder="little").tobytes())
    return digest.hexdigest()


def _track_points(
    track: Any,
    world_points: Any,
    world_points_conf: Any,
    *,
    conf_percentile: float = 50.0,
) -> Any | None:
    """Mirror the public RAS candidate point filtering for overlap checks."""
    import numpy as np

    points = np.asarray(world_points)
    confidence = np.asarray(world_points_conf)
    point_chunks = []
    confidence_chunks = []
    for item in track:
        frame_id = int(item["frame_id"])
        if frame_id < 0 or frame_id >= points.shape[0]:
            continue
        mask = np.asarray(item["mask"]).astype(bool, copy=False)
        if mask.shape != tuple(points.shape[1:3]) or not np.any(mask):
            continue
        point_chunks.append(points[frame_id][mask])
        confidence_chunks.append(confidence[frame_id][mask])
    if not point_chunks:
        return None
    merged_points = np.concatenate(point_chunks, axis=0)
    merged_confidence = np.concatenate(confidence_chunks, axis=0)
    threshold = float(np.percentile(merged_confidence, conf_percentile))
    keep = (merged_confidence >= threshold) & (merged_confidence > 1e-5)
    filtered = merged_points[keep]
    return filtered if len(filtered) >= 50 else None


def _tracks_overlap(
    left_points: Any | None,
    right_points: Any | None,
    overlap_fn: OverlapFn,
    threshold: float,
) -> bool:
    if left_points is None or right_points is None:
        return False
    try:
        left_ratio = float(overlap_fn(left_points, right_points))
        right_ratio = float(overlap_fn(right_points, left_points))
    except Exception:
        return False
    return (
        math.isfinite(left_ratio)
        and left_ratio >= threshold
        or math.isfinite(right_ratio)
        and right_ratio >= threshold
    )


def _merge_tracks(tracks: list[Any]) -> list[dict[str, Any]]:
    import numpy as np

    merged: dict[int, Any] = {}
    for track in tracks:
        for item in track:
            frame_id = int(item["frame_id"])
            mask = np.asarray(item["mask"]).astype(bool, copy=False)
            merged[frame_id] = (
                np.logical_or(merged[frame_id], mask)
                if frame_id in merged
                else mask.copy()
            )
    return [
        {"frame_id": frame_id, "mask": merged[frame_id]}
        for frame_id in sorted(merged)
    ]


def preserve_short_scene_parse_tracks(
    *,
    deduplicated_masks: dict[str, Any],
    category_masks: dict[str, Any],
    category_prompts: list[str],
    world_points: Any,
    world_points_conf: Any,
    overlap_fn: OverlapFn,
    overlap_threshold: float = SCENE_PARSE_DEDUP_OVERLAP_THRESHOLD,
) -> tuple[dict[str, list[Any]], int]:
    """Conservatively retain unmerged 1-2-frame hypotheses.

    Exact matches and candidates with RAS-level 3D overlap against an already
    returned object are not re-added. Remaining short candidates are clustered
    against each other before one merged hypothesis per group is appended.
    """
    prompt_order = {prompt: index for index, prompt in enumerate(category_prompts)}
    result: dict[str, list[Any]] = {
        prompt: list(deduplicated_masks.get(prompt, ()))
        for prompt in category_prompts
        if deduplicated_masks.get(prompt)
    }
    returned: list[dict[str, Any]] = []
    returned_signatures: set[str] = set()
    for prompt in category_prompts:
        for track in result.get(prompt, ()):
            signature = _track_mask_signature(track)
            returned_signatures.add(signature)
            returned.append(
                {
                    "prompt": prompt,
                    "track": track,
                    "points": _track_points(track, world_points, world_points_conf),
                }
            )

    candidates: list[dict[str, Any]] = []
    for prompt in category_prompts:
        tracks = category_masks.get(prompt, ())
        if not isinstance(tracks, (list, tuple)):
            continue
        for order, track in enumerate(tracks):
            if not 0 < _track_frame_count(track) < SCENE_PARSE_UPSTREAM_MIN_EVIDENCE:
                continue
            signature = _track_mask_signature(track)
            if signature in returned_signatures:
                continue
            points = _track_points(track, world_points, world_points_conf)
            if any(
                _tracks_overlap(points, existing["points"], overlap_fn, overlap_threshold)
                for existing in returned
            ):
                continue
            candidates.append(
                {
                    "prompt": prompt,
                    "track": track,
                    "points": points,
                    "signature": signature,
                    "sort": (prompt_order[prompt], order, signature),
                }
            )

    if not candidates:
        return result, 0
    candidates.sort(key=lambda item: item["sort"])
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if candidates[left]["signature"] == candidates[right]["signature"] or _tracks_overlap(
                candidates[left]["points"],
                candidates[right]["points"],
                overlap_fn,
                overlap_threshold,
            ):
                union(left, right)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(find(index), []).append(candidate)
    preserved = 0
    for group in groups.values():
        chosen_prompt = min(group, key=lambda item: item["sort"])["prompt"]
        result.setdefault(chosen_prompt, []).append(
            _merge_tracks([item["track"] for item in group])
        )
        preserved += 1
    return result, preserved


def bound_scene_parse_response(response: dict[str, Any]) -> dict[str, Any]:
    """Keep the RunPod inline result bounded while retaining complete objects."""
    warnings = response.setdefault("warnings", [])
    objects = response.get("objects")
    counts = response.get("counts")
    if not isinstance(objects, list) or not isinstance(counts, dict):
        raise ValueError("scene parse catalog response is malformed")
    original_returned_count = counts.get("returned_objects")
    while True:
        if len(objects) != original_returned_count:
            counts["returned_objects"] = len(objects)
            counts["omitted_objects"] = max(
                0, int(counts.get("deduplicated_objects") or 0) - len(objects)
            )
            counts["evidence_items"] = sum(len(item["evidence"]) for item in objects)
            response["truncated"] = True
            warning = "Inline response limit omitted complete object hypotheses."
            if warning not in warnings:
                warnings.append(warning)
        encoded = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) <= SCENE_PARSE_CATALOG_MAX_INLINE_BYTES:
            break
        if not objects:
            raise ValueError("scene parse catalog metadata exceeds the inline response limit")
        objects.pop()
    return response
