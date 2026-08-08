"""Small, dependency-light glTF 2.0 exporter for VGGT point clouds.

The file is intentionally implemented from the public glTF container format,
not copied from VGGT-Omega.  It emits one POINTS primitive with RGB colors and,
when camera extrinsics are available, one LINES primitive that draws the camera
frustums.  This is a visualization artifact, not a reconstructed surface mesh.
"""

from __future__ import annotations

import colorsys
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


_GLB_MAGIC = 0x46546C67
_GLB_VERSION = 2
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
# Agent Lab's signed upload policy allows a 16 MiB GLB. Each point occupies
# 16 bytes (float32 XYZ + normalized uint8 RGBA); 900k leaves over 2 MiB for
# camera lines, buffer alignment, and JSON container metadata.
GLB_MAX_BYTES = 16 * 1024 * 1024
GLB_HARD_MAX_POINTS = 900_000
GLB_HARD_MAX_CAMERAS = 160
_OUTLIER_RADIUS_FACTOR = 3.0
_OUTLIER_MAX_REMOVAL_FRACTION = 0.05


def _pad4(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((-len(data)) % 4)


def _rgb_to_rgba(rgb: np.ndarray) -> np.ndarray:
    """Return tightly packed, four-byte-aligned vertex colors for glTF."""
    rgba = np.empty((len(rgb), 4), dtype=np.uint8)
    rgba[:, :3] = rgb
    rgba[:, 3] = 255
    return rgba


def _bounded_true_indices(mask: np.ndarray, limit: int) -> tuple[np.ndarray, int]:
    """Uniformly choose true indices without materializing every match."""
    count = int(np.count_nonzero(mask))
    if count <= limit:
        return np.flatnonzero(mask), count
    ranks = np.linspace(0, count - 1, limit, dtype=np.int64)
    selected = np.empty(limit, dtype=np.int64)
    seen = 0
    cursor = 0
    chunk_size = 1_000_000
    for start in range(0, len(mask), chunk_size):
        local = np.flatnonzero(mask[start : start + chunk_size])
        next_seen = seen + len(local)
        next_cursor = int(np.searchsorted(ranks, next_seen, side="left"))
        if next_cursor > cursor:
            selected[cursor:next_cursor] = (
                local[ranks[cursor:next_cursor] - seen] + start
            )
        seen = next_seen
        cursor = next_cursor
    if cursor != limit:
        raise RuntimeError("failed to sample the requested point count")
    return selected, count


def _camera_lines(
    extrinsics: Any,
    scene_scale: float,
    scene_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build visible camera frustums from OpenCV world-to-camera matrices."""
    if extrinsics is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), 0

    matrices = np.asarray(extrinsics, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    if matrices.ndim != 3 or matrices.shape[-2:] not in {(3, 4), (4, 4)}:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), 0

    marker_scale = max(float(scene_scale) * 0.025, 1e-4)
    # OpenCV camera coordinates look down +Z.  Four corners plus the origin
    # become eight independent line segments for broad glTF viewer support.
    local = marker_scale * np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.75, -0.5, 1.0],
            [0.75, -0.5, 1.0],
            [0.75, 0.5, 1.0],
            [-0.75, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    segments = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    if len(matrices) > GLB_HARD_MAX_CAMERAS:
        camera_indices = np.linspace(
            0,
            len(matrices) - 1,
            GLB_HARD_MAX_CAMERAS,
            dtype=np.int64,
        )
    else:
        camera_indices = np.arange(len(matrices), dtype=np.int64)

    camera_worlds: list[tuple[int, np.ndarray]] = []
    for raw_index in camera_indices:
        index = int(raw_index)
        matrix = matrices[index]
        if matrix.shape == (3, 4):
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :4] = matrix
        else:
            world_to_camera = matrix
        if not np.isfinite(world_to_camera).all():
            continue
        try:
            camera_to_world = np.linalg.inv(world_to_camera)
        except np.linalg.LinAlgError:
            continue
        local_h = np.concatenate([local, np.ones((local.shape[0], 1), dtype=np.float64)], axis=1)
        world = (camera_to_world @ local_h.T).T[:, :3]
        if not np.isfinite(world).all():
            continue
        camera_worlds.append((index, world))

    if not camera_worlds:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), 0

    distances = np.asarray(
        [np.linalg.norm(world[0] - scene_center) for _, world in camera_worlds],
        dtype=np.float64,
    )
    if len(distances) >= 3:
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        group_limit = median + 8.0 * max(1.4826 * mad, scene_scale * 0.01, 1e-5)
        keep_camera = ~(
            (distances > max(scene_scale * 4.0, 1e-3))
            & (distances > group_limit)
        )
    else:
        keep_camera = distances <= max(scene_scale * 8.0, 1e-3)

    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    camera_count = 0
    for (index, world), keep in zip(camera_worlds, keep_camera):
        if not bool(keep):
            continue

        hue = (index / max(len(matrices), 1)) * 0.82
        rgb = np.array(colorsys.hsv_to_rgb(hue, 0.78, 1.0), dtype=np.float64)
        rgb = np.rint(rgb * 255.0).astype(np.uint8)
        for start, end in segments:
            positions.extend((world[start], world[end]))
            colors.extend((rgb, rgb))
        camera_count += 1

    if not positions:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), 0
    return np.asarray(positions, dtype=np.float32), np.asarray(colors, dtype=np.uint8), camera_count


def _trim_spatial_outliers(points: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Drop a small tail only when points are both robustly far and isolated."""
    if len(points) < 32:
        return np.ones(len(points), dtype=bool), 0, False
    lower, upper = np.percentile(points, [1.0, 99.0], axis=0)
    center = (lower + upper) * 0.5
    robust_diagonal = float(np.linalg.norm(upper - lower))
    numerical_floor = max(float(np.linalg.norm(center)) * 1e-6, 1e-5)
    radius = max(robust_diagonal * _OUTLIER_RADIUS_FACTOR, numerical_floor)
    far = np.linalg.norm(points - center, axis=1) > radius
    far_indices = np.flatnonzero(far)
    if far_indices.size == 0:
        return np.ones(len(points), dtype=bool), 0, False

    # Density is evaluated only for the small far tail, keeping memory bounded.
    # Valid remote geometry normally contributes several nearby points; a bad
    # VGGT spike tends to occupy a voxel alone.
    voxel_size = max(robust_diagonal * 0.05, numerical_floor)
    far_voxels = np.floor((points[far_indices] - center) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(
        far_voxels,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    candidates = far_indices[counts[inverse] < 4]
    if candidates.size == 0:
        return np.ones(len(points), dtype=bool), 0, False
    if candidates.size > len(points) * _OUTLIER_MAX_REMOVAL_FRACTION:
        # A large remote population may be a real secondary surface. Keep it
        # rather than silently cropping the reconstruction.
        return np.ones(len(points), dtype=bool), 0, True
    inliers = np.ones(len(points), dtype=bool)
    inliers[candidates] = False
    return inliers, int(candidates.size), False


def build_point_cloud_glb(
    world_points: Any,
    colors: Any,
    *,
    confidence: Any | None = None,
    extrinsics: Any | None = None,
    max_points: int = 300_000,
    confidence_percentile: float = 25.0,
) -> tuple[bytes, dict[str, int | float]]:
    """Return a GLB containing a bounded colored point cloud and camera markers."""
    points = np.asarray(world_points)
    rgb = np.asarray(colors)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("world_points must end in XYZ coordinates")
    if rgb.ndim < 2 or rgb.shape[-1] < 3:
        raise ValueError("colors must end in RGB channels")

    points = points.reshape(-1, 3)
    rgb = rgb[..., :3].reshape(-1, 3)
    if len(points) != len(rgb):
        raise ValueError("world_points and colors must contain the same number of samples")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    valid = np.isfinite(points).all(axis=1) & np.isfinite(rgb).all(axis=1)
    conf = None
    if confidence is not None:
        conf = np.asarray(confidence).reshape(-1)
        if len(conf) != len(points):
            raise ValueError("confidence must match the point count")
        valid &= np.isfinite(conf)
        confidence_sample, confidence_count = _bounded_true_indices(
            valid,
            GLB_HARD_MAX_POINTS,
        )
        if confidence_count:
            percentile = min(100.0, max(0.0, float(confidence_percentile)))
            threshold = float(np.percentile(conf[confidence_sample], percentile))
            valid &= conf >= threshold

    point_limit = min(int(max_points), GLB_HARD_MAX_POINTS)
    selected, source_point_count = _bounded_true_indices(valid, point_limit)
    if source_point_count == 0:
        raise ValueError("VGGT produced no finite point-cloud samples")

    # Bound the working set before robust statistics. Real videos can contain
    # tens of millions of VGGT samples; converting all of them to float64 just
    # for outlier analysis creates an avoidable host-memory spike.
    points = np.ascontiguousarray(points[selected], dtype=np.float32)
    rgb = rgb[selected]
    spatial_mask, spatial_outliers_removed, spatial_filter_fail_open = (
        _trim_spatial_outliers(points)
    )
    points = points[spatial_mask]
    rgb = rgb[spatial_mask]
    if np.issubdtype(rgb.dtype, np.floating) and (float(np.nanmax(rgb)) if rgb.size else 0.0) <= 1.0:
        rgb = rgb * 255.0
    rgb = np.ascontiguousarray(np.clip(np.rint(rgb), 0, 255), dtype=np.uint8)

    lower = np.percentile(points, 5, axis=0)
    upper = np.percentile(points, 95, axis=0)
    scene_scale = float(np.linalg.norm(upper - lower))
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        scene_scale = 1.0
    scene_center = (lower + upper) * 0.5
    camera_points, camera_colors, camera_count = _camera_lines(
        extrinsics,
        scene_scale,
        scene_center,
    )

    binary = bytearray()
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, Any]] = []

    def add_accessor(data: bytes, *, component_type: int, count: int, kind: str, normalized: bool = False, bounds: np.ndarray | None = None) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(data)
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": 34962})
        accessor: dict[str, Any] = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": count,
            "type": kind,
        }
        if normalized:
            accessor["normalized"] = True
        if bounds is not None:
            accessor["min"] = [float(x) for x in bounds[0]]
            accessor["max"] = [float(x) for x in bounds[1]]
        accessors.append(accessor)
        return len(accessors) - 1

    point_position = add_accessor(
        points.astype("<f4", copy=False).tobytes(),
        component_type=5126,
        count=len(points),
        kind="VEC3",
        bounds=np.stack([points.min(axis=0), points.max(axis=0)]),
    )
    point_color = add_accessor(
        _rgb_to_rgba(rgb).tobytes(),
        component_type=5121,
        count=len(rgb),
        kind="VEC4",
        normalized=True,
    )
    meshes: list[dict[str, Any]] = [
        {
            "name": "VGGT colored point cloud",
            "primitives": [{"attributes": {"POSITION": point_position, "COLOR_0": point_color}, "mode": 0}],
        }
    ]
    nodes: list[dict[str, Any]] = [{"mesh": 0, "name": "VGGT colored point cloud"}]

    if len(camera_points):
        camera_position = add_accessor(
            camera_points.astype("<f4", copy=False).tobytes(),
            component_type=5126,
            count=len(camera_points),
            kind="VEC3",
            bounds=np.stack([camera_points.min(axis=0), camera_points.max(axis=0)]),
        )
        camera_color = add_accessor(
            _rgb_to_rgba(camera_colors).tobytes(),
            component_type=5121,
            count=len(camera_colors),
            kind="VEC4",
            normalized=True,
        )
        meshes.append(
            {
                "name": "VGGT camera poses",
                "primitives": [{"attributes": {"POSITION": camera_position, "COLOR_0": camera_color}, "mode": 1}],
            }
        )
        nodes.append({"mesh": 1, "name": "VGGT camera poses"})

    gltf = {
        "asset": {"version": "2.0", "generator": "Palatial Stage2 point-cloud exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extras": {
            "artifactKind": "colored-point-cloud",
            "sourcePointCount": source_point_count,
            "pointCount": int(len(points)),
            "spatialOutliersRemoved": spatial_outliers_removed,
            "spatialFilterFailOpen": spatial_filter_fail_open,
            "hardPointCap": GLB_HARD_MAX_POINTS,
            "cameraCount": camera_count,
            "isSurfaceMesh": False,
        },
    }
    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":"), ensure_ascii=True).encode("utf-8"), b" ")
    bin_chunk = _pad4(bytes(binary), b"\x00")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    if total_length > GLB_MAX_BYTES:
        raise RuntimeError(
            f"GLB would exceed the {GLB_MAX_BYTES}-byte artifact contract"
        )
    glb = b"".join(
        [
            struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total_length),
            struct.pack("<II", len(json_chunk), _JSON_CHUNK),
            json_chunk,
            struct.pack("<II", len(bin_chunk), _BIN_CHUNK),
            bin_chunk,
        ]
    )
    return glb, {
        "source_point_count": source_point_count,
        "point_count": int(len(points)),
        "spatial_outliers_removed": spatial_outliers_removed,
        "spatial_filter_fail_open": spatial_filter_fail_open,
        "hard_point_cap": GLB_HARD_MAX_POINTS,
        "camera_count": camera_count,
        "bytes": len(glb),
    }


def write_point_cloud_glb(path: str | Path, *args: Any, **kwargs: Any) -> dict[str, int | float]:
    data, stats = build_point_cloud_glb(*args, **kwargs)
    Path(path).write_bytes(data)
    return stats
