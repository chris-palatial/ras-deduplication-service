import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import best_view


def _coco_counts(mask: np.ndarray) -> str:
    """Encode a boolean mask as COCO alternating run counts, column-major."""
    flat = np.asarray(mask, dtype=bool).flatten(order="F")
    counts: list[int] = []
    current = False
    run = 0
    for value in flat:
        if bool(value) == current:
            run += 1
            continue
        counts.append(run)
        current = bool(value)
        run = 1
    counts.append(run)
    return " ".join(str(count) for count in counts)


def _kaggle_pairs(mask: np.ndarray) -> str:
    """Encode a boolean mask as 1-based Kaggle start/length pairs, column-major."""
    flat = np.asarray(mask, dtype=bool).flatten(order="F")
    tokens: list[str] = []
    index = 0
    while index < flat.size:
        if not flat[index]:
            index += 1
            continue
        start = index
        while index < flat.size and flat[index]:
            index += 1
        tokens.extend([str(start + 1), str(index - start)])
    return " ".join(tokens)


def _grid_surface_area(pointmap, mask, max_triangle_size=2e-4):
    """Stand-in for the upstream Delaunay scorer on regularly gridded masks.

    Delaunay over a fully covered rectangular pixel block yields exactly two
    triangles per unit cell, so summing those two triangles reproduces the
    upstream total.  The same ``max_triangle_size`` outlier filter is applied,
    which is what the scale-relative threshold has to control.
    """
    points = np.asarray(pointmap, dtype=np.float64)
    binary = np.asarray(mask, dtype=bool)
    cells = binary[:-1, :-1] & binary[:-1, 1:] & binary[1:, :-1] & binary[1:, 1:]
    if not np.any(cells):
        return 0.0
    top_left = points[:-1, :-1][cells]
    top_right = points[:-1, 1:][cells]
    bottom_left = points[1:, :-1][cells]
    bottom_right = points[1:, 1:][cells]
    total = 0.0
    for a, b, c in (
        (top_left, top_right, bottom_left),
        (bottom_right, top_right, bottom_left),
    ):
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        total += float(np.sum(areas[(areas > 0) & (areas < max_triangle_size)]))
    return total


def _planar_frame(height: int, width: int, spacing: float) -> np.ndarray:
    grid_y, grid_x = np.mgrid[:height, :width]
    frame = np.zeros((height, width, 3), dtype=np.float64)
    frame[..., 0] = grid_x * spacing
    frame[..., 1] = grid_y * spacing
    return frame


def _block_mask(height: int, width: int, size: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:size, :size] = True
    return mask


def _disagreeing_scene():
    """Frame 0 covers more pixels; frame 1 covers more real surface."""
    height = width = 24
    world_points = np.stack(
        [_planar_frame(height, width, 0.01), _planar_frame(height, width, 0.1)]
    )
    track = [
        {"frame_id": 0, "mask": _block_mask(height, width, 20)},
        {"frame_id": 1, "mask": _block_mask(height, width, 10)},
    ]
    return world_points, track


class DecodeRleTest(unittest.TestCase):
    def setUp(self):
        self.height = 6
        self.width = 5
        self.mask = np.zeros((self.height, self.width), dtype=bool)
        self.mask[1:4, 1:3] = True
        self.mask[5, 4] = True

    def test_decodes_coco_alternating_run_counts(self):
        decoded = best_view.decode_rle(_coco_counts(self.mask), self.height, self.width)

        self.assertEqual(decoded.dtype, np.bool_)
        self.assertEqual(decoded.shape, (self.height, self.width))
        np.testing.assert_array_equal(decoded, self.mask)

    def test_decodes_kaggle_start_length_pairs(self):
        encoded = _kaggle_pairs(self.mask)
        self.assertNotEqual(
            sum(int(token) for token in encoded.split()),
            self.height * self.width,
            "the fixture must not also parse as COCO run counts",
        )

        decoded = best_view.decode_rle(encoded, self.height, self.width)

        np.testing.assert_array_equal(decoded, self.mask)

    def test_empty_and_full_masks_round_trip_in_both_encodings(self):
        for mask in (
            np.zeros((4, 3), dtype=bool),
            np.ones((4, 3), dtype=bool),
        ):
            with self.subTest(filled=bool(mask.all())):
                np.testing.assert_array_equal(
                    best_view.decode_rle(_coco_counts(mask), 4, 3), mask
                )

    def test_rejects_a_canvas_that_is_not_height_times_width(self):
        with self.assertRaises(ValueError):
            best_view.decode_rle("3 4 2 2 9", 4, 4)

    def test_rejects_overlapping_kaggle_pairs(self):
        with self.assertRaises(ValueError):
            best_view.decode_rle("3 4 4 2", 4, 5)

    def test_rejects_kaggle_pairs_past_the_canvas(self):
        with self.assertRaises(ValueError):
            best_view.decode_rle("18 9", 4, 5)

    def test_rejects_non_integer_and_empty_payloads(self):
        for value in ("", "   ", "1 -2 3", "1 2.5", "1 2 x", None, 7):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    best_view.decode_rle(value, 4, 5)

    def test_rejects_non_positive_dimensions(self):
        for height, width in ((0, 5), (5, 0), (-1, 5), (True, 5)):
            with self.subTest(height=height, width=width):
                with self.assertRaises(ValueError):
                    best_view.decode_rle("20", height, width)


class DeriveMaxTriangleSizeTest(unittest.TestCase):
    def test_threshold_is_one_hundred_times_the_squared_median_spacing(self):
        points = _planar_frame(8, 8, 0.25)
        mask = _block_mask(8, 8, 6)

        spacing, threshold = best_view.derive_max_triangle_size(points, mask)

        self.assertAlmostEqual(spacing, 0.25)
        self.assertAlmostEqual(threshold, 100.0 * 0.25 * 0.25)

    def test_single_pixel_mask_has_no_adjacent_spacing(self):
        points = _planar_frame(8, 8, 0.25)
        mask = np.zeros((8, 8), dtype=bool)
        mask[3, 3] = True

        self.assertEqual(best_view.derive_max_triangle_size(points, mask), (0.0, 0.0))

    def test_collapsed_and_non_finite_geometry_is_degenerate(self):
        mask = _block_mask(6, 6, 4)
        collapsed = np.zeros((6, 6, 3), dtype=np.float64)
        self.assertEqual(best_view.derive_max_triangle_size(collapsed, mask), (0.0, 0.0))

        broken = _planar_frame(6, 6, 0.1)
        broken[..., 0] = np.nan
        self.assertEqual(best_view.derive_max_triangle_size(broken, mask), (0.0, 0.0))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            best_view.derive_max_triangle_size(
                _planar_frame(6, 6, 0.1), np.ones((5, 5), dtype=bool)
            )


class ScoreTrackCandidatesTest(unittest.TestCase):
    def test_three_dimensional_and_two_dimensional_winners_can_disagree(self):
        world_points, track = _disagreeing_scene()

        ranking = best_view.score_track_candidates(
            world_points, track, _grid_surface_area
        )

        self.assertEqual(ranking["candidate_count"], 2)
        self.assertEqual(ranking["scored_candidate_count"], 2)
        self.assertEqual(ranking["best_frame_2d"], 0)
        self.assertEqual(ranking["best_frame_3d"], 1)
        self.assertEqual(ranking["best_frame"], 1)
        self.assertEqual(ranking["selection_method"], "surface_area_3d")
        self.assertIs(ranking["agreement"], False)
        self.assertEqual(
            [candidate["mask_pixel_count"] for candidate in ranking["candidates"]],
            [400, 100],
        )
        self.assertTrue(
            all(candidate["scored_3d"] for candidate in ranking["candidates"])
        )
        self.assertGreater(
            ranking["candidates"][1]["surface_area_3d"],
            ranking["candidates"][0]["surface_area_3d"],
        )

    def test_the_absolute_upstream_constant_would_rank_the_wrong_frame(self):
        world_points, track = _disagreeing_scene()

        def absolute_threshold_scorer(pointmap, mask, max_triangle_size=None):
            # Ignore the scale-relative threshold and use upstream's default.
            return _grid_surface_area(pointmap, mask, max_triangle_size=2e-4)

        absolute = best_view.score_track_candidates(
            world_points, track, absolute_threshold_scorer
        )
        relative = best_view.score_track_candidates(
            world_points, track, _grid_surface_area
        )

        self.assertEqual(absolute["best_frame_3d"], 0)
        self.assertEqual(relative["best_frame_3d"], 1)

    def test_candidate_records_carry_the_scale_policy_and_provenance(self):
        world_points, track = _disagreeing_scene()

        ranking = best_view.score_track_candidates(
            world_points,
            [{**evidence, "rle": f"rle-{evidence['frame_id']}"} for evidence in track],
            _grid_surface_area,
            source_frame_indices=[4, 19],
            source_frame_timestamps=[0.5, 2.25],
        )

        first = ranking["candidates"][0]
        self.assertEqual(first["scale_policy"], "relative_median_adjacent_spacing_v1")
        self.assertAlmostEqual(first["median_adjacent_spacing"], 0.01)
        self.assertAlmostEqual(first["max_triangle_size"], 100.0 * 0.01 * 0.01)
        self.assertEqual(first["bbox_xywh"], [0, 0, 20, 20])
        self.assertEqual(first["rle"], "rle-0")
        self.assertEqual(first["source_frame_index"], 4)
        self.assertEqual(first["source_timestamp_s"], 0.5)
        self.assertIsNone(first["unscored_reason"])

    def test_ranking_is_invariant_to_a_global_rescale(self):
        world_points, track = _disagreeing_scene()

        baseline = best_view.score_track_candidates(
            world_points, track, _grid_surface_area
        )
        rescaled = best_view.score_track_candidates(
            world_points * 1000.0, track, _grid_surface_area
        )

        self.assertEqual(baseline["best_frame_3d"], rescaled["best_frame_3d"])
        self.assertEqual(baseline["best_frame_2d"], rescaled["best_frame_2d"])
        self.assertEqual(baseline["agreement"], rescaled["agreement"])
        self.assertEqual(
            [candidate["scored_3d"] for candidate in baseline["candidates"]],
            [candidate["scored_3d"] for candidate in rescaled["candidates"]],
        )
        self.assertEqual(
            [candidate["sampled_frame_id"] for candidate in baseline["candidates"]],
            [candidate["sampled_frame_id"] for candidate in rescaled["candidates"]],
        )
        for before, after in zip(baseline["candidates"], rescaled["candidates"]):
            self.assertAlmostEqual(
                after["surface_area_3d"] / before["surface_area_3d"],
                1_000_000.0,
                places=3,
            )

    def test_degenerate_spacing_falls_back_to_the_two_dimensional_rule(self):
        world_points = np.zeros((2, 8, 8, 3), dtype=np.float64)
        big = np.zeros((8, 8), dtype=bool)
        big[1:5, 1:5] = True
        small = np.zeros((8, 8), dtype=bool)
        small[6, 6] = True
        track = [{"frame_id": 0, "mask": small}, {"frame_id": 1, "mask": big}]

        ranking = best_view.score_track_candidates(
            world_points, track, _grid_surface_area
        )

        self.assertEqual(ranking["scored_candidate_count"], 0)
        self.assertIsNone(ranking["best_frame_3d"])
        self.assertEqual(ranking["best_frame_2d"], 1)
        self.assertEqual(ranking["best_frame"], 1)
        self.assertEqual(ranking["selection_method"], "mask_pixel_count_fallback")
        self.assertIsNone(ranking["agreement"])
        self.assertEqual(
            [candidate["unscored_reason"] for candidate in ranking["candidates"]],
            ["degenerate_adjacent_spacing", "degenerate_adjacent_spacing"],
        )
        self.assertEqual(
            [candidate["median_adjacent_spacing"] for candidate in ranking["candidates"]],
            [None, None],
        )

    def test_a_failing_scorer_marks_the_candidate_unscored_without_failing_the_run(self):
        world_points, track = _disagreeing_scene()

        def failing(pointmap, mask, max_triangle_size=None):
            if pointmap[0, 1, 0] > 0.05:
                raise RuntimeError("Delaunay triangulation failed")
            return _grid_surface_area(pointmap, mask, max_triangle_size)

        ranking = best_view.score_track_candidates(world_points, track, failing)

        self.assertEqual(ranking["scored_candidate_count"], 1)
        self.assertEqual(ranking["best_frame_3d"], 0)
        self.assertEqual(
            ranking["candidates"][1]["unscored_reason"], "surface_area_failed"
        )

    def test_ties_resolve_to_the_earliest_sampled_frame(self):
        world_points = np.stack([_planar_frame(8, 8, 0.1) for _ in range(3)])
        mask = _block_mask(8, 8, 5)
        track = [{"frame_id": frame_id, "mask": mask} for frame_id in (2, 0, 1)]

        ranking = best_view.score_track_candidates(
            world_points, track, _grid_surface_area
        )

        self.assertEqual(ranking["best_frame_3d"], 0)
        self.assertEqual(ranking["best_frame_2d"], 0)
        self.assertIs(ranking["agreement"], True)
        self.assertEqual(ranking["first_sampled_frame_id"], 0)
        self.assertEqual(ranking["last_sampled_frame_id"], 2)

    def test_rejects_more_candidate_frames_than_the_sampling_cap(self):
        frames = best_view.BEST_VIEW_MAX_CANDIDATE_FRAMES + 1
        world_points = np.stack([_planar_frame(6, 6, 0.1) for _ in range(frames)])
        mask = _block_mask(6, 6, 4)
        track = [{"frame_id": frame_id, "mask": mask} for frame_id in range(frames)]

        with self.assertRaises(ValueError):
            best_view.score_track_candidates(world_points, track, _grid_surface_area)

    def test_rejects_malformed_tracks(self):
        world_points = np.stack([_planar_frame(6, 6, 0.1) for _ in range(2)])

        for track in (
            [],
            [{"frame_id": 5, "mask": _block_mask(6, 6, 3)}],
            [{"frame_id": 0, "mask": np.zeros((4, 4), dtype=bool)}],
            [{"frame_id": 0}],
        ):
            with self.subTest(track=track):
                with self.assertRaises(ValueError):
                    best_view.score_track_candidates(
                        world_points, track, _grid_surface_area
                    )


class BuildBestViewReportTest(unittest.TestCase):
    def _instances(
        self,
        count: int,
        *,
        height: int = 12,
        width: int = 12,
        rle_chars: int = 8,
    ):
        instances = []
        for index in range(count):
            size = 3 + (index % 6)
            instances.append(
                {
                    "category": "chair" if index % 2 == 0 else "table",
                    "track_id": index,
                    "track": [
                        {
                            "frame_id": frame_id,
                            "mask": _block_mask(height, width, size + frame_id),
                            "rle": f"{index}-{frame_id}".ljust(rle_chars, "7"),
                        }
                        for frame_id in range(2)
                    ],
                }
            )
        return instances

    def _build(self, out_dir, instances, **overrides):
        world_points = np.stack([_planar_frame(12, 12, 0.05) for _ in range(2)])
        arguments = {
            "instances": instances,
            "world_points": world_points,
            "surface_area_fn": _grid_surface_area,
            "out_dir": out_dir,
            "requested_categories": ["chair", "table"],
            "frames_used": 2,
            "model_frame_width": 12,
            "model_frame_height": 12,
            "source_frame_indices": [0, 11],
            "source_frame_timestamps": [0.0, 1.5],
        }
        arguments.update(overrides)
        return best_view.build_best_view_report(**arguments)

    def test_writes_the_negotiated_schema_and_returns_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            report = self._build(out_dir, self._instances(3))

            written = json.loads((out_dir / "best_view.json").read_text("utf-8"))

        self.assertEqual(report["schema"], "palatial.best_view.v1")
        self.assertEqual(report["coordinate_system"], "vggt_first_camera")
        self.assertEqual(report["scale_policy"], "relative_median_adjacent_spacing_v1")
        self.assertEqual(report["frames_used"], 2)
        self.assertEqual(report["model_frame_width"], 12)
        self.assertEqual(report["model_frame_height"], 12)
        self.assertEqual(report["total_instances"], 3)
        self.assertEqual(report["returned_instances"], 3)
        self.assertIs(report["truncated"], False)
        self.assertEqual(report["rle_scope"], "all_candidate_frames")
        self.assertEqual(written, report)
        self.assertEqual(
            [instance["instance_id"] for instance in report["instances"]],
            ["instance_0001", "instance_0002", "instance_0003"],
        )
        self.assertEqual(
            sum(report["agreement_counts"].values()), report["returned_instances"]
        )

    def test_caps_instances_by_descending_mask_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._build(Path(tmp), self._instances(40))

        self.assertEqual(report["total_instances"], 40)
        self.assertEqual(report["returned_instances"], best_view.BEST_VIEW_MAX_INSTANCES)
        self.assertIs(report["truncated"], True)
        sizes = [
            max(candidate["mask_pixel_count"] for candidate in instance["candidates"])
            for instance in report["instances"]
        ]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_reports_the_discovered_total_when_the_caller_pre_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._build(Path(tmp), self._instances(4), total_instances=91)

        self.assertEqual(report["total_instances"], 91)
        self.assertEqual(report["returned_instances"], 4)
        self.assertIs(report["truncated"], True)

    def test_rejects_a_total_smaller_than_the_scored_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._build(Path(tmp), self._instances(4), total_instances=2)

    def test_drops_non_winning_masks_before_truncating_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch.object(best_view, "BEST_VIEW_JSON_MAX_BYTES", 4_600):
                report = self._build(out_dir, self._instances(3, rle_chars=300))
            payload = (out_dir / "best_view.json").read_bytes()

        self.assertEqual(report["rle_scope"], "winning_frames_only")
        self.assertEqual(report["returned_instances"], 3)
        self.assertLessEqual(len(payload), 4_600)
        self.assertIn(
            "rle_dropped_for_size",
            [warning["code"] for warning in report["warnings"]],
        )
        for instance in report["instances"]:
            for candidate in instance["candidates"]:
                if candidate["sampled_frame_id"] == instance["best_frame"]:
                    self.assertIsNotNone(candidate["rle"])
                else:
                    self.assertIsNone(candidate["rle"])

    def test_truncates_instances_when_dropping_masks_is_not_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch.object(best_view, "BEST_VIEW_JSON_MAX_BYTES", 4_000):
                report = self._build(out_dir, self._instances(6, rle_chars=300))
            payload = (out_dir / "best_view.json").read_bytes()

        self.assertEqual(report["rle_scope"], "none")
        self.assertEqual(report["returned_instances"], 3)
        self.assertEqual(report["total_instances"], 6)
        self.assertIs(report["truncated"], True)
        self.assertLessEqual(len(payload), 4_000)
        self.assertEqual(json.loads(payload.decode("utf-8")), report)
        self.assertIn(
            "instances_truncated_for_size",
            [warning["code"] for warning in report["warnings"]],
        )

    def test_fails_when_even_one_instance_cannot_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(best_view, "BEST_VIEW_JSON_MAX_BYTES", 32):
                with self.assertRaises(ValueError):
                    self._build(Path(tmp), self._instances(3))


if __name__ == "__main__":
    unittest.main()
