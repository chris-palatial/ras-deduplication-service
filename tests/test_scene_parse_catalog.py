import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scene_parse_catalog as catalog
import stage2_service as stage2


def _mask(height=10, width=10, *, x=1, y=1, box_width=8, box_height=8):
    value = np.zeros((height, width), dtype=bool)
    value[y : y + box_height, x : x + box_width] = True
    return value


def _track(frame_ids, mask=None):
    value = _mask() if mask is None else mask
    return [{"frame_id": frame_id, "mask": value.copy()} for frame_id in frame_ids]


class SceneParseCatalogRetirementTest(unittest.TestCase):
    def test_typed_router_retires_the_legacy_profile_before_any_runner(self):
        payload = {
            "analysis_type": stage2.SCENE_PARSE_CATALOG_ANALYSIS_TYPE,
            "mode": "full",
        }
        with patch.object(stage2, "run_scene_parse_catalog") as runner, patch.object(
            stage2, "_ensure_ras_installed"
        ) as bootstrap:
            result = stage2.run_stage2(payload)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["analysis_type"], "scene_parse_catalog_v1")
        self.assertIn("legacy local-SAM profile is retired", result["error"])
        runner.assert_not_called()
        bootstrap.assert_not_called()

    def test_direct_legacy_runner_also_fails_closed(self):
        with patch.object(stage2, "_ensure_ras_installed") as bootstrap:
            result = stage2.run_scene_parse_catalog({})

        self.assertEqual(result["error_code"], "profile_retired")
        self.assertIn("Scene Parse must use its owning service", result["error"])
        bootstrap.assert_not_called()

    def test_local_scene_parse_model_entrypoints_are_absent(self):
        for name in (
            "_load_scene_parse_sam_video_model",
            "_preflight_scene_parse_sam_backend",
            "_ensure_scene_parse_model_weights",
            "_scene_parse_model_identity",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(stage2, name))


class SceneParseCatalogBuilderTest(unittest.TestCase):
    def test_catalog_has_deterministic_keys_complete_evidence_and_best_view(self):
        small = _mask(box_width=4, box_height=4)
        large = _mask(box_width=8, box_height=8)
        masks = {
            "chair": [[
                {"frame_id": 0, "mask": small},
                {"frame_id": 2, "mask": large},
            ]],
            "table": [_track([1])],
        }
        kwargs = {
            "all_masks": masks,
            "frame_shape": (3, 10, 10),
            "source_sha256": "b" * 64,
            "category_prompts": ["chair", "table"],
            "source_frame_indices": [0, 15, 30],
            "source_frame_timestamps_s": [0.0, 0.5, 1.0],
        }
        first = catalog.build_scene_parse_objects(**kwargs)
        second = catalog.build_scene_parse_objects(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first["returned_count"], 2)
        chair = first["objects"][0]
        self.assertRegex(chair["object_key"], r"^obj_[0-9a-f]{24}$")
        self.assertEqual(chair["evidence_strength"], "limited")
        self.assertEqual(len(chair["evidence"]), 2)
        self.assertEqual(chair["best_evidence_index"], 1)
        self.assertEqual(chair["evidence"][1]["source_frame_index"], 30)
        self.assertEqual(chair["evidence"][1]["timestamp_ms"], 1000)
        self.assertTrue(
            all(
                "mask" not in evidence
                for item in first["objects"]
                for evidence in item["evidence"]
            )
        )

    def test_one_frame_object_is_retained_with_limited_evidence(self):
        points = np.zeros((3, 10, 10, 3), dtype=np.float32)
        confidence = np.ones((3, 10, 10), dtype=np.float32)
        short = _track([0])
        recovered, preserved = catalog.preserve_short_scene_parse_tracks(
            deduplicated_masks={},
            category_masks={"chair": [short]},
            category_prompts=["chair"],
            world_points=points,
            world_points_conf=confidence,
            overlap_fn=lambda _left, _right: 0.0,
        )

        self.assertEqual(preserved, 1)
        built = catalog.build_scene_parse_objects(
            all_masks=recovered,
            frame_shape=(3, 10, 10),
            source_sha256="c" * 64,
            category_prompts=["chair"],
            source_frame_indices=[0, 1, 2],
            source_frame_timestamps_s=[0.0, 0.5, 1.0],
        )
        self.assertEqual(built["objects"][0]["evidence_strength"], "single_frame")

    def test_short_track_is_not_readded_when_it_overlaps_returned_merge(self):
        points = np.zeros((3, 10, 10, 3), dtype=np.float32)
        confidence = np.ones((3, 10, 10), dtype=np.float32)
        long_track = _track([0, 1, 2])
        short_track = _track([0])

        recovered, preserved = catalog.preserve_short_scene_parse_tracks(
            deduplicated_masks={"chair": [long_track]},
            category_masks={"chair": [long_track, short_track]},
            category_prompts=["chair"],
            world_points=points,
            world_points_conf=confidence,
            overlap_fn=lambda _left, _right: 0.8,
        )

        self.assertEqual(preserved, 0)
        self.assertEqual(recovered, {"chair": [long_track]})

    def test_inline_bound_rechecks_final_truncation_metadata(self):
        response = {
            "objects": [
                {"object_key": "obj_fixture", "evidence": [], "padding": "x" * 200}
            ],
            "counts": {
                "deduplicated_objects": 1,
                "returned_objects": 1,
                "omitted_objects": 0,
                "evidence_items": 0,
            },
            "truncated": False,
            "warnings": [],
        }
        with patch.object(catalog, "SCENE_PARSE_CATALOG_MAX_INLINE_BYTES", 256):
            bounded = catalog.bound_scene_parse_response(response)
        encoded = json.dumps(
            bounded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertLessEqual(len(encoded), 256)
        self.assertEqual(bounded["objects"], [])
        self.assertEqual(bounded["counts"]["returned_objects"], 0)
        self.assertTrue(bounded["truncated"])


if __name__ == "__main__":
    unittest.main()
