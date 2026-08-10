import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import object_catalog as catalog
import stage2_service as stage2


def _world_points(frame_count: int, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    points = np.zeros((frame_count, height, width, 3), dtype=np.float32)
    points[..., 0] = xx
    points[..., 1] = yy
    for frame_id in range(frame_count):
        points[frame_id, ..., 2] = frame_id
    return points


def _mask(height: int = 16, width: int = 16) -> np.ndarray:
    value = np.zeros((height, width), dtype=bool)
    value[4:12, 4:12] = True
    return value


def _track(frame_id: int = 0, *, height: int = 16, width: int = 16):
    return [{"frame_id": frame_id, "mask": _mask(height, width)}]


def _build_catalog(out_dir: Path, object_count: int, *, label: str = "chair"):
    colors = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    colors[..., 0] = 240
    points = _world_points(1, 16, 16)
    return catalog.build_object_catalog(
        all_masks={label: [_track() for _ in range(object_count)]},
        colors=colors,
        world_points=points,
        requested_categories=[label],
        source_frame_indices=[42],
        source_frame_timestamps=[1.25],
        surface_area_fn=lambda _points, mask: float(np.count_nonzero(mask)),
        out_dir=out_dir,
    )


class ObjectCatalogSelectorTest(unittest.TestCase):
    def test_surface_selector_is_bounded_and_deterministic(self):
        points = _world_points(20, 8, 8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        track = [{"frame_id": frame_id, "mask": mask} for frame_id in reversed(range(20))]
        calls = []

        def surface(pointmap, bounded_mask):
            calls.append(int(pointmap[0, 0, 2]))
            return float(pointmap[0, 0, 2] + 1)

        selected = catalog.select_representative_frame(points, track, surface)

        self.assertEqual(len(calls), catalog.OBJECT_CATALOG_MAX_SELECTOR_FRAMES)
        self.assertEqual(calls, list(range(catalog.OBJECT_CATALOG_MAX_SELECTOR_FRAMES)))
        self.assertEqual(selected["frame_id"], 15)
        self.assertEqual(
            selected["selection_method"],
            "bounded_ras_stage3_surface_area_v1",
        )
        self.assertEqual(selected["first_sampled_frame_id"], 0)
        self.assertEqual(selected["last_sampled_frame_id"], 19)

    def test_surface_area_tie_uses_earliest_sampled_frame(self):
        points = _world_points(4, 8, 8)
        track = [
            {"frame_id": frame_id, "mask": np.ones((8, 8), dtype=bool)}
            for frame_id in (3, 1, 2)
        ]

        selected = catalog.select_representative_frame(
            points,
            track,
            lambda _points, _mask: 1.0,
        )

        self.assertEqual(selected["frame_id"], 1)

    def test_failed_surface_scoring_falls_back_to_largest_mask_then_frame_id(self):
        points = _world_points(4, 8, 8)
        small = np.zeros((8, 8), dtype=bool)
        small[0:2, 0:2] = True
        large = np.zeros((8, 8), dtype=bool)
        large[1:4, 1:4] = True
        track = [
            {"frame_id": 3, "mask": small},
            {"frame_id": 2, "mask": large},
            {"frame_id": 1, "mask": large},
        ]

        with self.subTest("exceptions"):
            selected = catalog.select_representative_frame(
                points,
                track,
                lambda _points, _mask: (_ for _ in ()).throw(RuntimeError("bad geometry")),
            )
            self.assertEqual(selected["frame_id"], 1)
            self.assertEqual(selected["selection_method"], "largest_mask_fallback_v1")
            self.assertEqual(selected["mask_pixel_count"], 9)

        with self.subTest("non-finite scores"):
            selected = catalog.select_representative_frame(
                points,
                track,
                lambda _points, _mask: float("nan"),
            )
            self.assertEqual(selected["frame_id"], 1)
            self.assertEqual(selected["selection_method"], "largest_mask_fallback_v1")

    def test_surface_scorer_receives_at_most_25000_mask_pixels(self):
        points = _world_points(1, 300, 300)
        mask = np.ones((300, 300), dtype=bool)
        observed = []

        def bounded_surface(pointmap, bounded_mask):
            observed.append((pointmap.shape, int(np.count_nonzero(bounded_mask))))
            return 1.0

        selected = catalog.select_representative_frame(
            points,
            [{"frame_id": 0, "mask": mask}],
            bounded_surface,
        )

        self.assertEqual(selected["frame_id"], 0)
        self.assertLessEqual(observed[0][1], catalog.OBJECT_CATALOG_MAX_SURFACE_PIXELS)
        self.assertLess(observed[0][0][0], 300)

    def test_invalid_track_evidence_fails_closed(self):
        points = _world_points(2, 8, 8)
        valid_mask = np.ones((8, 8), dtype=bool)
        cases = (
            [],
            [{"frame_id": True, "mask": valid_mask}],
            [{"frame_id": 2, "mask": valid_mask}],
            [{"frame_id": 0, "mask": np.ones((7, 8), dtype=bool)}],
            [{"frame_id": 0, "mask": np.zeros((8, 8), dtype=bool)}],
            [{"frame_id": 0}],
        )
        for track in cases:
            with self.subTest(track=track), self.assertRaises(ValueError):
                catalog.select_representative_frame(
                    points,
                    track,
                    lambda _points, _mask: 1.0,
                )


class ObjectCatalogArtifactTest(unittest.TestCase):
    def test_empty_catalog_has_neutral_one_cell_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _build_catalog(Path(tmp), 0)
            json_bytes = (Path(tmp) / catalog.OBJECT_CATALOG_JSON_NAME).read_bytes()
            jpeg_bytes = (Path(tmp) / catalog.OBJECT_CROPS_ATLAS_NAME).read_bytes()
            decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertEqual(result["total_count"], 0)
        self.assertEqual(result["returned_count"], 0)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["objects"], [])
        self.assertEqual((decoded.shape[1], decoded.shape[0]), (224, 224))
        self.assertEqual(result["atlas"]["columns"], 1)
        self.assertEqual(result["atlas"]["rows"], 1)
        self.assertEqual(json.loads(json_bytes), result)
        self.assertEqual(hashlib.sha256(jpeg_bytes).hexdigest(), result["atlas"]["sha256"])
        self.assertNotIn("json_sha256", result)

    def test_one_object_has_source_mapping_crop_and_rgb_correct_atlas(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _build_catalog(Path(tmp), 1)
            jpeg_bytes = (Path(tmp) / catalog.OBJECT_CROPS_ATLAS_NAME).read_bytes()
            decoded_bgr = cv2.imdecode(
                np.frombuffer(jpeg_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

        obj = result["objects"][0]
        representative = obj["representative"]
        self.assertEqual(obj["id"], "object_0001")
        self.assertEqual(obj["label"], "chair")
        self.assertEqual(obj["label_source"], "requested_prompt")
        self.assertEqual(representative["sampled_frame_id"], 0)
        self.assertEqual(representative["source_frame_index"], 42)
        self.assertEqual(representative["source_timestamp_s"], 1.25)
        self.assertEqual(representative["model_frame_crop_xywh"], [0, 0, 16, 16])
        self.assertEqual(representative["atlas_tile_xywh"], [0, 0, 224, 224])
        center_bgr = decoded_bgr[112, 112]
        self.assertGreater(int(center_bgr[2]), 220)
        self.assertLess(int(center_bgr[0]), 20)

    def test_128_objects_fill_the_maximum_bounded_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _build_catalog(Path(tmp), 128)
            jpeg_path = Path(tmp) / catalog.OBJECT_CROPS_ATLAS_NAME
            decoded = cv2.imread(str(jpeg_path), cv2.IMREAD_COLOR)

        self.assertEqual(result["total_count"], 128)
        self.assertEqual(result["returned_count"], 128)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["objects"][-1]["id"], "object_0128")
        self.assertEqual(result["atlas"]["columns"], 12)
        self.assertEqual(result["atlas"]["rows"], 11)
        self.assertEqual(
            (decoded.shape[1], decoded.shape[0]),
            (catalog.OBJECT_CROPS_MAX_WIDTH, catalog.OBJECT_CROPS_MAX_HEIGHT),
        )
        tiles = [tuple(item["representative"]["atlas_tile_xywh"]) for item in result["objects"]]
        self.assertEqual(len(set(tiles)), 128)

    def test_129_objects_report_truthful_truncation_at_128(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _build_catalog(Path(tmp), 129)

        self.assertEqual(result["total_count"], 129)
        self.assertEqual(result["returned_count"], 128)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["objects"]), 128)

    def test_maximum_noise_atlas_stays_within_jpeg_byte_and_dimension_limits(self):
        rng = np.random.default_rng(7)
        noise = rng.integers(
            0,
            256,
            size=(catalog.OBJECT_CROPS_MAX_HEIGHT, catalog.OBJECT_CROPS_MAX_WIDTH, 3),
            dtype=np.uint8,
        )

        payload = catalog._encode_atlas(noise)
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertLessEqual(len(payload), catalog.OBJECT_CROPS_ATLAS_MAX_BYTES)
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        self.assertTrue(payload.endswith(b"\xff\xd9"))
        self.assertEqual(
            (decoded.shape[1], decoded.shape[0]),
            (catalog.OBJECT_CROPS_MAX_WIDTH, catalog.OBJECT_CROPS_MAX_HEIGHT),
        )

    def test_output_is_byte_deterministic_and_schema_contains_no_worker_location(self):
        label = "<script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_result = _build_catalog(Path(first), 1, label=label)
            second_result = _build_catalog(Path(second), 1, label=label)
            first_json = (Path(first) / catalog.OBJECT_CATALOG_JSON_NAME).read_bytes()
            second_json = (Path(second) / catalog.OBJECT_CATALOG_JSON_NAME).read_bytes()
            first_jpeg = (Path(first) / catalog.OBJECT_CROPS_ATLAS_NAME).read_bytes()
            second_jpeg = (Path(second) / catalog.OBJECT_CROPS_ATLAS_NAME).read_bytes()

        self.assertEqual(first_json, second_json)
        self.assertEqual(first_jpeg, second_jpeg)
        self.assertEqual(first_result, second_result)
        self.assertLessEqual(len(first_json), catalog.OBJECT_CATALOG_JSON_MAX_BYTES)
        self.assertLessEqual(len(first_jpeg), catalog.OBJECT_CROPS_ATLAS_MAX_BYTES)
        self.assertEqual(first_result["schema_version"], "palatial.object_catalog.v1")
        self.assertEqual(first_result["scope"], "requested_categories")
        self.assertFalse(first_result["exhaustive"])
        self.assertEqual(
            first_result["identity_semantics"],
            "ras_spatial_deduplicated_instance_hypothesis",
        )
        self.assertEqual(first_result["requested_categories"], [label])
        serialized = first_json.decode("utf-8")
        self.assertNotIn("http://", serialized)
        self.assertNotIn("https://", serialized)
        self.assertNotIn("/tmp/", serialized)
        self.assertNotIn("video_b64", serialized)

    def test_category_and_geometry_schema_validation_fails_closed(self):
        colors = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        points = _world_points(1, 8, 8)
        valid_track = [{"frame_id": 0, "mask": np.ones((8, 8), dtype=bool)}]
        base = {
            "all_masks": {"chair": [valid_track]},
            "colors": colors,
            "world_points": points,
            "requested_categories": ["chair"],
            "source_frame_indices": None,
            "source_frame_timestamps": None,
            "surface_area_fn": lambda _points, _mask: 1.0,
        }
        invalid_overrides = (
            {"requested_categories": ["chair", "chair"]},
            {"requested_categories": []},
            {"requested_categories": ["x" * 65]},
            {"all_masks": {"table": [valid_track]}},
            {"all_masks": {"chair": "not-tracks"}},
            {"colors": np.zeros((1, 8, 8), dtype=np.uint8)},
            {"world_points": np.zeros((1, 8, 7, 3), dtype=np.float32)},
        )
        for override in invalid_overrides:
            with self.subTest(override=tuple(override)), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    catalog.build_object_catalog(
                        **{**base, **override},
                        out_dir=Path(tmp),
                    )


class ObjectCatalogNegotiationTest(unittest.TestCase):
    @staticmethod
    def _full_success():
        return {"status": "ok", "mode": "full"}

    def test_negotiation_requires_exact_integer_one_before_runner_or_materialization(self):
        invalid_versions = (0, 2, "1", True, None, 1.0, {}, [])
        for version in invalid_versions:
            with self.subTest(version=version), patch.object(
                stage2,
                "run_stage2_full",
            ) as runner, patch.object(stage2, "_materialize_video") as materialize:
                result = stage2.run_stage2({
                    "mode": "full",
                    "analysis_type": "dedup_ras_vggt_sam3",
                    "object_catalog_version": version,
                    "categories": ["chair"],
                    "max_frames": 8,
                })

            self.assertEqual(result["status"], "error")
            self.assertIn("not owned by this RunPod worker", result["error"])
            runner.assert_not_called()
            materialize.assert_not_called()

    def test_negotiation_requires_full_ras_analysis_before_runner(self):
        payloads = (
            {
                "mode": "full",
                "object_catalog_version": 1,
                "categories": ["chair"],
                "max_frames": 8,
            },
            {
                "mode": "geometry",
                "analysis_type": "geometry_vggt_1b",
                "object_catalog_version": 1,
                "categories": ["chair"],
                "max_frames": 8,
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload), patch.object(
                stage2,
                "run_stage2_full",
            ) as full_runner, patch.object(
                stage2,
                "run_stage2_geometry",
            ) as geometry_runner, patch.object(stage2, "_materialize_video") as materialize:
                result = stage2.run_stage2(payload)

            self.assertEqual(result["status"], "error")
            full_runner.assert_not_called()
            geometry_runner.assert_not_called()
            materialize.assert_not_called()

    def test_retired_monolithic_negotiation_never_reaches_full_runner(self):
        with patch.object(
            stage2,
            "run_stage2_full",
            return_value=self._full_success(),
        ) as runner:
            result = stage2.run_stage2({
                "mode": "full",
                "analysis_type": "dedup_ras_vggt_sam3",
                "object_catalog_version": 1,
                "categories": ["chair"],
                "max_frames": 8,
            })

        self.assertEqual(result["status"], "error", result)
        self.assertIn("RAS finalizer composite", result["error"])
        runner.assert_not_called()


    def test_direct_full_runner_rejects_bad_negotiation_before_materialization(self):
        with patch.object(stage2, "_materialize_standard_video") as materialize:
            result = stage2.run_stage2_full({
                "mode": "full",
                "object_catalog_version": 1,
                "categories": ["chair"],
                "max_frames": 8,
            })

        self.assertEqual(result["status"], "error")
        self.assertIn("monolithic local-SAM Full RAS path is retired", result["error"])
        materialize.assert_not_called()

    def test_legacy_full_manifest_ignores_stray_catalog_files(self):
        upload = {
            "base": "https://edge.example",
            "runId": "legacy-run",
            "token": "secret",
            "exp": 9_999_999_999_999,
        }
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            for name in (
                "point_cloud.glb",
                "instance_masks.mp4",
                catalog.OBJECT_CATALOG_JSON_NAME,
                catalog.OBJECT_CROPS_ATLAS_NAME,
            ):
                (out_dir / name).write_bytes(b"fixture")

            def uploaded(_ticket, path, media_type):
                return {
                    "name": path.name,
                    "sha256": "a" * 64,
                    "bytes": path.stat().st_size,
                    "mediaType": media_type,
                }

            with patch("artifact_upload.upload_artifact_file", side_effect=uploaded):
                manifest = stage2._artifact_manifest(
                    out_dir,
                    work,
                    {"mode": "full", "upload": upload},
                )

        self.assertEqual(
            manifest["required_files"],
            ["point_cloud.glb", "instance_masks.mp4"],
        )
        self.assertEqual(
            [receipt["name"] for receipt in manifest["receipts"]],
            ["point_cloud.glb", "instance_masks.mp4"],
        )
        self.assertNotIn(catalog.OBJECT_CATALOG_JSON_NAME, manifest["files"])
        self.assertNotIn(catalog.OBJECT_CROPS_ATLAS_NAME, manifest["files"])

    def test_finalizer_manifest_requires_and_delivers_exactly_three_artifacts(self):
        upload = {
            "base": "https://edge.example",
            "runId": "catalog-run",
            "token": "secret",
            "exp": 9_999_999_999_999,
        }
        required = list(stage2.FULL_DEDUP_FINAL_ARTIFACTS)
        payload = {
            "mode": "full",
            "analysis_type": stage2.FULL_DEDUP_FINALIZE_ANALYSIS_TYPE,
            "object_catalog_version": 1,
            "upload": upload,
        }
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            for name in required:
                (out_dir / name).write_bytes(b"fixture")
            (out_dir / "not-allowlisted.txt").write_bytes(b"private")

            def uploaded(_ticket, path, media_type):
                return {
                    "name": path.name,
                    "sha256": "a" * 64,
                    "bytes": path.stat().st_size,
                    "mediaType": media_type,
                }

            with patch("artifact_upload.upload_artifact_file", side_effect=uploaded):
                manifest = stage2._artifact_manifest(out_dir, work, payload)

        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["required_files"], required)
        self.assertEqual([receipt["name"] for receipt in manifest["receipts"]], required)
        self.assertEqual(
            [receipt["mediaType"] for receipt in manifest["receipts"]],
            ["video/mp4", "application/json", "image/jpeg"],
        )
        self.assertNotIn("point_cloud.glb", manifest["files"])
        self.assertNotIn("not-allowlisted.txt", manifest["files"])

    def test_missing_either_negotiated_catalog_artifact_fails_delivery(self):
        upload = {
            "base": "https://edge.example",
            "runId": "catalog-run",
            "token": "secret",
            "exp": 9_999_999_999_999,
        }
        payload = {
            "mode": "full",
            "analysis_type": stage2.FULL_DEDUP_FINALIZE_ANALYSIS_TYPE,
            "object_catalog_version": 1,
            "upload": upload,
        }
        required = set(stage2.FULL_DEDUP_FINAL_ARTIFACTS)
        for missing in catalog.OBJECT_CATALOG_JSON_NAME, catalog.OBJECT_CROPS_ATLAS_NAME:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                out_dir = work / "output"
                out_dir.mkdir()
                for name in required - {missing}:
                    (out_dir / name).write_bytes(b"fixture")

                def uploaded(_ticket, path, media_type):
                    return {
                        "name": path.name,
                        "sha256": "a" * 64,
                        "bytes": path.stat().st_size,
                        "mediaType": media_type,
                    }

                with patch("artifact_upload.upload_artifact_file", side_effect=uploaded):
                    manifest = stage2._artifact_manifest(out_dir, work, payload)

            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["missing_required"], [missing])
            failure = stage2._artifact_delivery_error(payload, manifest)
            self.assertEqual(failure["error_code"], "artifact_delivery_failed")
            self.assertEqual(failure["artifact_delivery"]["missing_required"], [missing])


class ObjectCatalogTransportCanaryTest(unittest.TestCase):
    @staticmethod
    def _payload():
        return {
            "mode": "dry_run",
            "analysis_type": stage2.OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE,
            "object_catalog_transport_canary": True,
            "categories": [stage2.OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY],
            "upload": {
                "base": "https://edge.example",
                "runId": "internal-canary",
                "token": "privileged-ticket",
                "exp": 9_999_999_999_999,
            },
        }

    def test_canary_runs_real_catalog_and_manifest_without_video_or_models(self):
        captured = {}
        acknowledgements = []

        def uploaded(_ticket, path, media_type, *, require_put_acknowledgement=False):
            captured[path.name] = path.read_bytes()
            acknowledgements.append(require_put_acknowledgement)
            return {
                "name": path.name,
                "sha256": hashlib.sha256(captured[path.name]).hexdigest(),
                "bytes": len(captured[path.name]),
                "mediaType": media_type,
            }

        with patch("artifact_upload.upload_artifact_file", side_effect=uploaded), patch.object(
            stage2,
            "_materialize_video",
        ) as materialize, patch.object(
            stage2,
            "_ensure_ras_installed",
        ) as bootstrap, patch.object(
            stage2,
            "run_stage2_dry",
        ) as generic_dry_run:
            result = stage2.run_stage2(self._payload())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            result["analysis_type"],
            stage2.OBJECT_CATALOG_TRANSPORT_CANARY_ANALYSIS_TYPE,
        )
        self.assertTrue(result["synthetic_transport_canary"])
        self.assertEqual(result["frames_used"], 1)
        self.assertEqual(result["instance_count"], 0)
        self.assertEqual(result["instances"], [])
        self.assertEqual(
            result["artifacts"]["required_files"],
            [catalog.OBJECT_CATALOG_JSON_NAME, catalog.OBJECT_CROPS_ATLAS_NAME],
        )
        self.assertEqual(set(captured), set(result["artifacts"]["required_files"]))
        self.assertEqual(acknowledgements, [True, True])
        generated_json = json.loads(captured[catalog.OBJECT_CATALOG_JSON_NAME])
        self.assertEqual(generated_json["schema_version"], "palatial.object_catalog.v1")
        self.assertNotIn("synthetic_transport_canary", generated_json)
        self.assertEqual(
            generated_json["requested_categories"],
            [stage2.OBJECT_CATALOG_TRANSPORT_CANARY_CATEGORY],
        )
        self.assertEqual(generated_json["objects"], [])
        self.assertEqual(
            hashlib.sha256(captured[catalog.OBJECT_CROPS_ATLAS_NAME]).hexdigest(),
            generated_json["atlas"]["sha256"],
        )
        decoded = cv2.imdecode(
            np.frombuffer(captured[catalog.OBJECT_CROPS_ATLAS_NAME], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual((decoded.shape[1], decoded.shape[0]), (224, 224))
        materialize.assert_not_called()
        bootstrap.assert_not_called()
        generic_dry_run.assert_not_called()

    def test_canary_preflight_is_strict_and_happens_before_any_work(self):
        base = self._payload()
        invalid = (
            {**base, "object_catalog_transport_canary": False},
            {**base, "object_catalog_transport_canary": "true"},
            {key: value for key, value in base.items() if key != "object_catalog_transport_canary"},
            {**base, "categories": ["chair"]},
            {**base, "video_b64": "eA=="},
            {key: value for key, value in base.items() if key != "upload"},
            {**base, "analysis_type": "validation_v1"},
            {**base, "object_catalog_version": 1},
        )
        for payload in invalid:
            with self.subTest(payload=payload), patch.object(
                stage2,
                "run_object_catalog_transport_canary",
            ) as runner, patch.object(stage2, "_materialize_video") as materialize, patch.object(
                stage2,
                "build_object_catalog",
            ) as builder:
                result = stage2.run_stage2(payload)

            self.assertEqual(result["status"], "error")
            runner.assert_not_called()
            materialize.assert_not_called()
            builder.assert_not_called()

    def test_canary_upload_failure_is_a_sanitized_delivery_failure(self):
        with patch(
            "artifact_upload.upload_artifact_file",
            side_effect=RuntimeError("https://signed.example/?secret=do-not-leak"),
        ):
            result = stage2.run_stage2(self._payload())

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "artifact_delivery_failed")
        self.assertTrue(result["synthetic_transport_canary"])
        self.assertEqual(result["instance_count"], 0)
        serialized = json.dumps(result)
        self.assertNotIn("signed.example", serialized)
        self.assertNotIn("do-not-leak", serialized)

    def test_canary_success_provenance_rejects_any_physical_instance_claim(self):
        false_success = {
            "status": "ok",
            "mode": "dry_run",
            "synthetic_transport_canary": True,
            "frames_used": 1,
            "instance_count": 1,
            "instances": [{"instance_id": "fake"}],
            "object_catalog": {
                "schema_version": "palatial.object_catalog.v1",
                "total_count": 1,
                "returned_count": 1,
            },
            "artifacts": {
                "complete": True,
                "required_files": list(stage2.OBJECT_CATALOG_ARTIFACTS),
            },
        }
        with patch.object(
            stage2,
            "run_object_catalog_transport_canary",
            return_value=false_success,
        ):
            result = stage2.run_stage2(self._payload())

        self.assertEqual(result["status"], "error")
        self.assertIn("provenance", result["error"])
        self.assertNotIn("instances", result)


if __name__ == "__main__":
    unittest.main()
