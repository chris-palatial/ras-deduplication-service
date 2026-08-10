import hashlib
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import stage2_service as stage2


def _document(frames, *, width=4, height=4):
    return {
        "width": width,
        "height": height,
        "num_frames": len(frames),
        "frames": [
            {"frame_index": index, "objects": objects}
            for index, objects in enumerate(frames)
        ],
    }


def _request(*, categories=("chair",), documents=None):
    documents = documents or [_document([[], []]) for _ in categories]
    masks = []
    for index, (category, document) in enumerate(zip(categories, documents)):
        encoded = json.dumps(document, separators=(",", ":")).encode()
        masks.append({
            "category": category,
            "request_id": f"fal_req_{index:06d}",
            "url": f"https://artifacts.example/inputs/mask-{index}.json",
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        })
    return {
        "analysis_type": stage2.FULL_DEDUP_FINALIZE_ANALYSIS_TYPE,
        "mode": "full",
        "geometry_inputs_url": "https://artifacts.example/inputs/geometry_inputs.npz",
        "sampled_clip_url": "https://artifacts.example/inputs/sampled_clip.mp4",
        "masks": masks,
        "expected_frames_used": 2,
        "expected_model_frame_width": 4,
        "expected_model_frame_height": 4,
        "expected_geometry_model_id": stage2.DEFAULT_VGGT_MODEL_ID,
        "expected_geometry_source_revision": stage2.DEFAULT_VGGT_REVISION,
        "categories": list(categories),
        "sam_model": stage2.FAL_SAM3_RLE_OBJECTS_MODEL_ID,
        "room_align": False,
        "object_catalog_version": 1,
        "upload": {
            "base": "https://artifacts.example",
            "runId": "dedup-finalize-test",
            "token": "opaque-test-token",
        },
    }


class FullDedupFinalizeRequestTest(unittest.TestCase):
    def test_accepts_exact_fal_track_documents_and_receipts(self):
        normalized, error = stage2._validate_full_dedup_finalize_request(_request())

        self.assertIsNone(error)
        self.assertEqual(normalized["sam_model"], "fal-ai/sam-3/video-rle-objects")
        self.assertIs(normalized["room_align"], False)
        self.assertEqual(normalized["masks"][0]["request_id"], "fal_req_000000")

    def test_rejects_drift_before_any_worker_execution(self):
        cases = {
            "wrong_model": {"sam_model": "fal-ai/sam-3-1/video-rle-objects"},
            "room_alignment": {"room_align": True},
            "foreign_clip": {"sampled_clip_url": "https://foreign.example/clip.mp4"},
            "wrong_catalog": {"object_catalog_version": "1"},
            "too_many_shared_geometry_frames": {"expected_frames_used": 25},
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                normalized, error = stage2._validate_full_dedup_finalize_request({
                    **_request(),
                    **change,
                })
                self.assertIsNone(normalized)
                self.assertIsInstance(error, str)

    def test_rejects_missing_out_of_order_or_unreceipted_category_documents(self):
        base = _request(categories=("chair", "table"))
        payloads = []
        payloads.append({**base, "masks": base["masks"][:1]})
        payloads.append({**base, "masks": list(reversed(base["masks"]))})
        missing_hash = dict(base["masks"][0])
        missing_hash.pop("sha256")
        payloads.append({**base, "masks": [missing_hash, base["masks"][1]]})
        for payload in payloads:
            with self.subTest(masks=payload["masks"]):
                normalized, error = stage2._validate_full_dedup_finalize_request(payload)
                self.assertIsNone(normalized)
                self.assertIsInstance(error, str)

    def test_typed_router_dispatches_only_the_normalized_finalizer_contract(self):
        with patch.object(
            stage2,
            "run_full_dedup_finalize",
            return_value={"status": "error", "mode": "full", "error": "fixture"},
        ) as runner:
            result = stage2.run_stage2(_request())

        self.assertEqual(result["analysis_type"], stage2.FULL_DEDUP_FINALIZE_ANALYSIS_TYPE)
        runner.assert_called_once()
        submitted = runner.call_args.args[0]
        self.assertEqual(set(submitted), {
            "analysis_type",
            "mode",
            "geometry_inputs_url",
            "sampled_clip_url",
            "masks",
            "expected_frames_used",
            "expected_model_frame_width",
            "expected_model_frame_height",
            "expected_geometry_model_id",
            "expected_geometry_source_revision",
            "categories",
            "sam_model",
            "room_align",
            "object_catalog_version",
            "upload",
        })


class FullDedupFalTrackDecodeTest(unittest.TestCase):
    def test_track_ids_are_category_scoped_and_visibility_gaps_split_tracks(self):
        chair = _document([
            [{"track_id": 7, "rle": "0 16"}],
            [{"track_id": 7, "rle": "0 16"}],
            [],
            [{"track_id": 7, "rle": "1 16"}],
        ])
        table = _document([
            [{"track_id": 7, "rle": "0 16"}],
            [],
            [],
            [],
        ])

        tracks = stage2._full_dedup_tracks_from_masks(
            [("chair", chair), ("table", table)],
            frames_used=4,
            height=4,
            width=4,
        )

        self.assertEqual(
            [[item["frame_id"] for item in segment] for segment in tracks["chair"]],
            [[0, 1], [3]],
        )
        self.assertEqual(
            [[item["frame_id"] for item in segment] for segment in tracks["table"]],
            [[0]],
        )
        self.assertTrue(tracks["chair"][1][0]["mask"].all())

    def test_rejects_canvas_frame_order_duplicate_ids_and_ambiguous_rle(self):
        good = _document([
            [{"track_id": 1, "rle": "0 16"}],
            [],
        ])
        bad_documents = []
        bad_documents.append({**good, "width": 5})
        bad_documents.append({**good, "frames": [good["frames"][1], good["frames"][0]]})
        duplicate = _document([
            [
                {"track_id": 1, "rle": "0 16"},
                {"track_id": 1, "rle": "0 16"},
            ],
            [],
        ])
        bad_documents.append(duplicate)
        malformed = _document([[{"track_id": 1, "rle": "3 4 4 2"}], []])
        bad_documents.append(malformed)
        for document in bad_documents:
            with self.subTest(document=document), self.assertRaises(RuntimeError):
                stage2._full_dedup_tracks_from_masks(
                    [("chair", document)],
                    frames_used=2,
                    height=4,
                    width=4,
                )

    def test_rejects_duplicate_category_documents_and_decoded_mask_memory_overflow(self):
        document = _document([
            [{"track_id": 1, "rle": "0 16"}],
            [{"track_id": 1, "rle": "0 16"}],
        ])
        with self.assertRaisesRegex(RuntimeError, "categories must be unique"):
            stage2._full_dedup_tracks_from_masks(
                [("chair", document), ("chair", document)],
                frames_used=2,
                height=4,
                width=4,
            )
        with patch.object(stage2, "FULL_DEDUP_MAX_DECODED_MASK_BYTES", 20), self.assertRaisesRegex(
            RuntimeError, "memory bound"
        ):
            stage2._full_dedup_tracks_from_masks(
                [("chair", document)],
                frames_used=2,
                height=4,
                width=4,
            )


class FullDedupFinalizeRunnerTest(unittest.TestCase):
    def test_finalizer_is_weights_free_and_emits_no_geometry_artifact(self):
        document = _document([
            [{"track_id": 3, "rle": "0 16"}],
            [{"track_id": 3, "rle": "0 16"}],
        ])
        payload, error = stage2._validate_full_dedup_finalize_request(
            _request(documents=[document])
        )
        self.assertIsNone(error)
        encoded = json.dumps(document, separators=(",", ":")).encode()
        frames = 2
        height = width = 4
        colors = np.zeros((frames, height, width, 3), dtype=np.uint8)
        colors[..., 1] = 180
        depth = np.ones((frames, height, width), dtype=np.float32)
        confidence = np.ones_like(depth)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :3, :], frames, axis=0)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], frames, axis=0)
        grid_y, grid_x = np.mgrid[:height, :width]
        points = np.zeros((frames, height, width, 3), dtype=np.float32)
        points[..., 0] = grid_x
        points[..., 1] = grid_y
        points[..., 2] = 1

        src_package = types.ModuleType("src")
        src_package.__path__ = []
        geometry_module = types.ModuleType("src.geometry_utils")
        geometry_module.compute_surface_area_from_pointmap = (
            lambda _pointmap, mask, **_kwargs: float(np.count_nonzero(mask))
        )
        dedup_module = types.ModuleType("src.sg_deduplication")
        dedup_module.self_category_deduplicate = lambda tracks, *_args: tracks
        dedup_module.cross_category_deduplicate = lambda grouped, *_args: grouped
        utils_module = types.ModuleType("src.utils")
        utils_module.vis_instance_masks = (
            lambda _colors, _masks, path: Path(path).write_bytes(b"mask-video")
        )
        vggt_package = types.ModuleType("vggt")
        vggt_package.__path__ = []
        vggt_utils = types.ModuleType("vggt.utils")
        vggt_utils.__path__ = []
        vggt_geometry = types.ModuleType("vggt.utils.geometry")
        vggt_geometry.unproject_depth_map_to_point_map = lambda *_args: points
        runtime_modules = {
            "src": src_package,
            "src.geometry_utils": geometry_module,
            "src.sg_deduplication": dedup_module,
            "src.utils": utils_module,
            "vggt": vggt_package,
            "vggt.utils": vggt_utils,
            "vggt.utils.geometry": vggt_geometry,
        }

        def download(_url, destination, _limit):
            if destination.name.startswith("masks-"):
                destination.write_bytes(encoded)
            else:
                destination.write_bytes(b"fixture")
            return destination.stat().st_size

        captured_files = []

        def manifest(out_dir, _work, request):
            captured_files.extend(sorted(path.name for path in out_dir.iterdir()))
            return {
                "durable": True,
                "complete": True,
                "files": list(stage2.FULL_DEDUP_FINAL_ARTIFACTS),
                "required_files": list(stage2._required_artifacts(request)),
                "receipts": [],
                "errors": [],
            }

        with patch.dict(sys.modules, runtime_modules), patch.object(
            stage2, "_download_gateway_artifact", side_effect=download
        ), patch.object(stage2, "_ensure_ras_installed") as ensure, patch.object(
            stage2, "_ensure_ras_on_path", return_value=Path("/verified/ras")
        ), patch.object(
            stage2, "_verified_checkout_revision", return_value=stage2.DEFAULT_VGGT_REVISION
        ), patch.object(
            stage2,
            "_load_full_dedup_geometry_inputs",
            return_value=(
                depth,
                confidence,
                extrinsics,
                intrinsics,
                [10, 20],
                [0.0, 1.0],
                2.0,
            ),
        ), patch.object(
            stage2, "_load_sampled_clip_colors", return_value=colors
        ), patch.object(
            stage2, "_normalize_mask_video", return_value={"codec": "h264"}
        ), patch.object(stage2, "_artifact_manifest", side_effect=manifest):
            result = stage2.run_full_dedup_finalize(payload)

        self.assertEqual(result["status"], "ok", result)
        ensure.assert_called_once_with(ensure_weights=False)
        self.assertEqual(captured_files, [
            "instance_masks.mp4",
            "object_catalog.json",
            "object_crops.jpg",
        ])
        self.assertNotIn("point_cloud.glb", captured_files)
        self.assertEqual(result["artifacts"]["required_files"], list(stage2.FULL_DEDUP_FINAL_ARTIFACTS))
        self.assertEqual(result["sam"]["model_id"], "fal-ai/sam-3/video-rle-objects")
        self.assertEqual(result["sam"]["requests"], [{
            "category": "chair",
            "request_id": "fal_req_000000",
        }])
        self.assertIs(result["geometry"]["room_alignment_applied"], False)
        self.assertEqual(result["geometry"]["point_cloud_artifact_owner"], "geometry_leg")

        checked = stage2._attach_success_provenance(
            result,
            mode="full",
            analysis_type=stage2.FULL_DEDUP_FINALIZE_ANALYSIS_TYPE,
            object_catalog_version=stage2.OBJECT_CATALOG_VERSION,
        )
        self.assertEqual(checked["status"], "ok", checked)
        with_glb = {
            **result,
            "artifacts": {
                **result["artifacts"],
                "files": [*result["artifacts"]["files"], "point_cloud.glb"],
            },
        }
        rejected = stage2._attach_success_provenance(
            with_glb,
            mode="full",
            analysis_type=stage2.FULL_DEDUP_FINALIZE_ANALYSIS_TYPE,
            object_catalog_version=stage2.OBJECT_CATALOG_VERSION,
        )
        self.assertEqual(rejected["status"], "error")

    def test_digest_mismatch_fails_before_runtime_bootstrap(self):
        payload, error = stage2._validate_full_dedup_finalize_request(_request())
        self.assertIsNone(error)

        def download(_url, destination, _limit):
            destination.write_bytes(b"wrong")
            return destination.stat().st_size

        with patch.object(
            stage2, "_download_gateway_artifact", side_effect=download
        ), patch.object(stage2, "_ensure_ras_installed") as ensure:
            result = stage2.run_full_dedup_finalize(payload)

        self.assertEqual(result["status"], "error")
        self.assertIn("size", result["error"])
        ensure.assert_not_called()


class RetiredLocalSamRouteTest(unittest.TestCase):
    def test_finalizer_source_has_no_local_model_or_weight_entrypoint(self):
        source = inspect.getsource(stage2.run_full_dedup_finalize)
        for forbidden in (
            "src.models",
            "load_sam3",
            "HF_TOKEN",
            "huggingface_hub",
            "_download_vggt_weights",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn("_ensure_ras_installed(ensure_weights=False)", source)

    def test_old_full_and_scene_parse_profiles_fail_before_their_runners(self):
        for analysis_type in (
            "dedup_ras_vggt_sam3",
            stage2.SCENE_PARSE_CATALOG_ANALYSIS_TYPE,
        ):
            with self.subTest(analysis_type=analysis_type), patch.object(
                stage2, "run_stage2_full"
            ) as full, patch.object(stage2, "run_scene_parse_catalog") as scene_parse:
                result = stage2.run_stage2({
                    "analysis_type": analysis_type,
                    "mode": "full",
                })
            self.assertEqual(result["status"], "error")
            full.assert_not_called()
            scene_parse.assert_not_called()
