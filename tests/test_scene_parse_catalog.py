import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scene_parse_catalog as catalog
import stage2_service as stage2


def _request(**overrides):
    payload = {
        "analysis_type": stage2.SCENE_PARSE_CATALOG_ANALYSIS_TYPE,
        "mode": "full",
        "video_url": "https://objects.example/source.mp4?signature=redacted",
        "source_sha256": "a" * 64,
        "source_size_bytes": 1024,
        "category_prompts": ["chair", "table"],
    }
    payload.update(overrides)
    return payload


def _mask(height=10, width=10, *, x=1, y=1, box_width=8, box_height=8):
    value = np.zeros((height, width), dtype=bool)
    value[y : y + box_height, x : x + box_width] = True
    return value


def _track(frame_ids, mask=None):
    value = _mask() if mask is None else mask
    return [{"frame_id": frame_id, "mask": value.copy()} for frame_id in frame_ids]


class SceneParseCatalogContractTest(unittest.TestCase):
    def test_closed_request_accepts_only_the_production_contract(self):
        normalized, error = stage2._validate_scene_parse_catalog_request(_request())

        self.assertIsNone(error)
        self.assertEqual(normalized, _request())
        forbidden = (
            ("max_frames", 24),
            ("room_align", False),
            ("upload", {"ticket": "secret"}),
            ("object_catalog_version", 1),
            ("video_b64", "fixture"),
            ("categories", ["chair"]),
            ("expected_sam_model_id", stage2.SAM3_MODEL_ID),
        )
        for field, value in forbidden:
            with self.subTest(field=field):
                normalized, error = stage2._validate_scene_parse_catalog_request(
                    _request(**{field: value})
                )
                self.assertIsNone(normalized)
                self.assertIn(field, error)

    def test_request_rejects_untyped_or_duplicate_category_prompts(self):
        bad_prompts = (
            [],
            ["chair"] * 33,
            ["chair", "Chair"],
            [" chair"],
            ["chair\n"],
            ["x" * 65],
            [1],
        )
        for prompts in bad_prompts:
            with self.subTest(prompts=prompts):
                normalized, error = stage2._validate_scene_parse_catalog_request(
                    _request(category_prompts=prompts)
                )
                self.assertIsNone(normalized)
                self.assertIsInstance(error, str)

    def test_request_requires_exact_source_identity_and_optional_size(self):
        bad = (
            {"source_sha256": "A" * 64},
            {"source_sha256": "a" * 63},
            {"source_size_bytes": True},
            {"source_size_bytes": 0},
            {"source_size_bytes": stage2.SCENE_PARSE_MAX_REMOTE_VIDEO_BYTES + 1},
        )
        for change in bad:
            with self.subTest(change=change):
                normalized, error = stage2._validate_scene_parse_catalog_request(
                    _request(**change)
                )
                self.assertIsNone(normalized)
                self.assertIsInstance(error, str)

    def test_typed_router_does_not_enter_legacy_category_or_frame_coercion(self):
        runner_result = {"status": "error", "mode": "full", "error_code": "fixture", "error": "fixture"}
        with patch.object(stage2, "run_scene_parse_catalog", return_value=runner_result) as runner:
            result = stage2.run_stage2(_request())

        self.assertEqual(result["analysis_type"], stage2.SCENE_PARSE_CATALOG_ANALYSIS_TYPE)
        runner.assert_called_once_with(_request())

    def test_caller_sampling_and_artifact_knobs_fail_before_runner(self):
        for field, value in (
            ("max_frames", 96),
            ("room_align", False),
            ("upload", {"ticket": "fixture"}),
            ("object_catalog_version", 1),
        ):
            with self.subTest(field=field), patch.object(
                stage2, "run_scene_parse_catalog"
            ) as runner:
                result = stage2.run_stage2(_request(**{field: value}))
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_code"], "invalid_scene_parse_request")
            runner.assert_not_called()

    def test_sampling_policy_is_internal_adaptive_and_bounded(self):
        self.assertEqual(stage2._scene_parse_requested_frames(0.5, 1), 24)
        self.assertEqual(stage2._scene_parse_requested_frames(12, 1), 24)
        self.assertEqual(stage2._scene_parse_requested_frames(12.1, 1), 25)
        self.assertEqual(stage2._scene_parse_requested_frames(60, 1), 96)
        self.assertEqual(stage2._scene_parse_requested_frames(60, 8), 96)
        self.assertEqual(stage2._scene_parse_requested_frames(60, 16), 48)
        self.assertEqual(stage2._scene_parse_requested_frames(60, 32), 24)
        with self.assertRaisesRegex(stage2.SceneParseCatalogError, "60 second"):
            stage2._scene_parse_requested_frames(60.001, 1)

    def test_scene_parse_download_uses_2_gib_profile_limit_and_checks_identity(self):
        payload_bytes = b"video-fixture"
        payload = _request(
            source_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            source_size_bytes=len(payload_bytes),
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "downloaded.mp4"
            destination.write_bytes(payload_bytes)
            with patch.object(stage2, "_download_video", return_value=destination) as download:
                path, size = stage2._materialize_scene_parse_video(payload, Path(tmp))

        self.assertEqual(path, destination)
        self.assertEqual(size, len(payload_bytes))
        self.assertEqual(
            download.call_args.kwargs["max_bytes"],
            stage2.SCENE_PARSE_MAX_REMOTE_VIDEO_BYTES,
        )
        self.assertEqual(download.call_args.kwargs["expected_size_bytes"], len(payload_bytes))
        self.assertEqual(stage2.MAX_REMOTE_VIDEO_BYTES, 64 * 1024 * 1024)

    def test_scene_parse_source_mismatch_fails_before_model_bootstrap(self):
        with patch.object(
            stage2,
            "_materialize_scene_parse_video",
            side_effect=stage2.SceneParseCatalogError(
                "source_sha256_mismatch", "Downloaded video SHA-256 does not match."
            ),
        ), patch.object(stage2, "_ensure_ras_installed") as ensure:
            result = stage2.run_scene_parse_catalog(_request())

        self.assertEqual(result["error_code"], "source_sha256_mismatch")
        self.assertEqual(result["failed_stage"], "source_download")
        ensure.assert_not_called()

    def test_declared_source_size_requires_content_length(self):
        class DownloadResponse:
            status_code = 200
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"fixture"

        with tempfile.TemporaryDirectory() as tmp, patch(
            "requests.get", return_value=DownloadResponse()
        ), self.assertRaisesRegex(RuntimeError, "must include Content-Length"):
            stage2._download_video(
                "https://objects.example/source.mp4",
                Path(tmp),
                expected_size_bytes=7,
            )

    def test_cached_manifest_never_replaces_checkpoint_digest_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint = directory / "fixture.bin"
            checkpoint.write_bytes(b"trusted")
            expected_sha = hashlib.sha256(b"trusted").hexdigest()
            stage2._VERIFIED_MODEL_FILE_CACHE.clear()
            stage2._write_pinned_model_manifest(
                directory,
                model_id="fixture/model",
                revision="1" * 40,
                checkpoint_name=checkpoint.name,
                checkpoint_sha256=expected_sha,
                checkpoint_bytes=7,
            )
            self.assertTrue(
                stage2._pinned_model_ready(
                    directory,
                    model_id="fixture/model",
                    revision="1" * 40,
                    checkpoint_name=checkpoint.name,
                    checkpoint_sha256=expected_sha,
                    checkpoint_bytes=7,
                )
            )
            checkpoint.write_bytes(b"altered")
            stage2._VERIFIED_MODEL_FILE_CACHE.clear()
            self.assertFalse(
                stage2._pinned_model_ready(
                    directory,
                    model_id="fixture/model",
                    revision="1" * 40,
                    checkpoint_name=checkpoint.name,
                    checkpoint_sha256=expected_sha,
                    checkpoint_bytes=7,
                )
            )

    def test_source_plan_and_decoder_are_bounded_before_model_preprocessing(self):
        entries = [
            {
                "best_effort_timestamp_time": str(index / 10),
                "duration_time": "0.1",
            }
            for index in range(10)
        ]
        with patch.object(
            stage2,
            "_ffprobe_json",
            return_value={"frames": entries},
        ) as probe, patch.object(
            stage2,
            "_probe_video_duration",
            side_effect=AssertionError("container duration must not override decoded duration"),
        ):
            timeline, duration = stage2._scene_parse_decoded_timeline(
                Path("/source.mp4"),
                decoded_frame_count=10,
            )
        indices, timestamps = stage2._scene_parse_uniform_frame_plan(timeline, 4)
        self.assertEqual(indices, [0, 3, 6, 9])
        self.assertEqual(timestamps, [0.0, 0.3, 0.6, 0.9])
        self.assertEqual(duration, 1.0)
        self.assertIn("-max_pixels", probe.call_args.args)

        loaded_paths = []
        vggt = types.ModuleType("vggt")
        vggt.__path__ = []
        utils = types.ModuleType("vggt.utils")
        utils.__path__ = []
        load_fn = types.ModuleType("vggt.utils.load_fn")
        load_fn.load_and_preprocess_images = lambda paths: loaded_paths.extend(paths) or "tensor"

        def fake_ffmpeg(command, **_kwargs):
            output_dir = Path(command[-1]).parent
            for index in range(1, 5):
                (output_dir / f"frame-{index:06d}.jpg").write_bytes(b"jpg")
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules,
            {
                "vggt": vggt,
                "vggt.utils": utils,
                "vggt.utils.load_fn": load_fn,
            },
        ), patch("subprocess.run", side_effect=fake_ffmpeg) as run:
            result = stage2._load_scene_parse_sampled_frames(
                Path(tmp) / "source.mp4",
                Path(tmp),
                indices,
            )

        self.assertEqual(result, "tensor")
        self.assertEqual(len(loaded_paths), 4)
        command = run.call_args.args[0]
        video_filter = command[command.index("-vf") + 1]
        self.assertEqual(
            command[command.index("-max_pixels") + 1],
            str(stage2.SCENE_PARSE_MAX_SOURCE_PIXELS),
        )
        self.assertIn("select=eq(n\\,0)+eq(n\\,3)+eq(n\\,6)+eq(n\\,9)", video_filter)
        self.assertIn("scale=1036:1036", video_filter)

    def test_decoded_stream_rejects_frame_and_pixel_expansion(self):
        def completed(width, height, frames):
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "streams": [
                            {
                                "width": width,
                                "height": height,
                                "nb_read_frames": str(frames),
                            }
                        ]
                    }
                ),
            )

        with patch(
            "subprocess.run",
            return_value=completed(1920, 1080, stage2.SCENE_PARSE_MAX_DECODED_FRAMES + 1),
        ), self.assertRaisesRegex(stage2.SceneParseCatalogError, "decoded-frame count"):
            stage2._probe_scene_parse_decoded_stream(Path("/source.mp4"))
        with patch(
            "subprocess.run",
            return_value=completed(8192, 8192, 24),
        ), self.assertRaisesRegex(stage2.SceneParseCatalogError, "dimensions"):
            stage2._probe_scene_parse_decoded_stream(Path("/source.mp4"))
        with patch(
            "subprocess.run",
            return_value=completed(1920, 1080, 24),
        ) as run:
            self.assertEqual(
                stage2._probe_scene_parse_decoded_stream(Path("/source.mp4")),
                (24, 1920, 1080),
            )
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(
                command[command.index("-max_pixels") + 1],
                str(stage2.SCENE_PARSE_MAX_SOURCE_PIXELS),
            )

    def test_sam31_configuration_fails_early_when_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            stage2.os.environ,
            {
                "STAGE2_SCENE_PARSE_SAM_BACKEND": "sam3.1_multiplex",
                "STAGE2_MODELS_DIR": tmp,
            },
        ):
            with self.assertRaisesRegex(stage2.SceneParseCatalogError, "checkpoint is missing"):
                stage2._preflight_scene_parse_sam_backend()

    def test_model_loader_seam_uses_explicit_checkpoint_and_builder(self):
        calls = []
        sam3 = types.ModuleType("sam3")
        sam3.__path__ = []
        builder = types.ModuleType("sam3.model_builder")
        builder.build_sam3_video_predictor = lambda **kwargs: calls.append(
            ("sam3", kwargs)
        ) or "sam3-model"
        builder.build_sam3_multiplex_video_predictor = lambda **kwargs: calls.append(
            ("sam3.1", kwargs)
        ) or "sam31-model"
        modules = {"sam3": sam3, "sam3.model_builder": builder}
        with patch.dict(sys.modules, modules), patch.object(
            stage2,
            "_scene_parse_sam_checkpoint",
            side_effect=lambda _ras, backend: Path(f"/models/{backend}.pt"),
        ):
            baseline = stage2._load_scene_parse_sam_video_model(Path("/ras"), "sam3")
            multiplex = stage2._load_scene_parse_sam_video_model(
                Path("/ras"), "sam3.1_multiplex"
            )

        self.assertEqual(baseline, "sam3-model")
        self.assertEqual(calls[0], ("sam3", {"checkpoint_path": "/models/sam3.pt"}))
        self.assertEqual(multiplex, "sam31-model")
        self.assertEqual(calls[1][0], "sam3.1")
        self.assertEqual(
            calls[1][1]["checkpoint_path"],
            "/models/sam3.1_multiplex.pt",
        )
        self.assertEqual(
            calls[1][1]["max_num_objects"],
            catalog.SCENE_PARSE_CATALOG_MAX_OBJECTS,
        )
        self.assertFalse(calls[1][1]["use_fa3"])


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
        self.assertEqual(chair["evidence"][1]["bbox"], {
            "x": 0.1,
            "y": 0.1,
            "width": 0.8,
            "height": 0.8,
        })
        self.assertEqual(chair["evidence"][1]["quality_score"], 1.0)
        self.assertTrue(
            all("mask" not in evidence for item in first["objects"] for evidence in item["evidence"])
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
        self.assertEqual(len(recovered["chair"]), 1)
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


class SceneParseCatalogRunnerTest(unittest.TestCase):
    def test_mocked_pipeline_returns_only_inline_catalog_and_true_provenance(self):
        class Frames:
            shape = (3, 10, 10, 3)

            def to(self, _device):
                return self

        class Model:
            def to(self, _device):
                return self

        class SamVideo:
            def handle_request(self, request):
                self.request = request
                return {"session_id": "session"}

        track = _track([0, 1, 2])
        short_track = _track([0], _mask(x=0, y=0, box_width=1, box_height=1))
        prediction = {
            "colors": np.zeros((3, 10, 10, 3), dtype=np.uint8),
            "world_points": np.zeros((3, 10, 10, 3), dtype=np.float32),
            "world_points_conf": np.ones((3, 10, 10), dtype=np.float32),
        }
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_RGB2BGR = 1
        cv2.cvtColor = lambda image, _code: image
        def imwrite(path, _image):
            Path(path).write_bytes(b"jpg")
            return True
        cv2.imwrite = imwrite
        src = types.ModuleType("src")
        src.__path__ = []
        models = types.ModuleType("src.models")
        models.unload_model = lambda _model: None
        segmentation = types.ModuleType("src.object_segmentation")
        segmentation.segment_and_track = lambda _prompt, _model, _session: [
            track,
            short_track,
        ]
        dedup = types.ModuleType("src.sg_deduplication")
        dedup.self_category_deduplicate = lambda value, *_args: [value[0]]
        dedup.cross_category_deduplicate = lambda value, *_args: value
        dedup.get_overlap_ratio = lambda _left, _right: 0.0
        vggt_predict = types.ModuleType("src.vggt_predict")
        vggt_predict.vggt_predict = lambda _frames, _model: prediction
        modules = {
            "torch": torch,
            "cv2": cv2,
            "src": src,
            "src.models": models,
            "src.object_segmentation": segmentation,
            "src.sg_deduplication": dedup,
            "src.vggt_predict": vggt_predict,
        }
        sam_identity = stage2._scene_parse_model_identity("sam3")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), patch.object(
            stage2, "_preflight_scene_parse_sam_backend", return_value="sam3"
        ), patch.object(
            stage2,
            "_materialize_scene_parse_video",
            return_value=(Path(tmp) / "source.mp4", 1024),
        ), patch.object(
            stage2,
            "_probe_video_duration",
            side_effect=AssertionError("container duration must not drive Scene Parse"),
        ), patch.object(
            stage2, "_probe_scene_parse_decoded_stream", return_value=(3, 10, 10)
        ), patch.object(
            stage2,
            "_scene_parse_decoded_timeline",
            return_value=([0.0, 0.5, 1.0], 1.25),
        ), patch.object(
            stage2,
            "_scene_parse_uniform_frame_plan",
            return_value=([0, 12, 24], [0.0, 0.5, 1.0]),
        ), patch.object(
            stage2, "_ensure_ras_installed"
        ), patch.object(
            stage2, "_ensure_ras_on_path", return_value=Path(tmp) / "ras"
        ), patch.object(
            stage2,
            "_verified_checkout_revision",
            side_effect=[
                stage2.RAS_REVISION,
                stage2.DEFAULT_VGGT_REVISION,
                stage2.DEFAULT_SAM3_REVISION,
            ],
        ), patch.object(
            stage2, "_ensure_scene_parse_model_weights", return_value=sam_identity
        ), patch.object(
            stage2, "_load_scene_parse_sampled_frames", return_value=Frames()
        ), patch.object(
            stage2, "_load_scene_parse_vggt_model", return_value=Model()
        ), patch.object(
            stage2, "_load_scene_parse_sam_video_model", return_value=SamVideo()
        ):
            result = stage2.run_scene_parse_catalog(_request(category_prompts=["chair"]))

        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["schema"], catalog.SCENE_PARSE_CATALOG_SCHEMA)
        self.assertEqual(result["sampling"]["requested_frames"], 24)
        self.assertEqual(result["sampling"]["frames_used"], 3)
        self.assertEqual(result["counts"]["raw_tracks"], 2)
        self.assertEqual(result["counts"]["category_deduplicated_tracks"], 1)
        self.assertEqual(result["counts"]["deduplicated_objects"], 2)
        self.assertEqual(
            sorted(item["evidence_strength"] for item in result["objects"]),
            ["single_frame", "supported"],
        )
        self.assertEqual(
            result["sampling"]["policy"],
            stage2.SCENE_PARSE_SAMPLING_POLICY,
        )
        self.assertEqual(result["sampling"]["prompt_frame_budget"], 768)
        self.assertEqual(result["sampling"]["planned_prompt_frames"], 24)
        self.assertEqual(
            result["provenance"]["geometry"]["model_id"],
            stage2.COMMERCIAL_VGGT_MODEL_ID,
        )
        self.assertEqual(result["provenance"]["segmentation"]["model_id"], stage2.SAM3_MODEL_ID)
        for forbidden in ("artifacts", "geometry", "sam", "instances", "object_catalog"):
            self.assertNotIn(forbidden, result)
        verified = stage2._attach_success_provenance(
            result,
            mode="full",
            analysis_type=stage2.SCENE_PARSE_CATALOG_ANALYSIS_TYPE,
        )
        self.assertEqual(verified["status"], "ok", verified)
        self.assertEqual(
            verified["analysis_type"],
            stage2.SCENE_PARSE_CATALOG_ANALYSIS_TYPE,
        )


if __name__ == "__main__":
    unittest.main()
