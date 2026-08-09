import base64
import io
import json
import subprocess
import struct
import sys
import tempfile
import types
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests


import stage2_service as stage2
import artifact_upload as artifact_uploader
import handler as endpoint_handler

from point_cloud_glb import GLB_HARD_MAX_POINTS, GLB_MAX_BYTES, build_point_cloud_glb
from artifact_upload import upload_artifact_file


class Stage2ServiceTest(unittest.TestCase):
    def test_endpoint_response_reports_exact_code_revision(self):
        revision = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as tmp:
            revision_file = Path(tmp) / "revision"
            revision_file.write_text(revision)
            with patch.dict(endpoint_handler.os.environ, {
                "STAGE2_BUILD_REVISION_FILE": str(revision_file),
                "STAGE2_CODE_REV": "f" * 40,
            }), patch.object(
                endpoint_handler,
                "run_stage2",
                return_value={"status": "ok", "mode": "dry_run"},
            ):
                result = endpoint_handler.handler({"input": {}})

        self.assertEqual(result["stage2_code_revision"], revision)

    def test_endpoint_never_treats_an_environment_revision_as_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            endpoint_handler.os.environ,
            {
                "STAGE2_BUILD_REVISION_FILE": str(Path(tmp) / "missing"),
                "STAGE2_CODE_REV": "f" * 40,
            },
            clear=True,
        ), patch.object(
            endpoint_handler.subprocess,
            "check_output",
            side_effect=OSError("git unavailable"),
        ):
            self.assertEqual(endpoint_handler._runtime_code_revision(), "")

    def test_endpoint_git_fallback_requires_a_clean_tracked_checkout(self):
        revision = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            endpoint_handler.os.environ,
            {"STAGE2_BUILD_REVISION_FILE": str(Path(tmp) / "missing")},
            clear=True,
        ), patch.object(
            endpoint_handler.subprocess,
            "check_output",
            return_value=revision,
        ), patch.object(
            endpoint_handler.subprocess,
            "run",
            return_value=types.SimpleNamespace(returncode=1),
        ):
            self.assertEqual(endpoint_handler._runtime_code_revision(), "")

    def test_lazy_model_initialization_holds_cross_worker_lock(self):
        events = []

        @contextmanager
        def locked(_models_dir):
            events.append("lock_enter")
            yield
            events.append("lock_exit")

        def cloned(_url, dest, _revision):
            events.append(f"clone:{Path(dest).name}")

        with patch.object(stage2, "_ras_root", return_value=Path("/tmp/ras")), patch.object(
            stage2, "_models_dir", return_value=Path("/tmp/models")
        ), patch.object(stage2, "_stage2_initialization_lock", side_effect=locked), patch.object(
            stage2, "_clone_if_missing", side_effect=cloned
        ), patch.object(
            stage2, "_ensure_python_packages", side_effect=lambda *_args, **_kwargs: events.append("packages")
        ), patch.object(stage2, "_vggt_weights_ok", return_value=True):
            stage2._ensure_ras_installed(require_sam3=False)

        self.assertEqual(events[0], "lock_enter")
        self.assertEqual(events[-1], "lock_exit")
        self.assertEqual(events[1:-1], ["clone:ras", "clone:vggt", "packages"])

    def test_concurrent_first_jobs_create_one_models_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ras = root / "ras"
            models = root / "shared-models"
            ras.mkdir()
            with patch.dict(stage2.os.environ, {"STAGE2_MODELS_DIR": str(models)}):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda _index: stage2._link_models_dir(ras), range(2)))

            self.assertTrue((ras / "models").is_symlink())
            self.assertEqual((ras / "models").resolve(), models.resolve())

    def test_dirty_pinned_checkout_is_recreated_instead_of_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "checkout"
            checkout.mkdir()
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
            tracked = checkout / "tracked.txt"
            tracked.write_text("clean\n")
            subprocess.run(["git", "-C", str(checkout), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "fixture"], check=True)
            revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            tracked.write_text("dirty\n")

            commands = []
            with patch.object(stage2, "_run", side_effect=lambda command, **_kwargs: commands.append(command)):
                stage2._clone_if_missing("https://example.test/repo.git", checkout, revision)

            self.assertFalse(checkout.exists())
            self.assertEqual(commands[0][:2], ["git", "clone"])

    def test_source_indices_match_upstream_uniform_sampling(self):
        frames = {
            "frames": [
                {"best_effort_timestamp_time": f"{index / 30:.9f}"}
                for index in range(327)
            ]
        }
        with patch.object(stage2, "_ffprobe_json", return_value=frames):
            indices = stage2._sampled_source_frame_indices(Path("clip.mp4"), 8)
        self.assertEqual(indices, [0, 46, 93, 139, 186, 232, 279, 326])

    def test_source_frame_plan_uses_one_decoded_list_for_indices_and_pts(self):
        frames = {
            "frames": [
                {"best_effort_timestamp_time": value}
                for value in ("0.000", "0.100", "0.900", "1.200", "2.400")
            ]
        }
        with patch.object(stage2, "_ffprobe_json", return_value=frames) as probe:
            plan = stage2._source_frame_plan(Path("variable-frame-rate.mp4"), 4)

        self.assertEqual(plan, ([0, 1, 2, 4], [0.0, 0.1, 0.9, 2.4]))
        probe.assert_called_once()

    def test_rejects_unknown_mode_before_model_bootstrap(self):
        result = stage2.run_stage2(
            {"mode": "typo", "video_b64": "eA==", "categories": ["chair"], "max_frames": 8}
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("dry_run, geometry, or full", result["error"])

    def test_rejects_category_and_frame_boundaries(self):
        result = stage2.run_stage2(
            {"mode": "dry_run", "video_b64": "eA==", "categories": ["x" * 65], "max_frames": 8}
        )
        self.assertEqual(result["status"], "error")
        result = stage2.run_stage2(
            {"mode": "dry_run", "video_b64": "eA==", "categories": ["chair"], "max_frames": 161}
        )
        self.assertEqual(result["status"], "error")

    def test_inline_video_validation(self):
        self.assertEqual(stage2.MAX_INLINE_VIDEO_BYTES, 6 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            stage2, "MAX_INLINE_VIDEO_BYTES", 4
        ):
            path = stage2._materialize_video(
                {"video_b64": base64.b64encode(b"data").decode(), "media_type": "video/webm"},
                Path(tmp),
            )
            self.assertEqual(path.name, "input.webm")
            self.assertEqual(path.read_bytes(), b"data")
            with self.assertRaisesRegex(RuntimeError, "valid base64"):
                stage2._materialize_video({"video_b64": "***"}, Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "inline video exceeds"):
                stage2._materialize_video(
                    {"video_b64": base64.b64encode(b"12345").decode()}, Path(tmp)
                )

    def test_remote_video_download_is_streamed_and_size_bounded(self):
        class DownloadResponse:
            status_code = 200

            def __init__(self, chunks, content_length=None):
                self._chunks = chunks
                self.headers = {} if content_length is None else {"content-length": str(content_length)}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                self.test_chunk_size = chunk_size
                yield from self._chunks

        self.assertEqual(stage2.MAX_REMOTE_VIDEO_BYTES, 64 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            stage2, "MAX_REMOTE_VIDEO_BYTES", 4
        ):
            root = Path(tmp)
            within_limit = DownloadResponse([b"ab", b"cd"], 4)
            with patch.dict(stage2.os.environ, {
                "CF_ACCESS_CLIENT_ID": "must-not-leave-worker",
                "CF_ACCESS_CLIENT_SECRET": "must-not-leave-worker",
            }), patch("requests.get", return_value=within_limit) as get:
                path = stage2._download_video("https://input.example/clip.mp4?token=secret", root / "exact")
            self.assertEqual(path.read_bytes(), b"abcd")
            self.assertEqual(within_limit.test_chunk_size, 1 << 20)
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
            self.assertNotIn("headers", get.call_args.kwargs)

            declared_oversize = DownloadResponse([], 5)
            with patch("requests.get", return_value=declared_oversize), self.assertRaisesRegex(
                RuntimeError, "exceeds the .* service limit"
            ):
                stage2._download_video("https://input.example/declared.mp4", root / "declared")
            self.assertFalse(hasattr(declared_oversize, "test_chunk_size"))

            streamed_oversize = DownloadResponse([b"abc", b"de"])
            oversize_dir = root / "streamed"
            with patch("requests.get", return_value=streamed_oversize), self.assertRaisesRegex(
                RuntimeError, "exceeds the .* service limit"
            ):
                stage2._download_video("https://input.example/streamed.mp4", oversize_dir)
            self.assertFalse(
                (oversize_dir / "input.mp4").exists(),
                "a rejected partial download must be removed",
            )

            empty = DownloadResponse([])
            empty_dir = root / "empty"
            with patch("requests.get", return_value=empty), self.assertRaisesRegex(
                RuntimeError, "empty file"
            ):
                stage2._download_video("https://input.example/empty.mp4", empty_dir)
            self.assertFalse((empty_dir / "input.mp4").exists())

            redirected = DownloadResponse([])
            redirected.status_code = 307
            redirect_dir = root / "redirect"
            with patch("requests.get", return_value=redirected) as get, self.assertRaisesRegex(
                RuntimeError, "redirects are not allowed"
            ):
                stage2._download_video("https://input.example/redirect.mp4", redirect_dir)
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
            self.assertFalse((redirect_dir / "input.mp4").exists())

            for unsafe_url in (
                "http://input.example/plaintext.mp4",
                "https://user:password@input.example/credentials.mp4",
            ):
                with patch("requests.get") as get, self.assertRaisesRegex(
                    RuntimeError, "credential-free HTTPS URL"
                ):
                    stage2._download_video(unsafe_url, root / "unsafe")
                get.assert_not_called()

    def test_remote_video_download_does_not_leak_signed_query_in_errors(self):
        class FailedResponse:
            status_code = 500
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                raise requests.HTTPError(
                    "500 error for https://input.example/clip.mp4?token=do-not-leak"
                )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "requests.get", return_value=FailedResponse()
        ), self.assertRaisesRegex(RuntimeError, r"video download failed \(HTTPError\)") as caught:
            stage2._download_video(
                "https://input.example/clip.mp4?token=do-not-leak", Path(tmp)
            )

        self.assertNotIn("do-not-leak", str(caught.exception))

    def test_dry_run_never_claims_synthetic_instances(self):
        # Invalid video is enough to prove errors stay explicit; the real video
        # decode path is covered by the provider-backed smoke test.
        result = stage2.run_stage2(
            {
                "mode": "dry_run",
                "video_b64": base64.b64encode(b"not-a-video").decode(),
                "categories": ["chair"],
                "max_frames": 8,
            }
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "could not decode downloaded video")

    def test_ephemeral_geometry_skips_unusable_artifact_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            with patch.dict(
                stage2.os.environ,
                {"STAGE2_ARTIFACT_DIR": "", "STAGE2_KEEP_WORK": "0"},
            ):
                self.assertFalse(stage2._artifact_exports_enabled())
                manifest = stage2._artifact_manifest(out_dir, work)
        self.assertEqual(manifest["files"], [])
        self.assertIn("export was skipped", manifest["note"])

    def test_upload_ticket_enables_export_and_manifest_returns_receipts_only(self):
        future = 9_999_999_999_999
        upload = {"base": "https://edge.example", "runId": "stage2-a", "token": "t", "exp": future}
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            (out_dir / "point_cloud.glb").write_bytes(b"glb")
            (out_dir / "debug.tmp").write_bytes(b"private")

            def uploaded(_ticket, path, media_type):
                name = Path(path).name
                return {
                    "key": f"runs/stage2-a/hash-{name}",
                    "name": name,
                    "sha256": "a" * 64,
                    "md5": "b" * 32,
                    "bytes": Path(path).stat().st_size,
                    "mediaType": media_type,
                }

            with patch("artifact_upload.upload_artifact_file", side_effect=uploaded):
                self.assertTrue(stage2._artifact_exports_enabled({"upload": upload}))
                payload = {"mode": "geometry", "upload": upload}
                manifest = stage2._artifact_manifest(out_dir, work, payload)

        self.assertTrue(manifest["durable"])
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["required_files"], ["point_cloud.glb"])
        self.assertEqual(manifest["missing_required"], [])
        self.assertEqual(manifest["delivery"], "agent-lab-r2")
        self.assertEqual(manifest["files"], ["point_cloud.glb"])
        self.assertEqual(manifest["receipts"][0]["mediaType"], "model/gltf-binary")
        self.assertNotIn(str(work), json.dumps(manifest))
        self.assertNotIn("edge.example", json.dumps(manifest))

    def test_artifact_upload_failure_is_structured_without_signed_url_leak(self):
        upload = {"base": "https://edge.example", "runId": "stage2-a", "token": "t", "exp": 9_999_999_999_999}
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            (out_dir / "point_cloud.glb").write_bytes(b"glb")
            with patch(
                "artifact_upload.upload_artifact_file",
                side_effect=RuntimeError("https://signed.example/?X-Amz-Signature=secret"),
            ):
                payload = {"mode": "geometry", "upload": upload}
                manifest = stage2._artifact_manifest(out_dir, work, payload)

        self.assertFalse(manifest["durable"])
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["missing_required"], ["point_cloud.glb"])
        self.assertEqual(manifest["receipts"], [])
        self.assertEqual(manifest["errors"][0]["code"], "artifact_upload_failed")
        failure = stage2._artifact_delivery_error(payload, manifest)
        self.assertEqual(failure["error_code"], "artifact_delivery_failed")
        self.assertEqual(
            failure["error"],
            "Required result files could not be delivered to durable storage.",
        )
        self.assertNotIn("secret", json.dumps(manifest))
        self.assertNotIn("signed.example", json.dumps(manifest))

    def test_artifact_upload_failure_preserves_safe_phase_status_and_log(self):
        upload = {"base": "https://edge.example", "runId": "stage2-a", "token": "t", "exp": 9_999_999_999_999}
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            (out_dir / "point_cloud.glb").write_bytes(b"glb")
            failure = artifact_uploader.ArtifactUploadError(
                "artifact_put",
                http_status=409,
                retryable=False,
                attempts=1,
                cause_type="HTTPError",
            )
            log = io.StringIO()
            with patch(
                "artifact_upload.upload_artifact_file",
                side_effect=failure,
            ), redirect_stdout(log):
                manifest = stage2._artifact_manifest(
                    out_dir,
                    work,
                    {"mode": "geometry", "upload": upload},
                )

        self.assertEqual(manifest["errors"], [{
            "name": "point_cloud.glb",
            "code": "artifact_upload_failed",
            "phase": "artifact_put",
            "retryable": False,
            "attempts": 1,
            "detail": "artifact storage write returned HTTP 409",
            "http_status": 409,
        }])
        logged = json.loads(log.getvalue())
        self.assertEqual(logged["phase"], "artifact_put")
        self.assertEqual(logged["http_status"], 409)
        self.assertNotIn("token", log.getvalue().lower())
        self.assertNotIn("edge.example", log.getvalue())

    def test_full_delivery_requires_both_glb_and_mask_video_receipts(self):
        upload = {"base": "https://edge.example", "runId": "stage2-a", "token": "t", "exp": 9_999_999_999_999}
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out_dir = work / "output"
            out_dir.mkdir()
            (out_dir / "point_cloud.glb").write_bytes(b"glb")

            def uploaded(_ticket, path, media_type):
                name = Path(path).name
                return {
                    "key": f"runs/stage2-a/hash-{name}",
                    "name": name,
                    "sha256": "a" * 64,
                    "md5": "b" * 32,
                    "bytes": Path(path).stat().st_size,
                    "mediaType": media_type,
                }

            payload = {"mode": "full", "upload": upload}
            with patch("artifact_upload.upload_artifact_file", side_effect=uploaded):
                manifest = stage2._artifact_manifest(out_dir, work, payload)

        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["missing_required"], ["instance_masks.mp4"])
        self.assertEqual(manifest["errors"][0]["code"], "artifact_generation_missing")
        self.assertEqual(
            stage2._artifact_delivery_error(payload, manifest)["error_code"],
            "artifact_delivery_failed",
        )

    def test_artifact_uploader_returns_receipt_without_grant_url_or_ticket(self):
        class PutResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b""

        upload = {"base": "https://edge.example", "runId": "stage2-a", "token": "secret-ticket", "exp": 9_999_999_999_999}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            policy = {
                "outputs": [
                    {"name": "point_cloud.glb", "mediaType": "model/gltf-binary", "maxBytes": 16 * 1024 * 1024}
                ]
            }
            upload["policy"] = policy
            with patch(
                "artifact_upload._post_json",
                return_value={
                    "v": "2",
                    "key": "runs/stage2-a/hash-point_cloud.glb",
                    "url": "https://signed.example/?X-Amz-Signature=secret",
                    "headers": {
                        "content-length": "3",
                        "content-md5": "safe-md5",
                        "content-type": "model/gltf-binary",
                        "x-amz-checksum-sha256": "safe-sha256",
                        "User-Agent": "caller-controlled",
                    },
                },
            ) as post_json, patch("artifact_upload.urllib.request.urlopen", return_value=PutResponse()) as put:
                receipt = upload_artifact_file(upload, path, "model/gltf-binary")

        self.assertEqual(post_json.call_args.args[1]["policy"], policy)
        self.assertEqual(receipt["name"], "point_cloud.glb")
        self.assertEqual(receipt["bytes"], 3)
        self.assertNotIn("url", receipt)
        self.assertNotIn("token", receipt)
        self.assertNotIn("secret", json.dumps(receipt))
        put_request = put.call_args.args[0]
        self.assertEqual(
            put_request.get_header("User-agent"),
            artifact_uploader.UPLOADER_USER_AGENT,
        )
        self.assertEqual(put_request.get_header("Content-length"), "3")
        self.assertEqual(put_request.get_header("Content-md5"), "safe-md5")
        self.assertEqual(put_request.get_header("Content-type"), "model/gltf-binary")
        self.assertEqual(put_request.get_header("X-amz-checksum-sha256"), "safe-sha256")

    def test_artifact_post_uses_stable_uploader_identity_and_rejects_override(self):
        class JsonResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with patch(
            "artifact_upload.urllib.request.urlopen",
            return_value=JsonResponse(b'{"status":"ok"}'),
        ) as request:
            result = artifact_uploader._post_json(
                "https://edge.example/api/jobs/upload-grant",
                {"safe": True},
                {"User-Agent": "caller-controlled", "cf-access-client-id": "configured"},
                30,
            )

        self.assertEqual(result, {"status": "ok"})
        sent = request.call_args.args[0]
        self.assertEqual(sent.get_header("User-agent"), artifact_uploader.UPLOADER_USER_AGENT)
        self.assertEqual(sent.get_header("Cf-access-client-id"), "configured")

    def test_artifact_uploader_refreshes_grant_before_each_put_retry(self):
        class PutResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b""

        grants = [
            {
                "v": "2",
                "key": "runs/stage2-a/hash-point_cloud.glb",
                "url": "https://signed.example/attempt-1",
                "headers": {"content-length": "3"},
            },
            {
                "v": "2",
                "key": "runs/stage2-a/hash-point_cloud.glb",
                "url": "https://signed.example/attempt-2",
                "headers": {"content-length": "3"},
            },
        ]
        put_urls = []

        def put(request, **_kwargs):
            put_urls.append(request.full_url)
            if len(put_urls) == 1:
                raise OSError("first PUT lost")
            return PutResponse()

        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", side_effect=grants) as post_json, patch(
                "artifact_upload._stored_already", return_value=False
            ), patch("artifact_upload.urllib.request.urlopen", side_effect=put), patch(
                "artifact_upload.time.sleep"
            ):
                receipt = upload_artifact_file(upload, path, "model/gltf-binary")

        self.assertEqual(
            [call.args[0] for call in post_json.call_args_list],
            [
                "https://edge.example/api/jobs/upload-grant",
                "https://edge.example/api/jobs/upload-grant",
            ],
        )
        self.assertEqual(
            put_urls,
            ["https://signed.example/attempt-1", "https://signed.example/attempt-2"],
        )
        self.assertEqual(receipt["key"], "runs/stage2-a/hash-point_cloud.glb")

    def test_artifact_uploader_retries_unavailable_verification_without_replaying_put(self):
        grant = {
            "v": "2",
            "key": "runs/stage2-a/hash-point_cloud.glb",
            "url": "https://signed.example/attempt-1",
            "headers": {"content-length": "3"},
        }
        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", return_value=grant) as post_json, patch(
                "artifact_upload._stored_already", side_effect=[None, True]
            ) as verify, patch(
                "artifact_upload.urllib.request.urlopen", side_effect=OSError("response lost")
            ) as put, patch("artifact_upload.time.sleep"):
                receipt = upload_artifact_file(upload, path, "model/gltf-binary")

        self.assertEqual(receipt["key"], grant["key"])
        self.assertEqual(post_json.call_count, 1)
        self.assertEqual(put.call_count, 1)
        self.assertEqual(verify.call_count, 2)

    def test_artifact_uploader_never_replays_put_when_verification_stays_unavailable(self):
        grant = {
            "v": "2",
            "key": "runs/stage2-a/hash-point_cloud.glb",
            "url": "https://signed.example/attempt-1",
            "headers": {"content-length": "3"},
        }
        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", return_value=grant) as post_json, patch(
                "artifact_upload._stored_already", return_value=None
            ) as verify, patch(
                "artifact_upload.urllib.request.urlopen", side_effect=OSError("response lost")
            ) as put, patch("artifact_upload.time.sleep"):
                with self.assertRaisesRegex(
                    artifact_uploader.ArtifactUploadError,
                    "artifact storage verification failed",
                ):
                    upload_artifact_file(upload, path, "model/gltf-binary")

        self.assertEqual(post_json.call_count, 1)
        self.assertEqual(put.call_count, 1)
        self.assertEqual(verify.call_count, 5)

    def test_artifact_uploader_verifies_but_does_not_retry_deterministic_put_rejection(self):
        grant = {
            "v": "2",
            "key": "runs/stage2-a/hash-point_cloud.glb",
            "url": "https://signed.example/?X-Amz-Signature=secret",
            "headers": {"content-length": "3"},
        }
        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        http_error = urllib.error.HTTPError(
            grant["url"],
            409,
            "store rejected upload",
            {},
            None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", return_value=grant) as post_json, patch(
                "artifact_upload._stored_already", return_value=False
            ) as verify, patch(
                "artifact_upload.urllib.request.urlopen", side_effect=http_error
            ) as put, patch("artifact_upload.time.sleep") as sleep:
                with self.assertRaises(artifact_uploader.ArtifactUploadError) as caught:
                    upload_artifact_file(upload, path, "model/gltf-binary")

        self.assertEqual(caught.exception.phase, "artifact_put")
        self.assertEqual(caught.exception.http_status, 409)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(post_json.call_count, 1)
        self.assertEqual(put.call_count, 1)
        verify.assert_called_once()
        sleep.assert_not_called()
        self.assertNotIn("signed.example", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_artifact_uploader_recovers_when_rejected_put_already_landed(self):
        grant = {
            "v": "2",
            "key": "runs/stage2-a/hash-point_cloud.glb",
            "url": "https://signed.example/attempt-1",
            "headers": {"content-length": "3"},
        }
        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        http_error = urllib.error.HTTPError(grant["url"], 403, "forbidden", {}, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", return_value=grant), patch(
                "artifact_upload._stored_already", return_value=True
            ) as verify, patch(
                "artifact_upload.urllib.request.urlopen", side_effect=http_error
            ) as put, patch("artifact_upload.time.sleep") as sleep:
                receipt = upload_artifact_file(upload, path, "model/gltf-binary")

        self.assertEqual(receipt["key"], grant["key"])
        self.assertEqual(put.call_count, 1)
        verify.assert_called_once()
        sleep.assert_not_called()

    def test_artifact_uploader_strict_mode_surfaces_put_rejection_even_when_object_exists(self):
        grant = {
            "v": "2",
            "key": "runs/stage2-a/hash-point_cloud.glb",
            "url": "https://signed.example/?X-Amz-Signature=secret",
            "headers": {"content-length": "3"},
        }
        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        http_error = urllib.error.HTTPError(grant["url"], 403, "forbidden", {}, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", return_value=grant), patch(
                "artifact_upload._stored_already", return_value=True
            ) as verify, patch(
                "artifact_upload.urllib.request.urlopen", side_effect=http_error
            ) as put, patch("artifact_upload.time.sleep") as sleep:
                with self.assertRaises(artifact_uploader.ArtifactUploadError) as caught:
                    upload_artifact_file(
                        upload,
                        path,
                        "model/gltf-binary",
                        require_put_acknowledgement=True,
                    )

        self.assertEqual(caught.exception.phase, "artifact_put")
        self.assertEqual(caught.exception.http_status, 403)
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(put.call_count, 1)
        verify.assert_not_called()
        sleep.assert_not_called()
        self.assertNotIn("signed.example", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_artifact_uploader_strict_mode_retries_transient_put_with_fresh_grant(self):
        grants = [
            {
                "v": "2",
                "key": "runs/stage2-a/hash-point_cloud.glb",
                "url": "https://signed.example/attempt-1",
                "headers": {"content-length": "3"},
            },
            {
                "v": "2",
                "key": "runs/stage2-a/hash-point_cloud.glb",
                "url": "https://signed.example/attempt-2",
                "headers": {"content-length": "3"},
            },
        ]

        class PutResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b""

        put_urls = []

        def put(request, **_kwargs):
            put_urls.append(request.full_url)
            if len(put_urls) == 1:
                raise OSError("transient PUT failure")
            return PutResponse()

        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", side_effect=grants) as post_json, patch(
                "artifact_upload._stored_already"
            ) as verify, patch(
                "artifact_upload.urllib.request.urlopen", side_effect=put
            ), patch("artifact_upload.time.sleep") as sleep:
                receipt = upload_artifact_file(
                    upload,
                    path,
                    "model/gltf-binary",
                    require_put_acknowledgement=True,
                )

        self.assertEqual(receipt["key"], grants[0]["key"])
        self.assertEqual(post_json.call_count, 2)
        self.assertEqual(
            put_urls,
            ["https://signed.example/attempt-1", "https://signed.example/attempt-2"],
        )
        verify.assert_not_called()
        sleep.assert_called_once_with(1)

    def test_artifact_uploader_does_not_retry_deterministic_grant_rejection(self):
        upload = {
            "base": "https://edge.example",
            "runId": "stage2-a",
            "token": "secret-ticket",
            "exp": 9_999_999_999_999,
        }
        grant_error = urllib.error.HTTPError(
            "https://edge.example/api/jobs/upload-grant?token=do-not-leak",
            403,
            "forbidden",
            {},
            None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_cloud.glb"
            path.write_bytes(b"glb")
            with patch("artifact_upload._post_json", side_effect=grant_error) as post_json, patch(
                "artifact_upload.time.sleep"
            ) as sleep:
                with self.assertRaises(artifact_uploader.ArtifactUploadError) as caught:
                    upload_artifact_file(upload, path, "model/gltf-binary")

        self.assertEqual(caught.exception.phase, "upload_grant")
        self.assertEqual(caught.exception.http_status, 403)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(post_json.call_count, 1)
        sleep.assert_not_called()
        self.assertNotIn("do-not-leak", str(caught.exception))

    def test_artifact_uploader_preserves_only_allowlisted_gateway_rejection_code(self):
        body = io.BytesIO(json.dumps({
            "error": "invalid or expired upload ticket",
            "error_code": "upload_ticket_signature_invalid",
            "token": "must-not-leak",
        }).encode())
        error = urllib.error.HTTPError(
            "https://edge.example/api/jobs/upload-grant?token=must-not-leak",
            403,
            "forbidden",
            {"content-type": "application/json"},
            body,
        )
        failure = artifact_uploader._upload_error("upload_grant", error)
        self.assertEqual(failure.gateway_error_code, "upload_ticket_signature_invalid")
        self.assertEqual(failure.failure_record("point_cloud.glb"), {
            "name": "point_cloud.glb",
            "code": "artifact_upload_failed",
            "phase": "upload_grant",
            "retryable": False,
            "attempts": 1,
            "detail": "artifact upload authorization returned HTTP 403",
            "http_status": 403,
            "gateway_error_code": "upload_ticket_signature_invalid",
        })
        self.assertNotIn("must-not-leak", json.dumps(failure.failure_record("point_cloud.glb")))

        untrusted = urllib.error.HTTPError(
            "https://edge.example/api/jobs/upload-grant",
            403,
            "forbidden",
            {"content-type": "application/json"},
            io.BytesIO(b'{"error_code":"attacker_controlled","detail":"must-not-leak"}'),
        )
        sanitized = artifact_uploader._upload_error("upload_grant", untrusted)
        self.assertIsNone(sanitized.gateway_error_code)
        self.assertNotIn("attacker", json.dumps(sanitized.failure_record("point_cloud.glb")))

        non_string = urllib.error.HTTPError(
            "https://edge.example/api/jobs/upload-grant",
            403,
            "forbidden",
            {"content-type": "application/json"},
            io.BytesIO(b'{"error_code":[]}'),
        )
        non_string_failure = artifact_uploader._upload_error("upload_grant", non_string)
        self.assertIsNone(non_string_failure.gateway_error_code)
        self.assertEqual(non_string_failure.http_status, 403)

        cloudflare = urllib.error.HTTPError(
            "https://edge.example/api/jobs/upload-grant",
            403,
            "forbidden",
            {"content-type": "text/plain; charset=UTF-8"},
            io.BytesIO(b"error code: 1010\n"),
        )
        edge_failure = artifact_uploader._upload_error("upload_grant", cloudflare)
        self.assertEqual(edge_failure.edge_error_code, "cloudflare_1010")
        self.assertEqual(
            edge_failure.failure_record("point_cloud.glb")["edge_error_code"],
            "cloudflare_1010",
        )

    def test_ambiguous_put_verification_stops_at_ticket_expiry(self):
        payload = {"exp": 100_000}
        with patch("artifact_upload.time.time", return_value=100.0), patch(
            "artifact_upload._stored_already"
        ) as verify, patch("artifact_upload.time.sleep") as sleep:
            verdict = artifact_uploader._resolve_ambiguous_put(
                {"base": "https://edge.example"}, payload, {}, 300
            )

        self.assertIsNone(verdict)
        verify.assert_not_called()
        sleep.assert_not_called()

    def test_debug_artifacts_are_opt_in_and_default_glb_cap_is_300k(self):
        class PointCloud:
            vertices = np.array(
                [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32
            )
            colors = np.array(
                [[10, 20, 30, 255], [40, 50, 60, 255]], dtype=np.uint8
            )

            def export(self, path):
                Path(path).write_bytes(b"ply")

        pred = {
            "world_points": np.zeros((1, 2, 2, 3), dtype=np.float32),
            "colors": np.ones((1, 2, 2, 3), dtype=np.uint8),
            "world_points_conf": np.ones((1, 2, 2), dtype=np.float32),
            "extrinsics": np.eye(4, dtype=np.float32)[None, ...],
            "intrinsic": np.eye(3, dtype=np.float32)[None, ...],
            "point_cloud_data": PointCloud(),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            stage2.os.environ,
            {"STAGE2_EXPORT_DEBUG_ARTIFACTS": ""},
            clear=True,
        ), patch("point_cloud_glb.write_point_cloud_glb", return_value={"point_count": 4}) as write_glb:
            out_dir = Path(tmp)
            stage2._export_vggt_artifacts(pred, out_dir)
            self.assertEqual([p.name for p in out_dir.iterdir()], [])
            self.assertEqual(write_glb.call_args.kwargs["max_points"], 300_000)
            self.assertEqual(write_glb.call_args.kwargs["confidence_percentile"], 50.0)
            self.assertEqual(
                write_glb.call_args.kwargs["confidence_prefiltered_percentile"], 50.0
            )
            self.assertIsNone(write_glb.call_args.kwargs["confidence"])
            self.assertEqual(write_glb.call_args.kwargs["coordinate_system"], "vggt_first_camera")
            np.testing.assert_array_equal(write_glb.call_args.args[1], PointCloud.vertices)
            np.testing.assert_array_equal(
                write_glb.call_args.args[2], PointCloud.colors[:, :3]
            )

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            stage2.os.environ,
            {
                "STAGE2_EXPORT_DEBUG_ARTIFACTS": "",
                "STAGE2_POINT_CLOUD_MAX_POINTS": "2000000",
            },
            clear=True,
        ), patch("point_cloud_glb.write_point_cloud_glb", return_value={"point_count": 4}) as write_glb:
            stage2._export_vggt_artifacts(pred, Path(tmp))
            self.assertEqual(write_glb.call_args.kwargs["max_points"], GLB_HARD_MAX_POINTS)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            stage2.os.environ,
            {"STAGE2_EXPORT_DEBUG_ARTIFACTS": "1"},
            clear=True,
        ), patch("point_cloud_glb.write_point_cloud_glb", return_value={"point_count": 4}):
            out_dir = Path(tmp)
            stage2._export_vggt_artifacts(pred, out_dir)
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["camera_intrinsics.json", "point_cloud.ply"],
            )
            camera = json.loads((out_dir / "camera_intrinsics.json").read_text())
            self.assertEqual(camera["schema"], "vggt-camera-intrinsics-v1")

    def test_preview_glb_fallback_never_pairs_depth_points_with_pointmap_confidence(self):
        depth_conf = np.array([[[3.0, 4.0]]], dtype=np.float32)
        pred = {
            "world_points": np.zeros((1, 1, 2, 3), dtype=np.float32),
            "colors": np.ones((1, 1, 2, 3), dtype=np.uint8),
            "world_points_conf": np.array([[[100.0, 200.0]]], dtype=np.float32),
            "depth_conf": depth_conf,
            "point_cloud_data": object(),
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "point_cloud_glb.write_point_cloud_glb", return_value={"point_count": 2}
        ) as write_glb:
            stage2._export_vggt_artifacts(pred, Path(tmp))

        self.assertIs(write_glb.call_args.kwargs["confidence"], depth_conf)
        self.assertIsNone(
            write_glb.call_args.kwargs["confidence_prefiltered_percentile"]
        )

        without_depth_conf = dict(pred)
        without_depth_conf.pop("depth_conf")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "point_cloud_glb.write_point_cloud_glb"
        ) as write_glb:
            result = stage2._export_vggt_artifacts(without_depth_conf, Path(tmp))
        write_glb.assert_not_called()
        self.assertNotIn("point_cloud_glb", result)
        self.assertIn("matching depth confidence are unavailable", result["warnings"][0]["error"])

    def test_room_alignment_fallback_is_not_reported_as_applied(self):
        self.assertFalse(stage2._room_alignment_was_applied(np.eye(3), np.zeros(3)))
        self.assertFalse(stage2._room_alignment_was_applied(np.full((3, 3), np.nan), np.zeros(3)))
        rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        self.assertTrue(stage2._room_alignment_was_applied(rotation, np.zeros(3)))
        self.assertTrue(stage2._room_alignment_was_applied(np.eye(3), np.array([0.0, 0.0, 1.0])))

    def test_mask_video_is_h264_yuv420p_faststart_and_source_duration_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            mask = root / "instance_masks.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=96x64:r=10:d=2.4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=96x64:r=25:d=0.4",
                    "-c:v", "mpeg4", str(mask),
                ],
                check=True,
            )

            metadata = stage2._normalize_mask_video(mask, source)
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "stream=codec_name,pix_fmt,nb_frames:format=duration", "-of", "json", str(mask),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            media = json.loads(probe.stdout)
            payload = mask.read_bytes()

        self.assertEqual(media["streams"][0]["codec_name"], "h264")
        self.assertEqual(media["streams"][0]["pix_fmt"], "yuv420p")
        self.assertEqual(int(media["streams"][0]["nb_frames"]), 10)
        self.assertGreaterEqual(payload.find(b"moov"), 0)
        self.assertGreaterEqual(payload.find(b"mdat"), 0)
        self.assertLess(payload.find(b"moov"), payload.find(b"mdat"))
        self.assertTrue(metadata["duration_aligned"])
        self.assertEqual(metadata["timeline_mode"], "source_frame_pts")
        self.assertAlmostEqual(float(media["format"]["duration"]), 2.4, delta=0.15)

    def test_sampled_mask_timeline_preserves_nonuniform_source_pts(self):
        probed = {
            "frames": [
                {"best_effort_timestamp_time": value}
                for value in ("0.000", "0.100", "0.900", "1.200", "2.400")
            ]
        }
        with patch.object(stage2, "_ffprobe_json", return_value=probed):
            timestamps = stage2._sampled_source_frame_timestamps(
                Path("variable-frame-rate.mp4"),
                [0, 1, 3, 4],
            )

        self.assertEqual(timestamps, [0.0, 0.1, 1.2, 2.4])
        self.assertNotEqual(
            [right - left for left, right in zip(timestamps, timestamps[1:])],
            [0.8, 0.8, 0.8],
        )

    def test_vfr_mask_encoding_preserves_sparse_pts_and_container_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            mask = root / "instance_masks.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=96x64:r=25:d=2.56",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=96x64:r=25:d=0.16",
                    "-frames:v", "4", "-c:v", "mpeg4", str(mask),
                ],
                check=True,
            )

            metadata = stage2._normalize_mask_video(
                mask,
                source,
                [0, 1, 2, 3],
                [0.0, 0.12, 1.2, 2.52],
            )
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_frames", "-show_entries",
                    "frame=best_effort_timestamp_time", "-show_entries", "format=duration",
                    "-of", "json", str(mask),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            media = json.loads(probe.stdout)

        frame_pts = [float(frame["best_effort_timestamp_time"]) for frame in media["frames"]]
        self.assertEqual(len(frame_pts), 4)
        for actual, expected in zip(frame_pts, [0.0, 0.12, 1.2, 2.52]):
            self.assertAlmostEqual(actual, expected, delta=0.001)
        self.assertAlmostEqual(float(media["format"]["duration"]), 2.56, delta=0.01)
        self.assertTrue(metadata["duration_aligned"])

    def test_video_duration_falls_back_through_stream_packets_and_frames(self):
        responses = [
            types.SimpleNamespace(returncode=0, stdout=json.dumps({"format": {"duration": "N/A"}})),
            types.SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{"duration": "N/A"}]})),
            types.SimpleNamespace(returncode=0, stdout=json.dumps({"packets": []})),
            types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "frames": [
                            {"best_effort_timestamp_time": "0.1", "pkt_duration_time": "0.1"},
                            {"best_effort_timestamp_time": "2.4", "pkt_duration_time": "0.1"},
                        ]
                    }
                ),
            ),
        ]
        with patch("subprocess.run", side_effect=responses) as probe:
            duration = stage2._probe_video_duration(Path("source.mp4"))

        self.assertAlmostEqual(duration, 2.4)
        self.assertEqual(probe.call_count, 4)
        self.assertIn("format=duration", probe.call_args_list[0].args[0])
        self.assertIn("stream=duration", probe.call_args_list[1].args[0])
        self.assertIn("-show_packets", probe.call_args_list[2].args[0])
        self.assertIn("-show_frames", probe.call_args_list[3].args[0])

    def test_video_duration_uses_stream_or_packet_timeline_before_decoding_frames(self):
        stream_responses = [
            types.SimpleNamespace(returncode=0, stdout=json.dumps({"format": {}})),
            types.SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{"duration": "3.25"}]})),
        ]
        with patch("subprocess.run", side_effect=stream_responses) as stream_probe:
            self.assertEqual(stage2._probe_video_duration(Path("stream.mp4")), 3.25)
        self.assertEqual(stream_probe.call_count, 2)

        packet_responses = [
            types.SimpleNamespace(returncode=0, stdout=json.dumps({"format": {}})),
            types.SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{}]})),
            types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "packets": [
                            {"dts_time": "-0.1", "duration_time": "0.1"},
                            {"pts_time": "2.2", "duration_time": "0.1"},
                        ]
                    }
                ),
            ),
        ]
        with patch("subprocess.run", side_effect=packet_responses) as packet_probe:
            self.assertAlmostEqual(stage2._probe_video_duration(Path("packets.mp4")), 2.4)
        self.assertEqual(packet_probe.call_count, 3)
        self.assertIn("-show_packets", packet_probe.call_args_list[-1].args[0])

    def test_mask_normalization_fails_when_source_timeline_cannot_be_proved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            mask = root / "instance_masks.mp4"
            source.write_bytes(b"source")
            mask.write_bytes(b"mask")
            with patch.object(stage2, "_probe_video_duration", return_value=None), patch(
                "subprocess.run"
            ) as transcode:
                with self.assertRaisesRegex(RuntimeError, "refusing to publish an unsynchronized"):
                    stage2._normalize_mask_video(mask, source)

        transcode.assert_not_called()

    def test_main_deploy_chain_is_serialized_to_prevent_revision_rollback(self):
        workflow = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
        ).read_text()
        self.assertIn("concurrency:", workflow)
        self.assertIn("group: ${{ github.workflow }}-${{ github.ref }}", workflow)
        self.assertIn("cancel-in-progress: false", workflow)

    def test_markerless_existing_weights_remain_default_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.safetensors").write_bytes(b"weights")
            self.assertTrue(stage2._vggt_weights_ok(model_dir, "facebook/VGGT-1B"))
            self.assertFalse(stage2._vggt_weights_ok(model_dir, "facebook/VGGT-1B-Commercial"))
            (model_dir / stage2.VGGT_MODEL_MARKER).write_text("facebook/VGGT-1B-Commercial\n")
            self.assertTrue(stage2._vggt_weights_ok(model_dir, "facebook/VGGT-1B-Commercial"))

    def test_vggt_model_override_reports_license_scope(self):
        with patch.dict(stage2.os.environ, {}, clear=True):
            self.assertEqual(stage2._vggt_model_id(), "facebook/VGGT-1B")
            self.assertEqual(stage2._vggt_license_scope(), "research_noncommercial")
        with patch.dict(stage2.os.environ, {"VGGT_MODEL_ID": "facebook/VGGT-1B-Commercial"}, clear=True):
            self.assertEqual(stage2._vggt_model_id(), "facebook/VGGT-1B-Commercial")
            self.assertEqual(stage2._vggt_license_scope(), "commercial")

    def test_public_vggt_cache_miss_ignores_even_a_stale_hf_token(self):
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            destination = Path(kwargs["local_dir"])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "model.safetensors").write_bytes(b"weights")

        fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            stage2.os.environ,
            {"HF_TOKEN": "revoked-placeholder"},
            clear=True,
        ), patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            models = Path(tmp)
            stage2._download_vggt_weights(models)

            self.assertEqual(calls[0]["repo_id"], "facebook/VGGT-1B")
            self.assertIs(calls[0]["token"], False)
            self.assertEqual(
                (models / "VGGT" / stage2.VGGT_MODEL_MARKER).read_text().strip(),
                "facebook/VGGT-1B",
            )

    def test_gated_model_downloads_require_an_explicit_hf_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            stage2.os.environ,
            {"VGGT_MODEL_ID": "facebook/VGGT-1B-Commercial"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires an approved Hugging Face token"):
                stage2._download_vggt_weights(Path(tmp))

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            stage2.os.environ,
            {},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "SAM3 weights are gated"):
                stage2._download_sam3_weights(Path(tmp))

    def test_weight_download_script_never_converts_a_missing_token_to_true(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "download_weights.sh"
        ).read_text()
        self.assertNotRegex(script, r"HUGGING_FACE_HUB_TOKEN\"\) or True")
        self.assertIn('token=False if vggt_model_id == "facebook/VGGT-1B" else token', script)

    def test_glb_contains_bounded_colored_points_and_camera_lines(self):
        points = np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3)
        colors = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(2, 2, 3, 3)
        confidence = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
        extrinsics[1, 0, 3] = 1.0

        data, stats = build_point_cloud_glb(
            points,
            colors,
            confidence=confidence,
            extrinsics=extrinsics,
            max_points=5,
            confidence_percentile=0,
        )
        magic, version, total_length = struct.unpack_from("<III", data, 0)
        json_length, json_type = struct.unpack_from("<II", data, 12)
        document = json.loads(data[20 : 20 + json_length].decode().rstrip(" "))

        self.assertEqual(magic, 0x46546C67)
        self.assertEqual(version, 2)
        self.assertEqual(total_length, len(data))
        self.assertEqual(json_type, 0x4E4F534A)
        self.assertEqual(stats["point_count"], 5)
        self.assertEqual(stats["camera_count"], 2)
        self.assertEqual(document["meshes"][0]["primitives"][0]["mode"], 0)
        self.assertEqual(document["meshes"][1]["primitives"][0]["mode"], 1)
        self.assertFalse(document["extras"]["isSurfaceMesh"])
        self.assertEqual(document["extras"]["coordinateSystem"], "vggt-first-camera-opengl-y-up")
        self.assertEqual(document["extras"]["upAxis"], "Y")
        self.assertEqual(document["extras"]["colorSpace"], "linear-srgb")
        self.assertEqual(document["nodes"][0]["extras"]["role"], "point-cloud")
        self.assertEqual(document["nodes"][1]["extras"]["role"], "camera-poses")
        self.assertFalse(document["nodes"][1]["extras"]["defaultVisible"])

        bin_header_offset = 20 + json_length
        bin_length, bin_type = struct.unpack_from("<II", data, bin_header_offset)
        self.assertEqual(bin_type, 0x004E4942)
        binary = data[bin_header_offset + 8 : bin_header_offset + 8 + bin_length]
        for mesh in document["meshes"]:
            color_index = mesh["primitives"][0]["attributes"]["COLOR_0"]
            accessor = document["accessors"][color_index]
            view = document["bufferViews"][accessor["bufferView"]]
            self.assertEqual(accessor["componentType"], 5121)
            self.assertEqual(accessor["type"], "VEC4")
            self.assertTrue(accessor["normalized"])
            self.assertEqual(view["byteOffset"] % 4, 0)
            self.assertEqual(view["byteLength"], accessor["count"] * 4)
            color_data = binary[
                view["byteOffset"] : view["byteOffset"] + view["byteLength"]
            ]
            self.assertEqual(color_data[3::4], b"\xff" * accessor["count"])

    def test_glb_aligns_vggt_to_first_camera_and_encodes_linear_colors(self):
        points = np.array(
            [
                [10.0, -1.0, 2.0],
                [12.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        )
        colors = np.full((2, 3), 128, dtype=np.uint8)
        extrinsics = np.eye(4, dtype=np.float32)[None, ...]
        extrinsics[0, 0, 3] = -10.0

        data, _stats = build_point_cloud_glb(
            points,
            colors,
            extrinsics=extrinsics,
            confidence_percentile=0,
        )
        json_length = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + json_length].decode().rstrip(" "))
        bin_header_offset = 20 + json_length
        binary = data[bin_header_offset + 8 :]

        position_accessor = document["accessors"][0]
        position_view = document["bufferViews"][position_accessor["bufferView"]]
        position_bytes = binary[
            position_view["byteOffset"] : position_view["byteOffset"] + position_view["byteLength"]
        ]
        exported_points = np.frombuffer(position_bytes, dtype="<f4").reshape(-1, 3)
        np.testing.assert_allclose(
            exported_points,
            np.array([[0.0, 1.0, -2.0], [2.0, -1.0, -4.0]], dtype=np.float32),
        )

        color_accessor = document["accessors"][1]
        color_view = document["bufferViews"][color_accessor["bufferView"]]
        color_bytes = binary[
            color_view["byteOffset"] : color_view["byteOffset"] + color_view["byteLength"]
        ]
        self.assertEqual(color_bytes[:4], bytes([55, 55, 55, 255]))
        self.assertEqual(document["extras"]["sourceColorSpace"], "srgb")
        self.assertEqual(document["extras"]["colorSpace"], "linear-srgb")
        self.assertGreater(document["extras"]["previewRadius"], 0)
        self.assertEqual(len(document["extras"]["previewCenter"]), 3)

    def test_glb_camera_paths_share_the_rotated_first_camera_frame(self):
        points = np.array([[-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]], dtype=np.float32)
        colors = np.full((2, 3), 255, dtype=np.uint8)
        angle = np.radians(35.0)
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float32,
        )
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
        extrinsics[0, :3, :3] = rotation
        extrinsics[0, :3, 3] = [2.0, -1.0, 0.5]
        extrinsics[1, :3, 3] = [-1.0, 0.5, 2.0]

        data, _stats = build_point_cloud_glb(
            points,
            colors,
            extrinsics=extrinsics,
            confidence_percentile=0,
        )
        json_length = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + json_length].decode().rstrip(" "))
        binary = data[28 + json_length :]
        camera_accessor_index = document["meshes"][1]["primitives"][0]["attributes"]["POSITION"]
        camera_accessor = document["accessors"][camera_accessor_index]
        camera_view = document["bufferViews"][camera_accessor["bufferView"]]
        camera_bytes = binary[
            camera_view["byteOffset"] : camera_view["byteOffset"] + camera_view["byteLength"]
        ]
        camera_points = np.frombuffer(camera_bytes, dtype="<f4").reshape(-1, 3)

        np.testing.assert_allclose(camera_points[[0, 2, 4, 6]], np.zeros((4, 3)), atol=1e-5)
        self.assertLess(float(camera_points[1, 2]), 0.0)
        first_transform = np.diag([1.0, -1.0, -1.0, 1.0]) @ extrinsics[0]
        second_origin_world = np.linalg.inv(extrinsics[1]) @ np.array([0.0, 0.0, 0.0, 1.0])
        expected_second_origin = (first_transform @ second_origin_world)[:3]
        np.testing.assert_allclose(camera_points[16], expected_second_origin, atol=1e-5)
        np.testing.assert_allclose(camera_accessor["min"], camera_points.min(axis=0), atol=1e-5)
        np.testing.assert_allclose(camera_accessor["max"], camera_points.max(axis=0), atol=1e-5)

    def test_glb_reports_axis_only_fallback_for_invalid_first_camera(self):
        points = np.array([[-1.0, -1.0, 2.0], [1.0, 1.0, 4.0]], dtype=np.float32)
        colors = np.full((2, 3), 255, dtype=np.uint8)
        extrinsics = np.eye(4, dtype=np.float32)[None, ...]
        extrinsics[0, 0, 0] = 2.0

        data, stats = build_point_cloud_glb(points, colors, extrinsics=extrinsics)
        json_length = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + json_length].decode().rstrip(" "))
        self.assertEqual(document["extras"]["coordinateSystem"], "vggt-world-opengl-y-up")
        self.assertFalse(document["extras"]["firstCameraAlignmentApplied"])
        self.assertEqual(stats["camera_count"], 0)

    def test_glb_confidence_contract_matches_applied_filter(self):
        points = np.column_stack(
            (np.arange(4, dtype=np.float32), np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        )
        colors = np.full((4, 3), 255, dtype=np.uint8)
        confidence = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)

        data, stats = build_point_cloud_glb(points, colors, confidence=confidence)
        json_length = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + json_length].decode().rstrip(" "))
        self.assertTrue(document["extras"]["confidenceFilterApplied"])
        self.assertEqual(document["extras"]["confidencePercentile"], 50.0)
        self.assertEqual(document["extras"]["confidenceThreshold"], 0.5)
        self.assertEqual(document["extras"]["pointCount"], 2)
        self.assertTrue(stats["confidence_filter_applied"])
        self.assertEqual(stats["confidence_filter_source"], "exporter")
        self.assertEqual(stats["confidence_percentile"], 50.0)
        self.assertEqual(stats["confidence_threshold"], 0.5)

        no_confidence_data, no_confidence_stats = build_point_cloud_glb(points, colors)
        no_confidence_json_length = struct.unpack_from("<I", no_confidence_data, 12)[0]
        no_confidence_document = json.loads(
            no_confidence_data[20 : 20 + no_confidence_json_length].decode().rstrip(" ")
        )
        self.assertFalse(no_confidence_document["extras"]["confidenceFilterApplied"])
        self.assertIsNone(no_confidence_document["extras"]["confidencePercentile"])
        self.assertIsNone(no_confidence_stats["confidence_percentile"])

        prefiltered_data, prefiltered_stats = build_point_cloud_glb(
            points,
            colors,
            confidence_prefiltered_percentile=50,
        )
        prefiltered_json_length = struct.unpack_from("<I", prefiltered_data, 12)[0]
        prefiltered_document = json.loads(
            prefiltered_data[20 : 20 + prefiltered_json_length].decode().rstrip(" ")
        )
        self.assertTrue(prefiltered_document["extras"]["confidenceFilterApplied"])
        self.assertEqual(
            prefiltered_document["extras"]["confidenceFilterSource"],
            "upstream-depth-prefiltered",
        )
        self.assertEqual(prefiltered_document["extras"]["confidencePercentile"], 50.0)
        self.assertIsNone(prefiltered_document["extras"]["confidenceThreshold"])
        self.assertEqual(
            prefiltered_stats["confidence_filter_source"],
            "upstream-depth-prefiltered",
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_point_cloud_glb(
                points,
                colors,
                confidence=confidence,
                confidence_prefiltered_percentile=50,
            )

        with self.assertRaisesRegex(ValueError, "fewer than two"):
            build_point_cloud_glb(
                np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                np.array([[255, 255, 255]], dtype=np.uint8),
                confidence_prefiltered_percentile=50,
            )

    def test_glb_converts_room_aligned_z_up_geometry_to_gltf_y_up(self):
        points = np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )
        colors = np.full((2, 3), 255, dtype=np.uint8)

        data, _stats = build_point_cloud_glb(
            points,
            colors,
            confidence_percentile=0,
            coordinate_system="room_z_up",
        )
        json_length = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + json_length].decode().rstrip(" "))
        bin_header_offset = 20 + json_length
        binary = data[bin_header_offset + 8 :]
        accessor = document["accessors"][0]
        view = document["bufferViews"][accessor["bufferView"]]
        position_bytes = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        exported_points = np.frombuffer(position_bytes, dtype="<f4").reshape(-1, 3)

        np.testing.assert_allclose(
            exported_points,
            np.array([[0.0, 0.0, -2.0], [0.0, 3.0, -2.0]], dtype=np.float32),
        )
        self.assertEqual(document["extras"]["coordinateSystem"], "room-aligned-gltf-y-up")
        self.assertEqual(document["extras"]["upAxis"], "Y")
        self.assertFalse(document["extras"]["firstCameraAlignmentApplied"])

    def test_glb_hard_cap_stays_within_signed_16_mib_contract(self):
        count = GLB_HARD_MAX_POINTS + 101
        axis = np.linspace(-1.0, 1.0, count, dtype=np.float32)
        points = np.column_stack((axis, axis * 0.5, axis * -0.25))
        colors = np.zeros((count, 3), dtype=np.uint8)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 160, axis=0)

        data, stats = build_point_cloud_glb(
            points,
            colors,
            extrinsics=extrinsics,
            max_points=2_000_000,
            confidence_percentile=0,
        )

        self.assertEqual(stats["point_count"], GLB_HARD_MAX_POINTS)
        self.assertEqual(stats["camera_count"], 160)
        self.assertEqual(stats["hard_point_cap"], GLB_HARD_MAX_POINTS)
        self.assertLessEqual(len(data), GLB_MAX_BYTES)

    def test_glb_trims_catastrophic_spatial_and_camera_outliers(self):
        axis = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        points = np.column_stack((axis, np.sin(axis), np.cos(axis)))
        points = np.vstack((points, np.array([[1_000_000.0] * 3], dtype=np.float32)))
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
        extrinsics[1, :3, 3] = -1_000_000.0

        data, stats = build_point_cloud_glb(
            points,
            colors,
            extrinsics=extrinsics,
            confidence_percentile=0,
        )
        json_length = struct.unpack_from("<I", data, 12)[0]
        document = json.loads(data[20 : 20 + json_length].decode().rstrip(" "))
        point_accessor = document["accessors"][0]

        self.assertEqual(stats["spatial_outliers_removed"], 1)
        self.assertEqual(stats["camera_count"], 1)
        self.assertLess(max(abs(value) for value in point_accessor["min"]), 2.0)
        self.assertLess(max(abs(value) for value in point_accessor["max"]), 2.0)
        self.assertEqual(document["extras"]["spatialOutliersRemoved"], 1)

    def test_glb_preserves_a_dense_remote_cluster_but_removes_an_isolated_spike(self):
        axis = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        main = np.column_stack((axis, np.sin(axis), np.cos(axis)))
        remote_cluster = np.repeat(
            np.array([[20.0, 20.0, 20.0]], dtype=np.float32),
            8,
            axis=0,
        )
        isolated = np.array([[40.0, 40.0, 40.0]], dtype=np.float32)
        points = np.vstack((main, remote_cluster, isolated))
        colors = np.zeros((len(points), 3), dtype=np.uint8)

        _data, stats = build_point_cloud_glb(
            points,
            colors,
            confidence_percentile=0,
        )

        self.assertEqual(stats["spatial_outliers_removed"], 1)
        self.assertEqual(stats["point_count"], len(points) - 1)
        self.assertFalse(stats["spatial_filter_fail_open"])

    def test_glb_rejects_empty_or_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            build_point_cloud_glb(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            build_point_cloud_glb(
                np.full((2, 3), np.nan, dtype=np.float32),
                np.zeros((2, 3), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
