import base64
import contextlib
import fcntl
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import deploy_revision, prune_runtimes


REVISION = "0123456789abcdef0123456789abcdef01234567"
NEWER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"


class DeployRevisionTests(unittest.TestCase):
    def test_payload_pins_revision_and_preserves_existing_template_fields(self):
        current = {
            "name": "stage2",
            "imageName": "runpod/pytorch:test",
            "containerDiskInGb": 40,
            "env": {"HF_TOKEN": "not-returned", "STAGE2_MODE_DEFAULT": "geometry"},
        }

        payload = deploy_revision.build_payload(current, REVISION)
        assertions = deploy_revision.validate_payload(payload, REVISION)

        self.assertTrue(all(assertions.values()))
        self.assertEqual(payload["name"], current["name"])
        self.assertEqual(payload["imageName"], current["imageName"])
        self.assertEqual(payload["containerDiskInGb"], current["containerDiskInGb"])
        self.assertEqual(payload["env"]["HF_TOKEN"], "not-returned")
        self.assertEqual(payload["env"]["STAGE2_CODE_REV"], REVISION)

    def test_dry_run_builds_and_validates_full_synthetic_payload_without_values(self):
        report = deploy_revision.dry_run_report("template-test", REVISION)

        self.assertEqual(report["status"], "dry_run")
        self.assertTrue(all(report["assertions"].values()))
        self.assertEqual(
            report["payload_fields"],
            [
                "containerDiskInGb",
                "dockerEntrypoint",
                "dockerStartCmd",
                "env",
                "imageName",
                "name",
            ],
        )
        rendered = json.dumps(report)
        self.assertNotIn("SYNTHETIC_EXISTING_VALUE", rendered)
        self.assertNotIn("runpod/pytorch:synthetic", rendered)

    def test_current_revision_check_is_last_network_read_before_runpod_post(self):
        calls = []
        current = {"name": "stage2", "env": {"STAGE2_MODE_DEFAULT": "geometry"}}

        def runpod_request(method, _url, _api_key, payload=None):
            calls.append(method)
            if method == "GET":
                return current
            self.assertEqual(method, "POST")
            return payload

        def github_head(_token=""):
            calls.append("GITHUB_HEAD")
            return REVISION

        stdout = io.StringIO()
        with patch.object(deploy_revision, "request_json", side_effect=runpod_request), patch.object(
            deploy_revision, "github_branch_head", side_effect=github_head
        ), patch.object(
            deploy_revision,
            "run_post_deploy_smoke",
            return_value={"status": "passed", "attempts": 1, "revision": REVISION},
        ), patch.object(
            sys,
            "argv",
            ["deploy_revision.py", "--revision", REVISION, "--template-id", "template-test"],
        ), patch.dict(os.environ, {"RUNPOD_API_KEY": "test-key"}, clear=True), contextlib.redirect_stdout(stdout):
            deploy_revision.main()

        self.assertEqual(calls, ["GET", "GITHUB_HEAD", "POST"])
        self.assertEqual(json.loads(stdout.getvalue())["status"], "deployed")

    def test_post_deploy_smoke_requires_ok_dry_run_from_requested_revision(self):
        responses = [
            {"id": "smoke-job-1"},
            {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "mode": "dry_run",
                    "stage2_code_revision": REVISION,
                },
            },
        ]
        with patch.object(deploy_revision, "invoke_json", side_effect=responses) as invoke, patch.object(
            deploy_revision.time, "sleep"
        ):
            report = deploy_revision.run_post_deploy_smoke(
                "endpoint-test",
                REVISION,
                "test-key",
                timeout_seconds=60,
            )

        self.assertEqual(report, {"status": "passed", "attempts": 1, "revision": REVISION})
        self.assertEqual(invoke.call_args_list[0].args[0], "POST")
        self.assertEqual(invoke.call_args_list[1].args[0], "GET")
        self.assertNotIn("test-key", json.dumps(report))

    def test_post_deploy_smoke_retries_a_stale_warm_worker(self):
        responses = [
            {"id": "smoke-job-1"},
            {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "mode": "dry_run",
                    "stage2_code_revision": NEWER_REVISION,
                },
            },
            {"id": "smoke-job-2"},
            {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "mode": "dry_run",
                    "stage2_code_revision": REVISION,
                },
            },
        ]
        with patch.object(deploy_revision, "invoke_json", side_effect=responses), patch.object(
            deploy_revision.time, "sleep"
        ):
            report = deploy_revision.run_post_deploy_smoke(
                "endpoint-test",
                REVISION,
                "test-key",
                timeout_seconds=60,
            )

        self.assertEqual(report["attempts"], 2)

    def test_post_deploy_smoke_timeout_cancels_inflight_job(self):
        with patch.object(deploy_revision, "invoke_json", return_value={"id": "smoke-job-1"}), patch.object(
            deploy_revision.time, "monotonic", side_effect=[0.0, 31.0]
        ), patch.object(deploy_revision, "_cancel_smoke_job") as cancel:
            with self.assertRaisesRegex(TimeoutError, "bounded timeout"):
                deploy_revision.run_post_deploy_smoke(
                    "endpoint-test",
                    REVISION,
                    "test-key",
                    timeout_seconds=30,
                )

        cancel.assert_called_once_with("endpoint-test", "smoke-job-1", "test-key")

    def test_embedded_smoke_video_is_decodable(self):
        raw = base64.b64decode(deploy_revision.SMOKE_VIDEO_B64, validate=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.mp4"
            path.write_bytes(raw)
            completed = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("h264", completed.stdout)

    def test_stale_revision_skips_runpod_mutation(self):
        methods = []

        def runpod_request(method, _url, _api_key, payload=None):
            methods.append(method)
            if method == "POST":
                self.fail("stale revision must not mutate RunPod")
            return {"name": "stage2", "env": {}}

        stdout = io.StringIO()
        with patch.object(deploy_revision, "request_json", side_effect=runpod_request), patch.object(
            deploy_revision, "github_branch_head", return_value=NEWER_REVISION
        ), patch.object(
            sys,
            "argv",
            ["deploy_revision.py", "--revision", REVISION, "--template-id", "template-test"],
        ), patch.dict(os.environ, {"RUNPOD_API_KEY": "test-key"}, clear=True), contextlib.redirect_stdout(stdout):
            deploy_revision.main()

        report = json.loads(stdout.getvalue())
        self.assertEqual(methods, ["GET"])
        self.assertEqual(report["status"], "skipped_stale_revision")
        self.assertEqual(report["current_main_revision"], NEWER_REVISION)

    def test_start_script_rejects_unpinned_revision_before_bootstrap(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "start_serverless.sh"
        env = dict(os.environ)
        env["STAGE2_CODE_REV"] = "main"

        completed = subprocess.run(
            ["bash", str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertEqual(completed.returncode, 64)
        self.assertIn("must be an explicit 40-character commit SHA", completed.stderr)

    def test_start_script_uses_revision_scoped_runtime_and_source_precedence(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "start_serverless.sh"
        ).read_text()

        self.assertIn('runtime_dir="$STAGE2_RUNTIME_ROOT/$STAGE2_CODE_REV"', script)
        self.assertIn('export STAGE2_VENV="$runtime_dir/venv"', script)
        self.assertIn('export RAS_ROOT="$runtime_dir/ReplicateAnyScene"', script)
        self.assertIn('export PYTHONPATH="$RAS_ROOT/vggt:$RAS_ROOT/sam3', script)
        self.assertIn('.stage2_code_revision', script)
        self.assertIn('flock -s 8', script)
        self.assertIn('runtime_min_age_seconds', script)
        self.assertIn('diff-index --name-only HEAD', script)

    def test_runtime_pruning_respects_recent_use_and_active_worker_lease(self):
        now = time.time()
        revisions = [f"{index:040x}" for index in range(1, 7)]
        ages = [0, 100, 1_000, 2_000, 3_000, 4_000]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, revision in enumerate(revisions):
                runtime = root / revision
                runtime.mkdir()
                last_used = runtime / ".last_used"
                last_used.touch()
                (runtime / ".active_lease").touch()
                os.utime(last_used, (now - ages[index], now - ages[index]))
            (root / revisions[5] / ".last_used").unlink()
            (root / revisions[5] / ".active_lease").unlink()
            os.utime(root / revisions[5], (now - ages[5], now - ages[5]))

            active_runtime = root / revisions[3]
            active_handle = (active_runtime / ".active_lease").open("a+b")
            fcntl.flock(active_handle.fileno(), fcntl.LOCK_SH)
            try:
                result = prune_runtimes.prune_runtimes(
                    root,
                    revisions[0],
                    keep=2,
                    min_age_seconds=1_500,
                    now=now,
                )
            finally:
                fcntl.flock(active_handle.fileno(), fcntl.LOCK_UN)
                active_handle.close()

            self.assertTrue((root / revisions[2]).exists(), "recent runtime must remain")
            self.assertTrue(active_runtime.exists(), "actively leased runtime must remain")
            self.assertFalse((root / revisions[4]).exists(), "old unleased runtime should be pruned")
            self.assertTrue((root / revisions[5]).exists(), "legacy runtime without lease must remain")
            self.assertEqual(result["kept_active"], 1)
            self.assertEqual(result["kept_recent"], 1)
            self.assertEqual(result["kept_unmanaged"], 1)
            self.assertEqual(result["removed"], 1)


if __name__ == "__main__":
    unittest.main()
