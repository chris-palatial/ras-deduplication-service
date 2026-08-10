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
import urllib.error
from pathlib import Path
from unittest.mock import patch

import stage2_service as stage2
from scripts import deploy_revision, prune_runtimes


REVISION = "0123456789abcdef0123456789abcdef01234567"
NEWER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"
SOURCE_EVIDENCE = {
    "ras_revision": deploy_revision.EXPECTED_RAS_REVISION,
    "vggt_revision": deploy_revision.EXPECTED_VGGT_REVISION,
    "sam3_revision": deploy_revision.EXPECTED_SAM3_REVISION,
    "sam3_required": True,
    "weights_required": False,
}


class DeployRevisionTests(unittest.TestCase):
    def test_bootstrap_clones_the_canonical_deduplication_repository(self):
        root = Path(__file__).resolve().parents[1]
        start_script = (root / "scripts" / "start_serverless.sh").read_text()
        bootstrap = deploy_revision.bootstrap_command(REVISION)
        canonical_url = "https://github.com/chris-palatial/ras-deduplication-service.git"

        self.assertEqual(
            deploy_revision.GITHUB_REPOSITORY,
            "chris-palatial/ras-deduplication-service",
        )
        self.assertIn(canonical_url, start_script)
        self.assertIn(canonical_url, bootstrap)
        self.assertNotIn("ras-stage2-service", start_script)
        self.assertNotIn("ras-stage2-service", bootstrap)

    def test_deploy_source_smoke_pins_match_worker_contract(self):
        self.assertEqual(deploy_revision.EXPECTED_RAS_REVISION, stage2.RAS_REVISION)
        self.assertEqual(deploy_revision.EXPECTED_VGGT_REVISION, stage2.DEFAULT_VGGT_REVISION)
        self.assertEqual(deploy_revision.EXPECTED_SAM3_REVISION, stage2.DEFAULT_SAM3_REVISION)

    def test_docker_build_uses_a_secret_mount_for_hugging_face_credentials(self):
        dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()

        self.assertIn("--mount=type=secret,id=hf_token,required=false", dockerfile)
        self.assertIn("--secret id=hf_token,env=HF_TOKEN", dockerfile)
        self.assertNotRegex(dockerfile, r"(?m)^ARG HF_TOKEN")
        self.assertNotIn("--build-arg HF_TOKEN", dockerfile)

    def test_release_budget_exceeds_the_longest_worker_execution_window(self):
        root = Path(__file__).resolve().parents[1]
        start_script = (root / "scripts" / "start_serverless.sh").read_text()
        workflow = (root / ".github" / "workflows" / "ci.yml").read_text()

        max_execution = int(
            start_script.split('STAGE2_MAX_EXECUTION_SECONDS:-', 1)[1].split('}', 1)[0]
        )
        rollout_budget = int(
            workflow.split("--rollout-timeout-seconds ", 1)[1].splitlines()[0]
        )
        self.assertGreaterEqual(rollout_budget, max_execution + 600)
        self.assertIn("timeout-minutes: 60", workflow)

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

        def runpod_request(method, url, _api_key, payload=None):
            calls.append((method, url))
            if "/endpoints/" in url:
                return {"version": 7, "workers": []}
            if method == "GET":
                return current
            self.assertEqual(method, "POST")
            return payload

        def github_head(_token="", *, deadline=None):
            self.assertIsNotNone(deadline)
            calls.append(("GITHUB_HEAD", ""))
            return REVISION

        stdout = io.StringIO()
        with patch.object(deploy_revision, "request_json", side_effect=runpod_request), patch.object(
            deploy_revision, "github_branch_head", side_effect=github_head
        ), patch.object(
            deploy_revision,
            "wait_for_endpoint_rollout",
            return_value={"status": "converged", "version": "8"},
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

        self.assertEqual(
            [method for method, _url in calls],
            ["GET", "GET", "GITHUB_HEAD", "POST"],
        )
        self.assertIn("/templates/", calls[0][1])
        self.assertIn("/endpoints/", calls[1][1])
        self.assertIn("includeWorkers=true", calls[1][1])
        self.assertEqual(json.loads(stdout.getvalue())["status"], "deployed")

    def test_endpoint_rollout_waits_for_version_advance_and_every_live_worker(self):
        snapshots = [
            {"version": 7, "workers": [{"status": "RUNNING", "slsVersion": 7}]},
            {
                "version": 8,
                "workers": [
                    {"status": "RUNNING", "slsVersion": 7},
                    {"status": "RUNNING", "slsVersion": 8},
                ],
            },
            {
                "version": 8,
                "workers": [
                    {"status": "RUNNING", "slsVersion": 8},
                    {"status": "EXITED", "desiredStatus": "EXITED", "slsVersion": 7},
                ],
            },
        ]
        with patch.object(deploy_revision, "request_json", side_effect=snapshots), patch.object(
            deploy_revision.time, "sleep"
        ):
            report = deploy_revision.wait_for_endpoint_rollout(
                "endpoint-test",
                "7",
                "test-key",
                deadline=time.monotonic() + 60,
            )

        self.assertEqual(report["version"], "8")
        self.assertEqual(report["active_workers"], 1)
        self.assertEqual(report["polls"], 3)

    def test_endpoint_rollout_keeps_a_draining_old_worker_until_it_actually_exits(self):
        snapshots = [
            {
                "version": 8,
                "workers": [
                    {
                        "status": "RUNNING",
                        "desiredStatus": "EXITED",
                        "slsVersion": 7,
                    },
                    {"status": "RUNNING", "slsVersion": 8},
                ],
            },
            {
                "version": 8,
                "workers": [
                    {
                        "status": "EXITED",
                        "desiredStatus": "EXITED",
                        "slsVersion": 7,
                    },
                    {"status": "RUNNING", "slsVersion": 8},
                ],
            },
        ]
        with patch.object(deploy_revision, "request_json", side_effect=snapshots), patch.object(
            deploy_revision.time, "sleep"
        ):
            report = deploy_revision.wait_for_endpoint_rollout(
                "endpoint-test",
                "7",
                "test-key",
                deadline=time.monotonic() + 60,
            )

        self.assertEqual(report["active_workers"], 1)
        self.assertEqual(report["polls"], 2)

    def test_safe_get_caps_each_network_attempt_to_its_remaining_deadline(self):
        observed_timeouts = []

        def request_once(_method, _url, _api_key, _payload=None, *, timeout=30):
            observed_timeouts.append(timeout)
            return {"status": "ok"}

        with patch.object(deploy_revision, "_request_json_once", side_effect=request_once), patch.object(
            deploy_revision.time, "monotonic", side_effect=[100.0, 104.75]
        ):
            result = deploy_revision.request_json(
                "GET",
                "https://example.invalid/status",
                "test-key",
                deadline=105.0,
            )

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(observed_timeouts, [0.25])

    def test_endpoint_rollout_accepts_zero_workers_only_after_version_advance(self):
        snapshots = [
            {"version": "2", "workers": []},
            {"version": "3", "workers": []},
        ]
        with patch.object(deploy_revision, "request_json", side_effect=snapshots), patch.object(
            deploy_revision.time, "sleep"
        ):
            report = deploy_revision.wait_for_endpoint_rollout(
                "endpoint-test",
                "2",
                "test-key",
                deadline=time.monotonic() + 60,
            )

        self.assertEqual(report["version"], "3")
        self.assertEqual(report["active_workers"], 0)
        self.assertEqual(report["polls"], 2)

    def test_endpoint_rollout_retries_a_transient_safe_get(self):
        snapshot = {"version": 8, "workers": [{"status": "RUNNING", "slsVersion": 8}]}
        with patch.object(
            deploy_revision,
            "_request_json_once",
            side_effect=[urllib.error.URLError("temporary network failure"), snapshot],
        ) as read, patch.object(deploy_revision.time, "sleep"):
            report = deploy_revision.wait_for_endpoint_rollout(
                "endpoint-test",
                "7",
                "test-key",
                deadline=time.monotonic() + 60,
            )

        self.assertEqual(report["version"], "8")
        self.assertEqual(read.call_count, 2)

    def test_post_deploy_smoke_requires_ok_dry_run_from_requested_revision(self):
        responses = [
            {"id": "smoke-job-1"},
            {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "analysis_type": "validation_v1",
                    "mode": "dry_run",
                    "stage2_code_revision": REVISION,
                    "source": SOURCE_EVIDENCE,
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
        self.assertEqual(
            invoke.call_args_list[0].args[3]["input"]["analysis_type"],
            "validation_v1",
        )
        self.assertEqual(invoke.call_args_list[1].args[0], "GET")
        self.assertNotIn("test-key", json.dumps(report))

    def test_post_deploy_smoke_retries_only_the_safe_status_get_after_rate_limit(self):
        methods = []
        get_calls = 0

        def invoke_once(method, _url, _api_key, payload=None, *, timeout=30):
            nonlocal get_calls
            methods.append(method)
            if method == "POST":
                return {"id": "smoke-job-retry"}
            get_calls += 1
            if get_calls == 1:
                raise urllib.error.HTTPError(
                    "https://api.runpod.ai/status",
                    429,
                    "rate limited",
                    {"Retry-After": "0"},
                    None,
                )
            return {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "analysis_type": "validation_v1",
                    "mode": "dry_run",
                    "stage2_code_revision": REVISION,
                    "source": SOURCE_EVIDENCE,
                },
            }

        with patch.object(deploy_revision, "_invoke_json_once", side_effect=invoke_once), patch.object(
            deploy_revision.time, "sleep"
        ):
            report = deploy_revision.run_post_deploy_smoke(
                "endpoint-test",
                REVISION,
                "test-key",
                timeout_seconds=60,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(methods.count("POST"), 1, "paid /run must never be retried")
        self.assertEqual(methods.count("GET"), 2)

    def test_post_deploy_smoke_requires_pinned_source_bootstrap(self):
        responses = [
            {"id": "smoke-job-1"},
            {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "analysis_type": "validation_v1",
                    "mode": "dry_run",
                    "stage2_code_revision": REVISION,
                },
            },
        ]
        with patch.object(deploy_revision, "invoke_json", side_effect=responses), patch.object(
            deploy_revision.time, "sleep"
        ), self.assertRaisesRegex(RuntimeError, "pinned source bootstrap"):
            deploy_revision.run_post_deploy_smoke(
                "endpoint-test",
                REVISION,
                "test-key",
                timeout_seconds=60,
            )

    def test_post_deploy_smoke_retries_a_stale_warm_worker(self):
        responses = [
            {"id": "smoke-job-1"},
            {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "analysis_type": "validation_v1",
                    "mode": "dry_run",
                    "stage2_code_revision": NEWER_REVISION,
                    "source": SOURCE_EVIDENCE,
                },
            },
            {"id": "smoke-job-2"},
            {
                "status": "COMPLETED",
                "output": {
                    "status": "ok",
                    "analysis_type": "validation_v1",
                    "mode": "dry_run",
                    "stage2_code_revision": REVISION,
                    "source": SOURCE_EVIDENCE,
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

    def test_post_deploy_smoke_uses_the_deadline_instead_of_six_attempts(self):
        responses = []
        for attempt in range(1, 8):
            responses.extend([
                {"id": f"smoke-job-{attempt}"},
                {
                    "status": "COMPLETED",
                    "output": {
                        "status": "ok",
                        "analysis_type": "validation_v1",
                        "mode": "dry_run",
                        "stage2_code_revision": REVISION if attempt == 7 else NEWER_REVISION,
                        "source": SOURCE_EVIDENCE,
                    },
                },
            ])
        with patch.object(deploy_revision, "invoke_json", side_effect=responses), patch.object(
            deploy_revision.time, "sleep"
        ):
            report = deploy_revision.run_post_deploy_smoke(
                "endpoint-test",
                REVISION,
                "test-key",
                timeout_seconds=60,
            )

        self.assertEqual(report["attempts"], 7)

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

        def runpod_request(method, url, _api_key, payload=None):
            methods.append(method)
            if method == "POST":
                self.fail("stale revision must not mutate RunPod")
            if "/endpoints/" in url:
                return {"version": 3, "workers": []}
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
        self.assertEqual(methods, ["GET", "GET"])
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

    def test_start_script_refuses_to_rebuild_a_runtime_with_an_active_worker(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "start_serverless.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_root = root / "runtimes"
            runtime = runtime_root / REVISION
            runtime.mkdir(parents=True)
            sentinel = runtime / "worker-owned.txt"
            sentinel.write_text("still in use")
            lease_path = runtime / ".active_lease"
            lease = lease_path.open("a+b")
            fcntl.flock(lease.fileno(), fcntl.LOCK_SH)

            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_apt = fake_bin / "apt-get"
            fake_apt.write_text("#!/bin/sh\nexit 0\n")
            fake_apt.chmod(0o755)
            fake_flock = fake_bin / "flock"
            fake_flock.write_text(
                "#!/usr/bin/python3\n"
                "import fcntl, sys\n"
                "args = sys.argv[1:]\n"
                "unlock = '-u' in args\n"
                "shared = '-s' in args\n"
                "nonblocking = '-n' in args\n"
                "fd = int(args[-1])\n"
                "mode = fcntl.LOCK_UN if unlock else (fcntl.LOCK_SH if shared else fcntl.LOCK_EX)\n"
                "if nonblocking: mode |= fcntl.LOCK_NB\n"
                "try:\n"
                "    fcntl.flock(fd, mode)\n"
                "except BlockingIOError:\n"
                "    raise SystemExit(1)\n"
            )
            fake_flock.chmod(0o755)
            env = dict(os.environ)
            env.update({
                "PATH": f"{fake_bin}:{env['PATH']}",
                "STAGE2_CODE_REV": REVISION,
                "STAGE2_RUNTIME_ROOT": str(runtime_root),
                "STAGE2_MODELS_DIR": str(root / "models"),
                "HF_HOME": str(root / "hf-cache"),
            })
            try:
                completed = subprocess.run(
                    ["bash", str(script)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            self.assertEqual(completed.returncode, 75)
            self.assertIn("refusing to rebuild it in place", completed.stderr)
            self.assertEqual(sentinel.read_text(), "still in use")

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
        self.assertLess(script.index('flock -n 7'), script.index('rm -rf -- "$runtime_dir"'))
        self.assertIn('runtime_min_age_seconds', script)
        self.assertIn('diff-index --name-only HEAD', script)
        self.assertIn('export STAGE2_BUILD_REVISION_FILE="$marker"', script)

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
