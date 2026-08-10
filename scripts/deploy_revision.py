#!/usr/bin/env python3
"""Atomically pin the RunPod template environment and bootstrap to one commit."""

from __future__ import annotations

import argparse
import email.utils
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any


API_ROOT = "https://rest.runpod.io/v1"
GITHUB_API_ROOT = "https://api.github.com"
GITHUB_REPOSITORY = "chris-palatial/ras-deduplication-service"
GITHUB_BRANCH = "main"
DEFAULT_TEMPLATE_ID = "inapyg0va0"
DEFAULT_ENDPOINT_ID = "sp2oyuum48vk0j"
INVOKE_API_ROOT = "https://api.runpod.ai/v2"
EXPECTED_RAS_REVISION = "671191457e7244d9337ef3faf558ee92bbf9bf73"
EXPECTED_VGGT_REVISION = "9e4fa662a8893ed348d048e8b57816c12593448b"
SMOKE_VIDEO_B64 = (
    "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMrbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+gAAQAA"
    "AQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAlV0"
    "cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAA"
    "AABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAPoAAAAAAABAAAAAAHNbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABA"
    "AAAAQABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABeG1pbmYAAAAUdm1oZAAAAAEAAAAAAAAA"
    "AAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAThzdGJsAAAAuHN0c2QAAAAAAAAAAQAAAKhhdmMxAAAAAAAAAAEAAAAA"
    "AAAAAAAAAAAAAAAAABAAEABIAAAASAAAAAAAAAABFUxhdmM2Mi4yOC4xMDIgbGli eDI2NAAAAAAAAAAAAAAAGP//AAAALmF2Y0MBQsAK/"
    "+EAFmdCwArZHsBEAAADAAQAAAMAEDxImSABAAVoy4PLIAAAABBwYXNwAAAAAQAAAAEAAAAUYnRydAAAAAAAABRoAAAAAAAAABhzdHRzAAAA"
    "AAAAAAEAAAACAAAgAAAAABRzdHNzAAAAAAAAAAEAAAABAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAACAAAAAQAAABxzdHN6AAAAAAAAAAAAAAAC"
    "AAACgwAAAAoAAAAUc3RjbwAAAAAAAAABAAADWwAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAA"
    "AAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjIuMTIuMTAyAAAACGZyZWUAAAKVbWRhdAAAAnAGBf//bNxF6b3m2Ui3"
    "lizYINkj7u94MjY0IC0gY29yZSAxNjUgcjMyMjIgYjM1NjA1YSAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAz"
    "LTIwMjUgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0wIHJlZj0zIGRlYmxvY2s9MTow"
    "OjAgYW5hbHlzZT0weDE6MHgxMTEgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMiBtaXhlZF9yZWY9MSBtZV9yYW5n"
    "ZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTAgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9x"
    "cF9vZmZzZXQ9LTIgdGhyZWFkcz0xIGxvb2thaGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50"
    "ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MCB3ZWlnaHRwPTAga2V5aW50PTI1MCBr"
    "ZXlpbnRfbWluPTIgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVzaD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIz"
    "LjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAALZYiEBTyYoAA/"
    "v4AAAAAGQZo4CXqA"
).replace(" ", "")
UPDATE_FIELDS = (
    "containerDiskInGb",
    "containerRegistryAuthId",
    "dockerEntrypoint",
    "dockerStartCmd",
    "env",
    "imageName",
    "isPublic",
    "name",
    "ports",
    "readme",
    "volumeInGb",
    "volumeMountPath",
)


def bootstrap_command(revision: str) -> str:
    return (
        "set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq; apt-get install -y -qq git curl >/dev/null; "
        "BOOT_DIR=$(mktemp -d /tmp/ras-deduplication-boot.XXXXXX); "
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/chris-palatial/ras-deduplication-service.git \"$BOOT_DIR\"; "
        f"git -C \"$BOOT_DIR\" fetch --depth 1 origin {revision}; "
        "git -C \"$BOOT_DIR\" checkout --detach FETCH_HEAD; "
        f"export STAGE2_CODE_REV={revision}; "
        "exec bash \"$BOOT_DIR/scripts/start_serverless.sh\""
    )


def _request_json_once(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
    *,
    timeout: float = 30,
) -> dict:
    data = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("RunPod template API returned a non-object response")
    return result


def _invoke_json_once(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
    *,
    timeout: int = 30,
) -> dict:
    data = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("RunPod invoke API returned a non-object response")
    return result


def _retry_after_seconds(error: BaseException, attempt: int) -> float:
    if isinstance(error, urllib.error.HTTPError):
        raw = error.headers.get("Retry-After") if error.headers else None
        if raw:
            try:
                return max(0.0, min(float(raw), 30.0))
            except ValueError:
                try:
                    retry_at = email.utils.parsedate_to_datetime(raw).timestamp()
                    return max(0.0, min(retry_at - time.time(), 30.0))
                except (TypeError, ValueError, OverflowError):
                    pass
    return min(0.5 * (2 ** max(0, attempt - 1)), 10.0)


def _is_retryable_read_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 425, 429} or 500 <= error.code <= 599
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))


def _remaining_network_timeout(*, deadline: float, cap: float, description: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{description} exceeded its bounded deadline")
    return min(cap, remaining)


def _retry_safe_get(fetch, *, deadline: float, description: str) -> dict:
    attempt = 0
    while True:
        if deadline - time.monotonic() <= 0:
            raise TimeoutError(f"{description} did not recover before the deployment deadline")
        attempt += 1
        try:
            return fetch()
        except Exception as error:
            if not _is_retryable_read_error(error):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{description} did not recover before the deployment deadline"
                ) from error
            time.sleep(min(_retry_after_seconds(error, attempt), remaining))


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
    *,
    deadline: float | None = None,
) -> dict:
    if method != "GET":
        # Template updates are ambiguous writes and must never be replayed.
        return _request_json_once(method, url, api_key, payload)
    read_deadline = deadline if deadline is not None else time.monotonic() + 120
    return _retry_safe_get(
        lambda: _request_json_once(
            method,
            url,
            api_key,
            payload,
            timeout=_remaining_network_timeout(
                deadline=read_deadline,
                cap=30,
                description="RunPod control-plane read",
            ),
        ),
        deadline=read_deadline,
        description="RunPod control-plane read",
    )


def invoke_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
    *,
    timeout: int = 30,
    deadline: float | None = None,
) -> dict:
    if method != "GET":
        # Paid /run and cancellation POSTs are never retried implicitly.
        return _invoke_json_once(method, url, api_key, payload, timeout=timeout)
    read_deadline = deadline if deadline is not None else time.monotonic() + 120
    return _retry_safe_get(
        lambda: _invoke_json_once(
            method,
            url,
            api_key,
            payload,
            timeout=_remaining_network_timeout(
                deadline=read_deadline,
                cap=timeout,
                description="RunPod job-status read",
            ),
        ),
        deadline=read_deadline,
        description="RunPod job-status read",
    )


def invoke_api_root() -> str:
    root = os.environ.get("RUNPOD_INVOKE_URL", INVOKE_API_ROOT).strip().rstrip("/")
    if not (root.startswith("https://") or root.startswith("http://")):
        raise RuntimeError("RUNPOD_INVOKE_URL must be an HTTP URL")
    return root


def _cancel_smoke_job(endpoint_id: str, job_id: str, api_key: str) -> None:
    try:
        invoke_json(
            "POST",
            f"{invoke_api_root()}/{endpoint_id}/cancel/{job_id}",
            api_key,
            timeout=15,
        )
    except Exception:
        pass


def run_post_deploy_smoke(
    endpoint_id: str,
    revision: str,
    api_key: str,
    *,
    timeout_seconds: int,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Run bounded async dry jobs until the rolling endpoint serves revision."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", endpoint_id):
        raise RuntimeError("RunPod endpoint id is malformed")
    if timeout_seconds < 30 or timeout_seconds > 1200:
        raise RuntimeError("smoke timeout must be from 30 through 1200 seconds")

    now = time.monotonic()
    deadline = deadline if deadline is not None else now + timeout_seconds
    if deadline <= now:
        raise TimeoutError("RunPod rollout exhausted the bounded deployment timeout before smoke")
    invoke_root = invoke_api_root()
    last_observation = "no_completed_job"
    attempt = 0
    # The wall-clock deadline is the cost/reliability boundary. Do not exhaust
    # an arbitrary attempt count while the rolling endpoint can still have old
    # warm workers and most of the configured smoke budget remains.
    while attempt == 0 or time.monotonic() < deadline:
        attempt += 1
        started = invoke_json(
            "POST",
            f"{invoke_root}/{endpoint_id}/run",
            api_key,
            {
                "input": {
                    "analysis_type": "validation_v1",
                    "mode": "dry_run",
                    "video_b64": SMOKE_VIDEO_B64,
                    "media_type": "video/mp4",
                    "categories": ["deployment-smoke"],
                    "max_frames": 2,
                }
            },
        )
        job_id = started.get("id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{6,128}", job_id):
            raise RuntimeError("RunPod smoke submission returned an invalid job id")

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _cancel_smoke_job(endpoint_id, job_id, api_key)
                raise TimeoutError("RunPod post-deploy smoke exceeded its bounded timeout")
            status_result = invoke_json(
                "GET",
                f"{invoke_root}/{endpoint_id}/status/{job_id}",
                api_key,
                timeout=max(1, min(30, int(remaining))),
                deadline=deadline,
            )
            state = str(status_result.get("status") or "").upper()
            if state == "COMPLETED":
                break
            if state in {"FAILED", "TIMED_OUT", "CANCELLED"}:
                raise RuntimeError(f"RunPod post-deploy smoke ended with state {state}")
            poll_remaining = deadline - time.monotonic()
            if poll_remaining > 0:
                time.sleep(min(5.0, poll_remaining))

        output = status_result.get("output")
        if not isinstance(output, dict):
            last_observation = "non_object_output"
        elif output.get("stage2_code_revision") != revision:
            # A rolling release may briefly route to an old warm worker. Retry a
            # bounded number of fresh jobs without exposing the old response.
            last_observation = "stale_worker_revision"
        elif (
            output.get("status") != "ok"
            or output.get("mode") != "dry_run"
            or output.get("analysis_type") != "validation_v1"
        ):
            raise RuntimeError("RunPod post-deploy dry_run returned a non-ok result")
        elif not isinstance(output.get("source"), dict) or (
            output["source"].get("ras_revision") != EXPECTED_RAS_REVISION
            or output["source"].get("vggt_revision") != EXPECTED_VGGT_REVISION
            or output["source"].get("sam_provider") != "fal"
            or output["source"].get("sam3_required") is not False
            or output["source"].get("weights_required") is not False
        ):
            raise RuntimeError("RunPod post-deploy dry_run did not verify pinned source bootstrap")
        else:
            return {
                "status": "passed",
                "attempts": attempt,
                "revision": revision,
            }

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(10.0, remaining))

    raise RuntimeError(
        f"RunPod post-deploy smoke did not observe the deployed revision ({last_observation})"
    )


def _endpoint_version(snapshot: dict[str, Any]) -> str:
    version = snapshot.get("version")
    if isinstance(version, (str, int)) and str(version).strip():
        return str(version).strip()
    raise RuntimeError("RunPod endpoint API did not return a version")


def _active_worker_versions(snapshot: dict[str, Any]) -> list[str | None]:
    workers = snapshot.get("workers")
    if workers is None:
        return []
    if not isinstance(workers, list):
        raise RuntimeError("RunPod endpoint API returned malformed workers")

    versions: list[str | None] = []
    terminal = {"EXITED", "STOPPED", "TERMINATED"}
    for worker in workers:
        if not isinstance(worker, dict):
            raise RuntimeError("RunPod endpoint API returned a malformed worker")
        status = str(worker.get("status") or "").upper()
        # RunPod marks a draining worker's desired state EXITED before the
        # process has actually stopped. It remains part of the live fleet until
        # its observed status becomes terminal.
        if status in terminal:
            continue
        version = worker.get("slsVersion")
        versions.append(str(version).strip() if isinstance(version, (str, int)) else None)
    return versions


def _rollout_report(
    status: str,
    observation: str,
    *,
    previous_version: str,
    version: str,
    worker_versions: list[str | None],
    polls: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "observation": observation,
        "previous_version": previous_version,
        "version": version,
        "version_advanced": version != previous_version,
        "active_workers": len(worker_versions),
        "workers_on_target_version": sum(
            1 for worker_version in worker_versions if worker_version == version
        ),
        "polls": polls,
    }


def wait_for_endpoint_rollout(
    endpoint_id: str,
    previous_version: str,
    api_key: str,
    *,
    deadline: float,
    expect_version_change: bool = True,
) -> dict[str, Any]:
    """Observe the live fleet until it runs the endpoint's target version.

    Convergence is a statement about reaching the target state, not about
    catching a transition. RunPod does not bump the endpoint version for a
    template update that changes nothing, so a re-run against an already
    pinned template can never observe an advance, and demanding one made every
    retry of a timed-out rollout fail by construction.

    `expect_version_change` carries the only fact that separates the two cases:
    whether the template already held this revision's pins before the update.
    When it did, the endpoint's current version already reflects the target
    template and no advance is owed. When it did not, an advance is still
    required before a uniform fleet can mean anything, because a fleet that is
    uniformly on the previous version is uniformly on the previous code.

    The report is advisory. `run_post_deploy_smoke` is the authoritative proof
    that the target revision serves traffic, so a fleet that has not finished
    draining is reported rather than raised.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", endpoint_id):
        raise RuntimeError("RunPod endpoint id is malformed")

    polls = 0
    pending: dict[str, Any] | None = None
    while pending is None or time.monotonic() < deadline:
        polls += 1
        try:
            snapshot = request_json(
                "GET",
                f"{API_ROOT}/endpoints/{endpoint_id}?includeWorkers=true",
                api_key,
                deadline=deadline,
            )
        except TimeoutError:
            # A read that cannot finish before the fleet deadline is the fleet
            # deadline expiring. Report the last observation instead of losing
            # the deployment to it.
            if pending is None:
                raise
            return pending
        version = _endpoint_version(snapshot)
        worker_versions = _active_worker_versions(snapshot)
        version_advanced = version != previous_version
        endpoint_at_target = version_advanced or not expect_version_change
        fleet_on_version = all(
            worker_version == version for worker_version in worker_versions
        )
        if endpoint_at_target and fleet_on_version:
            return _rollout_report(
                "converged" if version_advanced else "already_converged",
                "fleet_on_target_version",
                previous_version=previous_version,
                version=version,
                worker_versions=worker_versions,
                polls=polls,
            )
        pending = _rollout_report(
            "fleet_not_converged",
            "mixed_worker_versions" if endpoint_at_target else "endpoint_version_not_advanced",
            previous_version=previous_version,
            version=version,
            worker_versions=worker_versions,
            polls=polls,
        )
        poll_remaining = deadline - time.monotonic()
        if poll_remaining > 0:
            time.sleep(min(5.0, poll_remaining))
    return pending


def github_branch_head(token: str = "", *, deadline: float | None = None) -> str:
    headers = {
        "accept": "application/vnd.github+json",
        "user-agent": "ras-deduplication-deployer",
        "x-github-api-version": "2022-11-28",
    }
    if token:
        headers["authorization"] = f"Bearer {token}"
    read_deadline = deadline if deadline is not None else time.monotonic() + 120

    def fetch() -> dict:
        req = urllib.request.Request(
            f"{GITHUB_API_ROOT}/repos/{GITHUB_REPOSITORY}/git/ref/heads/{GITHUB_BRANCH}",
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(
            req,
            timeout=_remaining_network_timeout(
                deadline=read_deadline,
                cap=30,
                description="GitHub branch read",
            ),
        ) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError("GitHub branch API returned a non-object response")
        return result

    result = _retry_safe_get(
        fetch,
        deadline=read_deadline,
        description="GitHub branch read",
    )
    revision = ((result.get("object") or {}).get("sha")) if isinstance(result, dict) else None
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("GitHub branch API returned an invalid revision")
    return revision


def build_payload(current: dict, revision: str) -> dict:
    payload = {field: current[field] for field in UPDATE_FIELDS if field in current and current[field] is not None}
    env = dict(current.get("env") or {})
    env["STAGE2_CODE_REV"] = revision
    payload.update({
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [bootstrap_command(revision)],
        "env": env,
    })
    return payload


def revision_pin_assertions(snapshot: dict[str, Any], revision: str) -> dict[str, bool]:
    """Report which revision pins a template snapshot already carries."""
    env = snapshot.get("env")
    entrypoint = snapshot.get("dockerEntrypoint")
    command = snapshot.get("dockerStartCmd")
    return {
        "environment_pin": isinstance(env, dict) and env.get("STAGE2_CODE_REV") == revision,
        "entrypoint_pin": entrypoint == ["bash", "-lc"],
        "bootstrap_pin": command == [bootstrap_command(revision)],
    }


def revision_pins_applied(snapshot: dict[str, Any], revision: str) -> bool:
    """Report whether the live template already pins exactly this revision.

    A template that already carries every pin makes the update a no-op, and
    RunPod does not issue a new endpoint version for a no-op update.
    """
    return all(revision_pin_assertions(snapshot, revision).values())


def validate_payload(payload: dict[str, Any], revision: str) -> dict[str, bool]:
    assertions = revision_pin_assertions(payload, revision)
    if not all(assertions.values()):
        failed = sorted(name for name, passed in assertions.items() if not passed)
        raise RuntimeError(f"RunPod template revision validation failed: {', '.join(failed)}")
    return assertions


def dry_run_report(template_id: str, revision: str) -> dict[str, Any]:
    current = {
        "name": "stage2-dry-run-template",
        "imageName": "runpod/pytorch:synthetic",
        "containerDiskInGb": 20,
        "env": {
            "STAGE2_MODE_DEFAULT": "geometry",
            "SYNTHETIC_EXISTING_VALUE": "preserved",
        },
    }
    payload = build_payload(current, revision)
    assertions = validate_payload(payload, revision)
    assertions["existing_environment_preserved"] = (
        payload["env"].get("SYNTHETIC_EXISTING_VALUE") == "preserved"
    )
    assertions["template_fields_preserved"] = all(
        payload.get(field) == current[field]
        for field in ("name", "imageName", "containerDiskInGb")
    )
    if not all(assertions.values()):
        raise RuntimeError("synthetic RunPod payload did not preserve existing template fields")
    return {
        "status": "dry_run",
        "template_id": template_id,
        "revision": revision,
        "assertions": assertions,
        "payload_fields": sorted(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument(
        "--endpoint-id",
        default=os.environ.get("STAGE2_ENDPOINT_ID", DEFAULT_ENDPOINT_ID),
    )
    parser.add_argument(
        "--smoke-timeout-seconds",
        type=int,
        default=int(os.environ.get("STAGE2_DEPLOY_SMOKE_TIMEOUT_SECONDS", "900")),
    )
    parser.add_argument(
        "--rollout-timeout-seconds",
        type=int,
        default=int(os.environ.get("STAGE2_DEPLOY_ROLLOUT_TIMEOUT_SECONDS", "2700")),
        help=(
            "total deployment budget after the template update. The smoke "
            "budget is reserved out of it and the remainder is the window the "
            "live fleet gets to finish draining onto the new version."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a full 40-character lowercase commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", args.template_id):
        raise SystemExit("--template-id is malformed")

    if args.dry_run:
        print(json.dumps(dry_run_report(args.template_id, args.revision), sort_keys=True))
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", args.endpoint_id):
        raise SystemExit("--endpoint-id is malformed")
    if args.smoke_timeout_seconds < 30 or args.smoke_timeout_seconds > 1200:
        raise SystemExit("--smoke-timeout-seconds must be from 30 through 1200")
    if args.rollout_timeout_seconds < 300 or args.rollout_timeout_seconds > 3600:
        raise SystemExit("--rollout-timeout-seconds must be from 300 through 3600")
    if args.rollout_timeout_seconds <= args.smoke_timeout_seconds:
        raise SystemExit(
            "--rollout-timeout-seconds must exceed --smoke-timeout-seconds so the "
            "authoritative smoke keeps a usable budget"
        )

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")
    base = f"{API_ROOT}/templates/{args.template_id}"
    current = request_json("GET", base + "?includeEndpointBoundTemplates=true", api_key)
    endpoint_before = request_json(
        "GET",
        f"{API_ROOT}/endpoints/{args.endpoint_id}?includeWorkers=true",
        api_key,
    )
    previous_endpoint_version = _endpoint_version(endpoint_before)
    # Read before the update: a template that already carries every pin makes
    # the update a no-op, so the endpoint owes no new version and the current
    # one already reflects the target template.
    already_pinned = revision_pins_applied(current, args.revision)
    # This authoritative branch read is intentionally the final network action
    # before the RunPod mutation. An out-of-order older workflow exits cleanly.
    branch_head = github_branch_head(
        os.environ.get("GITHUB_TOKEN", "").strip(),
        deadline=time.monotonic() + 120,
    )
    if branch_head != args.revision:
        print(json.dumps({
            "status": "skipped_stale_revision",
            "template_id": args.template_id,
            "revision": args.revision,
            "current_main_revision": branch_head,
        }, sort_keys=True))
        return
    updated = request_json("POST", base + "/update", api_key, build_payload(current, args.revision))
    validate_payload(updated, args.revision)
    deadline = time.monotonic() + args.rollout_timeout_seconds
    # The smoke is the only direct observation of the deployed revision, so its
    # budget is reserved up front. Sharing one deadline let a slow fleet consume
    # the whole deployment budget and leave the authoritative check no time at
    # all, which turned a draining worker into a failed deployment.
    rollout = wait_for_endpoint_rollout(
        args.endpoint_id,
        previous_endpoint_version,
        api_key,
        deadline=deadline - args.smoke_timeout_seconds,
        expect_version_change=not already_pinned,
    )
    if rollout["status"] == "fleet_not_converged":
        # Not fatal on its own: the template is pinned, so every worker started
        # from here runs the target revision, and the smoke below still has to
        # observe that revision serving traffic before this exits successfully.
        print(
            "[deploy] warning: endpoint fleet did not finish converging "
            f"({rollout['observation']}); "
            f"{rollout['workers_on_target_version']}/{rollout['active_workers']} "
            "live workers on the target version. The post-deploy smoke remains "
            "the authoritative check.",
            file=sys.stderr,
        )
    smoke = run_post_deploy_smoke(
        args.endpoint_id,
        args.revision,
        api_key,
        timeout_seconds=args.smoke_timeout_seconds,
        deadline=min(deadline, time.monotonic() + args.smoke_timeout_seconds),
    )
    print(json.dumps({
        "status": "deployed",
        "template_id": args.template_id,
        "revision": args.revision,
        "pins_match": True,
        "template_already_pinned": already_pinned,
        "rollout": rollout,
        "smoke": smoke,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
