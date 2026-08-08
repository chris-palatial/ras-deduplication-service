#!/usr/bin/env python3
"""Atomically pin the RunPod template environment and bootstrap to one commit."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from typing import Any


API_ROOT = "https://rest.runpod.io/v1"
GITHUB_API_ROOT = "https://api.github.com"
GITHUB_REPOSITORY = "chris-palatial/ras-stage2-service"
GITHUB_BRANCH = "main"
DEFAULT_TEMPLATE_ID = "inapyg0va0"
DEFAULT_ENDPOINT_ID = "sp2oyuum48vk0j"
INVOKE_API_ROOT = "https://api.runpod.ai/v2"
SMOKE_MAX_ATTEMPTS = 6
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
        "BOOT_DIR=$(mktemp -d /tmp/ras-stage2-boot.XXXXXX); "
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/chris-palatial/ras-stage2-service.git \"$BOOT_DIR\"; "
        f"git -C \"$BOOT_DIR\" fetch --depth 1 origin {revision}; "
        "git -C \"$BOOT_DIR\" checkout --detach FETCH_HEAD; "
        f"export STAGE2_CODE_REV={revision}; "
        "exec bash \"$BOOT_DIR/scripts/start_serverless.sh\""
    )


def request_json(method: str, url: str, api_key: str, payload: dict | None = None) -> dict:
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
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("RunPod template API returned a non-object response")
    return result


def invoke_json(
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
) -> dict[str, Any]:
    """Run bounded async dry jobs until the rolling endpoint serves revision."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", endpoint_id):
        raise RuntimeError("RunPod endpoint id is malformed")
    if timeout_seconds < 30 or timeout_seconds > 1200:
        raise RuntimeError("smoke timeout must be from 30 through 1200 seconds")

    deadline = time.monotonic() + timeout_seconds
    invoke_root = invoke_api_root()
    last_observation = "no_completed_job"
    for attempt in range(1, SMOKE_MAX_ATTEMPTS + 1):
        started = invoke_json(
            "POST",
            f"{invoke_root}/{endpoint_id}/run",
            api_key,
            {
                "input": {
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
            )
            state = str(status_result.get("status") or "").upper()
            if state == "COMPLETED":
                break
            if state in {"FAILED", "TIMED_OUT", "CANCELLED"}:
                raise RuntimeError(f"RunPod post-deploy smoke ended with state {state}")
            time.sleep(min(5.0, remaining))

        output = status_result.get("output")
        if not isinstance(output, dict):
            last_observation = "non_object_output"
        elif output.get("stage2_code_revision") != revision:
            # A rolling release may briefly route to an old warm worker. Retry a
            # bounded number of fresh jobs without exposing the old response.
            last_observation = "stale_worker_revision"
        elif output.get("status") != "ok" or output.get("mode") != "dry_run":
            raise RuntimeError("RunPod post-deploy dry_run returned a non-ok result")
        else:
            return {
                "status": "passed",
                "attempts": attempt,
                "revision": revision,
            }

        remaining = deadline - time.monotonic()
        if attempt < SMOKE_MAX_ATTEMPTS and remaining > 0:
            time.sleep(min(10.0, remaining))

    raise RuntimeError(
        f"RunPod post-deploy smoke did not observe the deployed revision ({last_observation})"
    )


def github_branch_head(token: str = "") -> str:
    headers = {
        "accept": "application/vnd.github+json",
        "user-agent": "ras-stage2-deployer",
        "x-github-api-version": "2022-11-28",
    }
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{GITHUB_API_ROOT}/repos/{GITHUB_REPOSITORY}/git/ref/heads/{GITHUB_BRANCH}",
        headers=headers,
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
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


def validate_payload(payload: dict[str, Any], revision: str) -> dict[str, bool]:
    env = payload.get("env")
    entrypoint = payload.get("dockerEntrypoint")
    command = payload.get("dockerStartCmd")
    assertions = {
        "environment_pin": isinstance(env, dict) and env.get("STAGE2_CODE_REV") == revision,
        "entrypoint_pin": entrypoint == ["bash", "-lc"],
        "bootstrap_pin": command == [bootstrap_command(revision)],
    }
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

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")
    base = f"{API_ROOT}/templates/{args.template_id}"
    current = request_json("GET", base + "?includeEndpointBoundTemplates=true", api_key)
    # This authoritative branch read is intentionally the final network action
    # before the RunPod mutation. An out-of-order older workflow exits cleanly.
    branch_head = github_branch_head(os.environ.get("GITHUB_TOKEN", "").strip())
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
    smoke = run_post_deploy_smoke(
        args.endpoint_id,
        args.revision,
        api_key,
        timeout_seconds=args.smoke_timeout_seconds,
    )
    print(json.dumps({
        "status": "deployed",
        "template_id": args.template_id,
        "revision": args.revision,
        "pins_match": True,
        "smoke": smoke,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
