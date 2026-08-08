#!/usr/bin/env python3
"""Atomically pin the RunPod template environment and bootstrap to one commit."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request


API_ROOT = "https://rest.runpod.io/v1"
DEFAULT_TEMPLATE_ID = "inapyg0va0"
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
        "rm -rf /workspace/boot; "
        "git clone --filter=blob:none --no-checkout "
        "https://github.com/chris-palatial/ras-stage2-service.git /workspace/boot; "
        f"git -C /workspace/boot fetch --depth 1 origin {revision}; "
        "git -C /workspace/boot checkout --detach FETCH_HEAD; "
        f"export STAGE2_CODE_REV={revision}; "
        "exec bash /workspace/boot/scripts/start_serverless.sh"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a full 40-character lowercase commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", args.template_id):
        raise SystemExit("--template-id is malformed")

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "template_id": args.template_id,
            "revision": args.revision,
            "atomic_fields": ["env.STAGE2_CODE_REV", "dockerStartCmd"],
        }, sort_keys=True))
        return

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")
    base = f"{API_ROOT}/templates/{args.template_id}"
    current = request_json("GET", base + "?includeEndpointBoundTemplates=true", api_key)
    updated = request_json("POST", base + "/update", api_key, build_payload(current, args.revision))
    actual_env = updated.get("env") or {}
    actual_cmd = updated.get("dockerStartCmd") or []
    if actual_env.get("STAGE2_CODE_REV") != args.revision or len(actual_cmd) != 1 or args.revision not in actual_cmd[0]:
        raise RuntimeError("RunPod accepted the update but the revision pins do not match")
    print(json.dumps({
        "status": "deployed",
        "template_id": args.template_id,
        "revision": args.revision,
        "pins_match": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
