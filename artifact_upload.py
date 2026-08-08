"""RunPod-to-R2 artifact delivery using Agent Lab's upload-ticket contract.

The Worker mints a per-run, expiring ticket.  This module hashes each finished
file, asks the Worker for a one-object upload grant, PUTs the bytes directly to
R2 (or the Worker's streaming local-dev proxy), and returns only a receipt.
Artifact bytes and signed upload URLs never enter the RunPod result JSON.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


ATTEMPTS = 3
LOST_PUT_VERIFY_ATTEMPTS = 5
LOST_PUT_VERIFY_WINDOW_SECONDS = 15


def _safe_error(exc: Exception | None) -> str:
    """Describe transport failures without echoing a signed URL or ticket."""
    if exc is None:
        return "unknown error"
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"{type(exc).__name__} ({type(exc.reason).__name__})"
    return type(exc).__name__


def digests(data: bytes) -> tuple[str, str]:
    return hashlib.sha256(data).hexdigest(), hashlib.md5(data, usedforsecurity=False).hexdigest()


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("artifact service returned a non-object response")
    return value


def _retrying(what: str, fn: Callable[[], dict[str, Any]], attempts: int = ATTEMPTS) -> dict[str, Any]:
    last: Exception | None = None
    for index in range(attempts):
        try:
            return fn()
        except Exception as exc:  # any transport failure is retryable at this boundary
            last = exc
            if index + 1 < attempts:
                time.sleep(min(2**index, 5))
    raise RuntimeError(f"{what} failed after {attempts} attempts: {_safe_error(last)}")


def _ticket_payload(
    upload: dict[str, Any],
    name: str,
    sha256: str,
    md5: str,
    size: int,
    media_type: str,
) -> dict[str, Any]:
    base = str(upload.get("base") or "").rstrip("/")
    run_id = str(upload.get("runId") or "")
    token = str(upload.get("token") or "")
    try:
        exp = int(upload.get("exp"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("artifact upload ticket has an invalid expiry") from exc
    if not (base.startswith("https://") or base.startswith("http://")):
        raise RuntimeError("artifact upload ticket has an invalid base URL")
    if not run_id or not token:
        raise RuntimeError("artifact upload ticket is missing runId or token")
    if exp <= int(time.time() * 1000):
        raise RuntimeError("artifact upload ticket has expired")
    return {
        "v": "2",
        "runId": run_id,
        "exp": exp,
        "token": token,
        # The policy is part of the ticket HMAC. Forward the exact object the
        # edge minted so grant verification remains bound to its output allowlist.
        "policy": upload.get("policy"),
        "name": name,
        "sha256": sha256,
        "md5": md5,
        "bytes": size,
        "mediaType": media_type,
    }


def _stored_already(
    upload: dict[str, Any],
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> bool | None:
    """Return stored/absent, or None when verification itself is unavailable."""
    try:
        verdict = _post_json(
            str(upload["base"]).rstrip("/") + "/api/jobs/upload-verify",
            payload,
            headers,
            timeout,
        )
        stored = verdict.get("stored")
        return stored if isinstance(stored, bool) else None
    except Exception:
        return None


def _resolve_ambiguous_put(
    upload: dict[str, Any],
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> bool | None:
    """Retry an unavailable verifier without replaying a possibly successful PUT."""
    now = time.time()
    ticket_deadline = int(payload["exp"]) / 1000
    deadline = min(ticket_deadline, now + LOST_PUT_VERIFY_WINDOW_SECONDS)
    for index in range(LOST_PUT_VERIFY_ATTEMPTS):
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        verify_timeout = max(0.1, min(float(timeout), remaining))
        verdict = _stored_already(upload, payload, headers, verify_timeout)
        if verdict is not None:
            return verdict
        if index + 1 < LOST_PUT_VERIFY_ATTEMPTS:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(2**index, 5, remaining))
    return None


def upload_artifact_file(
    upload: dict[str, Any],
    path: str | Path,
    media_type: str,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    """Upload one file and return a digest receipt suitable for result JSON."""
    file_path = Path(path)
    data = file_path.read_bytes()
    if not data:
        raise RuntimeError(f"refusing to upload empty artifact {file_path.name}")
    sha256, md5 = digests(data)
    payload = _ticket_payload(upload, file_path.name, sha256, md5, len(data), media_type)
    extra_headers = upload.get("headers") or {}
    if not isinstance(extra_headers, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in extra_headers.items()
    ):
        raise RuntimeError("artifact upload ticket headers must be strings")

    base = str(upload["base"]).rstrip("/")
    receipt_base = {
        "name": file_path.name,
        "sha256": sha256,
        "md5": md5,
        "bytes": len(data),
        "mediaType": media_type,
    }

    expected_key: str | None = None

    def fresh_grant() -> tuple[str, str, dict[str, str]]:
        if int(payload["exp"]) <= time.time() * 1000:
            raise RuntimeError("artifact upload ticket expired during upload")
        grant = _retrying(
            "upload grant",
            lambda: _post_json(base + "/api/jobs/upload-grant", payload, extra_headers, timeout),
        )
        version = str(grant.get("v") or "2")
        if version not in {"1", "2"}:
            raise RuntimeError(f"upload grant answered with unsupported protocol v{version}")
        key = grant.get("key")
        url = grant.get("url")
        if not isinstance(key, str) or not key or not isinstance(url, str) or not (
            url.startswith("https://") or url.startswith("http://")
        ):
            raise RuntimeError("upload grant is missing a valid key or HTTP URL")
        headers = grant.get("headers") or {}
        if not isinstance(headers, dict) or not all(
            isinstance(header, str) and isinstance(value, str)
            for header, value in headers.items()
        ):
            raise RuntimeError("upload grant headers must be strings")
        return key, url, headers

    def put(url: str, headers: dict[str, str]) -> None:
        req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read()

    last: Exception | None = None
    for index in range(ATTEMPTS):
        # A presigned PUT URL can be single-use or expire while a retry waits.
        # Ask for a fresh grant before every attempt instead of replaying a
        # stale URL. The content-addressed key must remain stable.
        key, url, headers = fresh_grant()
        if expected_key is None:
            expected_key = key
        elif key != expected_key:
            raise RuntimeError("upload grant key changed across retries")
        receipt = {"key": key, **receipt_base}
        try:
            put(url, headers)
            return receipt
        except Exception as exc:
            last = exc
            stored = _resolve_ambiguous_put(upload, payload, extra_headers, timeout)
            if stored is True:
                return receipt
            if stored is None:
                raise RuntimeError(
                    "artifact PUT outcome remained unverifiable; refusing to replay a possibly successful upload"
                ) from exc
            if index + 1 < ATTEMPTS:
                time.sleep(min(2**index, 5))
    raise RuntimeError(f"artifact PUT failed after {ATTEMPTS} attempts: {_safe_error(last)}")
