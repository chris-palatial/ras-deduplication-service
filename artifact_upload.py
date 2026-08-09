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

PHASE_LABELS = {
    "ticket_validation": "artifact upload ticket validation",
    "upload_grant": "artifact upload authorization",
    "artifact_put": "artifact storage write",
    "upload_verify": "artifact storage verification",
}


class ArtifactUploadError(RuntimeError):
    """Sanitized upload failure safe for logs and provider result JSON."""

    def __init__(
        self,
        phase: str,
        *,
        http_status: int | None = None,
        retryable: bool = False,
        attempts: int = 1,
        cause_type: str = "RuntimeError",
    ) -> None:
        self.phase = phase
        self.http_status = http_status
        self.retryable = retryable
        self.attempts = max(1, int(attempts))
        self.cause_type = cause_type if cause_type.isidentifier() else "Exception"
        super().__init__(self.safe_detail())

    def safe_detail(self) -> str:
        label = PHASE_LABELS.get(self.phase, "artifact upload")
        if self.http_status is not None:
            return f"{label} returned HTTP {self.http_status}"
        return f"{label} failed ({self.cause_type})"

    def with_attempts(self, attempts: int) -> "ArtifactUploadError":
        return ArtifactUploadError(
            self.phase,
            http_status=self.http_status,
            retryable=self.retryable,
            attempts=attempts,
            cause_type=self.cause_type,
        )

    def failure_record(self, name: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": name,
            "code": "artifact_upload_failed",
            "phase": self.phase,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "detail": self.safe_detail(),
        }
        if self.http_status is not None:
            record["http_status"] = self.http_status
        return record


def _upload_error(phase: str, exc: Exception, attempts: int = 1) -> ArtifactUploadError:
    """Classify retryability without copying URLs, response bodies, or tickets."""
    if isinstance(exc, ArtifactUploadError):
        return exc.with_attempts(max(attempts, exc.attempts))
    if isinstance(exc, urllib.error.HTTPError):
        status = int(exc.code)
        retryable = status in {408, 425, 429} or status >= 500
        return ArtifactUploadError(
            phase,
            http_status=status,
            retryable=retryable,
            attempts=attempts,
            cause_type=type(exc).__name__,
        )
    retryable = isinstance(
        exc,
        (urllib.error.URLError, TimeoutError, ConnectionError, OSError),
    )
    return ArtifactUploadError(
        phase,
        retryable=retryable,
        attempts=attempts,
        cause_type=type(exc).__name__,
    )


def artifact_failure_record(name: str, exc: Exception) -> dict[str, Any]:
    """Build one bounded result/log entry without arbitrary exception strings."""
    if isinstance(exc, ArtifactUploadError):
        return exc.failure_record(name)
    return {
        "name": name,
        "code": "artifact_upload_failed",
        "retryable": False,
        "attempts": 1,
        "detail": f"artifact upload failed ({type(exc).__name__})",
    }


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


def _retrying(phase: str, fn: Callable[[], dict[str, Any]], attempts: int = ATTEMPTS) -> dict[str, Any]:
    for index in range(attempts):
        try:
            return fn()
        except Exception as exc:
            failure = _upload_error(phase, exc, index + 1)
            if not failure.retryable or index + 1 >= attempts:
                raise failure from exc
            time.sleep(min(2**index, 5))
    raise AssertionError("retry loop exited without a result")


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
        raise ArtifactUploadError("ticket_validation", cause_type="InvalidExpiry") from exc
    if not (base.startswith("https://") or base.startswith("http://")):
        raise ArtifactUploadError("ticket_validation", cause_type="InvalidBaseUrl")
    if not run_id or not token:
        raise ArtifactUploadError("ticket_validation", cause_type="MissingCredential")
    if exp <= int(time.time() * 1000):
        raise ArtifactUploadError("ticket_validation", cause_type="ExpiredTicket")
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
        raise ArtifactUploadError("ticket_validation", cause_type="EmptyArtifact")
    sha256, md5 = digests(data)
    payload = _ticket_payload(upload, file_path.name, sha256, md5, len(data), media_type)
    extra_headers = upload.get("headers") or {}
    if not isinstance(extra_headers, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in extra_headers.items()
    ):
        raise ArtifactUploadError("ticket_validation", cause_type="InvalidHeaders")

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
            raise ArtifactUploadError("ticket_validation", cause_type="ExpiredTicket")
        grant = _retrying(
            "upload_grant",
            lambda: _post_json(base + "/api/jobs/upload-grant", payload, extra_headers, timeout),
        )
        version = str(grant.get("v") or "2")
        if version not in {"1", "2"}:
            raise ArtifactUploadError("upload_grant", cause_type="UnsupportedProtocol")
        key = grant.get("key")
        url = grant.get("url")
        if not isinstance(key, str) or not key or not isinstance(url, str) or not (
            url.startswith("https://") or url.startswith("http://")
        ):
            raise ArtifactUploadError("upload_grant", cause_type="InvalidGrant")
        headers = grant.get("headers") or {}
        if not isinstance(headers, dict) or not all(
            isinstance(header, str) and isinstance(value, str)
            for header, value in headers.items()
        ):
            raise ArtifactUploadError("upload_grant", cause_type="InvalidHeaders")
        return key, url, headers

    def put(url: str, headers: dict[str, str]) -> None:
        req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read()

    last: ArtifactUploadError | None = None
    for index in range(ATTEMPTS):
        # A presigned PUT URL can be single-use or expire while a retry waits.
        # Ask for a fresh grant before every attempt instead of replaying a
        # stale URL. The content-addressed key must remain stable.
        key, url, headers = fresh_grant()
        if expected_key is None:
            expected_key = key
        elif key != expected_key:
            raise ArtifactUploadError("upload_grant", cause_type="GrantKeyChanged")
        receipt = {"key": key, **receipt_base}
        try:
            put(url, headers)
            return receipt
        except Exception as exc:
            failure = _upload_error("artifact_put", exc, index + 1)
            last = failure
            # Even a deterministic HTTP response can follow an internal store
            # exception after the object landed. Confirm the durable state
            # before deciding whether this attempt failed.
            stored = (
                _resolve_ambiguous_put(upload, payload, extra_headers, timeout)
                if failure.retryable
                else _stored_already(upload, payload, extra_headers, timeout)
            )
            if stored is True:
                return receipt
            if not failure.retryable:
                raise failure from exc
            if stored is None:
                raise ArtifactUploadError(
                    "upload_verify",
                    retryable=True,
                    attempts=LOST_PUT_VERIFY_ATTEMPTS,
                    cause_type="VerificationUnavailable",
                ) from exc
            if index + 1 < ATTEMPTS:
                time.sleep(min(2**index, 5))
    if last is not None:
        raise last
    raise AssertionError("artifact PUT loop exited without a result")
