#!/usr/bin/env python3
"""Safely prune old revision runtimes while respecting live worker leases."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import time
from pathlib import Path


REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _last_used(runtime: Path) -> float:
    marker = runtime / ".last_used"
    try:
        return marker.stat().st_mtime
    except OSError:
        return runtime.stat().st_mtime


def prune_runtimes(
    root: Path,
    current_revision: str,
    *,
    keep: int,
    min_age_seconds: int,
    now: float | None = None,
) -> dict[str, int]:
    root = root.resolve()
    if root == Path("/") or not root.is_dir():
        raise ValueError("runtime root must be an existing non-root directory")
    if not REVISION_RE.fullmatch(current_revision):
        raise ValueError("current revision must be a full lowercase commit SHA")
    if keep < 2 or min_age_seconds <= 0:
        raise ValueError("keep must be at least 2 and minimum age must be positive")

    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and not child.is_symlink()
        and REVISION_RE.fullmatch(child.name)
    ]
    candidates.sort(key=_last_used, reverse=True)
    protected = {current_revision, *(path.name for path in candidates[:keep])}
    current_time = time.time() if now is None else now
    result = {
        "removed": 0,
        "kept_active": 0,
        "kept_recent": 0,
        "kept_newest": 0,
        "kept_unmanaged": 0,
    }

    for runtime in candidates:
        if runtime.name in protected:
            result["kept_newest"] += 1
            continue
        last_used_path = runtime / ".last_used"
        lease_path = runtime / ".active_lease"
        # Pre-lease runtimes may belong to workers started by an older script.
        # Their activity cannot be proved, so automated cleanup leaves them.
        if not last_used_path.is_file() or not lease_path.is_file():
            result["kept_unmanaged"] += 1
            continue
        if current_time - _last_used(runtime) < min_age_seconds:
            result["kept_recent"] += 1
            continue

        lease_fd = os.open(lease_path, os.O_RDWR)
        try:
            try:
                fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result["kept_active"] += 1
                continue
            # Re-check the destructive boundary after acquiring the lease.
            if runtime.parent.resolve() != root or not REVISION_RE.fullmatch(runtime.name):
                raise RuntimeError("runtime cleanup target escaped its validated root")
            shutil.rmtree(runtime)
            result["removed"] += 1
        finally:
            os.close(lease_fd)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--current-revision", required=True)
    parser.add_argument("--keep", type=int, required=True)
    parser.add_argument("--min-age-seconds", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prune_runtimes(
                args.root,
                args.current_revision,
                keep=args.keep,
                min_age_seconds=args.min_age_seconds,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
