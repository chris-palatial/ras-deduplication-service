"""
RunPod Serverless entry for the RAS Deduplication Service.

Does not reimplement the paper pipeline — stage2_service imports and runs the
public ReplicateAnyScene Stage 2 sequence.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from stage2_service import run_stage2


_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _runtime_code_revision() -> str:
    """Return revision evidence from runtime-owned files, never an env claim."""
    revision_file = Path(
        os.environ.get(
            "STAGE2_BUILD_REVISION_FILE",
            str(Path(__file__).resolve().with_name(".stage2_build_revision")),
        )
    )
    try:
        revision = revision_file.read_text(encoding="utf-8").strip()
        if _REVISION_RE.fullmatch(revision):
            return revision
    except OSError:
        pass

    app_dir = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(app_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        clean = subprocess.run(
            ["git", "-C", str(app_dir), "diff-index", "--quiet", "HEAD", "--"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return ""
    return revision if clean and _REVISION_RE.fullmatch(revision) else ""


def _with_code_revision(result: dict) -> dict:
    """Attach non-secret deployment provenance to every endpoint response."""
    return {
        **result,
        "stage2_code_revision": _runtime_code_revision(),
    }


def handler(job: dict):
    try:
        job_input = job.get("input", job) if isinstance(job, dict) else {}
        if isinstance(job_input, dict) and "input" in job_input and "video_url" not in job_input:
            job_input = job_input["input"]
        if not isinstance(job_input, dict):
            return _with_code_revision({"status": "error", "error": "input must be an object"})
        return _with_code_revision(run_stage2(job_input))
    except Exception as e:
        return _with_code_revision({
            "status": "error",
            "error": str(e),
        })


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-dry-run":
        import json

        out = run_stage2(
            {
                "video_url": "https://download.samplelib.com/mp4/sample-5s.mp4",
                "categories": ["person", "chair"],
                "max_frames": 4,
                "mode": "dry_run",
            }
        )
        print(json.dumps(out, indent=2)[:4000])
        sys.exit(0 if out.get("status") == "ok" else 1)

    import runpod

    runpod.serverless.start({"handler": handler})
