"""
RunPod Serverless entry for ReplicateAnyScene Stage 2 only.

Does not reimplement the paper pipeline — stage2_service imports and runs the
public ReplicateAnyScene Stage 2 sequence.
"""

from __future__ import annotations

import os
from stage2_service import run_stage2


def _with_code_revision(result: dict) -> dict:
    """Attach non-secret deployment provenance to every endpoint response."""
    return {
        **result,
        "stage2_code_revision": os.environ.get("STAGE2_CODE_REV", ""),
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
