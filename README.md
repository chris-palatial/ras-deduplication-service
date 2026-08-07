# ReplicateAnyScene Stage 2 service

Thin RunPod / HTTP wrapper around the **public paper repo** Stage 2 only.

We do **not** reimplement VGGT, SAM3, or spatial dedup here.
Those live in:

- https://github.com/xiac20/ReplicateAnyScene (`main.py` Stage 2 block + `src/*`)
- submodules `vggt` + `sam3` (facebookresearch)

## What we add

| File | Role |
| --- | --- |
| `stage2_service.py` | Calls RAS Stage 2 sequence; model-free and VGGT-only preflight modes |
| `handler.py` | RunPod Serverless entry |
| `scripts/download_weights.sh` | One-time HF weight download (build or volume) |
| `Dockerfile` | Installs RAS + submodules + wrapper |

## Modes

- `dry_run` — download video + sample frames; no model weights
- `geometry` — run real VGGT geometry; deliberately skip SAM3 and dedup
- `full` — exact RAS Stage 2 path (stops before Stage 3 mesh generation)

## Weights

Same as their README (not in git):

```bash
export STAGE2_MODELS_DIR=/models   # or /workspace/models on RunPod volumes
bash scripts/download_weights.sh
```

While SAM3 access is pending, pre-seed only the public VGGT weights:

```bash
STAGE2_DOWNLOAD_SAM3=0 bash scripts/download_weights.sh
```

The reviewed source revisions are pinned in the wrapper and Dockerfile. The
official SAM3 checkpoint is gated by Meta on Hugging Face; the service does not
fall back to unofficial mirrors or bypass its license approval.

## RunPod

Prefer a **prebuilt image** or **volume with weights already present**.
Do not reinstall the world on every cold start.

```json
{
  "input": {
    "video_url": "https://…/clip.mp4",
    "categories": ["chair", "table"],
    "max_frames": 24,
    "mode": "full"
  }
}
```

## Agent Lab

Point `STAGE2_ENDPOINT_ID` at this endpoint. Lab already async-polls `/status`.
Agent Lab accepts clips up to 6 MiB until a public object-store handoff is configured.
