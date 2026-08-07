# ReplicateAnyScene Stage 2 service

Thin RunPod / HTTP wrapper around the **public paper repo** Stage 2 only.

We do **not** reimplement VGGT, SAM3, or spatial dedup here.
Those live in:

- https://github.com/xiac20/ReplicateAnyScene (`main.py` Stage 2 block + `src/*`)
- submodules `vggt` + `sam3` (facebookresearch)

## What we add

| File | Role |
| --- | --- |
| `stage2_service.py` | Calls RAS Stage 2 sequence; `dry_run` for wire tests |
| `handler.py` | RunPod Serverless entry |
| `scripts/download_weights.sh` | One-time HF weight download (build or volume) |
| `Dockerfile` | Installs RAS + submodules + wrapper |

## Modes

- `dry_run` — download video + sample frames; no model weights
- `full` — exact RAS Stage 2 path (stops before Stage 3 mesh generation)

## Weights

Same as their README (not in git):

```bash
export STAGE2_MODELS_DIR=/models   # or /workspace/models on RunPod volumes
bash scripts/download_weights.sh
```

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
