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
| `point_cloud_glb.py` | Exports bounded colored VGGT points + camera frustums as glTF 2.0 GLB |
| `artifact_upload.py` | Uses Agent Lab upload tickets to deliver files to R2 and return receipts |
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

The internal demo currently defaults to `facebook/VGGT-1B`, whose license is
research/non-commercial. Every result reports the exact `geometry.model_id`
and `geometry.license_scope`; do not represent this default as a commercially
licensed output.

**Before any production/commercial release**, obtain checkpoint access and set
`VGGT_MODEL_ID=facebook/VGGT-1B-Commercial` on the endpoint, then run the real
geometry smoke test. The selected repo id is recorded beside newly downloaded
weights. Marker-less existing volumes remain compatible with the original
research checkpoint instead of being needlessly re-downloaded.

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

The thin stock-image bootstrap is fail-closed: `STAGE2_CODE_REV` must be an
explicit 40-character commit SHA and every package/clone/install step must
succeed. It never falls back to a moving branch. When rolling out a revision,
update both the template environment and its bootstrap command to the same SHA
before recycling workers, then verify `dry_run` and `geometry` through the
endpoint boundary.

`scripts/deploy_revision.py` performs that two-field pin in one RunPod template
update. CI runs it after a green push to `main`; the repository
`RUNPOD_API_KEY` secret must be configured. The script verifies the returned
template without printing any environment values or credentials. The complete
test→deploy workflow is serialized per Git ref, so an older slow run cannot
finish after a newer run and roll the endpoint revision backward.

```json
{
  "input": {
    "video_url": "https://…/clip.mp4",
    "categories": ["chair", "table"],
    "max_frames": 24,
    "mode": "geometry",
    "upload": {
      "base": "https://agent-lab-public-upload-origin.example",
      "runId": "stage2-…",
      "token": "<short-lived HMAC ticket>",
      "exp": 1780000000000,
      "policy": {
        "outputs": [
          {"name": "point_cloud.glb", "mediaType": "model/gltf-binary", "maxBytes": 16777216}
        ]
      }
    }
  }
}
```

Agent Lab supplies `upload`; callers should not mint it themselves. A geometry
job must durably receipt `point_cloud.glb`; a full job must durably receipt both
`point_cloud.glb` and `instance_masks.mp4`. Missing or failed required uploads
make the job explicitly fail with `artifact_delivery_failed` instead of
returning a partial `status: ok`. Its JSON contains digest receipts only. Agent
Lab verifies each receipt against R2 before exposing `/media/*` URLs; no signed
PUT URL, file bytes, base64 output, or temporary worker path is returned.
Every failed PUT is checked for an already-stored object and, when a retry is
needed, obtains a fresh short-lived upload grant instead of replaying its old
presigned URL.

Before upload, the full-mode mask visualization is normalized to browser-safe
H.264/yuv420p MP4 with fast-start metadata. Its sampled frames are retimed over
the source clip duration so Agent Lab's original/mask linked playback remains
aligned instead of finishing after ffmpeg's short default image-sequence rate.

The GLB is a visualization-friendly colored **point cloud**, not a watertight or
textured surface mesh. It defaults to at most 300,000 points and enforces a
900,000-point hard cap so the GLB remains below its signed 16 MiB upload
contract. Deployments may tune `STAGE2_POINT_CLOUD_MAX_POINTS` (10,000–900,000) and
`STAGE2_POINT_CLOUD_CONFIDENCE_PERCENTILE` (0–100).
The exporter also removes only the catastrophic tail beyond a generous robust
scene radius and ignores implausibly distant camera poses, preventing one bad
VGGT value from making the useful room appear microscopic in the viewer.

Large compatibility outputs are disabled by default. Set
`STAGE2_EXPORT_DEBUG_ARTIFACTS=1` only when the signed policy allows
`point_cloud.ply` and `camera_intrinsics.json`. The Python uploader hashes and
uploads one whole file at a time, and an artifact gateway without
`R2_S3_ACCOUNT_ID`, `R2_S3_ACCESS_KEY_ID`, and `R2_S3_SECRET_ACCESS_KEY` must
proxy those bytes through the Worker. Configure those S3 credentials for
direct-to-R2 grants before enabling large debug artifacts in a deployment.

## Agent Lab

Point `STAGE2_ENDPOINT_ID` at this endpoint. Lab already async-polls `/status`.
Agent Lab accepts clips up to 6 MiB until a public object-store handoff is configured.
