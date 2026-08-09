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
licensed output. Cache misses for this public checkpoint download anonymously,
so pending SAM3 approval or a missing/revoked Hugging Face token cannot block
geometry mode. SAM3 and the Commercial VGGT checkpoint still require an
explicit approved `HF_TOKEN`.

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
succeed. It never falls back to a moving branch. Each commit uses a separate,
marked runtime checkout and virtual environment, and its pinned VGGT/SAM3 source
paths take precedence over base-image packages. A bounded number of recent
runtimes is retained so rolling workers do not share mutable Python code. Every
worker holds a shared runtime lease until it exits; cleanup requires an exclusive
lease and a minimum age greater than `STAGE2_MAX_EXECUTION_SECONDS` (2100 seconds
by default), so active or recently used runtimes are never removed. Legacy
runtimes without lease metadata are left for manual inspection.

Lazy RAS/VGGT/SAM3 source installation and shared checkpoint downloads use one
cross-worker file lock under `STAGE2_MODELS_DIR`. Both the wrapper checkout and
the upstream source checkouts verify their commit and tracked-file cleanliness
before reuse; a dirty runtime-owned checkout is rebuilt.

Prebuilt images must pass
`--build-arg STAGE2_BUILD_REVISION="$(git rev-parse HEAD)"`. The handler reports
only that baked marker or the actual runtime Git checkout; it never treats an
environment variable alone as revision proof. If weights are baked, pass the
Hugging Face credential as a BuildKit secret (`--secret
id=hf_token,env=HF_TOKEN`), never as a build argument.

`scripts/deploy_revision.py` performs that two-field pin in one RunPod template
update. CI runs it after a green push to `main`; the repository
`RUNPOD_API_KEY` secret must be configured. The script verifies the returned
template without printing any environment values or credentials. The complete
test→deploy workflow is serialized per Git ref, and immediately before the
RunPod mutation the deploy script checks the authoritative GitHub `main` head.
An out-of-order stale workflow exits without changing the endpoint.
After a successful template update, deployment waits for the endpoint version to
advance and for every live worker reported by RunPod to carry that version. It
then submits a bounded real `dry_run` to endpoint `sp2oyuum48vk0j` (override with
`STAGE2_ENDPOINT_ID`), polls its async status, and requires the response's
`stage2_code_revision` to equal the deployed commit. Fleet convergence has a
2,700-second release budget (longer than the 2,100-second worker execution
window), while dry smoke is independently capped at 900 seconds and must also
fit inside the remaining release budget. Transient safe reads are retried, but
template updates and paid job submissions are never replayed. CI fails without
printing request credentials or endpoint output when verification does not pass.

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
Every failed PUT is checked for an already-stored object first. When a retry is
safe, the uploader obtains a fresh short-lived grant instead of replaying its
old presigned URL; explicit 4xx/409 rejections are verified once but never
retried when the object is absent. Results and worker logs retain only the
failure phase, HTTP status, retryability, and attempt count, never a ticket or
signed URL.

Before upload, the full-mode mask visualization is normalized to browser-safe
H.264/yuv420p MP4 with fast-start metadata. Each sampled mask frame keeps the
presentation timestamp of its corresponding source frame, including
variable-frame-rate clips, so Agent Lab's original/mask linked playback remains
aligned instead of merely sharing the same total duration.

The GLB is a visualization-friendly colored **point cloud**, not a watertight or
textured surface mesh. It defaults to at most 300,000 points and enforces a
900,000-point hard cap so the GLB remains below its signed 16 MiB upload
contract. Deployments may tune `STAGE2_POINT_CLOUD_MAX_POINTS` (10,000–900,000).
The preview consumes RAS's point cloud after its fixed 50th-percentile VGGT depth
confidence cutoff. Keeping that same point/confidence branch avoids mixing
depth-derived coordinates with point-map confidence. An unavailable or empty
filtered cloud fails the required preview artifact instead of publishing an
unfiltered or synthetic one-point result. The exporter then removes
only the catastrophic tail beyond a generous robust scene radius and ignores
implausibly distant camera poses. Geometry-mode files are aligned from VGGT's
OpenCV coordinates into first-camera, glTF Y-up preview space; room-aligned full
mode is converted from RAS Z-up into glTF Y-up only when RAS actually found a
usable floor and wall. The result reports room alignment as requested versus
applied; invalid or absent camera poses use a truthfully labeled axis-only
fallback instead of claiming first-camera alignment. GLB metadata includes
robust preview bounds, node roles, and whether confidence/camera alignment was
applied. Source-frame sRGB colors are encoded with glTF's linear vertex-color
semantics. Together these rules keep the useful room upright, correctly colored,
and framed independently of camera helpers.

Large compatibility outputs are disabled by default. Set
`STAGE2_EXPORT_DEBUG_ARTIFACTS=1` only when the signed policy allows
`point_cloud.ply` and `camera_intrinsics.json`. The Python uploader hashes and
uploads one whole file at a time, and an artifact gateway without
`R2_S3_ACCOUNT_ID`, `R2_S3_ACCESS_KEY_ID`, and `R2_S3_SECRET_ACCESS_KEY` must
proxy those bytes through the Worker. Configure those S3 credentials for
direct-to-R2 grants before enabling large debug artifacts in a deployment.

## Agent Lab

Point `STAGE2_ENDPOINT_ID` at this endpoint. Lab already async-polls `/status`.
Agent Lab passes uploaded clips through an expiring, exact-object signed HTTPS
`video_url`. The worker streams that response to disk and independently enforces
a 64 MiB limit using both `Content-Length` (when present) and the bytes actually
received. Legacy callers may continue to use `video_b64`, which remains limited
to 6 MiB; larger clips must use `video_url`.
