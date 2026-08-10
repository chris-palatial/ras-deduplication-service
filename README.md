# RAS Deduplication Service

RunPod worker for the Agent Lab deduplication research page. It wraps pinned
public ReplicateAnyScene (RAS) geometry and spatial-deduplication code, local
VGGT geometry, and Meta's hosted VGGT-Omega demo.

This service is **Agent-Lab-only**. It is not a Scene Parse production service.
The retired `scene_parse_catalog_v1` profile and the old monolithic
`dedup_ras_vggt_sam3` profile fail closed at the request router.

## Runtime ownership

The worker verifies exact revisions of:

- [ReplicateAnyScene](https://github.com/xiac20/ReplicateAnyScene) for geometry
  utilities and the original self/cross-category 3D deduplication functions.
- [VGGT](https://github.com/facebookresearch/vggt) for local geometry.

They are pinned runtime dependencies, not Git submodules. The image does not
clone, install, load, or download SAM. Agent Lab runs SAM through fal and hands
the resulting object tracks to this worker.

Existing `STAGE2_*` environment names and `stage2_code_revision` remain
compatibility contracts even though the product name is Deduplication.

## Supported analysis types

| Analysis type | Mode | Result |
| --- | --- | --- |
| `validation_v1` | `dry_run` | Input plus pinned RAS/VGGT source validation; no weights or inference |
| `validation_object_catalog_transport_v1` | `dry_run` | Privileged synthetic R2 artifact transport canary |
| `geometry_vggt_1b` | `geometry` | Local VGGT point-cloud GLB |
| `geometry_vggt_omega_1b` | `geometry` | Hosted VGGT-Omega point-cloud GLB |
| `best_view_geometry_v1` | `geometry` | Exact model-frame clip, reusable geometry inputs, and point-cloud GLB |
| `best_view_score_v1` | `full` | Weights-free best-view scoring over external fal masks |
| `dedup_ras_finalize_v1` | `full` | Weights-free RAS 3D identity dedup over external fal object tracks |

`mask_sam3` and `mask_sam31` belong to the Agent Lab fal adapters and are
rejected here. The same is true of the retired local-SAM Full RAS and Scene
Parse analysis IDs; the worker never silently falls back to Hugging Face.

## Full 3D deduplication

Full RAS is a bounded three-leg composite:

1. RunPod `best_view_geometry_v1` samples the exact VGGT model frames and
   returns `geometry_inputs.npz`, `sampled_clip.mp4`, and `point_cloud.glb`.
2. Agent Lab calls `fal-ai/sam-3/video-rle-objects` once per requested category
   on that sampled clip and stores each exact JSON result durably.
3. RunPod `dedup_ras_finalize_v1` reconstructs the VGGT point maps, validates
   and decodes the fal tracks, and runs the original RAS self/cross-category
   spatial deduplication.

Leg C does not regenerate, require, or upload a GLB. Agent Lab merges the
already-verified leg-A `point_cloud.glb` with these required leg-C artifacts:

- `instance_masks.mp4`
- `object_catalog.json`
- `object_crops.jpg`

Room wall/floor segmentation is intentionally skipped. RAS overlap-based
identity matching is invariant to a rigid scene transform, so the finalizer
reports `coordinate_system: vggt_first_camera`,
`room_alignment_applied: false`, and
`room_alignment_reason: semantic_room_alignment_skipped_external_masks`.

### Finalizer request

```json
{
  "analysis_type": "dedup_ras_finalize_v1",
  "mode": "full",
  "geometry_inputs_url": "https://artifact-gateway.example/inputs/geometry_inputs.npz",
  "sampled_clip_url": "https://artifact-gateway.example/inputs/sampled_clip.mp4",
  "expected_frames_used": 24,
  "expected_model_frame_width": 518,
  "expected_model_frame_height": 294,
  "expected_geometry_model_id": "facebook/VGGT-1B",
  "expected_geometry_source_revision": "9e4fa662a8893ed348d048e8b57816c12593448b",
  "categories": ["chair", "table"],
  "sam_model": "fal-ai/sam-3/video-rle-objects",
  "room_align": false,
  "object_catalog_version": 1,
  "masks": [
    {
      "category": "chair",
      "request_id": "fal-queue-request-id",
      "url": "https://artifact-gateway.example/inputs/chair.json",
      "sha256": "<64 lowercase hex characters>",
      "bytes": 12345
    },
    {
      "category": "table",
      "request_id": "fal-queue-request-id",
      "url": "https://artifact-gateway.example/inputs/table.json",
      "sha256": "<64 lowercase hex characters>",
      "bytes": 12345
    }
  ],
  "upload": {
    "base": "https://artifact-gateway.example",
    "runId": "dedup-…",
    "token": "<short-lived ticket>"
  }
}
```

Each downloaded fal result must be the official normalized object document:

```json
{
  "width": 518,
  "height": 294,
  "num_frames": 24,
  "frames": [
    {
      "frame_index": 0,
      "objects": [{"track_id": 1, "rle": "0 12 4 9 …"}]
    }
  ]
}
```

The finalizer fails closed unless:

- categories and mask documents are one-to-one and in exact request order;
- every URL uses the upload ticket's credential-free gateway origin;
- document byte count and SHA-256 match their receipts;
- width, height, frame count, and contiguous frame indices exactly match the
  sampled clip and geometry inputs;
- every decoded RLE contains exactly `width * height` binary pixels;
- track IDs are non-negative integers and unique within a frame.

Track identity is scoped to `(category index, track_id)`, because fal track IDs
are local to one category request. Missing visibility frames split a track into
separate RAS track segments. Masks are never resized, truncated, or shifted.

The official fal endpoint currently providing separable object tracks is SAM 3
`video-rle-objects`. SAM 3.1 remains available as an Agent Lab mask preview,
but it is not labeled as Full RAS until fal publishes an equivalent object-RLE
contract.

## Geometry and weights

Local geometry accepts a bounded video, uniformly samples frames across
the full decoded timeline, and exports a visualization-oriented point cloud,
not a watertight mesh. Geometry leg A also exports float16 depth and confidence,
camera matrices, exact source-frame evidence, and source duration for the
weights-free finalizer.

The shared `best_view_geometry_v1` → `dedup_ras_finalize_v1` composite uses
2–24 frames. Both legs enforce the same cap; the edge must not silently submit
or relabel a larger frame budget.

Only VGGT weights are stored locally:

```bash
export STAGE2_MODELS_DIR=/models
bash scripts/download_weights.sh
```

The default `facebook/VGGT-1B` checkpoint is public and is downloaded
anonymously. An explicitly selected gated/commercial VGGT checkpoint still
requires its own approved token. There is no SAM checkpoint directory,
`facebook/sam3` download, or `STAGE2_DOWNLOAD_SAM3` switch.

VGGT-Omega is a research-only hosted comparison through Meta's reviewed Space.
It is capped at 24 frames and reports hosted, unattested model provenance; it
never claims local VGGT or RAS dedup execution.

## Artifact delivery

Paid analysis requires a privileged Agent Lab upload ticket. Required output
files are uploaded first and verified through durable receipts. Missing or
failed required uploads return `artifact_delivery_failed`; responses never
contain signed PUT URLs, worker paths, or inline file bytes.

The mask visualization is normalized to browser-safe H.264/yuv420p with
fast-start metadata and retains the original source timeline represented by the
sampled frames. Catalog output is bounded to 128 result-scoped physical-object
hypotheses and a fixed JPEG crop atlas.

## RunPod and verification

`scripts/start_serverless.sh` bootstraps an exact 40-character wrapper revision.
`scripts/deploy_revision.py` pins the RunPod template and verifies the deployed
`validation_v1` response reports the requested wrapper revision, pinned RAS and
VGGT revisions, `sam_provider: fal`, `sam3_required: false`, and
`weights_required: false`.

Run deterministic tests without provider calls:

```bash
python -m unittest discover -s tests -v
bash -n scripts/download_weights.sh scripts/start_serverless.sh
python -m py_compile *.py scripts/*.py tests/*.py
```

A release still needs one authorized real fal + RunPod E2E. That acceptance
run must capture a real fal RLE document, verify the encoding against the strict
decoder, and confirm Agent Lab merges leg A's GLB with leg C's three receipts.
