#!/usr/bin/env bash
# One-time VGGT weight download into STAGE2_MODELS_DIR.
# SAM segmentation is always provided by fal and has no local checkpoint.
set -euo pipefail
ROOT="${STAGE2_MODELS_DIR:-./models}"
mkdir -p "$ROOT/VGGT"
echo "Downloading VGGT weights into $ROOT"

python - <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ.get("STAGE2_MODELS_DIR", "./models")).resolve()
token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)
vggt_model_id = os.environ.get("VGGT_MODEL_ID", "facebook/VGGT-1B").strip() or "facebook/VGGT-1B"
root.mkdir(parents=True, exist_ok=True)
print(vggt_model_id, "->", root / "VGGT")
if vggt_model_id != "facebook/VGGT-1B" and not token:
    sys.exit(f"{vggt_model_id} requires an approved Hugging Face token in HF_TOKEN")
snapshot_download(
    vggt_model_id,
    local_dir=str(root / "VGGT"),
    token=False if vggt_model_id == "facebook/VGGT-1B" else token,
)
(root / "VGGT" / ".stage2_model_id").write_text(vggt_model_id + "\n")
(root / "VGGT" / ".stage2_ready").touch()
print("Done.")
PY
