#!/usr/bin/env bash
# One-time weight download into STAGE2_MODELS_DIR (image build or Network Volume).
# Not per job. Paths match ReplicateAnyScene README layout.
set -euo pipefail
ROOT="${STAGE2_MODELS_DIR:-./models}"
mkdir -p "$ROOT/VGGT" "$ROOT/SAM3"
echo "Downloading into $ROOT (HF_TOKEN required for gated facebook/sam3)"

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
download_sam3 = os.environ.get("STAGE2_DOWNLOAD_SAM3", "1").lower() not in {"0", "false", "no"}
vggt_model_id = os.environ.get("VGGT_MODEL_ID", "facebook/VGGT-1B").strip() or "facebook/VGGT-1B"
root.mkdir(parents=True, exist_ok=True)
print(vggt_model_id, "->", root / "VGGT")
if vggt_model_id != "facebook/VGGT-1B" and not token:
    sys.exit(f"{vggt_model_id} requires an approved Hugging Face token in HF_TOKEN")
snapshot_download(
    vggt_model_id,
    local_dir=str(root / "VGGT"),
    token=None if vggt_model_id == "facebook/VGGT-1B" else token,
)
(root / "VGGT" / ".stage2_model_id").write_text(vggt_model_id + "\n")
if not download_sam3:
    print("SAM3 skipped (STAGE2_DOWNLOAD_SAM3=0); geometry mode is ready.")
    sys.exit(0)
if not token:
    sys.exit("facebook/sam3 is gated and requires an approved Hugging Face token in HF_TOKEN")
print("sam3 ->", root / "SAM3")
snapshot_download("facebook/sam3", local_dir=str(root / "SAM3"), token=token)
sam = root / "SAM3" / "sam3.pt"
if not sam.is_file():
    found = list((root / "SAM3").rglob("sam3.pt"))
    if not found:
        sys.exit("sam3.pt missing after download")
    sam.symlink_to(found[0].resolve()) if not sam.exists() else None
    if not sam.is_file():
        import shutil
        shutil.copy2(found[0], sam)
print("sam3.pt:", sam, "size", sam.stat().st_size)
(root / "VGGT" / ".stage2_ready").touch()
(root / "SAM3" / ".stage2_ready").touch()
print("Done.")
PY
