#!/usr/bin/env bash
# One-time weight download into STAGE2_MODELS_DIR (image build or Network Volume).
# Not per job. Paths match ReplicateAnyScene README layout.
set -euo pipefail
ROOT="${STAGE2_MODELS_DIR:-./models}"
mkdir -p "$ROOT"
echo "Downloading into $ROOT"
hf download facebook/VGGT-1B --local-dir "$ROOT/VGGT"
hf download facebook/sam3 --local-dir "$ROOT/SAM3"
# RAS models.py expects sam3.pt under models/SAM3/
if [ -f "$ROOT/SAM3/sam3.pt" ]; then
  echo "sam3.pt present"
else
  echo "NOTE: if sam3.pt is nested, symlink it to $ROOT/SAM3/sam3.pt"
  find "$ROOT/SAM3" -name 'sam3.pt' 2>/dev/null | head -5
fi
touch "$ROOT/VGGT/.stage2_ready" "$ROOT/SAM3/.stage2_ready"
echo "Done."
