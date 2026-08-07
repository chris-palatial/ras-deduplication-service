#!/usr/bin/env bash
# Start handler quickly. dry_run works without paper weights.
# Full mode installs RAS + weights on first need (or pre-seed volume).
set -uo pipefail
export PYTHONUNBUFFERED=1
export RAS_ROOT="${RAS_ROOT:-/workspace/ReplicateAnyScene}"
export STAGE2_MODELS_DIR="${STAGE2_MODELS_DIR:-/workspace/models}"
export STAGE2_MODE_DEFAULT="${STAGE2_MODE_DEFAULT:-full}"
export STAGE2_ROOM_ALIGN="${STAGE2_ROOM_ALIGN:-1}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore
export LIDRA_SKIP_INIT=true

mkdir -p /workspace/models /workspace/hf-cache /workspace/app
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 curl >/dev/null || true

rm -rf /workspace/app
git clone --depth 1 https://github.com/chris-palatial/ras-stage2-service.git /workspace/app
cd /workspace/app
python -m pip install -q --no-cache-dir --upgrade pip setuptools wheel || true
python -m pip install -q --no-cache-dir --ignore-installed blinker || true
python -m pip install -q --no-cache-dir --ignore-installed -r requirements.txt

# Optional pre-warm if PREPARE_FULL=1 or weights marker missing and PREPARE_FULL not 0
if [ "${PREPARE_FULL:-1}" = "1" ]; then
  echo "[start] preparing RAS full stack in background marker path"
  # Still do it before handler so first full job works; dry_run jobs wait too unless we flip PREPARE_FULL=0
  if [ ! -f "$RAS_ROOT/main.py" ]; then
    git clone --depth 1 https://github.com/xiac20/ReplicateAnyScene.git "$RAS_ROOT"
  fi
  cd "$RAS_ROOT"
  sed -i 's|org-16943930@github.com:facebookresearch/|https://github.com/facebookresearch/|g' .gitmodules 2>/dev/null || true
  [ -d vggt/.git ] || git clone --depth 1 https://github.com/facebookresearch/vggt.git vggt
  [ -d sam3/.git ] || git clone --depth 1 https://github.com/facebookresearch/sam3.git sam3
  python -m pip install -q --no-cache-dir --ignore-installed -e vggt || true
  python -m pip install -q --no-cache-dir --ignore-installed -e sam3 || true
  python -m pip install -q --no-cache-dir --ignore-installed open3d trimesh scipy matplotlib colorcet omegaconf hydra-core transformers || true
  if [ ! -f "$STAGE2_MODELS_DIR/VGGT/.stage2_ready" ] || [ ! -f "$STAGE2_MODELS_DIR/SAM3/.stage2_ready" ]; then
    echo "[start] downloading weights once"
    python -m pip install -q --no-cache-dir "huggingface_hub[cli]>=0.23"
    mkdir -p "$STAGE2_MODELS_DIR/VGGT" "$STAGE2_MODELS_DIR/SAM3"
    hf download facebook/VGGT-1B --local-dir "$STAGE2_MODELS_DIR/VGGT" || true
    hf download facebook/sam3 --local-dir "$STAGE2_MODELS_DIR/SAM3" || true
    touch "$STAGE2_MODELS_DIR/VGGT/.stage2_ready" "$STAGE2_MODELS_DIR/SAM3/.stage2_ready" || true
  fi
  cd "$RAS_ROOT"
  rm -f models 2>/dev/null || true
  [ -e models ] || ln -sfn "$STAGE2_MODELS_DIR" models
fi

cd /workspace/app
export RAS_ROOT
echo "[start] handler ready $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec python -u handler.py
