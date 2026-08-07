#!/usr/bin/env bash
# Start on runpod/pytorch: clone this wrapper + RAS paper repo, weights once on /workspace.
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
apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 >/dev/null || true

# Thin wrapper
rm -rf /workspace/app
git clone --depth 1 https://github.com/chris-palatial/ras-stage2-service.git /workspace/app
cd /workspace/app
python -m pip install -q --no-cache-dir --upgrade pip setuptools wheel || true
python -m pip install -q --no-cache-dir --ignore-installed blinker || true
python -m pip install -q --no-cache-dir --ignore-installed -r requirements.txt

# Official paper repo (Stage 2 source of truth)
if [ ! -f "$RAS_ROOT/main.py" ]; then
  git clone --depth 1 https://github.com/xiac20/ReplicateAnyScene.git "$RAS_ROOT"
fi
cd "$RAS_ROOT"
sed -i 's|org-16943930@github.com:facebookresearch/|https://github.com/facebookresearch/|g' .gitmodules 2>/dev/null || true
if [ ! -d vggt/.git ]; then
  git submodule update --init --depth 1 vggt 2>/dev/null || git clone --depth 1 https://github.com/facebookresearch/vggt.git vggt
fi
if [ ! -d sam3/.git ]; then
  git submodule update --init --depth 1 sam3 2>/dev/null || git clone --depth 1 https://github.com/facebookresearch/sam3.git sam3
fi
python -m pip install -q --no-cache-dir --ignore-installed -e vggt
python -m pip install -q --no-cache-dir --ignore-installed -e sam3 || true
python -m pip install -q --no-cache-dir --ignore-installed open3d trimesh scipy matplotlib colorcet omegaconf hydra-core transformers

# Weights once
if [ ! -f "$STAGE2_MODELS_DIR/VGGT/.stage2_ready" ] || [ ! -f "$STAGE2_MODELS_DIR/SAM3/.stage2_ready" ]; then
  echo "[start] downloading weights once to $STAGE2_MODELS_DIR"
  python -m pip install -q --no-cache-dir "huggingface_hub[cli]>=0.23"
  mkdir -p "$STAGE2_MODELS_DIR/VGGT" "$STAGE2_MODELS_DIR/SAM3"
  hf download facebook/VGGT-1B --local-dir "$STAGE2_MODELS_DIR/VGGT"
  hf download facebook/sam3 --local-dir "$STAGE2_MODELS_DIR/SAM3"
  touch "$STAGE2_MODELS_DIR/VGGT/.stage2_ready" "$STAGE2_MODELS_DIR/SAM3/.stage2_ready"
else
  echo "[start] weights already on volume"
fi

# RAS models.py expects ./models under RAS_ROOT
cd "$RAS_ROOT"
rm -f models 2>/dev/null || true
if [ ! -e models ]; then ln -s "$STAGE2_MODELS_DIR" models; fi

cd /workspace/app
export RAS_ROOT
echo "[start] handler"
exec python -u handler.py
