#!/usr/bin/env bash
# Fast Serverless start: thin wrapper only. dry_run works immediately.
# Full installs ReplicateAnyScene on first full job (or pre-seed volume).
set -uo pipefail
export PYTHONUNBUFFERED=1
export RAS_ROOT="${RAS_ROOT:-/workspace/ReplicateAnyScene}"
export STAGE2_MODELS_DIR="${STAGE2_MODELS_DIR:-/workspace/models}"
export STAGE2_MODE_DEFAULT="${STAGE2_MODE_DEFAULT:-full}"
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
python -m pip install -q --no-cache-dir --ignore-installed blinker || true
python -m pip install -q --no-cache-dir --ignore-installed -r requirements.txt
echo "[start] thin handler ready $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec python -u handler.py
