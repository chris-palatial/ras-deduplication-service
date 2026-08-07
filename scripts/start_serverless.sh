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
export STAGE2_VENV="${STAGE2_VENV:-/workspace/stage2-venv}"
mkdir -p /workspace/models /workspace/hf-cache /workspace/app
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 curl >/dev/null || true
rm -rf /workspace/app
git clone --filter=blob:none --no-checkout https://github.com/chris-palatial/ras-stage2-service.git /workspace/app
if [[ "${STAGE2_CODE_REV:-}" =~ ^[0-9a-f]{40}$ ]]; then
  git -C /workspace/app fetch --depth 1 origin "$STAGE2_CODE_REV"
  git -C /workspace/app checkout --detach FETCH_HEAD
else
  echo "[start] STAGE2_CODE_REV is not a commit SHA; using origin/main"
  git -C /workspace/app fetch --depth 1 origin main
  git -C /workspace/app checkout --detach FETCH_HEAD
fi
cd /workspace/app
if [ ! -x "$STAGE2_VENV/bin/python" ]; then
  python -m venv --system-site-packages "$STAGE2_VENV"
fi
"$STAGE2_VENV/bin/python" -m pip install -q --no-cache-dir -r requirements.txt
echo "[start] thin handler ready $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec "$STAGE2_VENV/bin/python" -u handler.py
