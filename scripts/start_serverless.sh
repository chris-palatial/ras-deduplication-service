#!/usr/bin/env bash
# Fast Serverless start: thin wrapper only. dry_run works immediately.
# Full installs ReplicateAnyScene on first full job (or pre-seed volume).
set -euo pipefail
export PYTHONUNBUFFERED=1
if [[ ! "${STAGE2_CODE_REV:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[start] STAGE2_CODE_REV must be an explicit 40-character commit SHA" >&2
  exit 64
fi

export STAGE2_RUNTIME_ROOT="${STAGE2_RUNTIME_ROOT:-/workspace/stage2-runtimes}"
if [[ "$STAGE2_RUNTIME_ROOT" != /* ]]; then
  echo "[start] STAGE2_RUNTIME_ROOT must be a non-root absolute path" >&2
  exit 64
fi
mkdir -p "$STAGE2_RUNTIME_ROOT"
export STAGE2_RUNTIME_ROOT="$(cd "$STAGE2_RUNTIME_ROOT" && pwd -P)"
if [[ "$STAGE2_RUNTIME_ROOT" == "/" ]]; then
  echo "[start] STAGE2_RUNTIME_ROOT must be a non-root absolute path" >&2
  exit 64
fi
runtime_keep="${STAGE2_RUNTIME_KEEP:-4}"
if [[ ! "$runtime_keep" =~ ^[2-8]$ ]]; then
  echo "[start] STAGE2_RUNTIME_KEEP must be an integer from 2 through 8" >&2
  exit 64
fi
max_execution_seconds="${STAGE2_MAX_EXECUTION_SECONDS:-2100}"
runtime_min_age_seconds="${STAGE2_RUNTIME_MIN_AGE_SECONDS:-2700}"
if [[ ! "$max_execution_seconds" =~ ^[0-9]+$ || ! "$runtime_min_age_seconds" =~ ^[0-9]+$ ]] \
  || (( max_execution_seconds < 60 || runtime_min_age_seconds <= max_execution_seconds )); then
  echo "[start] runtime minimum age must exceed the configured max execution time" >&2
  exit 64
fi

# Each deployed commit owns an immutable app checkout and venv.  The venv keeps
# the base image's CUDA/PyTorch packages, while PYTHONPATH gives the exact
# revision-scoped VGGT/SAM3 source trees precedence over any system copy.
runtime_dir="$STAGE2_RUNTIME_ROOT/$STAGE2_CODE_REV"
export STAGE2_VENV="$runtime_dir/venv"
export RAS_ROOT="$runtime_dir/ReplicateAnyScene"
export STAGE2_MODELS_DIR="${STAGE2_MODELS_DIR:-/workspace/models}"
export STAGE2_MODE_DEFAULT="${STAGE2_MODE_DEFAULT:-full}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PYTHONPATH="$RAS_ROOT/vggt:$RAS_ROOT/sam3${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore
export LIDRA_SKIP_INIT=true
export DEBIAN_FRONTEND=noninteractive

mkdir -p "$STAGE2_MODELS_DIR" "$HF_HOME"
apt-get update -qq
apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 curl util-linux >/dev/null

exec 9>"$STAGE2_RUNTIME_ROOT/.bootstrap.lock"
flock 9
marker="$runtime_dir/.stage2_code_revision"
app_dir="$runtime_dir/app"
ready=false
if [[ -x "$STAGE2_VENV/bin/python" && -f "$marker" && -d "$app_dir/.git" ]]; then
  installed_revision="$(<"$marker")"
  app_revision="$(git -C "$app_dir" rev-parse HEAD 2>/dev/null || true)"
  app_dirty="$(git -C "$app_dir" diff-index --name-only HEAD -- 2>/dev/null || echo dirty-check-failed)"
  if [[ "$installed_revision" == "$STAGE2_CODE_REV" \
    && "$app_revision" == "$STAGE2_CODE_REV" \
    && -z "$app_dirty" ]]; then
    ready=true
  fi
fi

if [[ "$ready" != true ]]; then
  # The target is constrained to <runtime-root>/<40-hex-revision>; never reuse a
  # partial or mismatched environment under that immutable key.
  if [[ "$runtime_dir" != "$STAGE2_RUNTIME_ROOT/$STAGE2_CODE_REV" ]]; then
    echo "[start] refusing unsafe runtime cleanup target" >&2
    exit 64
  fi
  mkdir -p "$runtime_dir"
  exec 7>"$runtime_dir/.active_lease"
  if ! flock -n 7; then
    echo "[start] runtime $STAGE2_CODE_REV is active; refusing to rebuild it in place" >&2
    exit 75
  fi
  rm -rf -- "$runtime_dir"
  mkdir -p "$runtime_dir"
  git clone --filter=blob:none --no-checkout https://github.com/chris-palatial/ras-stage2-service.git "$app_dir"
  git -C "$app_dir" fetch --depth 1 origin "$STAGE2_CODE_REV"
  git -C "$app_dir" checkout --detach FETCH_HEAD
  python -m venv --system-site-packages "$STAGE2_VENV"
  "$STAGE2_VENV/bin/python" -m pip install -q --no-cache-dir -r "$app_dir/requirements.txt"
  printf '%s\n' "$STAGE2_CODE_REV" >"$marker"
  flock -u 7
fi

# Hold a shared lease for the worker's entire process lifetime. Cleanup takes an
# exclusive non-blocking lease, so it can never remove a runtime in active use.
touch "$runtime_dir/.last_used"
exec 8>"$runtime_dir/.active_lease"
flock -s 8
export STAGE2_BUILD_REVISION_FILE="$marker"

# Count retention is only a first filter. A directory must also be older than
# the maximum job window and have no active worker lease before removal.
python "$app_dir/scripts/prune_runtimes.py" \
  --root "$STAGE2_RUNTIME_ROOT" \
  --current-revision "$STAGE2_CODE_REV" \
  --keep "$runtime_keep" \
  --min-age-seconds "$runtime_min_age_seconds"
flock -u 9

cd "$app_dir"
echo "[start] thin handler ready $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec "$STAGE2_VENV/bin/python" -u handler.py
