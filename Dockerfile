# syntax=docker/dockerfile:1.7
# Stage 2 service: official ReplicateAnyScene + thin RunPod handler.
# Build on linux/amd64. Prefer baking weights once or mounting STAGE2_MODELS_DIR.
#
#   docker build -t ras-stage2:full \
#     --build-arg STAGE2_BUILD_REVISION="$(git rev-parse HEAD)" \
#     --secret id=hf_token,env=HF_TOKEN \
#     --build-arg DOWNLOAD_WEIGHTS=1 .

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG RAS_REVISION=671191457e7244d9337ef3faf558ee92bbf9bf73
ARG VGGT_REVISION=44b3afbd1869d8bde4894dd8ea1e293112dd5eba
ARG SAM3_REVISION=bfbed072a07a6a52c8d5fdc75a7a186251a835b1
ARG STAGE2_BUILD_REVISION=""

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    RAS_ROOT=/app/vendor/ReplicateAnyScene \
    STAGE2_MODELS_DIR=/models \
    STAGE2_MODE_DEFAULT=full \
    STAGE2_ROOM_ALIGN=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    LIDRA_SKIP_INIT=true

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Wrapper
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -q --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install -q --no-cache-dir --ignore-installed blinker \
    && python -m pip install -q --no-cache-dir --ignore-installed -r /app/requirements.txt

# Pin all three source trees to the revisions reviewed with this wrapper.
RUN mkdir -p /app/vendor \
    && git clone --filter=blob:none --no-checkout https://github.com/xiac20/ReplicateAnyScene.git /app/vendor/ReplicateAnyScene \
    && git -C /app/vendor/ReplicateAnyScene fetch --depth 1 origin "$RAS_REVISION" \
    && git -C /app/vendor/ReplicateAnyScene checkout --detach FETCH_HEAD \
    && rm -rf /app/vendor/ReplicateAnyScene/vggt /app/vendor/ReplicateAnyScene/sam3 \
    && git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/vggt.git /app/vendor/ReplicateAnyScene/vggt \
    && git -C /app/vendor/ReplicateAnyScene/vggt fetch --depth 1 origin "$VGGT_REVISION" \
    && git -C /app/vendor/ReplicateAnyScene/vggt checkout --detach FETCH_HEAD \
    && git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/sam3.git /app/vendor/ReplicateAnyScene/sam3 \
    && git -C /app/vendor/ReplicateAnyScene/sam3 fetch --depth 1 origin "$SAM3_REVISION" \
    && git -C /app/vendor/ReplicateAnyScene/sam3 checkout --detach FETCH_HEAD \
    && cd /app/vendor/ReplicateAnyScene \
    && python -m pip install -q --no-cache-dir --ignore-installed /app/vendor/ReplicateAnyScene/vggt \
    && python -m pip install -q --no-cache-dir --ignore-installed /app/vendor/ReplicateAnyScene/sam3 \
    && python -m pip install -q --no-cache-dir --ignore-installed open3d trimesh scipy matplotlib colorcet omegaconf hydra-core transformers

COPY stage2_service.py handler.py artifact_upload.py point_cloud_glb.py /app/
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh || true \
    && if [ -n "$STAGE2_BUILD_REVISION" ]; then \
         printf '%s' "$STAGE2_BUILD_REVISION" | grep -Eq '^[0-9a-f]{40}$' \
           || { echo "STAGE2_BUILD_REVISION must be a full lowercase commit SHA" >&2; exit 64; }; \
         printf '%s\n' "$STAGE2_BUILD_REVISION" > /app/.stage2_build_revision; \
       else \
         printf '%s\n' unknown > /app/.stage2_build_revision; \
       fi
ENV STAGE2_BUILD_REVISION_FILE=/app/.stage2_build_revision

ARG DOWNLOAD_WEIGHTS=0
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then \
      if [ -s /run/secrets/hf_token ]; then \
        HF_TOKEN="$(cat /run/secrets/hf_token)"; \
        export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; \
      fi; \
      bash /app/scripts/download_weights.sh; \
    else \
      mkdir -p /models; \
      echo "Weights not baked; mount STAGE2_MODELS_DIR or run download_weights.sh"; \
    fi

WORKDIR /app
CMD ["python", "-u", "handler.py"]
