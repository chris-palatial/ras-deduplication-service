FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
WORKDIR /app
ENV PYTHONUNBUFFERED=1 RAS_ROOT=/app/vendor/ReplicateAnyScene \
    STAGE2_MODELS_DIR=/models STAGE2_MODE_DEFAULT=full STAGE2_ROOM_ALIGN=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore LIDRA_SKIP_INIT=true
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -q --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install -q --no-cache-dir --ignore-installed blinker \
 && python -m pip install -q --no-cache-dir --ignore-installed -r /app/requirements.txt
RUN mkdir -p /app/vendor \
 && git clone --depth 1 https://github.com/xiac20/ReplicateAnyScene.git /app/vendor/ReplicateAnyScene \
 && cd /app/vendor/ReplicateAnyScene \
 && sed -i 's|org-16943930@github.com:facebookresearch/|https://github.com/facebookresearch/|g' .gitmodules || true \
 && (git submodule update --init --depth 1 sam3 vggt || ( \
      git clone --depth 1 https://github.com/facebookresearch/vggt.git vggt && \
      git clone --depth 1 https://github.com/facebookresearch/sam3.git sam3)) \
 && python -m pip install -q --no-cache-dir --ignore-installed -e vggt \
 && (python -m pip install -q --no-cache-dir --ignore-installed -e sam3 || true) \
 && python -m pip install -q --no-cache-dir --ignore-installed open3d trimesh scipy matplotlib colorcet omegaconf hydra-core transformers
COPY stage2_service.py handler.py /app/
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh
ARG HF_TOKEN=""
ARG DOWNLOAD_WEIGHTS=0
RUN if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then \
      export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; \
      bash /app/scripts/download_weights.sh; \
    else mkdir -p /models; fi
CMD ["python", "-u", "handler.py"]
