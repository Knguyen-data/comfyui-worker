FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# HuggingFace token for gated model downloads (Flux Dev)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    xformers==0.0.28.post3 \
    numpy==1.26.4 \
    opencv-python \
    pillow \
    tqdm \
    requests \
    safetensors \
    huggingface_hub \
    einops \
    transformers==4.45.0 \
    accelerate \
    scipy \
    pandas \
    matplotlib \
    protobuf \
    aiohttp \
    runpod

# Install ComfyUI
WORKDIR /home/comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install -r requirements.txt

# Create models directories
RUN mkdir -p /home/comfyui/models/checkpoints \
             /home/comfyui/models/unet \
             /home/comfyui/models/clip \
             /home/comfyui/models/vae \
             /home/comfyui/models/loras \
             /home/comfyui/models/upscale_models \
             /home/comfyui/models/ipadapter \
             /home/comfyui/models/clip_vision

# Download Flux Dev model files
# 1. flux1-dev.safetensors (23.8 GB) - UNET, gated model
# 2. ae.safetensors (335 MB) - VAE
# 3. clip_l.safetensors (246 MB) - CLIP-L text encoder
# 4. t5xxl_fp16.safetensors (9.79 GB) - T5-XXL text encoder

# Flux Dev UNET (gated - needs HF_TOKEN)
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "${HF_TOKEN}") && \
    wget --progress=bar:force:noscroll \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /home/comfyui/models/unet/flux1-dev.safetensors \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"

# Flux VAE (gated - same repo)
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "${HF_TOKEN}") && \
    wget --progress=bar:force:noscroll \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    -O /home/comfyui/models/vae/ae.safetensors \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"

# CLIP-L and T5-XXL text encoders (public, no auth needed)
RUN wget --progress=bar:force:noscroll \
    -O /home/comfyui/models/clip/clip_l.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" && \
    wget --progress=bar:force:noscroll \
    -O /home/comfyui/models/clip/t5xxl_fp16.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"

# Download IPAdapter models for face reference
ARG CIVITAI_TOKEN
ENV CIVITAI_TOKEN=${CIVITAI_TOKEN}
RUN if [ -n "${CIVITAI_TOKEN}" ]; then \
    echo "Downloading IPAdapter FaceID..." && \
    curl -L -o /home/comfyui/models/ipadapter/ip-adapter-faceid_sdxl.bin \
        "https://civitai.com/api/download/models/215861?token=${CIVITAI_TOKEN}" && \
    echo "Downloading CLIP Vision..." && \
    curl -L -o /home/comfyui/models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors \
        "https://civitai.com/api/download/models/212354?token=${CIVITAI_TOKEN}"; \
    else echo "Skipping IPAdapter (no CIVITAI_TOKEN)"; fi

# Install custom nodes (IPAdapter Plus for face reference workflow)
RUN rm -rf /home/comfyui/custom_nodes/* && \
    mkdir -p /home/comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /home/comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    cd /home/comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    pip install --no-cache-dir -r requirements.txt || true && \
    cd /home/comfyui

# Copy handler
COPY handler.py /home/comfyui/handler.py

WORKDIR /home/comfyui

# Expose port
EXPOSE 8188

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8188')" || exit 1

# Start RunPod handler (starts ComfyUI internally)
CMD ["python", "-u", "handler.py"]
