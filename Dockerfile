FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

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
    uvicorn \
    fastapi \
    pydantic

# Install ComfyUI
WORKDIR /home/comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install -r requirements.txt

# Create models directories
RUN mkdir -p /home/comfyui/models/checkpoints \
             /home/comfyui/models/loras \
             /home/comfyui/models/upscale_models \
             /home/comfyui/models/ipadapter \
             /home/comfyui/models/clip_vision

# Download IPAdapter models (ARG only, not exposed as ENV)
ARG CIVITAI_TOKEN

RUN echo "Downloading IPAdapter FaceID..." && \
    curl -L -o /home/comfyui/models/ipadapter/ip-adapter-faceid_sdxl.bin \
        "https://civitai.com/api/download/models/215861?token=${CIVITAI_TOKEN}" && \
    echo "Downloading CLIP Vision..." && \
    curl -L -o /home/comfyui/models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors \
        "https://civitai.com/api/download/models/212354?token=${CIVITAI_TOKEN}" && \
    echo "Downloading IPAdapter Plus..." && \
    curl -L -o /home/comfyui/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors \
        "https://civitai.com/api/download/models/247653?token=${CIVITAI_TOKEN}"

# Download Lustify V7 (NSFW)
ARG LUSTIFY_MODEL_ID=2155386
RUN echo "Downloading Lustify SDXL V7..." && \
    curl -L -o /home/comfyui/models/checkpoints/lustifySDXLNSFW_ggwpV7.safetensors \
        "https://civitai.com/api/download/models/${LUSTIFY_MODEL_ID}?token=${CIVITAI_TOKEN}"

# Download base SDXL model
RUN echo "Downloading base SDXL model..." && \
    curl -L -o /home/comfyui/models/checkpoints/sd_xl_base_1.0.safetensors \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"

# Download T5 XXL CLIP model (required for workflow.json node #2)
# Note: t5xxl_fp16.safetensors may already be included with stable-diffusion-xl-base-1.0
# but we download it explicitly to ensure availability
RUN echo "Downloading T5 XXL CLIP model..." && \
    mkdir -p /home/comfyui/models/clip && \
    curl -L -o /home/comfyui/models/clip/t5xxl_fp16.safetensors \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/t5xxl_fp16.safetensors" || \
    echo "T5 XXL may already be bundled with base model"

# Copy custom nodes
RUN mkdir -p /home/comfyui/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /home/comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    git clone https://github.com/latent-consistency/latent-consistency.git /home/comfyui/custom_nodes/latent-consistency

# Copy workflow and handler
COPY --chmod=755 handler.py /home/comfyui/handler.py
COPY --chmod=755 workflow.json /home/comfyui/workflow.json

WORKDIR /home/comfyui

# Expose port
EXPOSE 8188

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8188')" || exit 1

# Start ComfyUI
CMD ["python", "main.py", "--disable-auto-launch", "--disable-metadata"]
