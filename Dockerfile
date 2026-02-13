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
    httpx \
    insightface \
    onnxruntime-gpu \
    runpod

# Install ComfyUI
WORKDIR /home/comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install -r requirements.txt

# Create models directories
RUN mkdir -p /home/comfyui/models/checkpoints \
             /home/comfyui/models/loras \
             /home/comfyui/models/upscale_models \
             /home/comfyui/models/ipadapter \
             /home/comfyui/models/clip_vision \
             /home/comfyui/models/insightface

# Download IPAdapter models (ARG only, not exposed as ENV)
ENV CIVITAI_TOKEN=715db9acbf5c71d8c82fc7cfc8ce2529

RUN echo "Downloading IPAdapter FaceID..." && \
    curl -L -o /home/comfyui/models/ipadapter/ip-adapter-faceid_sdxl.bin \
        "https://civitai.com/api/download/models/215861?token=715db9acbf5c71d8c82fc7cfc8ce2529" && \
    echo "Downloading CLIP Vision..." && \
    curl -L -o /home/comfyui/models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors \
        "https://civitai.com/api/download/models/212354?token=715db9acbf5c71d8c82fc7cfc8ce2529" && \
    echo "Downloading IPAdapter Plus..." && \
    curl -L -o /home/comfyui/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors \
        "https://civitai.com/api/download/models/247653?token=715db9acbf5c71d8c82fc7cfc8ce2529"

# Download Lustify V7 (NSFW)
ARG LUSTIFY_MODEL_ID=2155386
RUN echo "Downloading Lustify SDXL V7..." && \
    curl -L -o /home/comfyui/models/checkpoints/lustifySDXLNSFW_ggwpV7.safetensors \
        "https://civitai.com/api/download/models/${LUSTIFY_MODEL_ID}?token=715db9acbf5c71d8c82fc7cfc8ce2529"

# Install custom nodes (only IPAdapter Plus needed for this workflow)
# Clean any broken custom nodes first
RUN rm -rf /home/comfyui/custom_nodes/* && \
    mkdir -p /home/comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /home/comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    cd /home/comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    pip install --no-cache-dir -r requirements.txt || true && \
    cd /home/comfyui

# Copy workflow and handler
COPY handler.py /home/comfyui/handler.py
COPY workflow.json /home/comfyui/workflow.json

WORKDIR /home/comfyui

# Expose port
EXPOSE 8188

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8188')" || exit 1

# Start RunPod handler (starts ComfyUI internally)
CMD ["python", "-u", "handler.py"]
