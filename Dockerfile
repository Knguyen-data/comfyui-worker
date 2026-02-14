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

# CLIP-L and T5-XXL text encoders (public, no auth needed - download at build time)
RUN wget --progress=bar:force:noscroll \
    -O /home/comfyui/models/clip/clip_l.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" && \
    wget --progress=bar:force:noscroll \
    -O /home/comfyui/models/clip/t5xxl_fp16.safetensors \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"

# NOTE: Gated models (flux1-dev.safetensors, ae.safetensors) are downloaded
# at container startup via handler.py using HF_TOKEN environment variable.
# Set HF_TOKEN in RunPod endpoint environment variables.

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
