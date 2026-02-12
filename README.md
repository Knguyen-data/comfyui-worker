# ComfyUI Worker - SDXL Lustify with IPAdapter FaceID

Dockerized ComfyUI worker for RunPod with IPAdapter FaceID and Lustify V7.

## Features

- 🎨 **SDXL Support** - Stable Diffusion XL base model
- 👤 **IPAdapter FaceID** - Consistent face generation
- 🔞 **Lustify V7** - NSFW image generation
- 🐳 **Docker Ready** - Self-contained build with all models

## Setup

### 1. GitHub Secrets

Add these secrets at https://github.com/Knguyen-data/comfyui-worker/settings/secrets/actions:

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | `kie1` |
| `DOCKERHUB_TOKEN` | [Docker Hub password/token] |
| `CIVITAI_TOKEN` | `715db9acbf5c71d8c82fc7cfc8ce2529` |

### 2. GitHub Actions

Push to `main` branch → Docker image auto-built → Pushed to Docker Hub

### 3. Deploy to RunPod

1. Go to https://runpod.io/serverless
2. Create endpoint:
   - **Container**: `kie1/comfyui-worker:latest`
   - **GPU**: RTX A4500 or A100 40GB
   - **Timeout**: 600s
3. Copy endpoint ID

### 4. Update Frontend

```typescript
// src/services/comfyui-service.ts
export const DEFAULT_ENDPOINT_URL = 'https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync';
```

## Local Development

```bash
# Build
docker build --build-arg CIVITAI_TOKEN=your_token -t kie1/comfyui-worker:test .

# Run
docker run -p 8000:8000 kie1/comfyui-worker:test

# Test endpoint
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{"prompt": {...}}'
```

## Models Included

- Lustify SDXL V7 (NSFW)
- IPAdapter FaceID SDXL
- CLIP Vision (ViT-bigG)
- IPAdapter Plus SDXL

## License

MIT
