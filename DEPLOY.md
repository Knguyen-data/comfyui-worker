# ComfyUI Worker v2.0 - Deployment to RunPod

## Quick Deploy

### Step 1: Build & Push Docker Image
```bash
cd workers/comfyui-worker
docker build --build-arg CIVITAI_TOKEN=715db9acbf5c71d8c82fc7cfc8ce2529 -t kie1/comfyui-worker:v2.0 .
docker push kie1/comfyui-worker:v2.0
```

### Step 2: Create RunPod Endpoint
1. Go to https://runpod.io/serverless
2. Create new endpoint:
   - **Name**: ComfyUI-Lustify-v2
   - **Container**: `kie1/comfyui-worker:v2.0`
   - **GPU**: RTX A4500 or A100 40GB
   - **Timeout**: 600s
3. Save endpoint ID

### Step 3: Update Frontend
Update `src/services/comfyui-service.ts`:
```typescript
export const DEFAULT_ENDPOINT_URL = 'https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync';
```

## Models Included
- Lustify SDXL V7 (NSFW)
- IPAdapter FaceID SDXL
- CLIP Vision (ViT-bigG)
- IPAdapter Plus SDXL
