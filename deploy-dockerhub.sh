#!/bin/bash
# Deploy ComfyUI Worker to Docker Hub
# Usage: ./deploy-dockerhub.sh

set -e

# Configuration
DOCKERHUB_USER="kie1"
IMAGE_NAME="comfyui-worker"
TAG="v2.0"
CIVITAI_TOKEN="${1:-$CIVITAI_TOKEN}"

echo "🐳 Deploying ComfyUI Worker to Docker Hub"
echo "=========================================="

# Check for CIVITAI_TOKEN
if [ -z "$CIVITAI_TOKEN" ]; then
    echo "❌ CIVITAI_TOKEN required"
    echo "Usage: CIVITAI_TOKEN=your_token ./deploy-dockerhub.sh"
    exit 1
fi

echo "🔨 Building Docker image..."
cd "$(dirname "$0")"

docker build \
    --build-arg CIVITAI_TOKEN="$CIVITAI_TOKEN" \
    --build-arg LUSTIFY_MODEL_ID=2155386 \
    -t "${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}" \
    -t "${DOCKERHUB_USER}/${IMAGE_NAME}:latest" \
    .

echo "🏷️  Tags created:"
echo "   - ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo "   - ${DOCKERHUB_USER}/${IMAGE_NAME}:latest"

echo "📤 Pushing to Docker Hub..."
docker push "${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
docker push "${DOCKERHUB_USER}/${IMAGE_NAME}:latest"

echo ""
echo "✅ Deployed successfully!"
echo "   Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo ""
echo "📝 Next steps:"
echo "   1. Go to https://runpod.io/serverless"
echo "   2. Create endpoint with container: ${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"
echo "   3. Save endpoint ID"
