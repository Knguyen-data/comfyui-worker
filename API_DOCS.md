# ComfyUI RunPod Worker API Documentation

## Overview

This document describes the complete API contract between the frontend client and the RunPod serverless worker for the ComfyUI image generation pipeline.

**Architecture Flow:**
```
Frontend → RunPod API → Serverless Handler → ComfyUI → Output Images
```

---

## 1. RunPod API Endpoints

All endpoints are accessed via the base URL `https://api.runpod.ai/v2/{endpoint_id}` with authentication.

### 1.1 Submit Async Job

**Endpoint:** `POST /v2/{endpoint_id}/run`

**Headers:**
```
Content-Type: application/json
Authorization: Bearer {RUNPOD_API_KEY}
```

**Request Body:**
```json
{
  "input": {
    // ComfyUI API-format prompt (node graph)
  }
}
```

**Response (202 Accepted):**
```json
{
  "id": "job-uuid-here",
  "status": "IN_QUEUE"
}
```

**Status Codes:**
- `202`: Job accepted and queued
- `401`: Invalid API key
- `403`: API key lacks endpoint access
- `429`: Rate limit exceeded

### 1.2 Poll Job Status

**Endpoint:** `GET /v2/{endpoint_id}/status/{job_id}`

**Headers:**
```
Authorization: Bearer {RUNPOD_API_KEY}
```

**Response:**
```json
{
  "id": "job-uuid-here",
  "status": "COMPLETED",
  "output": {
    "images": [
      {
        "filename": "ComfyUI_0001.png",
        "data": "base64-encoded-image-data",
        "type": "image/png"
      }
    ]
  }
}
```

### 1.3 Cancel Job

**Endpoint:** `POST /v2/{endpoint_id}/cancel/{job_id}`

**Headers:**
```
Authorization: Bearer {RUNPOD_API_KEY}
```

**Response:** Empty body (204 No Content on success)

### 1.4 Job Status Values

| Status | Description |
|--------|-------------|
| `IN_QUEUE` | Job is waiting to be processed |
| `IN_PROGRESS` | Job is currently executing |
| `COMPLETED` | Job finished successfully |
| `FAILED` | Job encountered an error |
| `CANCELLED` | Job was cancelled by user |

---

## 2. Job Input Payload

The frontend constructs a ComfyUI API-format prompt structure as the `input` field in the RunPod job body.

### 2.1 Type Definitions (Frontend)

```typescript
// From src/types/index.ts

export type ComfyUISampler = 'euler' | 'euler_ancestral' | 'dpmpp_2m' | 'dpmpp_sde';
export type ComfyUIScheduler = 'normal' | 'karras' | 'sgm_uniform';

export interface ComfyUISettings {
  steps: number;              // 15-50, default 20
  cfg: number;                // 1-15, default 8
  denoise: number;            // 0-1, default 1.0
  sampler: ComfyUISampler;   // default 'euler'
  scheduler: ComfyUIScheduler; // default 'normal'
  seed: number;               // -1 = random, otherwise specific seed
  ipAdapterWeight: number;    // 0-2, default 1.0 (face strength)
  ipAdapterFaceidWeight: number; // 0-2, default 1.0
}

export type ComfyUIDimensions = { width: number; height: number };

export interface ComfyUIRunPodJob {
  id: string;
  status: 'IN_QUEUE' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  output?: {
    success: boolean;
    outputs?: Record<string, unknown>;
    error?: string;
  };
  error?: string;
}
```

### 2.2 Aspect Ratio Mapping

```typescript
export const mapAspectRatioToDimensions = (ratio: AspectRatio): ComfyUIDimensions => {
  const map: Record<AspectRatio, ComfyUIDimensions> = {
    '1:1':  { width: 1024, height: 1024 },
    '16:9': { width: 1344, height: 768 },
    '9:16': { width: 768,  height: 1344 },
    '4:3':  { width: 1152, height: 896 },
    '3:4':  { width: 896,  height: 1152 },
    '4:5':  { width: 896,  height: 1152 },
  };
  return map[ratio] || { width: 1024, height: 1024 };
};
```

### 2.3 Basic Mode (Text-to-Image) Payload

When no face image is provided, the frontend builds this simplified workflow:

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "lustifySDXLNSFW_ggwpV7.safetensors"
    }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "beautiful scenery nature",
      "clip": ["1", 1]
    }
  },
  "8": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "ugly, blurry, low quality, deformed",
      "clip": ["1", 1]
    }
  },
  "13": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    }
  },
  "10": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["7", 0],
      "negative": ["8", 0],
      "latent_image": ["13", 0],
      "seed": 123456789,
      "steps": 20,
      "cfg": 8.0,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0
    }
  },
  "11": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["10", 0],
      "vae": ["1", 2]
    }
  },
  "12": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["11", 0],
      "filename_prefix": "ComfyUI"
    }
  }
}
```

### 2.4 IPAdapter FaceID Mode Payload

When a face image is provided, additional nodes are injected for face reference:

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "lustifySDXLNSFW_ggwpV7.safetensors"
    }
  },
  "3": {
    "class_type": "CLIPVisionLoader",
    "inputs": {
      "clip_name": "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"
    }
  },
  "4": {
    "class_type": "IPAdapterModelLoader",
    "inputs": {
      "ipadapter_file": "ip-adapter-faceid_sdxl.bin"
    }
  },
  "6": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "base64-encoded-face-image"
    }
  },
  "5": {
    "class_type": "IPAdapterAdvanced",
    "inputs": {
      "model": ["1", 0],
      "clip": ["1", 1],
      "ipadapter": ["4", 0],
      "clip_vision": ["3", 0],
      "image": ["6", 0],
      "weight": 1.0,
      "weight_faceidv2": 1.0,
      "combine_embeds": "Average",
      "embeds_scaling": "V only"
    }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "beautiful scenery nature",
      "clip": ["1", 1]
    }
  },
  "8": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "ugly, blurry, low quality, deformed",
      "clip": ["1", 1]
    }
  },
  "13": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    }
  },
  "10": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["5", 0],
      "positive": ["7", 0],
      "negative": ["8", 0],
      "latent_image": ["13", 0],
      "seed": 123456789,
      "steps": 20,
      "cfg": 8.0,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0
    }
  },
  "11": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["10", 0],
      "vae": ["1", 2]
    }
  },
  "12": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["11", 0],
      "filename_prefix": "ComfyUI"
    }
  }
}
```

### 2.5 Node Reference

| Node ID | Class Type | Purpose | Inputs / Outputs |
|---------|-----------|---------|------------------|
| 1 | `CheckpointLoaderSimple` | Loads SDXL checkpoint | Output: MODEL[0], CLIP[1], VAE[2] |
| 3 | `CLIPVisionLoader` | Loads CLIP vision for IPAdapter | Output: CLIP_VISION |
| 4 | `IPAdapterModelLoader` | Loads IPAdapter FaceID model | Output: IPADAPTER |
| 5 | `IPAdapterAdvanced` | Applies face reference | Inputs: model, clip, ipadapter, clip_vision, image; Output: MODEL |
| 6 | `LoadImage` | Loads face reference image | Input: image (base64); Output: IMAGE, MASK |
| 7 | `CLIPTextEncode` | Positive prompt encoding | Inputs: text, clip; Output: CONDITIONING |
| 8 | `CLIPTextEncode` | Negative prompt encoding | Inputs: text, clip; Output: CONDITIONING |
| 10 | `KSampler` | Main sampling loop | Multiple inputs; Output: LATENT |
| 11 | `VAEDecode` | Decodes latent to image | Inputs: samples, vae; Output: IMAGE |
| 12 | `SaveImage` | Saves output to disk | Input: images |
| 13 | `EmptyLatentImage` | Creates latent canvas | Inputs: width, height, batch_size; Output: LATENT |

---

## 3. Worker Processing Flow

### 3.1 Handler Entry Point

The RunPod handler (`handler.py`) receives the job and processes it:

```python
def handler(event):
    """RunPod serverless handler."""
    try:
        start_comfyui()

        # RunPod delivers event["input"] = whatever the caller sent as "input"
        # The frontend sends the ComfyUI node graph directly as "input"
        prompt = event.get("input", {})

        if not prompt:
            return {"error": "No prompt provided in input"}

        # Submit prompt
        prompt_id = queue_prompt(prompt)

        # Poll for completion
        history = poll_history(prompt_id)

        # Check for errors
        status_info = history.get("status", {})
        if status_info.get("status_str") == "error":
            msgs = status_info.get("messages", [])
            return {"error": f"ComfyUI error: {msgs}"}

        # Collect output images
        images = collect_outputs(history)

        if not images:
            return {"error": "No output images produced"}

        return {"images": images}

    except Exception as e:
        return {"error": str(e)}
```

### 3.2 ComfyUI Internal API Calls

The handler communicates with the ComfyUI server running internally:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `http://127.0.0.1:8188/prompt` | POST | Submit the prompt node graph |
| `http://127.0.0.1:8188/history/{prompt_id}` | GET | Poll for completion status |
| `http://127.0.0.1:8188/system_stats` | GET | Health check (used during startup) |

**Submit Prompt Request:**
```python
def queue_prompt(prompt):
    """Submit a prompt to ComfyUI and return prompt_id."""
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    return result["prompt_id"]
```

**Poll History Request:**
```python
def poll_history(prompt_id, timeout=600):
    """Poll ComfyUI history endpoint until the prompt completes."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10
            )
            history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")
```

### 3.3 Output Collection

```python
def collect_outputs(history):
    """Extract output images from ComfyUI history as base64."""
    images = []
    outputs = history.get("outputs", {})

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for img_info in node_output["images"]:
                filename = img_info.get("filename", "")
                subfolder = img_info.get("subfolder", "")
                img_path = os.path.join(COMFYUI_OUTPUT, subfolder, filename)
                if os.path.isfile(img_path):
                    with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    images.append({
                        "filename": filename,
                        "data": b64,
                        "type": "image/png",
                    })

    return images
```

---

## 4. Job Output Payload

### 4.1 Success Response

```json
{
  "images": [
    {
      "filename": "ComfyUI_0001.png",
      "data": "iVBORw0KGgoAAAANSUhEUgAABVYAAASVCAYAA...",
      "type": "image/png"
    }
  ]
}
```

### 4.2 Error Response

```json
{
  "error": "ComfyUI error: [\"Error loading node\", \"Some node failed to load\"]"
}
```

### 4.3 Error Types

| Error Message | Cause |
|---------------|-------|
| `No prompt provided in input` | Frontend sent empty `input` field |
| `ComfyUI error: [...]` | ComfyUI node execution failed |
| `No output images produced` | Generation completed but SaveImage node didn't run |
| `Prompt {id} did not complete within {timeout}s` | Timeout (default 600s) |
| `ComfyUI process died...` | ComfyUI server crashed |

---

## 5. Payload Validation

### 5.1 Frontend vs Handler Comparison

| Field | Frontend Sends | Handler Expects | Match? |
|-------|---------------|-----------------|--------|
| `input` | Full ComfyUI prompt object | Full ComfyUI prompt object | ✅ Yes |
| `input` | Empty object error | No special validation | ✅ Yes |
| `input.prompt_id` | Not sent (internal to handler) | Extracted from /prompt response | N/A |
| `output.images` | Extracted by frontend | Generated by handler | ✅ Yes |

### 5.2 Potential Issues & Mismatches

**⚠️ CRITICAL: Node Type Mismatch Between workflow.json and Frontend Code**

The `workflow.json` file uses different node types than what the frontend code actually sends:

| Component | workflow.json Uses | Frontend Code Uses | Impact |
|-----------|-------------------|-------------------|--------|
| Checkpoint Loader | `CheckPointLoaderSDXL` | `CheckpointLoaderSimple` | ⚠️ Different node |
| CLIP Text Encode | `CLIPTextEncodeSDXL` (x2) | `CLIPTextEncode` (x1) | ⚠️ Different node |
| Latent Image | `EmptySDXLLatentImage` | `EmptyLatentImage` | ⚠️ Different node |

**Analysis:**
- The `workflow.json` is marked as "NOT used at runtime" - it's only a reference for UI design
- The frontend code (`buildWorkflowPrompt()`) uses `CheckpointLoaderSimple`, `CLIPTextEncode`, and `EmptyLatentImage`
- **This is intentional** - the frontend uses simpler node types that should work with the installed models

**Potential Risk:** The `CheckpointLoaderSimple` may not properly configure SDXL-specific features compared to `CheckPointLoaderSDXL`. The CLIP text encoding is simplified (single CLIPTextEncode vs dual CLIPTextEncodeSDXL for base+refiner).

### 5.3 Model Filename Validation

| Model Path (Dockerfile) | Frontend Reference | Status |
|-------------------------|-------------------|--------|
| `/home/comfyui/models/checkpoints/lustifySDXLNSFW_ggwpV7.safetensors` | `lustifySDXLNSFW_ggwpV7.safetensors` | ✅ Match |
| `/home/comfyui/models/ipadapter/ip-adapter-faceid_sdxl.bin` | `ip-adapter-faceid_sdxl.bin` | ✅ Match |
| `/home/comfyui/models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` | `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` | ✅ Match |
| `/home/comfyui/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors` | Not referenced in frontend | ℹ️ Extra model |

**✅ All model filenames match between frontend and Dockerfile**

### 5.4 Missing CLIP Model

**Observation:** The Dockerfile downloads `t5xxl_fp16.safetensors` via `CIVITAI_TOKEN` in the commented curl commands but does NOT actually download it.

```dockerfile
# Not executed (commented out in workflow.json but not Dockerfile):
# curl -L ... t5xxl_fp16.safetensors ...

# Frontend uses CheckpointLoaderSimple which extracts CLIP from checkpoint
# So no standalone CLIP loader needed
```

**Verdict:** The frontend uses `CheckpointLoaderSimple` which extracts CLIP from the checkpoint. This should work as long as the checkpoint file contains CLIP weights (which SDXL checkpoints do).

---

## 6. Available Models in Container

### 6.1 Models Downloaded in Dockerfile

```dockerfile
# IPAdapter Models
ENV CIVITAI_TOKEN=715db9acbf5c71d8c82fc7cfc8ce2529

RUN echo "Downloading IPAdapter FaceID..." && \
    curl -L -o /home/comfyui/models/ipadapter/ip-adapter-faceid_sdxl.bin \
        "https://civitai.com/api/download/models/215861?token=715db9acbf5c71d8c82fc7cfc8ce2529" && \
    curl -L -o /home/comfyui/models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors \
        "https://civitai.com/api/download/models/212354?token=715db9acbf5c71d8c82fc7cfc8ce2529" && \
    curl -L -o /home/comfyui/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors \
        "https://civitai.com/api/download/models/247653?token=715db9acbf5c71d8c82fc7cfc8ce2529"

# Main Model
ARG LUSTIFY_MODEL_ID=2155386
RUN curl -L -o /home/comfyui/models/checkpoints/lustifySDXLNSFW_ggwpV7.safetensors \
        "https://civitai.com/api/download/models/${LUSTIFY_MODEL_ID}?token=715db9acbf5c71d8c82fc7cfc8ce2529"
```

### 6.2 Complete Model List

| Model Type | Filename | Path | Used By |
|------------|----------|------|---------|
| Checkpoint | `lustifySDXLNSFW_ggwpV7.safetensors` | `models/checkpoints/` | Node 1 (`CheckpointLoaderSimple`) |
| IPAdapter FaceID | `ip-adapter-faceid_sdxl.bin` | `models/ipadapter/` | Node 4 (`IPAdapterModelLoader`) |
| CLIP Vision | `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors` | `models/clip_vision/` | Node 3 (`CLIPVisionLoader`) |
| IPAdapter Plus | `ip-adapter-plus_sdxl_vit-h.safetensors` | `models/ipadapter/` | Not used (extra) |

### 6.3 Custom Nodes

```dockerfile
# Installed IPAdapter Plus custom node
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
        /home/comfyui/custom_nodes/ComfyUI_IPAdapter_plus
```

This custom node provides:
- `IPAdapterModelLoader`
- `IPAdapterAdvanced`
- `CLIPVisionLoader`

---

## 7. Complete Example Request

### 7.1 Frontend Request to RunPod

```javascript
// submitRunPodJob() in comfyui-runpod-service.ts
const response = await fetch(`${RUNPOD_BASE}/v2/${endpointId}/run`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
  },
  body: JSON.stringify({
    input: {
      "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": { "ckpt_name": "lustifySDXLNSFW_ggwpV7.safetensors" }
      },
      "3": {
        "class_type": "CLIPVisionLoader",
        "inputs": { "clip_name": "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" }
      },
      "4": {
        "class_type": "IPAdapterModelLoader",
        "inputs": { "ipadapter_file": "ip-adapter-faceid_sdxl.bin" }
      },
      "6": {
        "class_type": "LoadImage",
        "inputs": { "image": "base64-encoded-face-image-data" }
      },
      "5": {
        "class_type": "IPAdapterAdvanced",
        "inputs": {
          "model": ["1", 0],
          "clip": ["1", 1],
          "ipadapter": ["4", 0],
          "clip_vision": ["3", 0],
          "image": ["6", 0],
          "weight": 1.0,
          "weight_faceidv2": 1.0,
          "combine_embeds": "Average",
          "embeds_scaling": "V only"
        }
      },
      "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "a portrait of a beautiful woman, high quality, detailed",
          "clip": ["1", 1]
        }
      },
      "8": {
        "class_type": "CLIPTextEncode",
        "inputs": {
          "text": "ugly, blurry, low quality, deformed, bad anatomy",
          "clip": ["1", 1]
        }
      },
      "13": {
        "class_type": "EmptyLatentImage",
        "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }
      },
      "10": {
        "class_type": "KSampler",
        "inputs": {
          "model": ["5", 0],
          "positive": ["7", 0],
          "negative": ["8", 0],
          "latent_image": ["13", 0],
          "seed": 123456789,
          "steps": 20,
          "cfg": 8.0,
          "sampler_name": "euler",
          "scheduler": "normal",
          "denoise": 1.0
        }
      },
      "11": {
        "class_type": "VAEDecode",
        "inputs": { "samples": ["10", 0], "vae": ["1", 2] }
      },
      "12": {
        "class_type": "SaveImage",
        "inputs": { "images": ["11", 0], "filename_prefix": "ComfyUI" }
      }
    }
  })
});
```

### 7.2 Handler Response

```json
{
  "id": "abc123-uuid",
  "status": "COMPLETED",
  "output": {
    "images": [
      {
        "filename": "ComfyUI_0001.png",
        "data": "iVBORw0KGgoAAAANSUhEUgAAAnAAAASVCAYAAAB9...",
        "type": "image/png"
      }
    ]
  }
}
```

### 7.3 Frontend Image Extraction

```typescript
// extractImageFromOutput() in comfyui-runpod-service.ts
const extractImageFromOutput = async (
  output: ComfyUIRunPodJob['output'],
): Promise<{ base64: string; mimeType: string }> => {
  if (!output) throw new Error('No output from ComfyUI worker');
  if (output.error) throw new Error(`ComfyUI error: ${output.error}`);
  
  const images = output.images as Array<{
    filename: string;
    data: string;
    type: string;
  }> | undefined;
  
  if (images && images.length > 0 && images[0].data) {
    return { base64: images[0].data, mimeType: images[0].type || 'image/png' };
  }
  
  throw new Error('No images in ComfyUI output');
};
```

---

## 8. Summary

### ✅ What's Working
1. **Model paths match** between frontend references and Dockerfile downloads
2. **RunPod API endpoints** are correctly implemented in frontend service
3. **Handler flow** properly submits to ComfyUI, polls, and collects outputs
4. **Error handling** covers common failure cases

### ⚠️ Items to Monitor
1. **Node type mismatch** between `workflow.json` (reference) and actual `buildWorkflowPrompt()` implementation - verify the simpler nodes work correctly
2. **No standalone CLIP model** - relying on checkpoint-embedded CLIP weights
3. **IPAdapter Plus model** downloaded but not used in current workflow

### 📝 Configuration Notes
- **Default settings:** steps=20, cfg=8, sampler='euler', scheduler='normal', denoise=1.0
- **Seed handling:** -1 = random, otherwise uses specified seed
- **IPAdapter weights:** default 1.0 for both regular and faceid weights
- **Timeout:** 600 seconds (10 minutes) for ComfyUI generation
- **Polling interval:** 5 seconds between status checks (max 120 attempts)
