#!/usr/bin/env python3
"""
RunPod Handler for ComfyUI Worker
Supports optional LoRA loading via lora_name/lora_weight/lora_url input params.
"""
import os
import sys
import json
import time
import asyncio
import subprocess
import tempfile
import shutil
from datetime import datetime

import runpod

# ComfyUI paths
COMFYUI_PATH = "/home/comfyui"
LORA_DIR = os.path.join(COMFYUI_PATH, "models", "loras")
sys.path.insert(0, COMFYUI_PATH)

# Global state
comfyui_process = None
initialized = False
_init_lock = asyncio.Lock()


async def start_comfyui():
    """Start ComfyUI server"""
    global comfyui_process, initialized
    
    async with _init_lock:
        if initialized:
            return True
    
        print("Starting ComfyUI server...")
        comfyui_process = await asyncio.create_subprocess_exec(
            sys.executable, "main.py",
            "--disable-auto-launch",
            "--disable-metadata",
            "--port", "8188",
            cwd=COMFYUI_PATH,
            stdout=None,
            stderr=None
        )
        
        # Wait for server to be ready
        max_wait = 120
        wait_time = 0
        while wait_time < max_wait:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:8188", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            print("ComfyUI server ready!")
                            initialized = True
                            return True
            except:
                pass
            await asyncio.sleep(1)
            wait_time += 1
            print(f"Waiting for ComfyUI... ({wait_time}s)")
        
        print("Failed to start ComfyUI within timeout")
        return False
    
    return True


async def run_workflow(prompt: dict) -> dict:
    """Execute a ComfyUI workflow"""
    global comfyui_process
    
    if not initialized:
        if not await start_comfyui():
            return {"success": False, "error": "ComfyUI failed to start"}
    
    try:
        import aiohttp
        
        # Submit workflow
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:8188/api/prompt",
                json={"prompt": prompt},
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"API error: {error}")
                result = await resp.json()
                prompt_id = result["prompt_id"]
        
        # Poll for completion with timeout
        poll_start = time.time()
        poll_timeout = 600  # 10 minutes max
        while True:
            # Check timeout
            if time.time() - poll_start > poll_timeout:
                return {"success": False, "error": "Polling timeout exceeded (600s)"}

            await asyncio.sleep(2)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:8188/history/{prompt_id}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        history = await resp.json()
                        if prompt_id in history:
                            entry = history[prompt_id]
                            status_info = entry.get("status", {})
                            if status_info.get("completed", False):
                                outputs = entry.get("outputs", {})
                                # Extract image filenames from outputs
                                images = []
                                for node_id, node_output in outputs.items():
                                    if "images" in node_output:
                                        for img in node_output["images"]:
                                            images.append({
                                                "filename": img["filename"],
                                                "subfolder": img.get("subfolder", ""),
                                                "type": img.get("type", "output"),
                                            })
                                return {"success": True, "outputs": outputs, "images": images}
                            elif status_info.get("status_str") == "error":
                                msgs = entry.get("status", {}).get("messages", [])
                                error_msg = str(msgs) if msgs else "Workflow execution failed"
                                return {"success": False, "error": error_msg}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def download_lora(url: str, filename: str) -> str:
    """Download a LoRA .safetensors file to ComfyUI's loras directory.
    Skips if already cached. Uses atomic write (temp -> rename)."""
    os.makedirs(LORA_DIR, exist_ok=True)
    dest = os.path.join(LORA_DIR, filename)

    if os.path.exists(dest):
        print(f"LoRA already cached: {filename}")
        return dest

    print(f"Downloading LoRA: {url} -> {dest}")
    import urllib.request

    tmp_fd, tmp_path = tempfile.mkstemp(dir=LORA_DIR, suffix=".tmp")
    try:
        os.close(tmp_fd)
        urllib.request.urlretrieve(url, tmp_path)
        shutil.move(tmp_path, dest)
        print(f"LoRA downloaded: {filename} ({os.path.getsize(dest)} bytes)")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"LoRA download failed: {e}") from e

    return dest


def inject_lora_into_prompt(prompt: dict, lora_name: str, lora_weight: float) -> dict:
    """Inject a LoraLoader node between model/clip loaders and downstream nodes.

    Supports Flux Dev workflow:
    - UNETLoader "1" provides MODEL[0]
    - DualCLIPLoader "2" provides CLIP[0]

    Inserts node "20" (LoraLoader) that takes model+clip and rewires
    KSampler "10", CLIP nodes "7"/"8", and IPAdapter "33" if present.
    """
    prompt = dict(prompt)  # shallow copy

    # Detect model and clip source nodes
    model_src = ["1", 0]  # UNETLoader or CheckpointLoaderSimple
    clip_src = ["2", 0]   # DualCLIPLoader for Flux

    # Fallback: if no node "2", CLIP comes from checkpoint "1"
    if "2" not in prompt:
        clip_src = ["1", 1]

    prompt["20"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": lora_name,
            "strength_model": lora_weight,
            "strength_clip": lora_weight,
            "model": model_src,
            "clip": clip_src,
        },
    }

    # Rewire CLIP text encode nodes to use LoRA clip output
    for nid in ("7", "8"):
        if nid in prompt:
            node = prompt[nid]
            if isinstance(node, dict) and "inputs" in node:
                inputs = node["inputs"]
                if isinstance(inputs.get("clip"), list):
                    src = inputs["clip"][0]
                    if src == model_src[0] or src == clip_src[0]:
                        inputs["clip"] = ["20", 1]

    # Rewire KSampler model input
    if "10" in prompt:
        ks_inputs = prompt["10"].get("inputs", {})
        model_ref = ks_inputs.get("model")
        if isinstance(model_ref, list) and model_ref[0] == model_src[0]:
            ks_inputs["model"] = ["20", 0]

    # Rewire IPAdapter if present (node "33" for Flux, "5" for legacy)
    for ipa_nid in ("33", "5"):
        if ipa_nid in prompt:
            ipa_inputs = prompt[ipa_nid].get("inputs", {})
            model_ref = ipa_inputs.get("model")
            if isinstance(model_ref, list) and model_ref[0] == model_src[0]:
                ipa_inputs["model"] = ["20", 0]

    return prompt


async def upload_base64_images(prompt: dict) -> dict:
    """Scan prompt for LoadImage nodes with base64 data and upload them to ComfyUI."""
    import aiohttp
    import base64

    for node_id, node in list(prompt.items()):
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "LoadImage":
            continue
        image_val = node.get("inputs", {}).get("image", "")
        if not isinstance(image_val, str) or len(image_val) < 200:
            continue  # Already a filename, not base64

        # Decode base64 image data
        try:
            # Strip data URI prefix if present
            b64_data = image_val
            if b64_data.startswith("data:"):
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
        except Exception as e:
            print(f"Failed to decode base64 for node {node_id}: {e}")
            continue

        # Determine extension from magic bytes
        ext = "jpg"
        if img_bytes[:4] == b'\x89PNG':
            ext = "png"
        elif img_bytes[:4] == b'RIFF':
            ext = "webp"

        filename = f"upload_{node_id}_{int(time.time())}.{ext}"

        # Upload to ComfyUI via /upload/image API
        try:
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field('image', img_bytes, filename=filename,
                               content_type=f'image/{ext}')
                async with session.post(
                    "http://127.0.0.1:8188/upload/image",
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        uploaded_name = result.get("name", filename)
                        node["inputs"]["image"] = uploaded_name
                        print(f"Uploaded base64 image for node {node_id} -> {uploaded_name}")
                    else:
                        error = await resp.text()
                        print(f"Upload failed for node {node_id}: {error}")
        except Exception as e:
            print(f"Failed to upload image for node {node_id}: {e}")

    return prompt


async def handler(event: dict) -> dict:
    """Main handler for RunPod.
    
    Accepts optional top-level input fields for LoRA:
      - lora_name: filename of the .safetensors in models/loras/
      - lora_weight: float (default 0.8)
      - lora_url: URL to download the LoRA from (cached after first download)
    """
    try:
        raw_input = event.get("input", {})
        
        if not raw_input:
            return {"success": False, "error": "No input provided"}

        # Extract LoRA params (top-level, not inside the prompt)
        lora_name = raw_input.pop("lora_name", None)
        lora_weight = float(raw_input.pop("lora_weight", 0.8))
        lora_url = raw_input.pop("lora_url", None)

        # Download LoRA if URL provided
        if lora_url and lora_name:
            try:
                download_lora(lora_url, lora_name)
            except Exception as e:
                return {"success": False, "error": f"LoRA download failed: {e}"}

        # The remaining input is the workflow prompt
        prompt = raw_input

        # Upload any base64 images embedded in LoadImage nodes
        prompt = await upload_base64_images(prompt)

        # Inject LoRA node into workflow if requested
        if lora_name:
            prompt = inject_lora_into_prompt(prompt, lora_name, lora_weight)

        # Run workflow
        result = await run_workflow(prompt)
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import sys
    
    if "--local" in sys.argv:
        # For local testing with FastAPI
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        
        app = FastAPI()
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        class InputModel(BaseModel):
            prompt: dict
        
        @app.post("/runsync")
        async def run_sync(input_data: InputModel):
            result = await handler({"input": input_data.dict()})
            return result
        
        @app.get("/health")
        async def health():
            return {"status": "ok", "initialized": initialized}
        
        print("Starting local server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # RunPod serverless production entrypoint
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
