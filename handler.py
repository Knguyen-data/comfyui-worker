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


async def start_comfyui():
    """Start ComfyUI server"""
    global comfyui_process, initialized
    
    if not initialized:
        print("Starting ComfyUI server...")
        comfyui_process = await asyncio.create_subprocess_exec(
            sys.executable, "main.py",
            "--disable-auto-launch",
            "--disable-metadata",
            "--port", "8188",
            cwd=COMFYUI_PATH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
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
        await start_comfyui()
    
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

            await asyncio.sleep(1)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:8188/api/prompt_status/{prompt_id}",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        if status["status"] == "success":
                            outputs = status["outputs"]
                            return {"success": True, "outputs": outputs}
                        elif status["status"] == "failed":
                            error_msg = status.get("error", "Unknown error")
                            return {"success": False, "error": error_msg}
                    elif resp.status != 404:
                        raise Exception(f"Status check failed: {resp.status}")
        
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


def process_base64_images(prompt: dict) -> dict:
    """Find LoadImage nodes with base64 data, save to ComfyUI input dir, replace with filename."""
    import base64
    input_dir = os.path.join(COMFYUI_PATH, "input")
    os.makedirs(input_dir, exist_ok=True)

    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "LoadImage":
            continue

        inputs = node.get("inputs", {})
        image_data = inputs.get("image", "")

        # Skip if already a filename (not base64 and not a URL)
        if not image_data or len(image_data) < 200:
            continue

        # Handle URL-based images (uploaded to storage first)
        if image_data.startswith("http://") or image_data.startswith("https://"):
            filename = f"ref_{node_id}_{int(time.time())}.jpg"
            filepath = os.path.join(input_dir, filename)
            try:
                import httpx
                with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                    resp = client.get(image_data)
                    resp.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                print(f"Downloaded ref image for node {node_id}: {filename} ({os.path.getsize(filepath)} bytes)")
                inputs["image"] = filename
            except Exception as e:
                print(f"Failed to download image for node {node_id}: {e}")
            continue

        # Strip data URL prefix if present
        raw_b64 = image_data
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        # Detect mime type from header
        ext = "png"
        if image_data.startswith("data:image/jpeg"):
            ext = "jpg"
        elif image_data.startswith("data:image/webp"):
            ext = "webp"

        filename = f"ref_{node_id}_{int(time.time())}.{ext}"
        filepath = os.path.join(input_dir, filename)

        try:
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(raw_b64))
            print(f"Saved base64 image for node {node_id}: {filename} ({os.path.getsize(filepath)} bytes)")
            inputs["image"] = filename
        except Exception as e:
            print(f"Failed to decode base64 for node {node_id}: {e}")

    return prompt


def inject_lora_into_prompt(prompt: dict, lora_name: str, lora_weight: float) -> dict:
    """Inject a LoraLoader node between the checkpoint and downstream nodes.

    Inserts node "20" (LoraLoader) that:
    - Takes model+clip from checkpoint node (CheckpointLoaderSimple)
    - Rewires KSampler and CLIP nodes to use LoRA outputs
    - If IPAdapter node exists, rewires its model input too
    """
    prompt = dict(prompt)  # shallow copy

    # 1. Find the CheckpointLoaderSimple node
    checkpoint_node_id = None
    for nid, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") == "CheckpointLoaderSimple":
            checkpoint_node_id = nid
            break

    if not checkpoint_node_id:
        print("WARNING: No CheckpointLoaderSimple found. Skipping LoRA injection.")
        return prompt

    print(f"Injecting LoRA '{lora_name}' after checkpoint node {checkpoint_node_id}")

    # 2. Create LoraLoader node
    prompt["20"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": lora_name,
            "strength_model": lora_weight,
            "strength_clip": lora_weight,
            "model": [checkpoint_node_id, 0],
            "clip": [checkpoint_node_id, 1],
        },
    }

    # 3. Rewire CLIP text encode nodes to use LoRA clip output
    for nid, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue

        if node["class_type"] in ("CLIPTextEncode", "CLIPTextEncodeSDXL"):
            inputs = node.get("inputs", {})
            # If currently using the original checkpoint, switch to LoRA
            if "clip" in inputs and isinstance(inputs["clip"], list):
                if inputs["clip"][0] == checkpoint_node_id:
                    inputs["clip"] = ["20", 1]

    # 4. Rewire KSampler model input
    for nid, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue

        if node["class_type"] == "KSampler":
            inputs = node.get("inputs", {})
            if "model" in inputs and isinstance(inputs["model"], list):
                if inputs["model"][0] == checkpoint_node_id:
                    inputs["model"] = ["20", 0]

    # 5. Rewire IPAdapter if present
    for nid, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue

        if node["class_type"] == "IPAdapter":
            inputs = node.get("inputs", {})
            if "model" in inputs and isinstance(inputs["model"], list):
                if inputs["model"][0] == checkpoint_node_id:
                    inputs["model"] = ["20", 0]

    return prompt


async def handler(event: dict) -> dict:
    """Main handler for RunPod.

    Accepts optional top-level input fields for LoRA:
      - lora_name: filename of the .safetensors in models/loras/
      - lora_weight: float (default 0.8)
      - lora_url: URL to download the LoRA from (cached after first download)

    Base64 images in LoadImage nodes are automatically saved to ComfyUI's
    input directory and replaced with filenames.
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

        # Process base64 images in LoadImage nodes
        prompt = process_base64_images(prompt)

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
