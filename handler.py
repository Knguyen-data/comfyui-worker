#!/usr/bin/env python3
"""
RunPod Handler for ComfyUI Worker
"""
import os
import sys
import json
import time
import asyncio
import subprocess
from datetime import datetime

# ComfyUI paths
COMFYUI_PATH = "/home/comfyui"
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


async def handler(event: dict) -> dict:
    """Main handler for RunPod"""
    try:
        # Extract input
        prompt = event.get("input", {})
        
        if not prompt:
            return {"success": False, "error": "No prompt provided"}
        
        # Run workflow
        result = await run_workflow(prompt)
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # For local testing
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
