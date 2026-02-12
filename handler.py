#!/usr/bin/env python3
"""
RunPod Serverless Handler for ComfyUI Worker
"""
import os
import sys
import json
import time
import base64
import subprocess
import urllib.request
import urllib.error

import runpod

# ComfyUI paths
COMFYUI_PATH = "/home/comfyui"
COMFYUI_OUTPUT = os.path.join(COMFYUI_PATH, "output")
COMFYUI_LOG = "/tmp/comfyui.log"

# Global state
comfyui_process = None
_log_file = None


def start_comfyui():
    """Start ComfyUI server as a subprocess.

    Redirects stdout/stderr to a log file to prevent pipe buffer deadlock.
    Previously used subprocess.PIPE without draining, which caused ComfyUI
    to block on write when the 64KB OS pipe buffer filled up.
    """
    global comfyui_process, _log_file

    if comfyui_process is not None:
        # Check if the existing process is still alive
        if comfyui_process.poll() is None:
            return
        # Process died — restart it
        print(f"ComfyUI process died with exit code {comfyui_process.returncode}, restarting...")
        comfyui_process = None
        if _log_file:
            _log_file.close()
            _log_file = None

    print("Starting ComfyUI server...")

    # Redirect to a log file instead of PIPE to avoid pipe buffer deadlock.
    # The log file never fills up / blocks the subprocess.
    _log_file = open(COMFYUI_LOG, "w")

    comfyui_process = subprocess.Popen(
        [sys.executable, "main.py", "--disable-auto-launch", "--disable-metadata", "--port", "8188"],
        cwd=COMFYUI_PATH,
        stdout=_log_file,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    max_wait = 120
    for i in range(max_wait):
        # Check if the process crashed during startup
        if comfyui_process.poll() is not None:
            # Process exited — dump tail of log for diagnostics
            _log_file.flush()
            try:
                with open(COMFYUI_LOG, "r") as lf:
                    log_tail = lf.read()[-2000:]
            except Exception:
                log_tail = "(could not read log)"
            raise RuntimeError(
                f"ComfyUI process exited with code {comfyui_process.returncode} during startup.\n"
                f"Log tail:\n{log_tail}"
            )

        try:
            req = urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            if req.status == 200:
                print(f"ComfyUI server ready after {i+1}s")
                return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(1)
        if i % 10 == 0:
            print(f"Waiting for ComfyUI... ({i}s)")

    raise RuntimeError("ComfyUI failed to start within 120s")


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


def poll_history(prompt_id, timeout=600):
    """Poll ComfyUI history endpoint until the prompt completes.

    Also checks whether the ComfyUI process is still alive — if it crashed,
    we fail fast instead of waiting the full timeout.
    """
    start = time.time()
    poll_count = 0
    while time.time() - start < timeout:
        # Check if ComfyUI process crashed
        if comfyui_process is not None and comfyui_process.poll() is not None:
            elapsed = time.time() - start
            try:
                with open(COMFYUI_LOG, "r") as lf:
                    log_tail = lf.read()[-2000:]
            except Exception:
                log_tail = "(could not read log)"
            raise RuntimeError(
                f"ComfyUI process died (exit code {comfyui_process.returncode}) "
                f"while waiting for prompt {prompt_id} after {elapsed:.0f}s.\n"
                f"Log tail:\n{log_tail}"
            )

        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10
            )
            history = json.loads(resp.read())
            if prompt_id in history:
                elapsed = time.time() - start
                print(f"Prompt {prompt_id} completed after {elapsed:.1f}s")
                return history[prompt_id]
        except (urllib.error.URLError, OSError):
            pass

        poll_count += 1
        if poll_count % 30 == 0:
            elapsed = time.time() - start
            print(f"Still waiting for prompt {prompt_id}... ({elapsed:.0f}s elapsed)")

        time.sleep(1)

    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


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


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
