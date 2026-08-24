"""Thin SDXL adapter on RunPod's first-party ComfyUI worker image."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import runpod

from common.comfy_runner import ComfyRunner
from comfy_workflow import build_workflow


CHECKPOINT = Path(os.environ["MODEL_CHECKPOINT"]).name
_RUNNER: ComfyRunner | None = None
_RUNNER_LOCK = threading.Lock()


def _runner() -> ComfyRunner:
    """Create the cached-model runner after the worker has joined the queue.

    Keeping cache validation inside the job boundary makes a missing or partial
    model bundle fail the request instead of leaving it queued behind a worker
    process that exited before registering with RunPod.
    """
    global _RUNNER
    if _RUNNER is not None:
        return _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = ComfyRunner(
                repo_id=os.environ["MODEL_REPO_ID"],
                port=int(os.environ.get("COMFY_PORT", "8188")),
                model_layout={
                    "checkpoints": "Stable-diffusion",
                    "loras": "Lora",
                    "controlnet": "ControlNet",
                    "clip_vision": "clip_vision",
                },
            )
    return _RUNNER


def _init_image(payload: dict) -> tuple[str | None, list[dict]]:
    rows = payload.get("init_images") or []
    if not rows:
        return None, []
    encoded = str(rows[0] or "")
    return "site-init.png", [{"name": "site-init.png", "data": encoded}]


def _run(payload: dict, *, timeout: int, warmup: bool = False) -> dict:
    total_started = time.monotonic()
    runner = _runner()
    initialized = runner.ensure_started()
    init_name, images = _init_image(payload)
    if images:
        runner.upload_images(images)
    workflow, metadata = build_workflow(
        payload,
        checkpoint=CHECKPOINT,
        available_loras=runner.list_loras(),
        init_image_name=init_name,
    )
    artifacts, inference = runner.run_workflow(workflow, timeout=timeout)
    outputs = [
        item["data"] for item in artifacts if item["mime_type"].startswith("image/")
    ]
    if not outputs:
        raise RuntimeError("ComfyUI workflow returned no image")
    return {
        "images": outputs,
        "parameters": payload,
        "info": json.dumps(metadata, ensure_ascii=False),
        "metrics": {
            "runtime": "runpod-worker-comfyui",
            "initialization_seconds": initialized,
            "inference_seconds": inference,
            "total_handler_seconds": round(time.monotonic() - total_started, 3),
            "warmup": warmup,
        },
    }


def handler(job: dict) -> dict:
    request = job.get("input") or {}
    operation = str(request.get("operation") or "")
    if operation == "list_loras":
        return {"loras": _runner().list_loras()}
    if operation == "warmup":
        result = _run(
            {
                "prompt": "warmup",
                "negative_prompt": "",
                "width": 256,
                "height": 256,
                "steps": 1,
                "cfg_scale": 1,
                "sampler_name": "Euler",
                "batch_size": 1,
                "seed": 1,
            },
            timeout=300,
            warmup=True,
        )
        return {
            "ready": True,
            "checkpoint": CHECKPOINT,
            "metrics": result["metrics"],
        }
    if operation not in {"txt2img", "img2img"}:
        raise ValueError("Unsupported SDXL worker operation")
    payload = request.get("payload") or {}
    if operation == "img2img" and not payload.get("init_images"):
        raise ValueError("img2img requires init_images")
    timeout = max(30, min(900, int(request.get("timeout") or 600)))
    return _run(payload, timeout=timeout)


runpod.serverless.start({"handler": handler})
