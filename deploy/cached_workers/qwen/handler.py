"""RunPod cached-model worker for the current Qwen Image ComfyUI workflows."""

from __future__ import annotations

import os
import time

import runpod

from common.comfy_runner import ComfyRunner


RUNNER = ComfyRunner(
    repo_id=os.environ["MODEL_REPO_ID"],
    port=int(os.environ.get("COMFY_PORT", "8188")),
)


def handler(job: dict) -> dict:
    request = job.get("input") or {}
    operation = str(request.get("operation") or "")
    if operation == "list_loras":
        return {"loras": RUNNER.list_loras()}
    if operation != "workflow":
        raise ValueError("Unsupported Qwen worker operation")
    total_started = time.monotonic()
    initialized = RUNNER.ensure_started()
    RUNNER.upload_images(list(request.get("images") or []))
    artifacts, inference = RUNNER.run_workflow(
        request.get("workflow") or {}, timeout=int(request.get("timeout") or 600)
    )
    images = [item for item in artifacts if item["mime_type"].startswith("image/")]
    if not images:
        raise RuntimeError("Qwen workflow returned no image")
    return {
        "schema": 1,
        "artifacts": images,
        "metrics": {
            "initialization_seconds": initialized,
            "inference_seconds": inference,
            "total_handler_seconds": round(time.monotonic() - total_started, 3),
        },
    }


runpod.serverless.start({"handler": handler})
