"""RunPod queue worker for the persistent Qwen/ComfyUI edit engine."""

from __future__ import annotations

import base64
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
import runpod


PORT = int(os.environ.get("COMFY_PORT", "7860"))
BASE = f"http://127.0.0.1:{PORT}"
START_TIMEOUT = int(os.environ.get("COMFY_START_TIMEOUT", "900"))
_start_lock = threading.Lock()
_process: subprocess.Popen | None = None


def _ready() -> bool:
    try:
        response = httpx.get(f"{BASE}/system_stats", timeout=3)
        return response.status_code == 200 and "devices" in response.text
    except Exception:
        return False


def ensure_comfy() -> None:
    global _process
    if _ready():
        return
    with _start_lock:
        if _ready():
            return
        directories = [Path("/runpod-volume/ComfyUI"), Path("/workspace/ComfyUI")]
        pythons = [
            Path("/runpod-volume/venvs/comfyui/bin/python"),
            Path("/workspace/venvs/comfyui/bin/python"),
        ]
        directory = next((p for p in directories if (p / "main.py").is_file()), None)
        python = next((p for p in pythons if p.is_file()), None)
        if not directory or not python:
            raise RuntimeError("Persistent ComfyUI installation is missing")
        log = open("/tmp/serverless-comfy.log", "a", encoding="utf-8")
        command = [
            str(python), "main.py", "--listen", "127.0.0.1",
            "--port", str(PORT),
        ]
        if os.environ.get("COMFY_LOWVRAM", "0") in ("1", "true", "yes"):
            command.append("--lowvram")
        _process = subprocess.Popen(
            command, cwd=str(directory), stdout=log, stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            if _ready():
                return
            if _process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited during boot with code {_process.returncode}"
                )
            time.sleep(3)
        raise TimeoutError(f"ComfyUI did not become ready in {START_TIMEOUT}s")


def _upload(item: dict) -> None:
    name = Path(str(item.get("name") or "input.png")).name
    raw = base64.b64decode(str(item.get("data") or "").split(",", 1)[-1])
    if not raw or len(raw) > 25 * 1024 * 1024:
        raise ValueError("invalid or oversized input image")
    response = httpx.post(
        f"{BASE}/upload/image",
        files={"image": (name, raw, "image/png")},
        data={"type": "input", "overwrite": "true"},
        timeout=120,
    )
    response.raise_for_status()


def _list_loras() -> list[str]:
    """List every LoRA on the persistent volume without starting ComfyUI."""
    roots = [
        Path("/runpod-volume/ComfyUI/models/loras"),
        Path("/workspace/ComfyUI/models/loras"),
    ]
    names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}:
                names.add(path.relative_to(root).as_posix())
    return sorted(names, key=str.casefold)


def _run_workflow(workflow: dict, timeout: int) -> bytes:
    if not isinstance(workflow, dict) or not workflow or len(workflow) > 160:
        raise ValueError("invalid ComfyUI workflow")
    client_id = uuid.uuid4().hex
    queued = httpx.post(
        f"{BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    if queued.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected workflow: {queued.text[:500]}")
    prompt_id = (queued.json() or {}).get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI returned no prompt id")
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = httpx.get(f"{BASE}/history/{prompt_id}", timeout=30).json()
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed: {status.get('messages')!s:.500}")
            for node in (entry.get("outputs") or {}).values():
                for image in node.get("images") or []:
                    result = httpx.get(
                        f"{BASE}/view",
                        params={
                            "filename": image.get("filename"),
                            "subfolder": image.get("subfolder", ""),
                            "type": image.get("type", "output"),
                        },
                        timeout=120,
                    )
                    result.raise_for_status()
                    return result.content
        time.sleep(1.5)
    raise TimeoutError(f"ComfyUI job exceeded {timeout}s")


def handler(job: dict) -> dict:
    request = job.get("input") or {}
    operation = request.get("operation")
    if operation == "list_loras":
        return {"loras": _list_loras()}
    if operation != "workflow":
        raise ValueError("unsupported ComfyUI operation")
    ensure_comfy()
    for item in request.get("images") or []:
        _upload(item)
    timeout = max(30, min(600, int(request.get("timeout") or 420)))
    started = time.time()
    image = _run_workflow(request.get("workflow") or {}, timeout)
    return {
        "image": base64.b64encode(image).decode("ascii"),
        "elapsed": round(time.time() - started, 1),
    }


runpod.serverless.start({"handler": handler})
