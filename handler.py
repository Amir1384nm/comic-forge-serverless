"""RunPod queue-worker adapter for the project's existing Forge volume."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import runpod


PORT = int(os.environ.get("FORGE_PORT", "3001"))
BASE = f"http://127.0.0.1:{PORT}"
START_TIMEOUT = int(os.environ.get("FORGE_START_TIMEOUT", "900"))
ALLOWED_OPERATIONS = {
    "txt2img": "/sdapi/v1/txt2img",
    "img2img": "/sdapi/v1/img2img",
}

_start_lock = threading.Lock()
_forge_process: subprocess.Popen | None = None
LOG_PATH = Path("/tmp/serverless-forge.log")


def _log_tail() -> str:
    try:
        return LOG_PATH.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        return "Forge log is unavailable"


def _ready() -> bool:
    try:
        response = httpx.get(f"{BASE}/sdapi/v1/sd-models", timeout=3)
        data = response.json() if response.status_code == 200 else None
        return isinstance(data, list) and bool(data)
    except Exception:
        return False


def _paths() -> tuple[Path, Path]:
    dirs = [
        Path("/runpod-volume/stable-diffusion-webui-forge"),
        Path("/workspace/stable-diffusion-webui-forge"),
        Path("/stable-diffusion-webui-forge"),
    ]
    pythons = [
        Path("/venv/bin/python"),
        Path("/runpod-volume/venvs/stable-diffusion-webui-forge/bin/python"),
        Path("/workspace/venvs/stable-diffusion-webui-forge/bin/python"),
    ]
    directory = next((p for p in dirs if (p / "launch.py").is_file()), None)
    python = next((p for p in pythons if p.is_file()), None)
    if not directory or not python:
        raise RuntimeError("Forge code or Python environment is missing")
    return directory, python


def ensure_forge() -> None:
    global _forge_process
    if _ready():
        return
    with _start_lock:
        if _ready():
            return
        directory, python = _paths()
        log = open(LOG_PATH, "a", encoding="utf-8")
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(python.parent.parent)
        env["PATH"] = f"{python.parent}:{env.get('PATH', '')}"
        print(f"[forge] starting directory={directory} python={python}", flush=True)
        _forge_process = subprocess.Popen(
            [
                str(python), "launch.py", "--listen", "--port", str(PORT),
                "--api", "--xformers", "--enable-insecure-extension-access",
                "--no-half-vae",
            ],
            cwd=str(directory), env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            if _ready():
                return
            if _forge_process.poll() is not None:
                raise RuntimeError(
                    f"Forge exited during boot with code {_forge_process.returncode}; "
                    f"log tail:\n{_log_tail()}"
                )
            time.sleep(3)
        raise TimeoutError(
            f"Forge did not become ready in {START_TIMEOUT}s; log tail:\n{_log_tail()}"
        )


def _validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    for key in ("width", "height"):
        if int(payload.get(key, 1024)) > 2048:
            raise ValueError(f"{key} exceeds the safe limit")
    if int(payload.get("steps", 28)) > 80:
        raise ValueError("steps exceeds the safe limit")
    if int(payload.get("batch_size", 1)) > 4:
        raise ValueError("batch_size exceeds the safe limit")


def handler(job: dict) -> dict:
    request = job.get("input") or {}
    operation = str(request.get("operation") or "")
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError("unsupported Forge operation")
    payload = request.get("payload") or {}
    _validate_payload(payload)
    ensure_forge()
    timeout = float(request.get("timeout") or 600)
    timeout = max(10, min(600, timeout))
    response = httpx.post(
        BASE + ALLOWED_OPERATIONS[operation], json=payload, timeout=timeout
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Forge returned HTTP {response.status_code}: {response.text[:500]}"
        )
    data = response.json()
    if not isinstance(data, dict) or not data.get("images"):
        raise RuntimeError("Forge returned no images")
    return data


runpod.serverless.start({"handler": handler})
