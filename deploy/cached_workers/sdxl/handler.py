"""RunPod cached-model worker for one isolated SDXL site model."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import runpod

from common.cached_model import (
    force_offline_mode,
    link_tree,
    resolve_snapshot_path,
    verify_bundle,
)


PORT = int(os.environ.get("FORGE_PORT", "3001"))
BASE = f"http://127.0.0.1:{PORT}"
REPO_ID = os.environ["MODEL_REPO_ID"]
CHECKPOINT = Path(os.environ["MODEL_CHECKPOINT"]).name
ALLOWED_OPERATIONS = {
    "txt2img": "/sdapi/v1/txt2img",
    "img2img": "/sdapi/v1/img2img",
}
LOG_PATH = Path("/tmp/cached-forge.log")
_lock = threading.Lock()
_process: subprocess.Popen | None = None
_initialized_seconds: float | None = None

force_offline_mode()
SNAPSHOT = resolve_snapshot_path(REPO_ID)
MANIFEST = verify_bundle(SNAPSHOT)


def _forge_root() -> Path:
    candidates = (
        Path("/opt/stable-diffusion-webui-forge"),
        Path("/stable-diffusion-webui-forge"),
        Path("/workspace/stable-diffusion-webui-forge"),
    )
    root = next((path for path in candidates if (path / "launch.py").is_file()), None)
    if not root:
        raise RuntimeError("Forge runtime is missing from worker image")
    return root


def _python() -> Path:
    candidates = (Path("/venv/bin/python"), Path("/opt/venv/bin/python"))
    python = next((path for path in candidates if path.is_file()), None)
    if not python:
        raise RuntimeError("Forge Python runtime is missing from worker image")
    return python


def _prepare_models(root: Path) -> Path:
    link_tree(SNAPSHOT / "models", root / "models")
    checkpoint = root / "models" / "Stable-diffusion" / CHECKPOINT
    if not checkpoint.is_file():
        raise RuntimeError(f"Configured checkpoint is absent from cached bundle: {CHECKPOINT}")
    return checkpoint


def _ready() -> bool:
    try:
        response = httpx.get(f"{BASE}/sdapi/v1/sd-models", timeout=3)
        return response.status_code == 200 and bool(response.json())
    except Exception:
        return False


def _log_tail() -> str:
    try:
        return LOG_PATH.read_text(encoding="utf-8", errors="replace")[-3000:]
    except OSError:
        return "Forge log unavailable"


def ensure_forge() -> float:
    global _process, _initialized_seconds
    if _ready():
        return float(_initialized_seconds or 0.0)
    with _lock:
        if _ready():
            return float(_initialized_seconds or 0.0)
        root = _forge_root()
        checkpoint = _prepare_models(root)
        started = time.monotonic()
        log = LOG_PATH.open("w", encoding="utf-8")
        command = [
            str(_python()), "launch.py", "--listen", "--port", str(PORT),
            "--api", "--xformers", "--enable-insecure-extension-access",
            "--no-half-vae", "--skip-install", "--ckpt", str(checkpoint),
        ]
        _process = subprocess.Popen(
            command, cwd=root, env=os.environ.copy(), stdout=log,
            stderr=subprocess.STDOUT,
        )
        timeout = int(os.environ.get("RUNPOD_INIT_TIMEOUT", "800"))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _ready():
                _initialized_seconds = round(time.monotonic() - started, 3)
                return _initialized_seconds
            if _process.poll() is not None:
                raise RuntimeError(f"Forge exited during initialization; {_log_tail()}")
            time.sleep(2)
        raise TimeoutError(f"Forge did not initialize in {timeout}s; {_log_tail()}")


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
    override = payload.get("override_settings") or {}
    requested = str(override.get("sd_model_checkpoint") or "").strip()
    if requested and Path(requested).name != CHECKPOINT:
        raise ValueError("This endpoint serves a different isolated checkpoint")


def handler(job: dict) -> dict:
    request = job.get("input") or {}
    operation = str(request.get("operation") or "")
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError("Unsupported Forge operation")
    payload = request.get("payload") or {}
    _validate_payload(payload)
    payload.setdefault("override_settings", {})["sd_model_checkpoint"] = CHECKPOINT
    total_started = time.monotonic()
    initialized = ensure_forge()
    inference_started = time.monotonic()
    timeout = max(10, min(900, int(request.get("timeout") or 600)))
    response = httpx.post(
        BASE + ALLOWED_OPERATIONS[operation], json=payload, timeout=timeout
    )
    if response.status_code != 200:
        raise RuntimeError(f"Forge returned HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    if not isinstance(data, dict) or not data.get("images"):
        raise RuntimeError("Forge returned no images")
    data["cached_worker_metrics"] = {
        "initialization_seconds": initialized,
        "inference_seconds": round(time.monotonic() - inference_started, 3),
        "total_handler_seconds": round(time.monotonic() - total_started, 3),
    }
    return data


runpod.serverless.start({"handler": handler})
