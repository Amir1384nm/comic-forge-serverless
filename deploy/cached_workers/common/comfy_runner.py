"""Local ComfyUI process and API client shared by Qwen/H3 workers."""

from __future__ import annotations

import base64
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx

from .cached_model import (
    force_offline_mode,
    resolve_snapshot_path,
    verify_bundle,
    write_comfy_extra_paths,
)


class ComfyRunner:
    def __init__(self, *, repo_id: str, port: int = 8188):
        force_offline_mode()
        self.repo_id = repo_id
        self.port = int(port)
        self.base = f"http://127.0.0.1:{self.port}"
        self.comfy_root = Path(os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
        self.snapshot = resolve_snapshot_path(repo_id)
        self.manifest = verify_bundle(self.snapshot)
        self.extra_paths = write_comfy_extra_paths(
            self.snapshot, Path("/tmp/cached-extra-model-paths.yaml")
        )
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._initialized_seconds: float | None = None
        self.log_path = Path("/tmp/cached-comfy.log")

    def ready(self) -> bool:
        try:
            response = httpx.get(f"{self.base}/system_stats", timeout=3)
            return response.status_code == 200 and isinstance(response.json(), dict)
        except Exception:
            return False

    def _log_tail(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            return "ComfyUI log unavailable"

    def ensure_started(self) -> float:
        if self.ready():
            return float(self._initialized_seconds or 0.0)
        with self._lock:
            if self.ready():
                return float(self._initialized_seconds or 0.0)
            if not (self.comfy_root / "main.py").is_file():
                raise RuntimeError("ComfyUI runtime is missing from worker image")
            started = time.monotonic()
            log = self.log_path.open("w", encoding="utf-8")
            command = [
                os.environ.get("COMFY_PYTHON", "/opt/venv/bin/python"),
                "main.py", "--listen", "127.0.0.1", "--port", str(self.port),
                "--extra-model-paths-config", str(self.extra_paths),
            ]
            memory_mode = os.environ.get("COMFY_MEMORY_MODE", "").strip()
            if memory_mode in {"lowvram", "novram", "highvram"}:
                command.append("--" + memory_mode)
            self._process = subprocess.Popen(
                command,
                cwd=self.comfy_root,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            timeout = int(os.environ.get("RUNPOD_INIT_TIMEOUT", "800"))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.ready():
                    self._initialized_seconds = round(time.monotonic() - started, 3)
                    return self._initialized_seconds
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"ComfyUI exited during initialization; {self._log_tail()}"
                    )
                time.sleep(2)
            raise TimeoutError(
                f"ComfyUI did not initialize in {timeout}s; {self._log_tail()}"
            )

    def list_loras(self) -> list[str]:
        root = self.snapshot / "models" / "loras"
        if not root.is_dir():
            return []
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}
        )

    def upload_images(self, items: list[dict]) -> None:
        for item in items:
            name = Path(str(item.get("name") or "input.png")).name
            encoded = str(item.get("data") or "").split(",", 1)[-1]
            try:
                raw = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError("Invalid base64 input image") from exc
            if not raw or len(raw) > 25 * 1024 * 1024:
                raise ValueError("Input image is empty or exceeds 25 MB")
            media = str(item.get("mime_type") or "image/png")
            response = httpx.post(
                f"{self.base}/upload/image",
                files={"image": (name, raw, media)},
                data={"type": "input", "overwrite": "true"},
                timeout=120,
            )
            response.raise_for_status()

    def run_workflow(self, workflow: dict, *, timeout: int) -> tuple[list[dict], float]:
        if not isinstance(workflow, dict) or not workflow or len(workflow) > 240:
            raise ValueError("Invalid ComfyUI workflow")
        timeout = max(30, min(3600, int(timeout)))
        started = time.monotonic()
        queued = httpx.post(
            f"{self.base}/prompt",
            json={"prompt": workflow, "client_id": uuid.uuid4().hex},
            timeout=60,
        )
        if queued.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow: {queued.text[:500]}")
        prompt_id = str((queued.json() or {}).get("prompt_id") or "")
        if not prompt_id:
            raise RuntimeError("ComfyUI returned no prompt id")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = httpx.get(f"{self.base}/history/{prompt_id}", timeout=30).json()
            entry = history.get(prompt_id)
            if not entry:
                time.sleep(1.5)
                continue
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed: {str(status.get('messages'))[:800]}")
            artifacts = self._download_outputs(entry.get("outputs") or {})
            if artifacts:
                return artifacts, round(time.monotonic() - started, 3)
            raise RuntimeError("ComfyUI completed without an output artifact")
        raise TimeoutError(f"ComfyUI workflow exceeded {timeout}s")

    def _download_outputs(self, outputs: dict) -> list[dict]:
        artifacts: list[dict] = []
        total = 0
        for node in outputs.values():
            if not isinstance(node, dict):
                continue
            for kind in ("images", "video", "videos", "audio"):
                rows = node.get(kind) or []
                if isinstance(rows, dict):
                    rows = [rows]
                for item in rows:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    response = httpx.get(
                        f"{self.base}/view",
                        params={
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        },
                        timeout=300,
                    )
                    response.raise_for_status()
                    total += len(response.content)
                    if total > 80 * 1024 * 1024:
                        raise RuntimeError("Output artifacts exceed the 80 MB safety limit")
                    filename = Path(str(item["filename"])).name
                    artifacts.append({
                        "filename": filename,
                        "mime_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
                        "data": base64.b64encode(response.content).decode("ascii"),
                        "size": len(response.content),
                    })
        return artifacts

