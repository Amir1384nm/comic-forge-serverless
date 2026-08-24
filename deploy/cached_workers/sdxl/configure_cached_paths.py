"""Point the first-party ComfyUI worker at RunPod's cached HF snapshot."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from common.cached_model import force_offline_mode, resolve_snapshot_path
except ModuleNotFoundError:  # Local test/import path.
    from deploy.cached_workers.common.cached_model import (
        force_offline_mode,
        resolve_snapshot_path,
    )


def configure(output: Path | str = "/comfyui/extra_model_paths.yaml") -> Path:
    force_offline_mode()
    snapshot = resolve_snapshot_path(os.environ["MODEL_REPO_ID"])
    model_root = snapshot / "models"
    output = Path(output)
    paths = {
        "checkpoints": "Stable-diffusion",
        "loras": "Lora",
        "controlnet": "ControlNet",
        "clip_vision": "clip_vision",
        "diffusion_models": "diffusion_models",
        "unet": "unet",
        "text_encoders": "text_encoders",
        "clip": "text_encoders",
        "vae": "vae",
    }
    lines = ["cached_bundle:", f"  base_path: {model_root.as_posix()}"]
    lines.extend(f"  {key}: {value}" for key, value in paths.items())
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"cached-model: configured {os.environ['MODEL_REPO_ID']}", flush=True)
    return output


if __name__ == "__main__":
    configure()
