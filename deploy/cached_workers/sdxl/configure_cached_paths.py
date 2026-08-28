"""Point the first-party ComfyUI worker at RunPod's cached HF snapshot."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

try:
    from common.cached_model import force_offline_mode, resolve_model_source
except ModuleNotFoundError:  # Local test/import path.
    from deploy.cached_workers.common.cached_model import (
        force_offline_mode,
        resolve_model_source,
    )


DEPTH_EXPECTED_SIZE = 99_218_434
DEPTH_EXPECTED_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"


def _verify_depth_asset(path: Path) -> None:
    """Verify the post-bundle depth asset that is not in the core manifest."""
    if not path.is_file():
        raise RuntimeError(f"Cached depth annotator is missing: {path}")
    if path.stat().st_size != DEPTH_EXPECTED_SIZE:
        raise RuntimeError(f"Cached depth annotator has the wrong size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != DEPTH_EXPECTED_SHA256:
        raise RuntimeError(f"Cached depth annotator failed SHA256 verification: {path}")


def configure(output: Path | str = "/comfyui/extra_model_paths.yaml") -> Path:
    force_offline_mode()
    source = resolve_model_source(os.environ.get("MODEL_REPO_ID"))
    model_root = source.model_root
    output = Path(output)
    paths = {
        "checkpoints": "Stable-diffusion",
        "loras": "Lora",
        "controlnet": "ControlNet",
        "ipadapter": "ControlNet",
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

    # The source Forge bundle already contains the exact DWPose ONNX files.
    # Link them into the directory layout expected by comfyui_controlnet_aux so
    # the offline production worker never downloads annotators at inference.
    openpose = model_root / "ControlNetPreprocessor" / "openpose"
    dwpose = Path(os.environ.get(
        "AUX_ANNOTATOR_CKPTS_PATH",
        "/comfyui/custom_nodes/comfyui_controlnet_aux/ckpts",
    )) / "yzd-v" / "DWPose"
    dwpose.mkdir(parents=True, exist_ok=True)
    for filename in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        source = openpose / filename
        if not source.is_file():
            raise RuntimeError(f"Cached DWPose asset is missing: {source}")
        target = dwpose / filename
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

    depth_source = (
        model_root / "ControlNetPreprocessor" / "depth" /
        "depth_anything_v2_vits.pth"
    )
    _verify_depth_asset(depth_source)
    depth_target = (
        Path(os.environ.get(
            "AUX_ANNOTATOR_CKPTS_PATH",
            "/comfyui/custom_nodes/comfyui_controlnet_aux/ckpts",
        )) / "depth-anything" / "Depth-Anything-V2-Small" /
        "depth_anything_v2_vits.pth"
    )
    depth_target.parent.mkdir(parents=True, exist_ok=True)
    if not (depth_target.is_symlink() and depth_target.resolve() == depth_source.resolve()):
        if depth_target.exists() or depth_target.is_symlink():
            depth_target.unlink()
        depth_target.symlink_to(depth_source)
    print(f"model-source: configured {source.kind} at {model_root}", flush=True)
    return output


if __name__ == "__main__":
    configure()
