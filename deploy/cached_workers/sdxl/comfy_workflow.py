"""Translate the site's Forge-compatible SDXL payload into core ComfyUI nodes."""

from __future__ import annotations

import re
import secrets
from pathlib import Path


_LORA_TAG = re.compile(
    r"<lora:([^:>]+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))>", re.IGNORECASE
)

_SAMPLERS = {
    "dpm++ 2m": "dpmpp_2m",
    "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ sde": "dpmpp_sde",
    "dpm++ 3m sde": "dpmpp_3m_sde",
    "euler": "euler",
    "euler a": "euler_ancestral",
    "heun": "heun",
    "lms": "lms",
    "ddim": "ddim",
    "unipc": "uni_pc",
}


def _normalized_asset(value: str) -> str:
    name = Path(str(value or "").replace("\\", "/")).name.casefold()
    for suffix in (".safetensors", ".ckpt", ".pt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def extract_loras(prompt: str, available: list[str]) -> tuple[str, list[tuple[str, float]]]:
    index: dict[str, str] = {}
    for relative in available:
        key = _normalized_asset(relative)
        if key and key not in index:
            index[key] = relative

    selected: list[tuple[str, float]] = []
    for match in _LORA_TAG.finditer(str(prompt or "")):
        requested = match.group(1).strip()
        resolved = index.get(_normalized_asset(requested))
        if not resolved:
            raise ValueError(f"Requested LoRA is absent from cached bundle: {requested}")
        weight = max(-2.0, min(2.0, float(match.group(2))))
        selected.append((resolved, weight))
        if len(selected) > 12:
            raise ValueError("At most 12 LoRAs are allowed per SDXL request")
    clean = _LORA_TAG.sub(" ", str(prompt or ""))
    return " ".join(clean.split()), selected


def sampler_settings(value: str) -> tuple[str, str]:
    raw = " ".join(str(value or "DPM++ 2M Karras").strip().split())
    scheduler = "karras" if raw.casefold().endswith(" karras") else "normal"
    if scheduler == "karras":
        raw = raw[: -len(" Karras")].strip()
    sampler = _SAMPLERS.get(raw.casefold())
    if not sampler:
        raise ValueError(f"Unsupported sampler for ComfyUI worker: {value}")
    return sampler, scheduler


def _bounded_int(payload: dict, key: str, default: int, low: int, high: int) -> int:
    value = int(payload.get(key, default))
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def build_workflow(
    payload: dict,
    *,
    checkpoint: str,
    available_loras: list[str],
    init_image_name: str | None = None,
) -> tuple[dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    width = _bounded_int(payload, "width", 1024, 64, 2048)
    height = _bounded_int(payload, "height", 1024, 64, 2048)
    if width % 8 or height % 8:
        raise ValueError("width and height must be divisible by 8")
    steps = _bounded_int(payload, "steps", 28, 1, 100)
    batch = _bounded_int(payload, "batch_size", 1, 1, 4)
    cfg = max(0.0, min(30.0, float(payload.get("cfg_scale", 7.0))))
    raw_seed = int(payload.get("seed", -1))
    seed = secrets.randbelow(2**63) if raw_seed < 0 else raw_seed % (2**64)
    sampler, scheduler = sampler_settings(str(payload.get("sampler_name") or ""))
    prompt, loras = extract_loras(str(payload.get("prompt") or ""), available_loras)
    negative = str(payload.get("negative_prompt") or "")

    graph: dict[str, dict] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": Path(checkpoint).name},
        }
    }
    model_ref: list[object] = ["1", 0]
    clip_ref: list[object] = ["1", 1]
    next_id = 2
    for lora_name, weight in loras:
        node_id = str(next_id)
        graph[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": weight,
                "strength_clip": weight,
            },
        }
        model_ref = [node_id, 0]
        clip_ref = [node_id, 1]
        next_id += 1

    positive_id, negative_id = str(next_id), str(next_id + 1)
    graph[positive_id] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": clip_ref},
    }
    graph[negative_id] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": clip_ref},
    }
    next_id += 2

    if init_image_name:
        load_id, latent_id = str(next_id), str(next_id + 1)
        graph[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": init_image_name},
        }
        graph[latent_id] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [load_id, 0], "vae": ["1", 2]},
        }
        latent_ref: list[object] = [latent_id, 0]
        denoise = max(0.0, min(1.0, float(payload.get("denoising_strength", 0.75))))
        next_id += 2
    else:
        latent_id = str(next_id)
        graph[latent_id] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch},
        }
        latent_ref = [latent_id, 0]
        denoise = 1.0
        next_id += 1

    sampler_id, decode_id, save_id = str(next_id), str(next_id + 1), str(next_id + 2)
    graph[sampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
            "model": model_ref,
            "positive": [positive_id, 0],
            "negative": [negative_id, 0],
            "latent_image": latent_ref,
        },
    }
    graph[decode_id] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [sampler_id, 0], "vae": ["1", 2]},
    }
    graph[save_id] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "site-sdxl", "images": [decode_id, 0]},
    }
    metadata = {
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "loras": [{"name": name, "weight": weight} for name, weight in loras],
        "mode": "img2img" if init_image_name else "txt2img",
    }
    return graph, metadata

