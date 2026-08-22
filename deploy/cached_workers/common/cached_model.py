"""Resolve and validate RunPod Hugging Face cached-model snapshots.

The cache hash is deliberately never hardcoded. RunPod writes Hugging Face's
normal ``refs/main`` and ``snapshots/<revision>`` layout before starting a
worker; production workers run offline and fail closed if that cache is not
complete.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


DEFAULT_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
_REPO_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CachedModelError(RuntimeError):
    pass


def force_offline_mode() -> None:
    """Prevent hidden model downloads during worker startup/inference."""
    for key in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE",
    ):
        os.environ[key] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _repo_parts(repo_id: str) -> tuple[str, str]:
    parts = str(repo_id or "").strip().split("/", 1)
    if len(parts) != 2 or not all(_REPO_PART.fullmatch(part) for part in parts):
        raise CachedModelError("MODEL_REPO_ID must be in the form owner/name")
    return parts[0], parts[1]


def repo_cache_dir(repo_id: str, cache_root: Path | str | None = None) -> Path:
    owner, name = _repo_parts(repo_id)
    root = Path(cache_root or os.environ.get("HF_CACHE_ROOT") or DEFAULT_CACHE_ROOT)
    return root / f"models--{owner}--{name}"


def resolve_snapshot_path(
    repo_id: str,
    *,
    cache_root: Path | str | None = None,
    revision: str = "main",
) -> Path:
    model_root = repo_cache_dir(repo_id, cache_root)
    snapshots = model_root / "snapshots"
    ref = model_root / "refs" / revision

    if ref.is_file():
        snapshot_hash = ref.read_text(encoding="utf-8").strip()
        if not snapshot_hash or any(ch in snapshot_hash for ch in "/\\"):
            raise CachedModelError(f"Invalid cached ref for {repo_id}@{revision}")
        candidate = snapshots / snapshot_hash
        if candidate.is_dir():
            return candidate.resolve()
        raise CachedModelError(
            f"Cached ref exists but snapshot is missing: {repo_id}@{snapshot_hash}"
        )

    candidates = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if snapshots.is_dir() else []
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise CachedModelError(f"No cached snapshot found for {repo_id}")
    raise CachedModelError(
        f"Multiple cached snapshots exist for {repo_id}, but refs/{revision} is missing"
    )


def verify_bundle(snapshot: Path | str) -> dict:
    """Verify paths and sizes without re-hashing tens of GB during cold start."""
    snapshot = Path(snapshot).resolve()
    manifest_path = snapshot / "bundle-manifest.json"
    if not manifest_path.is_file():
        raise CachedModelError("bundle-manifest.json is missing from cached repo")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CachedModelError("Cached bundle manifest is invalid") from exc
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CachedModelError("Cached bundle manifest has no files")

    missing: list[str] = []
    bad_sizes: list[str] = []
    for item in files:
        relative = str((item or {}).get("target") or "").strip()
        if not relative or relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
            raise CachedModelError("Cached bundle manifest contains an unsafe path")
        path = snapshot / relative
        if not path.is_file():
            missing.append(relative)
            continue
        expected = item.get("size")
        if expected is not None and path.stat().st_size != int(expected):
            bad_sizes.append(relative)
    if missing or bad_sizes:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing[:8]))
        if bad_sizes:
            details.append("size_mismatch=" + ",".join(bad_sizes[:8]))
        raise CachedModelError("Cached bundle is incomplete: " + "; ".join(details))
    return manifest


def link_tree(source: Path | str, destination: Path | str) -> None:
    """Link a cached directory into a runtime model tree without copying it."""
    source = Path(source).resolve()
    destination = Path(destination)
    if not source.is_dir():
        raise CachedModelError(f"Cached model directory is missing: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() == path.resolve():
                continue
            target.unlink()
        elif target.exists():
            raise CachedModelError(f"Runtime model path already exists: {target}")
        target.symlink_to(path)


def write_comfy_extra_paths(snapshot: Path | str, output: Path | str) -> Path:
    """Point ComfyUI at ``snapshot/models``; only a tiny YAML file is written."""
    model_root = Path(snapshot).resolve() / "models"
    if not model_root.is_dir():
        raise CachedModelError("Cached bundle has no models directory")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [
        "cached_bundle:",
        f"  base_path: {model_root.as_posix()}",
        "  checkpoints: checkpoints",
        "  diffusion_models: diffusion_models",
        "  unet: unet",
        "  text_encoders: text_encoders",
        "  clip: text_encoders",
        "  vae: vae",
        "  loras: loras",
        "  clip_vision: clip_vision",
        "  controlnet: controlnet",
    ]
    output.write_text("\n".join(values) + "\n", encoding="utf-8")
    return output

