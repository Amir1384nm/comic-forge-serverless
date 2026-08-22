# Cached-model workers

Three runtime images preserve the current inference contracts while removing
production model-weight dependence on Network Volumes:

- `Dockerfile.sdxl`: Forge runtime for each of the three isolated SDXL bundles.
- `Dockerfile.qwen`: pinned ComfyUI + ComfyUI-GGUF for Qwen Image.
- `Dockerfile.video`: pinned Blackwell-capable ComfyUI for H3 and, only after a
  complete validated bundle/workflow exists, WAN.

All images contain code and dependencies only. At runtime `MODEL_REPO_ID`
selects the single RunPod Cached Model attached to the endpoint. The resolver
reads Hugging Face `refs/main`, validates `bundle-manifest.json`, and exposes
weights through symlinks or ComfyUI extra paths. Offline flags are mandatory;
missing files fail the worker instead of downloading them.

Build contexts must be the repository root, for example:

```text
deploy/cached_workers/Dockerfile.qwen
```

Required endpoint environment variables:

- `MODEL_REPO_ID=owner/private-repo`
- SDXL only: `MODEL_CHECKPOINT=<exact checkpoint filename>`
- Optional: `RUNPOD_INIT_TIMEOUT=800`

Do not attach a Network Volume for model weights. Configure one Cached Model,
workers min `0`, workers max `1`, idle timeout `5`, Queue Delay scaling, and
FlashBoot.
