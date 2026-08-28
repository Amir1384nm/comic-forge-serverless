# Serverless model workers

Three runtime images preserve the current inference contracts. Cached Models
remain supported for rollback; production replacements can use the existing
regional Network Volumes when first-host cache population is too slow.

- `Dockerfile.sdxl`: Forge runtime for each of the three isolated SDXL bundles.
- `Dockerfile.qwen`: pinned ComfyUI + ComfyUI-GGUF for Qwen Image.
- `Dockerfile.video`: pinned Blackwell-capable ComfyUI for H3 and, only after a
  complete validated bundle/workflow exists, WAN.

All images contain code and dependencies only. At runtime a worker chooses one
offline source: Cached Model (`MODEL_REPO_ID=owner/repo`) or Network Volume
(`MODEL_SOURCE=network_volume`, `MODEL_ROOT=/runpod-volume/<model-tree>`, and
`MODEL_REQUIRED_PATHS=<JSON filename/minimum-size manifest>`). The latter
validates each required weight before registering work. Missing files never
trigger a runtime download.

Build contexts must be the repository root, for example:

```text
deploy/cached_workers/Dockerfile.qwen
```

Cached endpoint environment variables:

- `MODEL_REPO_ID=owner/private-repo`
- SDXL only: `MODEL_CHECKPOINT=<exact checkpoint filename>`
- Optional: `RUNPOD_INIT_TIMEOUT=800`

Network Volume replacement endpoints must use the same data centre as the
volume, workers min `0`, max `1`, a finite idle timeout, and native execution
and TTL ceilings. Do not attach a Cached Model to a replacement endpoint.
