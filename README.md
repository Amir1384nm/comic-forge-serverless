# Comic Forge Serverless worker

This repository contains two isolated workers backed by the same Network
Volume:

- `Dockerfile` + `handler.py`: Forge burst generation.
- `Dockerfile.comfy` + `comfy_handler.py`: Qwen/ComfyUI image editing.

ComfyUI never runs in the primary Pod. This prevents Forge and Qwen from
competing for VRAM and makes advanced editing independently scalable.

Safe endpoint configuration:

- Active workers: `0`
- Max workers: `1` on each endpoint (Forge + Comfy = at most 2 Serverless GPUs)
- GPUs per worker: `1`
- Idle timeout: `40` seconds
- Scaler: `QUEUE_DELAY`, value `4`
- FlashBoot: enabled
- Execution timeout: `600000` ms
- Network Volume: the same volume used by the primary Pod (mounted at
  `/runpod-volume` on Serverless)
- Data center: the Network Volume's data center

The application keeps Serverless disabled until the built image and endpoint
ID are configured. Merely adding an endpoint ID does not bypass the global
three-GPU capacity policy.
