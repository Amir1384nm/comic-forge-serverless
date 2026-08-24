#!/usr/bin/env bash
set -euo pipefail

/opt/venv/bin/python -u /opt/cached-worker/configure_cached_paths.py
exec /start.sh
