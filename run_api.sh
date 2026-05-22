#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
uvicorn app.main:app --app-dir "$(pwd)" --host 0.0.0.0 --port "$PORT" --reload
