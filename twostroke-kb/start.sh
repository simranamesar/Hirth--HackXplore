#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== TwoStroke-KB Startup ==="

# Ensure .env exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env. Please configure it and rerun."
    exit 1
fi

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting FastAPI (uvicorn)..."
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  > uvicorn.log 2>&1 &

echo $! > .uvicorn.pid

sleep 3

echo "Starting Cloudflare tunnel..."

if [ -x "./cloudflared" ]; then
    ./cloudflared tunnel --url http://localhost:8000
else
    echo "cloudflared not found — running backend only"
    wait $(cat .uvicorn.pid)
fi