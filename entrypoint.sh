#!/bin/bash

set -e

echo "🔧 Entrypoint starting..."
echo "Starting Cloudflare TCP tunnel for PostgreSQL..."
cloudflared access tcp --hostname db.leandrohome.de --url localhost:5432 &



# Check for GPU_WORKER_NAME
if [[ -z "$GPU_WORKER_NAME" ]]; then
  echo "❌ GPU_WORKER_NAME not set! Exiting..."
  exit 1
fi

echo "💡 GPU_WORKER_NAME = $GPU_WORKER_NAME"

# Dynamically resolve corresponding tunnel token env var
TOKEN_VAR_NAME="${GPU_WORKER_NAME}_TUNNEL_TOKEN"
CLOUDFLARE_TUNNEL_TOKEN="${!TOKEN_VAR_NAME}"

if [[ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]]; then
  echo "🔐 Launching cloudflared tunnel for GPU_WORKER_NAME=$GPU_WORKER_NAME..."
  cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" &
else
  echo "⚠️ No tunnel token found for GPU_WORKER_NAME=$GPU_WORKER_NAME (env var $TOKEN_VAR_NAME missing)"
  exit 1
fi

sleep 2  # Optional: allow tunnel to stabilize

echo "🚀 Launching GPU Worker Python entrypoint..."
python -m run
