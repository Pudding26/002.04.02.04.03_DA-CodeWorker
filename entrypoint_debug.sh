#!/bin/bash
set -e

echo "🔧 Entrypoint starting (cloudflared only mode)..."
echo "Starting Cloudflare TCP tunnel for PostgreSQL..."
cloudflared access tcp --hostname db.leandrohome.de --url localhost:5432 &


if [[ -z "$GPU_WORKER_NAME" ]]; then
  echo "❌ GPU_WORKER_NAME not set! Exiting..."
  exit 1
fi

echo "💡 GPU_WORKER_NAME = $GPU_WORKER_NAME"

TOKEN_VAR_NAME="${GPU_WORKER_NAME}_TUNNEL_TOKEN"
CLOUDFLARE_TUNNEL_TOKEN="${!TOKEN_VAR_NAME}"

echo "DEBUG: Resolved TOKEN_VAR_NAME=$TOKEN_VAR_NAME with value=$CLOUDFLARE_TUNNEL_TOKEN"

if [[ -n "$CLOUDFLARE_TUNNEL_TOKEN" ]]; then
  echo "🔐 Launching cloudflared tunnel (foreground, persistent)..."
  exec cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN"
else
  echo "⚠️ No tunnel token found for GPU_WORKER_NAME=$GPU_WORKER_NAME (env var $TOKEN_VAR_NAME missing)"
  exit 1
fi
