#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: scripts/build_images.sh <acr-login-server> [tag]"
  exit 1
fi

acr_login_server="$1"
tag="${2:-latest}"

services=(
  "llm-service:services/llm-service"
  "book-catalog:services/book-catalog"
  "user-profile:services/user-profile"
  "embedding-worker:services/embedding-worker"
  "recommendation:services/recommendation"
)

for item in "${services[@]}"; do
  name="${item%%:*}"
  path="${item#*:}"
  docker build \
    -f Dockerfile.service \
    --build-arg "SERVICE_PATH=${path}" \
    -t "${acr_login_server}/${name}:${tag}" \
    .
done

docker build -f frontend/streamlit/Dockerfile -t "${acr_login_server}/frontend:${tag}" .
