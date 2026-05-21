#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  docker compose -f docker-compose.ci.yml down >/dev/null 2>&1 || true
}
trap cleanup EXIT

python_bin="${PYTHON_BIN:-python}"

docker compose -f docker-compose.ci.yml up -d --build
"$python_bin" scripts/wait_for_services.py
"$python_bin" scripts/smoke_test.py
