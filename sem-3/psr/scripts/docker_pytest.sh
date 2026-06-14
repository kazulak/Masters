#!/usr/bin/env bash
set -euo pipefail

project_name="${COMPOSE_PROJECT_NAME:-book-ai-test}"
compose=(docker compose -p "$project_name" -f docker-compose.test.yml)

cleanup() {
  "${compose[@]}" down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${compose[@]}" up --build --abort-on-container-exit --exit-code-from pytest
