#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  "${compose[@]}" down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

python_bin="${PYTHON_BIN:-python}"
project_name="${COMPOSE_PROJECT_NAME:-book-ai-ci}"
compose=(docker compose -p "$project_name" -f docker-compose.ci.yml)

export USER_PROFILE_URL="${USER_PROFILE_URL:-http://127.0.0.1:18001}"
export BOOK_CATALOG_URL="${BOOK_CATALOG_URL:-http://127.0.0.1:18002}"
export EMBEDDING_WORKER_URL="${EMBEDDING_WORKER_URL:-http://127.0.0.1:18003}"
export RECOMMENDATION_URL="${RECOMMENDATION_URL:-http://127.0.0.1:18004}"
export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://127.0.0.1:18005}"

"${compose[@]}" up -d --build
"$python_bin" scripts/wait_for_services.py
"$python_bin" scripts/llm_smoke.py
"$python_bin" scripts/smoke_test.py
