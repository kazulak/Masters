#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export APP_STATE_FILE="${APP_STATE_FILE:-.local/smoke_state.json}"
export BOOK_CATALOG_URL="${BOOK_CATALOG_URL:-http://127.0.0.1:8002}"
export USER_PROFILE_URL="${USER_PROFILE_URL:-http://127.0.0.1:8001}"
export RECOMMENDATION_URL="${RECOMMENDATION_URL:-http://127.0.0.1:8004}"
export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://127.0.0.1:8005}"

mkdir -p .local
rm -f "$APP_STATE_FILE"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

.venv/bin/uvicorn main:app --app-dir services/llm-service/app --host 127.0.0.1 --port 8005 >/tmp/book-ai-llm.log 2>&1 &
pids+=("$!")
.venv/bin/uvicorn main:app --app-dir services/book-catalog/app --host 127.0.0.1 --port 8002 >/tmp/book-ai-catalog.log 2>&1 &
pids+=("$!")
.venv/bin/uvicorn main:app --app-dir services/user-profile/app --host 127.0.0.1 --port 8001 >/tmp/book-ai-user.log 2>&1 &
pids+=("$!")
.venv/bin/uvicorn main:app --app-dir services/embedding-worker/app --host 127.0.0.1 --port 8003 >/tmp/book-ai-embedding.log 2>&1 &
pids+=("$!")
.venv/bin/uvicorn main:app --app-dir services/recommendation/app --host 127.0.0.1 --port 8004 >/tmp/book-ai-recommendation.log 2>&1 &
pids+=("$!")

.venv/bin/python scripts/wait_for_services.py
.venv/bin/python scripts/smoke_test.py
