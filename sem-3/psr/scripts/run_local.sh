#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export APP_STATE_FILE="${APP_STATE_FILE:-.local/app_state.json}"
export BOOK_CATALOG_URL="${BOOK_CATALOG_URL:-http://127.0.0.1:8002}"
export USER_PROFILE_URL="${USER_PROFILE_URL:-http://127.0.0.1:8001}"
export RECOMMENDATION_URL="${RECOMMENDATION_URL:-http://127.0.0.1:8004}"
export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://127.0.0.1:8005}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

mkdir -p .local

.venv/bin/uvicorn main:app --app-dir services/llm-service/app --host 127.0.0.1 --port 8005 &
.venv/bin/uvicorn main:app --app-dir services/book-catalog/app --host 127.0.0.1 --port 8002 &
.venv/bin/uvicorn main:app --app-dir services/user-profile/app --host 127.0.0.1 --port 8001 &
.venv/bin/uvicorn main:app --app-dir services/embedding-worker/app --host 127.0.0.1 --port 8003 &
.venv/bin/uvicorn main:app --app-dir services/recommendation/app --host 127.0.0.1 --port 8004 &
.venv/bin/streamlit run frontend/streamlit/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
