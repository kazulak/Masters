#!/usr/bin/env bash
set -euo pipefail

if [ -f .local/pids ]; then
  echo "Local app appears to be running. Use scripts/stop_local.sh before starting again."
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export APP_STATE_FILE="${APP_STATE_FILE:-.local/app_state.json}"
export BOOK_CATALOG_URL="${BOOK_CATALOG_URL:-http://127.0.0.1:8002}"
export USER_PROFILE_URL="${USER_PROFILE_URL:-http://127.0.0.1:8001}"
export RECOMMENDATION_URL="${RECOMMENDATION_URL:-http://127.0.0.1:8004}"
export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://127.0.0.1:8005}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

mkdir -p .local/logs
: > .local/pids

nohup .venv/bin/uvicorn main:app --app-dir services/llm-service/app --host 127.0.0.1 --port 8005 >.local/logs/llm-service.log 2>&1 &
echo "$!" >> .local/pids
nohup .venv/bin/uvicorn main:app --app-dir services/book-catalog/app --host 127.0.0.1 --port 8002 >.local/logs/book-catalog.log 2>&1 &
echo "$!" >> .local/pids
nohup .venv/bin/uvicorn main:app --app-dir services/user-profile/app --host 127.0.0.1 --port 8001 >.local/logs/user-profile.log 2>&1 &
echo "$!" >> .local/pids
nohup .venv/bin/uvicorn main:app --app-dir services/embedding-worker/app --host 127.0.0.1 --port 8003 >.local/logs/embedding-worker.log 2>&1 &
echo "$!" >> .local/pids
nohup .venv/bin/uvicorn main:app --app-dir services/recommendation/app --host 127.0.0.1 --port 8004 >.local/logs/recommendation.log 2>&1 &
echo "$!" >> .local/pids

.venv/bin/python scripts/wait_for_services.py

nohup .venv/bin/streamlit run frontend/streamlit/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true >.local/logs/streamlit.log 2>&1 &
echo "$!" >> .local/pids

echo "Book AI Library is running:"
echo "Frontend: http://127.0.0.1:8501"
echo "User Profile API: http://127.0.0.1:8001/docs"
echo "Book Catalog API: http://127.0.0.1:8002/docs"
echo "Embedding Worker API: http://127.0.0.1:8003/docs"
echo "Recommendation API: http://127.0.0.1:8004/docs"
echo "LLM Service API: http://127.0.0.1:8005/docs"
