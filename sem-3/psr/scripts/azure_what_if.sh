#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: scripts/azure_what_if.sh <resource-group> <location> [app-name] [tag]"
  echo
  echo "optional env:"
  echo "  DEPLOY_APPS=true|false"
  echo "  DEPLOY_POSTGRES=true|false"
  echo "  DEPLOY_OLLAMA=true|false"
  echo "  OLLAMA_GENERATE_MODEL=qwen3:0.6b"
  echo "  OLLAMA_EMBED_MODEL=embeddinggemma"
  echo "  POSTGRES_ADMIN_USER=bookadmin"
  echo "  POSTGRES_ADMIN_PASSWORD=..."
  echo "  AZURE_OPENAI_ENDPOINT=https://..."
  echo "  AZURE_OPENAI_API_KEY=..."
  echo "  AZURE_OPENAI_EMBED_DEPLOYMENT=..."
  echo "  AZURE_OPENAI_CHAT_DEPLOYMENT=..."
  echo "  EXTERNAL_LLM_SERVICE_URL=https://..."
  echo "  OPEN_LIBRARY_TIMEOUT_SECONDS=25"
  exit 1
fi

resource_group="$1"
location="$2"
app_name="${3:-book-ai-library}"
tag="${4:-latest}"
deploy_apps="${DEPLOY_APPS:-false}"
deploy_postgres="${DEPLOY_POSTGRES:-false}"
deploy_ollama="${DEPLOY_OLLAMA:-false}"
postgres_admin_user="${POSTGRES_ADMIN_USER:-bookadmin}"
postgres_admin_password="${POSTGRES_ADMIN_PASSWORD:-}"

parameters=(
  "location=$location"
  "appName=$app_name"
  "imageTag=$tag"
  "deployApps=$deploy_apps"
  "deployPostgres=$deploy_postgres"
  "deployOllama=$deploy_ollama"
  "ollamaGenerateModel=${OLLAMA_GENERATE_MODEL:-qwen3:0.6b}"
  "ollamaEmbedModel=${OLLAMA_EMBED_MODEL:-embeddinggemma}"
  "postgresAdminUser=$postgres_admin_user"
  "azureOpenAIEndpoint=${AZURE_OPENAI_ENDPOINT:-}"
  "azureOpenAIApiKey=${AZURE_OPENAI_API_KEY:-}"
  "azureOpenAIEmbedDeployment=${AZURE_OPENAI_EMBED_DEPLOYMENT:-}"
  "azureOpenAIChatDeployment=${AZURE_OPENAI_CHAT_DEPLOYMENT:-}"
  "externalLlmServiceUrl=${EXTERNAL_LLM_SERVICE_URL:-}"
  "openLibraryTimeoutSeconds=${OPEN_LIBRARY_TIMEOUT_SECONDS:-25}"
)

if [ "$deploy_postgres" = "true" ]; then
  if [ -z "$postgres_admin_password" ]; then
    echo "POSTGRES_ADMIN_PASSWORD is required when DEPLOY_POSTGRES=true"
    exit 1
  fi
  parameters+=("postgresAdminPassword=$postgres_admin_password")
fi

az group create --name "$resource_group" --location "$location" >/dev/null
az deployment group what-if \
  --resource-group "$resource_group" \
  --template-file infra/bicep/main.bicep \
  --parameters "${parameters[@]}"
