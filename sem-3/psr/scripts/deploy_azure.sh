#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: scripts/deploy_azure.sh <resource-group> <location> [app-name] [tag]"
  echo
  echo "optional env:"
  echo "  DEPLOY_APPS=true|false                 default: true"
  echo "  DEPLOY_POSTGRES=true|false             default: true when POSTGRES_ADMIN_PASSWORD is set, otherwise false"
  echo "  POSTGRES_ADMIN_USER=bookadmin"
  echo "  POSTGRES_ADMIN_PASSWORD=..."
  echo "  AZURE_OPENAI_ENDPOINT=https://..."
  echo "  AZURE_OPENAI_API_KEY=..."
  echo "  AZURE_OPENAI_EMBED_DEPLOYMENT=..."
  echo "  AZURE_OPENAI_CHAT_DEPLOYMENT=..."
  exit 1
fi

resource_group="$1"
location="$2"
app_name="${3:-book-ai-library}"
tag="${4:-latest}"
deploy_apps="${DEPLOY_APPS:-true}"
postgres_admin_user="${POSTGRES_ADMIN_USER:-bookadmin}"
postgres_admin_password="${POSTGRES_ADMIN_PASSWORD:-}"
deploy_postgres="${DEPLOY_POSTGRES:-}"

if [ -z "$deploy_postgres" ]; then
  if [ -n "$postgres_admin_password" ]; then
    deploy_postgres="true"
  else
    deploy_postgres="false"
  fi
fi

if [ "$deploy_postgres" = "true" ] && [ -z "$postgres_admin_password" ]; then
  echo "POSTGRES_ADMIN_PASSWORD is required when DEPLOY_POSTGRES=true"
  exit 1
fi

common_parameters=(
  "appName=$app_name"
  "imageTag=$tag"
  "deployPostgres=$deploy_postgres"
  "postgresAdminUser=$postgres_admin_user"
  "azureOpenAIEndpoint=${AZURE_OPENAI_ENDPOINT:-}"
  "azureOpenAIApiKey=${AZURE_OPENAI_API_KEY:-}"
  "azureOpenAIEmbedDeployment=${AZURE_OPENAI_EMBED_DEPLOYMENT:-}"
  "azureOpenAIChatDeployment=${AZURE_OPENAI_CHAT_DEPLOYMENT:-}"
)

if [ "$deploy_postgres" = "true" ]; then
  common_parameters+=("postgresAdminPassword=$postgres_admin_password")
fi

az group create --name "$resource_group" --location "$location"
az deployment group create \
  --name main \
  --resource-group "$resource_group" \
  --template-file infra/bicep/main.bicep \
  --parameters "${common_parameters[@]}" deployApps=false

acr_login_server="$(az deployment group show \
  --resource-group "$resource_group" \
  --name main \
  --query properties.outputs.acrLoginServer.value \
  --output tsv)"

if [ "$deploy_apps" != "true" ]; then
  echo "Infrastructure deployment finished. DEPLOY_APPS is not true, so image build/push and Container Apps deployment were skipped."
  exit 0
fi

az acr login --name "${acr_login_server%%.*}"
scripts/build_images.sh "$acr_login_server" "$tag"

docker push "$acr_login_server/llm-service:$tag"
docker push "$acr_login_server/book-catalog:$tag"
docker push "$acr_login_server/user-profile:$tag"
docker push "$acr_login_server/embedding-worker:$tag"
docker push "$acr_login_server/recommendation:$tag"
docker push "$acr_login_server/frontend:$tag"

az deployment group create \
  --name main \
  --resource-group "$resource_group" \
  --template-file infra/bicep/main.bicep \
  --parameters "${common_parameters[@]}" deployApps=true
