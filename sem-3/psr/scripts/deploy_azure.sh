#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: scripts/deploy_azure.sh <resource-group> <location> [app-name] [tag]"
  exit 1
fi

resource_group="$1"
location="$2"
app_name="${3:-book-ai-library}"
tag="${4:-latest}"

az group create --name "$resource_group" --location "$location"
az deployment group create \
  --name main \
  --resource-group "$resource_group" \
  --template-file infra/bicep/main.bicep \
  --parameters appName="$app_name" imageTag="$tag" deployApps=false

acr_login_server="$(az deployment group show \
  --resource-group "$resource_group" \
  --name main \
  --query properties.outputs.acrLoginServer.value \
  --output tsv)"

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
  --parameters appName="$app_name" imageTag="$tag" deployApps=true
