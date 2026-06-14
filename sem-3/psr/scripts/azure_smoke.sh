#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: scripts/azure_smoke.sh <resource-group> <app-name>"
  exit 1
fi

resource_group="$1"
app_name="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"

recommendation_fqdn="$(az containerapp show \
  --resource-group "$resource_group" \
  --name "${app_name}-recommendation" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

frontend_fqdn="$(az containerapp show \
  --resource-group "$resource_group" \
  --name "${app_name}-frontend" \
  --query properties.configuration.ingress.fqdn \
  --output tsv)"

echo "Frontend: https://${frontend_fqdn}"
echo "Recommendation: https://${recommendation_fqdn}"

curl -fsS "https://${recommendation_fqdn}/health"
echo
curl -fsS "https://${recommendation_fqdn}/recommendations?user_id=demo-user&type=similar"
echo
