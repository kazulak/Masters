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

echo "Checking frontend..."
curl -fsS "https://${frontend_fqdn}/" >/dev/null
echo "ok"

echo "Checking recommendation health..."
curl -fsS "https://${recommendation_fqdn}/health"
echo

echo "Checking cache-only recommendations endpoint..."
curl -fsS "https://${recommendation_fqdn}/recommendations?user_id=demo-user&type=similar"
echo

echo "Checking public LLM adapter path through Recommendation service..."
curl -fsS -X POST "https://${recommendation_fqdn}/profile/summary" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user","limit":3}'
echo

if [ "${RUN_INTERNAL_FLOW:-true}" = "true" ]; then
  echo "Checking internal Azure flow from the User Profile container..."
  internal_command="python -c \"import os, requests; u='azure-smoke@example.edu'; h={'X-User-Id':u}; c=os.environ['BOOK_CATALOG_URL']; p=os.environ['USER_PROFILE_URL']; w=os.environ['EMBEDDING_WORKER_URL']; r=os.environ['RECOMMENDATION_URL']; requests.post(c+'/catalog/seed/demo', timeout=90).raise_for_status(); a=requests.post(p+'/me/books', headers=h, json={'title':'Dune','author':'Frank Herbert','description':'A desert planet, politics, ecology, prophecy, and power.','genres':['science fiction','adventure'],'rating':5}, timeout=90); a.raise_for_status(); b=requests.get(p+'/me/books', headers=h, timeout=90); b.raise_for_status(); books=b.json().get('books', []); assert books, b.text; requests.post(w+'/work', timeout=180).raise_for_status(); requests.post(r+'/work', timeout=120).raise_for_status(); o=requests.get(r+'/recommendations', params={'user_id':u,'type':'similar'}, timeout=90); o.raise_for_status(); recs=o.json().get('books', []); assert recs, o.text; ids={x['id'] for x in books}; assert not [x['title'] for x in recs if x.get('id') in ids]; ol=requests.get(c+'/external/openlibrary/search', params={'query':'Ursula Le Guin','limit':1}, timeout=90); ol.raise_for_status(); assert ol.json(); print('ok internal flow books=%s recs=%s' % (len(books), len(recs)))\""
  az containerapp exec \
    --resource-group "$resource_group" \
    --name "${app_name}-user-profile" \
    --command "$internal_command"
fi
