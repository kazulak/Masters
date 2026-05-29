# Book AI Library - POC Handoff

Book AI Library is a university POC for a cloud-native SOA book recommendation system. It currently runs locally as five FastAPI services plus a Streamlit frontend. The architecture document is [index.html](./index.html).

## Current Status

Implemented and verified locally:

- Microservices: User Profile, Book Catalog, Embedding Worker, Recommendation, LLM Service, Frontend.
- Async boundary: `BookCreated`, `BookEmbedded`, and `UserBookAdded` events are still processed outside the user request path.
- Open Library integration: search, enrich manually added books, and seed a bounded catalog of recommendation candidates.
- Local LLM path: LLM Service can use Ollama `/api/embed` and `/api/generate`; deterministic embeddings remain available for fast tests.
- Local persistence:
  - Python/dev mode uses `.local/app_state.json`.
  - Docker Compose mode uses normalized PostgreSQL tables via `DATABASE_URL`, persisted in the `postgres-data` Docker volume.
  - PostgreSQL schema includes `users`, `books`, `reading_list`, `book_embeddings`, `recommendations`, `events`, and `event_deliveries`.
  - `book_embeddings.embedding` uses the pgvector `vector` type, and Recommendation uses pgvector cosine ordering when PostgreSQL is enabled.
  - Services now use granular repository functions for books, users, reading list rows, embeddings, recommendations, and events. The old snapshot storage API remains only as a JSON fallback/test utility.
  - Ollama models are persisted in the `ollama-models` Docker volume, so models are not downloaded on every restart.
  - The Ollama container is internal-only; the public LLM microservice is `llm-service` on port `8005`, so host port `11434` does not need to be free.
- Event contracts: `BookCreated`, `BookEmbedded`, and `UserBookAdded` payloads are validated before publishing.
- Azure-ready adapters: `shared/events.py` has an optional Azure Service Bus path behind the same `publish`/`pull` functions; LLM Service has an Azure OpenAI REST path behind `LLM_PROVIDER=azure-openai`.
- Tests: `pytest -q`, PostgreSQL integration test, and CI Compose smoke pass locally.
- Docker images: service images build successfully through the Compose smoke path.

Not production-ready yet:

- Azure Service Bus and Azure OpenAI adapters are wired but not exercised against real Azure resources in this session.
- The Azure Service Bus adapter requires the `azure-servicebus` package before deploying that mode.
- Azure Bicep is improved with Service Bus topics/subscriptions and a manual CI deployment gate, but it is still not a hardened production deployment.
- `shared/storage.py` still contains the legacy snapshot compatibility helpers for JSON/dev tests; application services no longer call them directly.

## Latest Verification

Last handoff verification:

- `timeout 45s .venv/bin/python -m pytest -q`: passes with `15 passed, 1 skipped`.
- PostgreSQL repository integration: `TEST_DATABASE_URL=postgresql://book_ai:book_ai_password@127.0.0.1:5432/book_ai_library .venv/bin/python -m pytest -q tests/test_postgres_storage.py` passes against `docker-compose.ci.yml` PostgreSQL.
- CI Compose smoke: `PYTHON_BIN=.venv/bin/python scripts/compose_smoke.sh` passes and returns one recommendation.
- `docker compose -f docker-compose.yml config`: valid for the main stack.
- `docker-compose.yml`: Ollama is internal-only and does not publish host port `11434`.
- `docker-compose.yml`: `ollama-pull` uses `entrypoint: ["/bin/sh", "-lc"]`, so the pull commands are executed by a shell instead of being parsed as Ollama subcommands.
- `az version`: not available on this machine, so Bicep was prepared but not built locally.

The full Ollama model download was not rerun during this handoff because it can be large. The Compose wiring and LLM adapter behavior are covered by tests.

## Requirements Mapping

1. Microservices: User Profile, Book Catalog, Embedding Worker, Recommendation, LLM Service, Frontend.
2. Asynchronous communication: local event adapter now; Azure Service Bus target.
3. Cloud SaaS services: Azure OpenAI, PostgreSQL, Service Bus, ACR in target architecture.
4. Serverless / Kubernetes: Azure Container Apps target.
5. Minimal frontend: Streamlit.
6. Infrastructure as Code: Bicep starter in [infra/bicep](./infra/bicep).
7. CI/CD: GitHub Actions starter in [.github/workflows/ci.yml](./.github/workflows/ci.yml).
8. Architecture diagrams: [index.html](./index.html).

## Application Structure

- [services/user-profile](./services/user-profile): user profile and reading list API.
- [services/book-catalog](./services/book-catalog): metadata store, Open Library search/enrichment, `BookCreated` publisher.
- [services/embedding-worker](./services/embedding-worker): consumes `BookCreated`, calls LLM Service, writes embeddings, publishes `BookEmbedded`.
- [services/recommendation](./services/recommendation): consumes embedding/user events, precomputes cached recommendations.
- [services/llm-service](./services/llm-service): adapter for deterministic local mode, Ollama, or Azure OpenAI.
- [frontend/streamlit](./frontend/streamlit): minimal UI with Discover, Add book, Reading list, Recommendations tabs.
- [shared](./shared): config, repositories, event contracts/adapters, Open Library client, vector helpers.
- [docker-compose.yml](./docker-compose.yml): local container stack with persistent PostgreSQL and Ollama volumes.
- [docker-compose.ci.yml](./docker-compose.ci.yml): deterministic-LLM Compose stack for CI smoke tests.
- [infra/db/001_init.sql](./infra/db/001_init.sql): normalized PostgreSQL + pgvector schema.

## Run Locally - Fast Python Mode

Use this for development and tests. It does not require Docker models.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
./scripts/run_local.sh
```

Open:

- Frontend: <http://127.0.0.1:8501>
- User Profile API: <http://127.0.0.1:8001/docs>
- Book Catalog API: <http://127.0.0.1:8002/docs>
- Embedding Worker API: <http://127.0.0.1:8003/docs>
- Recommendation API: <http://127.0.0.1:8004/docs>
- LLM Service API: <http://127.0.0.1:8005/docs>

Verify:

```bash
pytest -q
./scripts/run_backend_smoke.sh
```

## Run Locally - Container Mode

Use this to prepare for Azure. It runs services as containers, stores app state in PostgreSQL, and uses Ollama for local embeddings/generation.

```bash
docker compose build
docker compose up
```

First startup pulls the configured Ollama models:

- `OLLAMA_EMBED_MODEL=embeddinggemma`
- `OLLAMA_GENERATE_MODEL=gemma3:1b`

Those downloads are stored in the persistent `ollama-models` Docker volume. PostgreSQL data is stored in `postgres-data`.
The `ollama-pull` helper runs via a shell entrypoint, so it actually executes `ollama pull ...` instead of being misparsed by the Ollama binary.

Useful commands:

```bash
docker compose ps
docker compose logs -f llm-service
docker compose logs -f ollama-pull
docker compose down
```

Keep volumes when stopping:

```bash
docker compose down
```

Delete all local persisted data and models:

```bash
docker compose down -v
```

## Local LLM Requests

After `docker compose up`, you can call the local LLM adapter directly:

```bash
curl -s http://127.0.0.1:8005/health
```

```bash
curl -s http://127.0.0.1:8005/v1/embed \
  -H 'Content-Type: application/json' \
  -d '{"text":"Dune by Frank Herbert, desert ecology and politics"}'
```

```bash
curl -s http://127.0.0.1:8005/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain why Dune is a good recommendation for a science fiction reader."}'
```

The LLM Service calls Ollama’s official `POST /api/embed` endpoint for embeddings. Source: <https://docs.ollama.com/api/embed>.
If Ollama is unavailable, the LLM service still has a deterministic fallback mode for tests and offline development; the production container path uses Ollama when it is healthy.

## Open Library Flow

Book Catalog exposes:

- `GET /external/openlibrary/search?query=...`
- `POST /books`
- `POST /catalog/seed/openlibrary`

Design intent:

- User can add a title with minimal metadata.
- Book Catalog enriches missing author/description/genres/ISBN/cover from Open Library.
- The catalog can be seeded with bounded Open Library searches, creating unread recommendation candidates.
- Seeded books emit `BookCreated`, are embedded asynchronously, and are then available for recommendation ranking.

Do not turn Open Library integration into a crawler. Keep it bounded by query and limit; use official Open Library dumps for a large offline corpus.

## Testing

All tests:

```bash
pytest -q
```

Single service or integration area:

```bash
pytest -q tests/test_llm_service.py
pytest -q tests/test_open_library.py
pytest -q tests/test_shared_flow.py
pytest -q tests/test_event_contracts.py
pytest -q tests/test_service_health.py
```

PostgreSQL-backed integration test:

```bash
docker compose -f docker-compose.ci.yml up -d postgres
TEST_DATABASE_URL=postgresql://book_ai:book_ai_password@127.0.0.1:5432/book_ai_library \
  pytest -q tests/test_postgres_storage.py
docker compose -f docker-compose.ci.yml down -v
```

Backend smoke test:

```bash
./scripts/run_backend_smoke.sh
```

Container smoke test with deterministic LLM and pgvector PostgreSQL:

```bash
PYTHON_BIN=.venv/bin/python ./scripts/compose_smoke.sh
```

Manual service checks:

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8002/health
curl -s http://127.0.0.1:8003/health
curl -s http://127.0.0.1:8004/health
curl -s http://127.0.0.1:8005/health
```

Current caveat:

- `tests/test_service_health.py` calls route functions directly instead of using FastAPI `TestClient`; `TestClient` hung in this sandbox with the installed FastAPI/httpx/anyio stack. Revisit after dependency upgrades.
- `docker-compose.yml` keeps Ollama internal-only to avoid host port collisions. The user-facing LLM entry point is the `llm-service` container.

Needed test upgrades:

- Add endpoint-level tests once the TestClient/runtime issue is resolved.
- Add failure-path tests for Open Library, Ollama, and database outages.
- Add direct HTTP contract tests for the LLM service once the `TestClient` stack is upgraded or replaced with a stable transport.

## CI/CD Plan

Current CI:

- `test`: installs dependencies, compiles Python files, and runs `pytest -q`.
- `postgres-integration`: starts pgvector PostgreSQL and runs `tests/test_postgres_storage.py`.
- `compose-smoke`: validates Compose config, builds service images, starts the deterministic-LLM stack, and runs `scripts/smoke_test.py`.
- `bicep-build`: runs `az bicep build --file infra/bicep/main.bicep` in GitHub Actions.
- `azure-deploy-gate`: manual-only `workflow_dispatch` job gated by the `azure-manual` GitHub environment and Azure OIDC secrets.

Recommended next CI stages:

1. Lint and format:
   - `ruff check .`
   - `ruff format --check .`
2. API schema/HTTP contract tests once the TestClient/runtime issue is resolved.
3. Optional nightly Ollama smoke with the real local embedding model.
4. Azure deployment gate:
   - only on protected branch or manual workflow dispatch
   - Azure OIDC login
   - Bicep what-if
   - build and push images to ACR
   - deploy/update Azure Container Apps

Do not store Azure secrets in GitHub. Use OIDC federated credentials.

Current LLM issue solved:

- `ollama` is no longer published to host port `11434`, so it cannot collide with a local Ollama install.
- `ollama-pull` now uses `entrypoint: ["/bin/sh", "-lc"]` and runs the `ollama pull` commands correctly.
- The public LLM microservice remains `llm-service:8005`, which is what the app should call.

## Azure Deployment Path

Do not push yet unless the missing production pieces below are addressed.

Starter command:

```bash
./scripts/deploy_azure.sh <resource-group> <location> [app-name] [tag]
```

Current script behavior:

- Creates resource group.
- Deploys ACR, Log Analytics, ACA environment, Service Bus namespace/topics/subscriptions, and Container Apps skeleton.
- Builds and pushes service images.
- Re-runs the Bicep deployment with app images.

Before real Azure deployment:

- Install/package `azure-servicebus` for Azure deployments or replace the adapter with a REST/managed-identity implementation.
- Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBED_DEPLOYMENT`, and `AZURE_OPENAI_CHAT_DEPLOYMENT` for LLM Service.
- Replace Service Bus connection strings with managed identity/RBAC before production.
- Add pgvector ANN index after embedding dimension is fixed for the Azure OpenAI model.
- Use managed identities instead of ACR admin credentials.
- Move secrets to Key Vault or Container Apps secrets.
- Add App Insights/Log Analytics queries and alerts.

## Next Worker Notes

- Keep this README current after every architectural or runtime change.
- Preserve the async recommendation boundary.
- Keep LLM Service as an adapter; business services must not import Ollama/Azure OpenAI SDKs directly.
- Keep Open Library calls bounded by query and limit.
- Keep new service code on granular repository functions; do not reintroduce direct `read_state`/`update_state` calls in services.
- Decide whether to delete or quarantine the legacy snapshot helpers in `shared/storage.py` once all older tests and scripts are migrated.
- Add pgvector ANN indexes once the deployed embedding dimension is fixed.
- Exercise Azure Service Bus and Azure OpenAI adapters against real Azure resources from the manual CI gate.
- Harden Dockerfiles: split runtime/test requirements and reduce frontend image size.
- Add `azure-servicebus` to the Azure runtime image/dependency set before using `EVENT_BUS_PROVIDER=azure-service-bus`.
- Validate Bicep with `az bicep build` and `az deployment group what-if` once Azure CLI is available locally.

## Prompt For Next Elevation

Use this prompt for the next implementation pass:

```text
You are a senior software developer and DevOps engineer. Continue the Book AI Library POC. Keep README.md up to date. The services now use granular repository functions instead of the PostgreSQL snapshot rewrite path, event payload contracts are enforced, CI has a manual Azure deployment gate, and Azure-ready Service Bus/Azure OpenAI adapter paths exist but have not been exercised against real Azure resources. Do not deploy automatically. Next: package/test the Azure Service Bus dependency, validate Bicep with az bicep build and deployment what-if, add pgvector ANN indexes after fixing the deployed embedding dimension, add failure-path tests for Open Library/Ollama/PostgreSQL, and replace the hanging FastAPI TestClient experiment with stable HTTP-level contract tests.
```
