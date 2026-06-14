# Book AI Library - POC Handoff

Book AI Library is a university POC for a cloud-native SOA book recommendation system. It currently runs locally as five FastAPI services plus a Streamlit frontend. The architecture document is [index.html](./index.html).

For a teaching-style continuation plan with Azure, staging, local LLM, and next-step guidance, read [PROJECT_CONTINUATION_GUIDE.md](./PROJECT_CONTINUATION_GUIDE.md).

## Current Status

Implemented and verified locally:

- Microservices: User Profile, Book Catalog, Embedding Worker, Recommendation, LLM Service, Frontend.
- Async boundary: `BookCreated`, `BookEmbedded`, and `UserBookAdded` events are still processed outside the user request path.
- Open Library integration: search, enrich manually added books, and seed a bounded catalog of recommendation candidates.
- Offline/local demo catalog: `POST /catalog/seed/demo` seeds a curated recommendation corpus without depending on Open Library.
- Local LLM path: LLM Service can use Ollama `/api/embed` and `/api/generate`; deterministic embeddings remain available for fast tests.
- Local generation model: the default Ollama generation model is `gemma4:e2b`, chosen as the practical Gemma 4 default for an AMD RX 6600-class local GPU. Larger Gemma 4 models remain opt-in through `OLLAMA_GENERATE_MODEL`.
- Standalone local LLM path: [docker-compose.llm.yml](./docker-compose.llm.yml) runs only Ollama + LLM Service for prompt/embedding experiments.
- AMD GPU path: [docker-compose.amd.yml](./docker-compose.amd.yml) and [docker-compose.llm.amd.yml](./docker-compose.llm.amd.yml) switch Ollama to `ollama/ollama:rocm` and pass `/dev/kfd` plus `/dev/dri` into the container.
- Local persistence:
  - Python/dev mode uses `.local/app_state.json`.
  - Docker Compose mode uses normalized PostgreSQL tables via `DATABASE_URL`, persisted in the `postgres-data` Docker volume.
  - PostgreSQL schema includes `users`, `books`, `reading_list`, `book_embeddings`, `recommendations`, `events`, and `event_deliveries`.
  - `book_embeddings.embedding` uses the pgvector `vector` type, and Recommendation uses pgvector cosine ordering when PostgreSQL is enabled.
  - Services now use granular repository functions for books, users, reading list rows, embeddings, recommendations, and events. The old snapshot storage API remains only as a JSON fallback/test utility.
  - Ollama models are persisted in the `ollama-models` Docker volume, so models are not downloaded on every restart.
  - The Ollama container is internal-only; the public LLM microservice is `llm-service` on port `8005`, so host port `11434` does not need to be free.
- Event contracts: `BookCreated`, `BookEmbedded`, and `UserBookAdded` payloads are validated before publishing.
- Azure-ready adapters: `shared/events.py` has an optional Azure Service Bus path behind the same `publish`/`pull` functions; LLM Service has an Azure OpenAI REST path behind `LLM_PROVIDER=azure-openai`. Both paths now have mocked adapter tests.
- Azure dependency packaging: `azure-servicebus==7.13.0` is in `requirements.txt` and was verified through the Docker Compose smoke build.
- Azure IaC: Bicep builds locally with Azure CLI and `what-if` validates in an allowed Azure for Students region.
- Tests: `pytest -q`, PostgreSQL integration test, Bicep build, Azure what-if, local app smoke, and LLM smoke pass locally. CI Compose smoke is configured to include LLM smoke; it was not rerun after that script change because the main Docker stack was left running on the same published ports.
- Docker images: service images build successfully through the Compose smoke path.
- Recommendation reads now defensively filter cached recommendation rows against the user's current reading list. This prevents the just-added book from appearing as the top recommendation while async recomputation catches up.

Not production-ready yet:

- Azure Service Bus and Azure OpenAI adapters are tested with mocks and Bicep what-if, but not exercised through a live deployed app yet.
- No Container Apps, PostgreSQL server, or pushed Azure images were deployed in this session.
- Azure Bicep now includes Service Bus topics/subscriptions, optional PostgreSQL Flexible Server, and a manual CI deployment gate, but it is still not a hardened production deployment.
- `shared/storage.py` still contains the legacy snapshot compatibility helpers for JSON/dev tests; application services no longer call them directly.

## Latest Verification

Latest local verification:

- `.venv/bin/python -m pytest -q`: passes with `27 passed, 1 skipped`.
- PostgreSQL repository integration: `TEST_DATABASE_URL=postgresql://book_ai:book_ai_password@127.0.0.1:5432/book_ai_library .venv/bin/python -m pytest -q tests/test_postgres_storage.py` passes against `docker-compose.ci.yml` PostgreSQL.
- CI Compose smoke: `PYTHON_BIN=.venv/bin/python scripts/compose_smoke.sh` passed earlier after rebuilding images with `azure-servicebus`; the script now also runs `scripts/llm_smoke.py`, but that updated CI-smoke script was not rerun because the main Docker stack is currently running on the same ports.
- Running Docker app smoke: `.venv/bin/python scripts/local_app_smoke.py` passes against the main Compose stack with `ok: local app demo_total=22 recommendations=10`. The smoke now fails if the just-added book appears in recommendations.
- Running LLM smoke: `.venv/bin/python scripts/llm_smoke.py` passes against the main Compose stack with Ollama, `provider=ollama embedding_dims=768 generated_chars=227`.
- Browser-friendly LLM GET prompt: `GET /v1/generate?prompt=...` returns an Ollama response through `llm-service` with `provider=ollama:gemma4:e2b`.
- Running model check: `GET /v1/models` reports `ollama_generate_model=gemma4:e2b`.
- Main stack was rebuilt and restarted after pulling `gemma4:e2b`; all eight containers are running.
- `az account show`: subscription is `Azure for Students`.
- Azure policy assignment `sys.regionrestriction` allows only `italynorth`, `switzerlandnorth`, `swedencentral`, `spaincentral`, and `germanywestcentral`.
- `az bicep build --file infra/bicep/main.bicep`: passes.
- `DEPLOY_APPS=false DEPLOY_POSTGRES=false scripts/azure_what_if.sh book-ai-library-stage-rg spaincentral book-ai-library local-test`: passes; predicts 9 infrastructure resources.
- `DEPLOY_APPS=false DEPLOY_POSTGRES=true POSTGRES_ADMIN_PASSWORD=... scripts/azure_what_if.sh book-ai-library-stage-rg spaincentral book-ai-library local-test`: passes; predicts 13 resources including PostgreSQL Flexible Server, database, pgvector allow-list, and firewall rule.
- `docker compose -f docker-compose.yml config`: valid for the main stack.
- `docker compose -f docker-compose.llm.yml config`: valid for the standalone local LLM stack.
- `docker compose -f docker-compose.yml -f docker-compose.amd.yml config`: valid for the AMD ROCm main stack override.
- `docker compose -f docker-compose.llm.yml -f docker-compose.llm.amd.yml config`: valid for the AMD ROCm standalone LLM override.
- `docker-compose.yml`: Ollama is internal-only and does not publish host port `11434`.
- `docker-compose.yml`: `ollama-pull` uses `entrypoint: ["/bin/sh", "-lc"]`, so the pull commands are executed by a shell instead of being parsed as Ollama subcommands.

The Gemma 4 model download may be large. `gemma4:e2b` is about 7.2 GB in Ollama's library, `gemma4:e4b` is about 9.6 GB, and `gemma4:26b` is the MoE option at about 18 GB. The Compose wiring and LLM adapter behavior are covered by tests.

Current AMD GPU note: the host has an AMD Radeon RX 6600-class GPU visible through `lspci`, but this sandbox currently does not expose `/dev/kfd` or `/dev/dri`. The normal Compose stack therefore ran Ollama on CPU; Ollama logs reported `offloaded 0/36 layers to GPU`. The AMD ROCm Compose overrides are present and validated, but a live GPU smoke requires those devices to be visible to Docker.

Azure staging note: the what-if probes created empty resource groups `book-ai-library-rg` in `westeurope`, `book-ai-library-pl-rg` in `polandcentral`, and `book-ai-library-stage-rg` in `spaincentral`. The first two regions are policy-blocked for actual resources in this subscription. The groups are empty and can be deleted if you want a clean Azure portal.

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
- [docker-compose.llm.yml](./docker-compose.llm.yml): standalone local LLM stack with persistent Ollama models and public LLM Service on port `8005`.
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
- `OLLAMA_GENERATE_MODEL=gemma4:e2b`

Those downloads are stored in the persistent `ollama-models` Docker volume. PostgreSQL data is stored in `postgres-data`.
The `ollama-pull` helper runs via a shell entrypoint, so it actually executes `ollama pull ...` instead of being misparsed by the Ollama binary.

Gemma 4 model choice:

- Default: `gemma4:e2b`. Best local-first choice for an AMD RX 6600-class 8 GB GPU because it is the smallest current Gemma 4 edge model.
- Higher quality edge option: `OLLAMA_GENERATE_MODEL=gemma4:e4b`, but expect more memory pressure.
- MoE option: `OLLAMA_GENERATE_MODEL=gemma4:26b`. This is the Gemma 4 Mixture-of-Experts model with about 4B active parameters, but the Ollama package is much larger and is not the safe default for an 8 GB local GPU.
- Workstation dense option: `OLLAMA_GENERATE_MODEL=gemma4:31b`, only for machines with substantially more memory.

References:

- Gemma 4 model overview: <https://deepmind.google/models/gemma/gemma-4/>
- Ollama Gemma 4 tags and model sizes: <https://ollama.com/library/gemma4>
- Ollama Docker AMD ROCm usage: <https://hub.docker.com/r/ollama/ollama>

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

## Run Locally - AMD GPU Ollama

Use this only when the host exposes AMD ROCm devices to Docker. Check first:

```bash
ls -l /dev/kfd /dev/dri
```

Run the full app with the ROCm Ollama image:

```bash
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d --build
```

Run only Ollama + LLM Service with the ROCm Ollama image:

```bash
docker compose -f docker-compose.llm.yml -f docker-compose.llm.amd.yml up -d --build
```

If `/dev/kfd` or `/dev/dri` is missing, Docker cannot pass the AMD GPU into the container. Use the normal CPU compose files until ROCm/driver/container permissions are fixed.

## Local LLM Requests

After `docker compose up`, you can call the local LLM adapter directly:

```bash
curl -s http://127.0.0.1:8005/health
```

Browser/quick prompt:

```bash
curl -s "http://127.0.0.1:8005/v1/generate?prompt=Recommend%20one%20short%20science%20fiction%20book"
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

Command helper:

```bash
./scripts/llm_prompt.py "Recommend one short science fiction book"
```

The LLM Service calls Ollama’s official `POST /api/embed` endpoint for embeddings. Source: <https://docs.ollama.com/api/embed>.
If Ollama is unavailable, the LLM service still has a deterministic fallback mode for tests and offline development; the production container path uses Ollama when it is healthy.

## Run Only The Local LLM

Use this when you only want Ollama plus the `llm-service` API, not the whole application:

```bash
docker compose -f docker-compose.llm.yml build
docker compose -f docker-compose.llm.yml up
```

Open:

- LLM Service docs: <http://127.0.0.1:8005/docs>
- LLM Service usage JSON: <http://127.0.0.1:8005/>

Verify:

```bash
./scripts/llm_smoke.py
./scripts/llm_prompt.py "Recommend one short science fiction book"
```

The Ollama model volume is still `ollama-models`, so models are not downloaded on every restart. Ollama itself remains internal-only to avoid host port `11434` conflicts.

## Open Library Flow

Book Catalog exposes:

- `GET /external/openlibrary/search?query=...`
- `POST /books`
- `POST /catalog/seed/openlibrary`
- `POST /catalog/seed/demo`

Design intent:

- User can add a title with minimal metadata.
- Book Catalog enriches missing author/description/genres/ISBN/cover from Open Library.
- The catalog can be seeded with bounded Open Library searches, creating unread recommendation candidates.
- The demo catalog can be seeded locally when Open Library is slow, unavailable, or blocked.
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
pytest -q tests/test_azure_adapters.py
pytest -q tests/test_book_catalog_demo_seed.py
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

Full local app smoke against the main Docker stack:

```bash
docker compose up -d --build
.venv/bin/python scripts/local_app_smoke.py
```

Standalone LLM smoke:

```bash
docker compose -f docker-compose.llm.yml up -d --build
.venv/bin/python scripts/llm_smoke.py
```

Compose config checks:

```bash
docker compose -f docker-compose.yml config
docker compose -f docker-compose.llm.yml config
docker compose -f docker-compose.yml -f docker-compose.amd.yml config
docker compose -f docker-compose.llm.yml -f docker-compose.llm.amd.yml config
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
- `compose-smoke`: validates deterministic, standalone LLM, and AMD ROCm Compose config; builds service images; starts the deterministic-LLM stack; runs `scripts/llm_smoke.py`; and runs `scripts/smoke_test.py`.
- `bicep-build`: runs `az bicep build --file infra/bicep/main.bicep` in GitHub Actions.
- `azure-deploy-gate`: manual-only `workflow_dispatch` job gated by the `azure-manual` GitHub environment and Azure OIDC secrets. It now takes resource group, region, app name, image tag, `deploy_apps`, and `deploy_postgres` inputs.

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
   - optional PostgreSQL
   - build and push images to ACR
   - deploy/update Azure Container Apps

Do not store Azure secrets in GitHub. Use OIDC federated credentials.

Current LLM issue solved:

- `ollama` is no longer published to host port `11434`, so it cannot collide with a local Ollama install.
- `ollama-pull` now uses `entrypoint: ["/bin/sh", "-lc"]` and runs the `ollama pull` commands correctly.
- The public LLM microservice remains `llm-service:8005`, which is what the app should call.
- The default generation model is Gemma 4, specifically `gemma4:e2b`; larger Gemma 4 models are explicit opt-ins.
- AMD GPU mode is provided as a Compose override using `ollama/ollama:rocm`.

## Azure Deployment Path

Do not push yet unless the missing production pieces below are addressed.

Starter command:

```bash
./scripts/deploy_azure.sh <resource-group> <location> [app-name] [tag]
```

Safe preview command:

```bash
DEPLOY_APPS=false DEPLOY_POSTGRES=false \
  scripts/azure_what_if.sh book-ai-library-stage-rg spaincentral book-ai-library local-test
```

PostgreSQL preview command:

```bash
DEPLOY_APPS=false DEPLOY_POSTGRES=true POSTGRES_ADMIN_PASSWORD='<strong-password>' \
  scripts/azure_what_if.sh book-ai-library-stage-rg spaincentral book-ai-library local-test
```

For this Azure for Students subscription, use one of the allowed policy regions: `italynorth`, `switzerlandnorth`, `swedencentral`, `spaincentral`, or `germanywestcentral`.

Current script behavior:

- Creates resource group.
- Deploys ACR, Log Analytics, ACA environment, Service Bus namespace/topics/subscriptions, and optionally PostgreSQL Flexible Server.
- If `DEPLOY_APPS=true`, builds and pushes service images.
- Re-runs the Bicep deployment with Container Apps enabled only when `DEPLOY_APPS=true`.

Before real Azure deployment:

- Decide whether to use connection-string Service Bus for the POC or replace it with managed identity/RBAC before grading.
- Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBED_DEPLOYMENT`, and `AZURE_OPENAI_CHAT_DEPLOYMENT` for LLM Service.
- Add pgvector ANN index after embedding dimension is fixed for the Azure OpenAI model.
- Use managed identities instead of ACR admin credentials.
- Move secrets to Key Vault or Container Apps secrets.
- Add App Insights/Log Analytics queries and alerts.

## Next Worker Notes

- Keep this README current after every architectural or runtime change.
- Preserve the async recommendation boundary.
- Keep LLM Service as an adapter; business services must not import Ollama/Azure OpenAI SDKs directly.
- Keep Open Library calls bounded by query and limit.
- Keep the demo catalog path working; it is the reliable local fallback when Open Library is slow or unavailable.
- Keep the standalone LLM Compose path working; it is the fastest way to test local prompts without the full app.
- Keep `gemma4:e2b` as the local default unless the target machine has enough GPU memory for a larger Gemma 4 model.
- If experimenting with Gemma 4 MoE, use `OLLAMA_GENERATE_MODEL=gemma4:26b` explicitly and expect much higher memory requirements than the default.
- Keep new service code on granular repository functions; do not reintroduce direct `read_state`/`update_state` calls in services.
- Decide whether to delete or quarantine the legacy snapshot helpers in `shared/storage.py` once all older tests and scripts are migrated.
- Add pgvector ANN indexes once the deployed embedding dimension is fixed.
- Exercise Azure Service Bus and Azure OpenAI adapters through a live deployed app from the manual CI gate.
- Harden Dockerfiles: split runtime/test requirements and reduce frontend image size.
- Replace connection-string Service Bus and ACR admin credentials with managed identity/RBAC.
- Clean up empty exploratory Azure resource groups if you do not want them in the portal.

## Prompt For Next Elevation

Use this prompt for the next implementation pass:

```text
You are a senior software developer and DevOps engineer. Continue the Book AI Library POC. Keep README.md and PROJECT_CONTINUATION_GUIDE.md up to date. Focus locally before Azure unless explicitly asked to deploy. The local LLM default is Gemma 4 via OLLAMA_GENERATE_MODEL=gemma4:e2b, with AMD ROCm Compose overrides in docker-compose.amd.yml and docker-compose.llm.amd.yml. The recommendation API filters cached rows against the user's current reading list so the just-added book is not returned while async recomputation catches up. Next: pull/run the Gemma 4 model locally, verify AMD GPU availability with /dev/kfd and /dev/dri, run scripts/llm_smoke.py and scripts/local_app_smoke.py, keep hardening the local UX and tests, add failure-path tests for Open Library/Ollama/PostgreSQL, add stable HTTP-level endpoint tests, then return to Azure only when the local app is consistently usable.
```
