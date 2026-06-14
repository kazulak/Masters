# Book AI Library - POC Handoff

Book AI Library is a university POC for a cloud-native SOA book recommendation system. It currently runs locally as five FastAPI services plus a Streamlit frontend. The architecture document is [index.html](./index.html).

For a teaching-style continuation plan with Azure, staging, local LLM, and next-step guidance, read [PROJECT_CONTINUATION_GUIDE.md](./PROJECT_CONTINUATION_GUIDE.md). For the current technical review and priority order, read [PROJECT_REVIEW_AND_PRIORITIES.md](./PROJECT_REVIEW_AND_PRIORITIES.md). For the single canonical continuation prompt, read [NEXT_PROMPT.md](./NEXT_PROMPT.md).

## Current Status

Implemented and verified locally:

- Microservices: User Profile, Book Catalog, Embedding Worker, Recommendation, LLM Service, Frontend.
- Async boundary: `BookCreated`, `BookEmbedded`, and `UserBookAdded` events are still processed outside the user request path.
- Open Library integration: search, enrich manually added books, and seed a bounded catalog of recommendation candidates.
- Offline/local demo catalog: `POST /catalog/seed/demo` seeds a curated recommendation corpus without depending on Open Library.
- Local LLM path: LLM Service can use Ollama `/api/embed` and `/api/generate`; deterministic embeddings remain available for fast tests. Docker defaults to `LLM_PROVIDER=ollama-with-fallback`, which tries Ollama first and falls back only if Ollama fails, times out, or returns empty text.
- Local generation model: the default Ollama generation model is `gemma4:e2b`, chosen as the practical Gemma 4 default for an AMD RX 6600-class local GPU. The service sends `OLLAMA_THINK=false` by default because Gemma 4 otherwise spends the full `/api/generate` token budget on hidden thinking and may return an empty `response`. Larger Gemma 4 models remain opt-in through `OLLAMA_GENERATE_MODEL`.
- Standalone local LLM path: [docker-compose.llm.yml](./docker-compose.llm.yml) runs only Ollama + LLM Service for prompt/embedding experiments.
- AMD GPU path: [docker-compose.amd.yml](./docker-compose.amd.yml) and [docker-compose.llm.amd.yml](./docker-compose.llm.amd.yml) switch Ollama to `ollama/ollama:rocm` and pass `/dev/kfd` plus `/dev/dri` into the container.
- Frontend demo console: Streamlit now has Discover, Add book, Reading list, Recommendations, and System flow tabs. The Recommendations tab has a presentation-focused cache view, side-by-side mode comparison, explicit async processing controls, filtering metrics, empty states, and addable recommendation cards. The System flow tab renders a topology-style microservice map, shows catalog/reading/recommendation counts, displays the active LLM model, shows async event backlog/timestamps, worker run/failure/latency state, runs live smoke checks, can run a one-click end-to-end demo scenario, and can send a real prompt through `llm-service`.
- Frontend user model: the backend supports multiple users via `X-User-Id`; Streamlit is a demo client with a selectable user ID in the sidebar.
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
- Tests: host pytest, Dockerized pytest, PostgreSQL integration, isolated Compose smoke, Bicep build, Azure what-if, local app smoke, and LLM smoke pass locally.
- Docker images: service images build successfully through the Compose smoke path.
- Recommendation reads and recomputation now defensively filter against the user's current reading list by exact ID and logical book identity (`isbn`, `openlibrary_key`, normalized `title + author`). Responses include `filter_summary`, so the frontend can show how many cached recommendation rows were hidden because they are already owned.
- Recommendation modes are intentionally distinct: `similar` prioritizes closest vector/genre matches, `widen` boosts strong candidates from less-read genres, and `mood` biases toward the user's stored mood profile.
- CI/local Docker test isolation: [scripts/docker_pytest.sh](./scripts/docker_pytest.sh) runs pytest inside Docker with its own PostgreSQL service, and [scripts/compose_smoke.sh](./scripts/compose_smoke.sh) now uses a separate Compose project plus `18001`-`18005` host ports so it can run while the main app is up.
- Real-process HTTP endpoint tests: [tests/test_http_process_endpoints.py](./tests/test_http_process_endpoints.py) starts Uvicorn services on localhost ports and calls endpoints over HTTP. It replaces the earlier skipped in-process harness as the stable endpoint test path. In restricted sandboxes where socket binding is forbidden, those tests skip; in normal local/CI environments they run.
- Browser regression hook: [tests/test_streamlit_browser.py](./tests/test_streamlit_browser.py) captures the Streamlit System flow page. CI now has a dedicated `streamlit-browser` job that starts the deterministic Compose stack, installs Playwright Chromium, and runs the screenshot test. Local runs still skip cleanly when browser dependencies are not installed.
- Async worker observability: Embedding Worker and Recommendation `/status` responses include `last_success_at`, `last_error`, `last_duration_ms`, `total_runs`, `total_failures`, `consecutive_failures`, and `last_result`.

Not production-ready yet:

- Azure Service Bus and Azure OpenAI adapters are tested with mocks and Bicep what-if, but not exercised through a live deployed app yet.
- No Container Apps, PostgreSQL server, or pushed Azure images were deployed in this session.
- Azure Bicep now includes Service Bus topics/subscriptions, optional PostgreSQL Flexible Server, and a manual CI deployment gate, but it is still not a hardened production deployment.
- `shared/storage.py` still contains the legacy snapshot compatibility helpers for JSON/dev tests; application services no longer call them directly.

## Latest Verification

Latest local verification:

- `.venv/bin/python -m pytest -q`: passes with `49 passed, 5 skipped`; real-process HTTP endpoint tests pass when localhost socket binding is available. The Streamlit browser screenshot test is one of the skipped tests unless Playwright/Chromium are installed.
- `scripts/docker_pytest.sh`: passes in Docker with PostgreSQL, `50 passed, 4 skipped`.
- PostgreSQL repository integration: `TEST_DATABASE_URL=postgresql://book_ai:book_ai_password@127.0.0.1:15432/book_ai_library .venv/bin/python -m pytest -q tests/test_postgres_storage.py` is the current local host-port path for `docker-compose.ci.yml`.
- CI Compose smoke: `PYTHON_BIN=.venv/bin/python scripts/compose_smoke.sh` passes with isolated project `book-ai-ci`, deterministic LLM, frontend build, alternate ports, and `ok: 1 recommendations returned`.
- Running Docker app smoke: `.venv/bin/python scripts/local_app_smoke.py` passes against the rebuilt main Compose stack with `ok: local app demo_total=31 recommendations=9`. The smoke fails if recommendations overlap with the reading list by ID or logical book identity.
- Running LLM smoke: `.venv/bin/python scripts/llm_smoke.py` passes against the main Compose stack with `provider=ollama-with-fallback embedding_dims=768 generated_chars=25`.
- Browser-friendly LLM GET prompt: `GET /v1/generate?prompt=...` returns a real Gemma 4 response through `llm-service` with `provider=ollama:gemma4:e2b`.
- Running model check: `GET /v1/models` reports `ollama_generate_model=gemma4:e2b` and `ollama_think=false`.
- Worker status checks: `GET /status` on Embedding Worker and Recommendation return event backlog rows plus worker observability. Latest live payloads showed `pending=0`, delivered counts, `last_error=null`, and successful worker run counters.
- Manual overlap check against the running stack: recommendation smoke returns unread books only; `similar`, `widen`, and `mood` use distinct scoring paths and explanations.
- Main stack was rebuilt and restarted after the recommendation/System flow changes; all eight containers are running.
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

## Architecture Clarifications

- Is it REST? Mostly yes. User-facing and service-facing synchronous APIs are HTTP REST-ish FastAPI endpoints. The architecture is not pure REST because it deliberately also uses asynchronous events for the embedding/recommendation pipeline.
- Is it one-user only? No at the backend level. User Profile and Recommendation accept user identity through `X-User-Id` or `user_id` query parameters. The frontend was originally hard-coded to `demo-user`; it now has a sidebar user selector for local demos.
- Is the frontend a production UI? No. It is a Streamlit POC UI required by the course. It should be good enough for presentation, but the production evolution path is still React/Next.js or another proper frontend.

## Application Structure

- [services/user-profile](./services/user-profile): user profile and reading list API.
- [services/book-catalog](./services/book-catalog): metadata store, Open Library search/enrichment, `BookCreated` publisher.
- [services/embedding-worker](./services/embedding-worker): consumes `BookCreated`, calls LLM Service, writes embeddings, publishes `BookEmbedded`.
- [services/recommendation](./services/recommendation): consumes embedding/user events, precomputes cached recommendations.
- [services/llm-service](./services/llm-service): adapter for deterministic local mode, Ollama, or Azure OpenAI.
- [frontend/streamlit](./frontend/streamlit): local demo UI with Discover, Add book, Reading list, Recommendations, and live System flow tabs.
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

Demo the live architecture:

1. Open the Streamlit frontend and go to **System flow**.
2. Use **Run demo scenario** to seed candidate books, add Dune to the reading list, run the async workers, and read cached recommendations.
3. Use **Refresh flow** to show the current microservice topology and which services are online.
4. Use **Run async pass** after adding books to process pending events without waiting for the background loop.
5. Use **Run smoke checks** to show a live demo check table against the running services.
6. Use the prompt box to send a real request through `llm-service`; the response caption shows whether Gemma 4 or the fallback answered.

The System flow smoke checks are presentation checks against the running app. Full tests still run from the terminal and in CI.

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
- `OLLAMA_THINK=false`

Those downloads are stored in the persistent `ollama-models` Docker volume. PostgreSQL data is stored in `postgres-data`.
The `ollama-pull` helper runs via a shell entrypoint, so it actually executes `ollama pull ...` instead of being misparsed by the Ollama binary.

Gemma 4 model choice:

- Default: `gemma4:e2b`. Best local-first choice for an AMD RX 6600-class 8 GB GPU because it is the smallest current Gemma 4 edge model.
- Higher quality edge option: `OLLAMA_GENERATE_MODEL=gemma4:e4b`, but expect more memory pressure.
- MoE option: `OLLAMA_GENERATE_MODEL=gemma4:26b`. This is the Gemma 4 Mixture-of-Experts model with about 4B active parameters, but the Ollama package is much larger and is not the safe default for an 8 GB local GPU.
- Workstation dense option: `OLLAMA_GENERATE_MODEL=gemma4:31b`, only for machines with substantially more memory.

Generation setting:

- Keep `OLLAMA_THINK=false` for the app path. With `gemma4:e2b`, Ollama `/api/generate` can otherwise spend the whole `num_predict` budget on thinking tokens and return an empty `response`. The CLI may still appear to work because it displays thinking output differently.
- Set `OLLAMA_THINK=true` only when intentionally experimenting with reasoning traces and after increasing `OLLAMA_NUM_PREDICT`.

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
If Ollama is unavailable, times out, or returns an empty generation, the `ollama-with-fallback` mode logs the reason and returns a deterministic template response so the local demo does not break. When Ollama succeeds, `/v1/generate` returns `provider=ollama:gemma4:e2b`.

### `POST /v1/generate` vs `GET /v1/generate`

Both endpoints call the same generation adapter.

- `POST /v1/generate` is the application contract. It accepts JSON, supports future structured fields like `context`, and is what scripts, tests, and other services should use.
- `GET /v1/generate?prompt=...` is a browser/manual convenience endpoint. It exists so you can paste a URL or use a simple `curl` command while checking the LLM container.

Do not build backend service-to-service logic on the GET endpoint. Keep GET for manual inspection and POST for real application calls.

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
TEST_DATABASE_URL=postgresql://book_ai:book_ai_password@127.0.0.1:15432/book_ai_library \
  pytest -q tests/test_postgres_storage.py
docker compose -f docker-compose.ci.yml down -v
```

Full pytest inside Docker:

```bash
scripts/docker_pytest.sh
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
docker compose -f docker-compose.test.yml config
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

- `tests/test_service_health.py` still calls route functions directly for fast smoke coverage. `tests/test_http_process_endpoints.py` is the stable endpoint path: it starts real Uvicorn processes and calls HTTP endpoints, including user-profile -> book-catalog, the async recommendation pipeline, and an LLM dependency failure path for the embedding worker. `tests/test_http_endpoints.py` remains quarantined as an experimental in-process ASGI harness and is skipped unless `RUN_EXPERIMENTAL_HTTP_TESTS=1`.
- `tests/test_streamlit_browser.py` is a browser screenshot regression check for the Streamlit System flow page. It requires Playwright and a Chromium browser install locally; CI installs those dependencies in the `streamlit-browser` job.
- `docker-compose.yml` keeps Ollama internal-only to avoid host port collisions. The user-facing LLM entry point is the `llm-service` container.

Needed test upgrades:

- Expand real-process HTTP tests with more failure paths and edge cases.
- Expand failure-path coverage beyond the current Open Library timeout, Ollama fallback, PostgreSQL connection failure, and embedding-worker LLM dependency failure tests.
- Extend browser/screenshot regression beyond System flow to cover the Recommendations tab and one-click demo path.
- Keep `scripts/local_app_smoke.py` checking recommendation overlap by logical book identity, not only database ID.

## CI/CD Plan

Current CI:

- `test`: installs dependencies, compiles Python files, and runs `pytest -q`.
- `postgres-integration`: starts pgvector PostgreSQL and runs `tests/test_postgres_storage.py`.
- `docker-pytest`: runs [scripts/docker_pytest.sh](./scripts/docker_pytest.sh), which executes pytest in a dedicated Docker image with PostgreSQL.
- `compose-smoke`: validates deterministic, standalone LLM, and AMD ROCm Compose config; builds service images; starts the deterministic-LLM stack; runs `scripts/llm_smoke.py`; and runs `scripts/smoke_test.py`.
- `streamlit-browser`: starts the deterministic Compose stack with the frontend on port `18501`, installs Playwright Chromium, and runs `tests/test_streamlit_browser.py`.
- `bicep-build`: runs `az bicep build --file infra/bicep/main.bicep` in GitHub Actions.
- `azure-deploy-gate`: manual-only `workflow_dispatch` job gated by the `azure-manual` GitHub environment and Azure OIDC secrets. It now takes resource group, region, app name, image tag, `deploy_apps`, and `deploy_postgres` inputs. The default region is `spaincentral`, which is allowed by the current Azure for Students policy.

Recommended next CI stages:

1. Lint and format:
   - `ruff check .`
   - `ruff format --check .`
2. API schema/HTTP contract tests using a stable real-process or browser-level harness.
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
- Gemma 4 generation now sends `think: false` to Ollama by default through `OLLAMA_THINK=false`. This fixes the observed `/api/generate` behavior where Gemma 4 returned HTTP 200 with an empty `response` after spending the token budget on thinking.
- AMD GPU mode is provided as a Compose override using `ollama/ollama:rocm`.
- `ollama-with-fallback` now falls back for generation as well as embeddings and logs the fallback reason.
- The Streamlit Recommendations tab has `Refresh` and `Process updates` controls. `Process updates` invokes the embedding and recommendation workers explicitly for local demos; recommendation reads still use precomputed rows and do not call the LLM.
- The Streamlit System flow tab shows a topology-style map of frontend, REST services, local event bus, async worker, LLM adapter, PostgreSQL/pgvector, Open Library, live service health, model config, event backlog/timestamps, runtime smoke checks, and a real LLM prompt path.

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
- Keep `scripts/docker_pytest.sh` and `scripts/compose_smoke.sh` isolated from the main app; they should be safe to run while the demo stack is already up.
- Keep the System flow tab live-data driven; it should reflect real service health/model/counts, not a static architecture picture.
- Keep recommendation filtering based on logical book identity as well as ID; Open Library/demo/manual entries can represent the same book under different IDs.
- Keep `gemma4:e2b` as the local default unless the target machine has enough GPU memory for a larger Gemma 4 model.
- Keep `OLLAMA_THINK=false` for Gemma 4 app calls unless you deliberately increase token limits and want reasoning traces.
- If experimenting with Gemma 4 MoE, use `OLLAMA_GENERATE_MODEL=gemma4:26b` explicitly and expect much higher memory requirements than the default.
- Keep new service code on granular repository functions; do not reintroduce direct `read_state`/`update_state` calls in services.
- Decide whether to delete or quarantine the legacy snapshot helpers in `shared/storage.py` once all older tests and scripts are migrated.
- Add pgvector ANN indexes once the deployed embedding dimension is fixed.
- Exercise Azure Service Bus and Azure OpenAI adapters through a live deployed app from the manual CI gate.
- Harden Dockerfiles: split runtime/test requirements and reduce frontend image size.
- Replace connection-string Service Bus and ACR admin credentials with managed identity/RBAC.
- Clean up empty exploratory Azure resource groups if you do not want them in the portal.

## Next Prompt

There is only one continuation prompt for the project: [NEXT_PROMPT.md](./NEXT_PROMPT.md).
