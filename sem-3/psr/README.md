# Book AI Library

Local-first microservice book recommendation app for the PSR university project. The project is intentionally small: five FastAPI services, one Streamlit frontend, PostgreSQL/pgvector, Ollama locally, and Azure-ready infrastructure.

Architecture diagram: [index.html](./index.html)

Presentation/deployment briefing: [AZURE_PRESENTATION_GUIDE.md](./AZURE_PRESENTATION_GUIDE.md)

## Requirements Mapping

| Requirement | Implementation |
| --- | --- |
| Microservices, min. 3 nodes | `user-profile`, `book-catalog`, `embedding-worker`, `recommendation`, `llm-service`, plus `frontend` |
| Async communication | `BookCreated`, `BookEmbedded`, `UserBookAdded` events through `shared/events.py`; Azure Service Bus adapter path exists |
| Cloud SaaS | Target: Azure OpenAI, Azure Service Bus, Azure Database for PostgreSQL, ACR |
| Serverless / Kubernetes-like | Target: Azure Container Apps via Bicep |
| Minimal frontend | Streamlit |
| Infrastructure as Code | `infra/bicep/main.bicep` |
| CI/CD | `.github/workflows/ci.yml` |
| Architecture diagram | `index.html` |

## Current App Shape

- Streamlit starts with a local sign-in/create-account screen using email + password.
- User profile and reading list persist in PostgreSQL in Docker mode.
- Home loads saved profile/library state from PostgreSQL first. It calls the LLM only when `Regenerate profile summary with LLM` is clicked, then saves the generated summary back into the profile.
- Explore provides curated shelves that can be added to the user's library.
- Recommendations has two paths:
  - `GET /recommendations`: fast cache-only read of precomputed recommendations.
  - `POST /recommendations/ask`: explicit LLM command using the user's prompt and already-filtered unread candidates.
- The recommendation engine is isolated behind `RECOMMENDATION_ENGINE`; the current engine is `vector-similarity`.
- Adding a book stays responsive. Embeddings and recommendation recomputation happen asynchronously.
- Book reads support basic REST caching: `GET /books/{id}` returns `ETag` and honors `If-None-Match`.

Do not move LLM calls into `GET /recommendations`; that endpoint is deliberately cache-only.

## Recommendation Design

The recommendation path is intentionally split into two layers:

1. Candidate engine: `vector-similarity` uses pgvector/cosine-style candidate retrieval and mode-specific scoring.
2. LLM guide: `POST /recommendations/ask` sends the user's reading list, saved recommendation instructions, request text, and engine candidates to `llm-service`.

The three modes use the same candidate engine but different ranking logic:

| Mode | Intent |
| --- | --- |
| `similar` | stay close to the user's current library and overlapping genres |
| `widen` | prefer useful novelty and underrepresented genres |
| `mood` | bias candidates toward the user's saved mood keywords |

The LLM is not limited to the vector candidate list by default. The candidates ground the prompt, but the model may recommend an outside book if it explains why it is a better fit. Users can save persistent recommendation instructions in the frontend; those instructions are stored in their profile and included in every LLM recommendation prompt.

Swapping the recommendation engine should happen behind the service boundary, not in the frontend. A future engine can replace `vector-similarity` with collaborative filtering, a graph recommender, a learning-to-rank model, or a dedicated vector database while preserving:

- `GET /recommendations`
- `POST /recommendations/ask`
- the async event boundary
- the frontend contract

## Architecture Notes

This is a REST-based microservice app with one async pipeline:

- Browser users interact with the Streamlit frontend only.
- Streamlit calls backend FastAPI services over REST.
- `user-profile`, `book-catalog`, `recommendation`, and `llm-service` expose HTTP APIs.
- `embedding-worker` is a worker service with HTTP health/admin endpoints, but its main work is async event processing.
- `BookCreated`, `BookEmbedded`, and `UserBookAdded` events decouple writes from embeddings/recommendation recomputation.
- PostgreSQL/pgvector stores users, books, reading lists, embeddings, cached recommendations, and local event state.

The local Docker stack runs Ollama plus Gemma through `llm-service`, so the LLM path is visible without cloud inference cost. The Azure deployment path can run the same `llm-service` in deterministic mode when Azure OpenAI is not configured; that proves the microservice wiring without spending on model inference. When Azure OpenAI settings are provided, only `llm-service` changes provider.

## LLM Deployment Choices

The application deliberately keeps all model calls behind `llm-service`, so the rest of the microservices do not care which provider is active.

| Option | Where it runs | When to use it | Trade-off |
| --- | --- | --- | --- |
| Local Ollama/Gemma | Your machine through Docker Compose | Best final demo path when you want real model output without cloud inference cost. | Requires local model download and enough CPU/GPU/RAM. |
| Deterministic provider | Local or Azure `llm-service` container | Proves microservice wiring and keeps Azure cost low when no Azure OpenAI is configured. | Not real AI inference; acceptable only as a technical fallback/demo mode. |
| Azure OpenAI | Azure SaaS behind `llm-service` | Cleanest cloud/SaaS story for the final requirements. | Requires endpoint, key, and deployment names; consumes Azure OpenAI quota/credits. |
| Hybrid local LLM | Azure services call a public `llm-service` URL tunneled to your PC | Good demo option when Azure should run the app but your local machine should provide stronger model inference. | Requires a stable HTTPS tunnel such as Cloudflare Tunnel/ngrok, your PC must stay online, and it is not a production security model. |
| Ollama in Azure | Optional internal Container App running `ollama/ollama` | Good enough for a POC with a tiny model such as `qwen3:0.6b`. | Can be slow, may produce weak output, auto-pulls models after cold start, and uses more Azure credits. Not production-hardened. |

For this project, the recommended presentation is: use local Ollama/Gemma to show real LLM behavior, then show the Azure deployment with deterministic or Azure OpenAI mode to satisfy the cloud/serverless/SaaS requirements. Running Ollama/Gemma in Azure is a valid future extension, but it is not the cheapest or simplest path for a student POC.

Hybrid Azure-to-local LLM deployment is supported through `EXTERNAL_LLM_SERVICE_URL`. Example:

```bash
# 1. Run local LLM service and expose it with a secure HTTPS tunnel.
# 2. Deploy Azure services pointing to that public llm-service URL.
EXTERNAL_LLM_SERVICE_URL='https://your-tunnel.example.com' \
DEPLOY_APPS=true DEPLOY_POSTGRES=true POSTGRES_ADMIN_PASSWORD='<strong-password>' \
  scripts/deploy_azure.sh book-ai-library-stage-rg spaincentral book-ai-library azure-hybrid
```

In that mode, Azure Container Apps still run the microservices, Service Bus, PostgreSQL, ACR, and Log Analytics. The LLM Service dependency is remote, which is acceptable for a POC architecture demonstration because it proves the service boundary is real.

Tiny self-hosted Azure model POC:

```bash
DEPLOY_APPS=true DEPLOY_POSTGRES=true DEPLOY_OLLAMA=true \
OLLAMA_GENERATE_MODEL='qwen3:0.6b' \
OLLAMA_EMBED_MODEL='embeddinggemma' \
POSTGRES_ADMIN_PASSWORD='<strong-password>' \
  scripts/deploy_azure.sh book-ai-library-stage-rg spaincentral book-ai-library azure-qwen-poc
```

This creates an internal `book-ai-library-ollama` Container App and points `llm-service` at it with `LLM_PROVIDER=ollama-with-fallback` and `OLLAMA_AUTO_PULL=true`. The first request can be slow because the model is downloaded after startup. For a demo, that is acceptable; for production, use Azure OpenAI, managed model serving, or a properly sized GPU workload.

## Repository Structure

```text
.
├── services/
│   ├── user-profile/        # users and reading lists
│   ├── book-catalog/        # metadata, Open Library, BookCreated events
│   ├── embedding-worker/    # BookCreated -> vector -> BookEmbedded
│   ├── recommendation/      # cached recommendations + LLM-guided ask command
│   └── llm-service/         # Ollama / Azure OpenAI adapter
├── frontend/streamlit/      # web UI
├── shared/                  # config, repositories, events, Open Library, text/vector helpers
├── infra/
│   ├── db/001_init.sql      # PostgreSQL + pgvector schema
│   └── bicep/main.bicep     # Azure target infrastructure
├── scripts/                 # CI, smoke, and Azure helper scripts
├── tests/                   # unit, integration, real-process HTTP, browser hooks
├── docker-compose.yml       # local app stack
├── docker-compose.ci.yml    # deterministic CI smoke stack
├── docker-compose.test.yml  # Docker pytest stack
└── .github/workflows/ci.yml
```

Generated local folders such as `.venv/`, `.local/`, `.pytest_cache/`, and `__pycache__/` are ignored and should not be committed.

## Run Locally

```bash
docker compose up -d --build
```

Open:

- Frontend: <http://127.0.0.1:8501>
- User Profile API: <http://127.0.0.1:8001/docs>
- Book Catalog API: <http://127.0.0.1:8002/docs>
- Embedding Worker API: <http://127.0.0.1:8003/docs>
- Recommendation API: <http://127.0.0.1:8004/docs>
- LLM Service API: <http://127.0.0.1:8005/docs>

The only port a normal user needs is `8501`. Ports `8001-8005` are developer API ports:

| Port | Service | Use |
| --- | --- | --- |
| `8001` | User Profile | test account/profile/reading-list endpoints |
| `8002` | Book Catalog | inspect/search/create books and seed catalog |
| `8003` | Embedding Worker | health/status/admin `POST /work` for async processing |
| `8004` | Recommendation | cached recommendations, `Ask AI`, profile summary |
| `8005` | LLM Service | direct embedding/generation adapter tests |

You usually do not open ports `8001-8005` while presenting the app. They exist so a reviewer or developer can inspect each microservice independently through `/docs`, prove the services are separately deployable, and test REST contracts without the frontend.

Example:

```bash
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8005/v1/generate?prompt=Recommend%20one%20book
```

In Azure, only the frontend and Recommendation service are public. The other service URLs are internal Container Apps DNS names.

Useful local flow:

1. Create a local user with email + password, or sign in to an existing local user.
2. Open Explore and add a few books.
3. Open Recommendations and click `Process async updates`.
4. Compare cached modes or use `Ask AI`.
5. Open Home and click `Regenerate profile summary with LLM` only when you want a fresh generated profile.
6. Open Architecture to show service health, event status, and the Azure/ACA-style component layout.

## LLM Service Usage

`GET /v1/generate?prompt=...` is a browser-friendly test helper.

`POST /v1/generate` is the real service command used by the app:

```bash
curl -fsS -X POST http://127.0.0.1:8005/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Recommend one book","context":{"title":"Dune"}}'
```

Expected response shape:

```json
{"text":"...","provider":"ollama:gemma4:e2b"}
```

If Ollama is slow or unavailable and `LLM_PROVIDER=ollama-with-fallback`, the service falls back to deterministic local text so the demo stays usable. With `LLM_PROVIDER=ollama`, failures are returned as upstream errors.

Stop containers without deleting data:

```bash
docker compose down
```

Reset PostgreSQL/Ollama volumes:

```bash
docker compose down -v
```

## Verify

Host tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Docker pytest:

```bash
scripts/docker_pytest.sh
```

Isolated deterministic Compose smoke:

```bash
PYTHON_BIN=python scripts/compose_smoke.sh
```

Smoke the already-running main stack:

```bash
python scripts/local_app_smoke.py
```

Manual recommendation command check against a running Docker stack:

```bash
curl -fsS -X POST http://127.0.0.1:8004/recommendations/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id":"demo-user",
    "type":"widen",
    "prompt":"Recommend something ambitious but readable.",
    "allow_outside_candidates":true
  }'
```

Latest verified state:

- Host pytest: not run in this shell because pytest is not installed globally; use the install command above or Docker pytest.
- Docker pytest: `53 passed, 5 skipped`
- Compose smoke: passed
- Main stack smoke: `ok: local app demo_total=39 recommendations=10`
- Live Docker recommendation command: fresh user + async processing + `POST /recommendations/ask` passed with `source=llm-over-engine-candidates`, `engine=vector-similarity`, `provider=ollama:gemma4:e2b`
- Direct LLM POST inside Compose: passed with `provider=ollama:gemma4:e2b`
- Local auth endpoints inside Compose: signup/signin passed
- Frontend robustness patch: add-book actions no longer crash if an Azure service returns an unexpected payload; demo catalog seed falls back from `POST /catalog/seed/demo` to stable `POST /books` calls when an older Book Catalog image returns HTTP 405.
- 2026-06-17 validation after Azure frontend crash fix: `python3 -m py_compile frontend/streamlit/app.py services/llm-service/app/main.py tests/test_llm_service.py`, shell syntax checks, `az bicep build --file infra/bicep/main.bicep`, `docker compose -f docker-compose.yml config`, `scripts/docker_pytest.sh`, and `python3 scripts/local_app_smoke.py` passed.
- 2026-06-17 Azure fix: all Container Apps updated to image tag `azure-20260617-1`; internal service URLs changed from `http://...internal...` to `https://...internal...` to avoid Azure Container Apps 301 redirects changing POST requests into GET requests. Live internal `POST /me/books` smoke passed and public `scripts/azure_smoke.sh book-ai-library-stage-rg book-ai-library` passed.
- 2026-06-17 Azure database fix: service containers now use shared Azure PostgreSQL through `DATABASE_URL=secretref:database-url`; full internal Azure flow passed: seed demo catalog -> add Dune -> run workers -> read non-empty recommendations. User Profile hotfix image is `azure-20260617-2`.
- 2026-06-17 Azure stage fix: current live image tag is `azure-qwen-poc`; Book Catalog, User Profile, Embedding Worker, and Recommendation have `DATABASE_URL=secretref:database-url`, `OPEN_LIBRARY_TIMEOUT_SECONDS=25`, and internal HTTPS service URLs. Live smoke passed with `scripts/azure_smoke.sh book-ai-library-stage-rg book-ai-library`: frontend OK, Recommendation health OK, public LLM adapter OK with `provider=ollama:qwen3:0.6b`, internal add/read/recommend flow OK with `books=1 recs=10`, and Open Library discovery OK.
- Local LLM generation budget is now `OLLAMA_NUM_PREDICT=512`, so Ask AI responses are less likely to stop mid-sentence.

Browser screenshots are in `tests/test_streamlit_browser.py`. They run in CI with Playwright/Chromium; locally they skip unless browser dependencies are installed.

## Presentation Readiness

Local Docker is ready to present as the application demo. It proves the microservice split, REST APIs, async event flow, PostgreSQL/pgvector persistence, local LLM adapter, frontend, and tests.

For the final university requirement set, also redeploy the Azure stage before submission if the reviewer expects to see cloud resources live. The local app demonstrates the behavior; the Azure deployment demonstrates SaaS/serverless/Kubernetes-style hosting.

Recommended final demo flow:

1. Start locally with `docker compose up -d --build`.
2. Run `scripts/docker_pytest.sh` to show the test suite.
3. Open <http://127.0.0.1:8501>, create/sign in to a user, add books from Explore, run `Process async updates`, then use `Ask AI`.
4. Open the Architecture tab and show each service node, health state, LLM provider, and async worker/event status.
5. Open <http://127.0.0.1:8004/docs> and <http://127.0.0.1:8005/docs> to show independent service APIs.
6. Open [index.html](./index.html) to show the architecture diagram matching the running system.
7. For Azure, run `scripts/azure_smoke.sh book-ai-library-stage-rg book-ai-library` and show Azure Container Apps, Service Bus, PostgreSQL, ACR, and Log Analytics in the Azure portal or CLI output.

What is still not production-grade:

- Local email/password auth is intentionally simple and should be replaced by Entra External ID, Azure AD B2C, Clerk, or another real identity provider.
- Azure deployment still uses connection strings and ACR admin credentials; managed identity/RBAC is the next hardening step.
- Local Ollama is for a cost-free demo. Azure OpenAI is supported behind `llm-service`, but real deployment settings must be provided to use it.
- Only the frontend and Recommendation API are public in Azure. Internal services remain private by design.

## Requirement Demonstration

| Polish requirement | How this project addresses it | How to demonstrate it |
| --- | --- | --- |
| `architektura mikroserwisów (min. 3 węzły)` | Six deployable containers: `frontend`, `user-profile`, `book-catalog`, `embedding-worker`, `recommendation`, `llm-service`. | Run `docker compose ps`; open service Swagger pages on ports `8001-8005`; show `services/` folders. |
| `komunikacja asynchroniczna` | Adding books publishes `BookCreated`/`UserBookAdded`; workers later consume events and compute embeddings/recommendations. | Add a book, then click `Process async updates`; show Architecture tab event/worker status; inspect `shared/events.py`. |
| `usługi SaaS w chmurze` | Azure target uses Azure Service Bus, Azure Database for PostgreSQL, ACR, Log Analytics, and optionally Azure OpenAI. | After redeploy, show these resources in resource group `book-ai-library-stage-rg`; run `scripts/azure_smoke.sh`. |
| `serverless lub Kubernetes` | Azure Container Apps hosts each container app in a managed serverless Kubernetes-like environment with scale settings. | Show Container Apps resources in Azure or `infra/bicep/main.bicep`; explain local Docker maps to ACA. |
| `minimalny frontend` | Streamlit frontend in `frontend/streamlit`, public on port `8501` locally and as a public Container App in Azure. | Open <http://127.0.0.1:8501>; sign in, add books, ask for recommendations. |
| `Infrastructure as Code` | Azure resources are defined in Bicep at `infra/bicep/main.bicep`. | Run `az bicep build --file infra/bicep/main.bicep`; show Bicep resources. |
| `CI/CD` | GitHub Actions runs tests, Docker pytest, Compose smoke, browser checks, Bicep build, and a manual Azure deployment gate. | Open `.github/workflows/ci.yml`; show completed CI jobs or run the workflow manually. |
| `Diagram architektury` | Static architecture document is `index.html`; frontend Architecture tab shows the live topology. | Open `index.html`; compare it to the Architecture tab in Streamlit. |

## CI/CD

GitHub Actions jobs:

- `test`: install deps, compile, run pytest
- `postgres-integration`: run PostgreSQL/pgvector integration tests
- `docker-pytest`: run tests inside Docker with PostgreSQL
- `compose-smoke`: build/run deterministic local stack and smoke it
- `streamlit-browser`: run browser screenshot checks against the CI stack
- `bicep-build`: validate Bicep syntax
- `azure-deploy-gate`: manual Azure deployment gate only

Azure deployment is manual by design. Do not deploy paid resources automatically.

## Azure Path

Target Azure resources are defined in `infra/bicep/main.bicep`:

- Azure Container Apps environment
- Container Apps for services/frontend
- Azure Container Registry
- Azure Service Bus topics/subscriptions
- Optional Azure Database for PostgreSQL Flexible Server
- Log Analytics
- Azure OpenAI configuration through environment variables/secrets

Current local-to-Azure mapping:

- Local `postgres` container -> Azure Database for PostgreSQL Flexible Server with pgvector.
- Local event table adapter -> Azure Service Bus topics/subscriptions behind the same `publish`/`pull` semantics.
- Local Ollama/Gemma through `llm-service` -> Azure OpenAI or optional internal Azure Ollama behind the same `/v1/embed` and `/v1/generate` API.
- Local Docker Compose services -> Azure Container Apps containers.
- Local Streamlit frontend -> public Azure Container App.

## Current Azure Deployment

Deployed on 2026-06-16 and fixed/validated again on 2026-06-17 in the Azure for Students subscription.

| Item | Value |
| --- | --- |
| Resource group | `book-ai-library-stage-rg` |
| Region | `spaincentral` |
| Image tag | `azure-qwen-poc` |
| LLM mode | Internal Azure Ollama Container App with `qwen3:0.6b` behind `llm-service`; `llm-service` still exposes `/v1/embed` and `/v1/generate` |
| ACR | `bookaibookailibrary4biet5q2bsamwacr.azurecr.io` |
| PostgreSQL server | `book-ai-library-4biet5q2bsamw-pg` |
| Service Bus namespace | `book-ai-library-4biet5q2bsamw-bus` |
| Frontend URL | Current observed URL: `https://book-ai-library-frontend.wittydesert-682b90bb.spaincentral.azurecontainerapps.io/`; previous deleted URL was `https://book-ai-library-frontend.whitehill-edc41080.spaincentral.azurecontainerapps.io` |
| Recommendation URL | Current observed URL: `https://book-ai-library-recommendation.wittydesert-682b90bb.spaincentral.azurecontainerapps.io`; previous deleted URL was `https://book-ai-library-recommendation.whitehill-edc41080.spaincentral.azurecontainerapps.io` |

The current stage was updated in place on 2026-06-17. To recreate the same stage deployment, run:

```bash
DEPLOY_APPS=true DEPLOY_POSTGRES=true DEPLOY_OLLAMA=true \
OLLAMA_GENERATE_MODEL='qwen3:0.6b' \
OLLAMA_EMBED_MODEL='embeddinggemma' \
POSTGRES_ADMIN_PASSWORD='<strong-password>' \
  scripts/deploy_azure.sh book-ai-library-stage-rg spaincentral book-ai-library azure-qwen-poc
```

Deployment smoke after redeploy:

```bash
scripts/azure_smoke.sh book-ai-library-stage-rg book-ai-library
```

The smoke script checks:

- Frontend public URL returns HTTP 200.
- `GET /health` on Recommendation returns `{"status":"ok","service":"recommendation"}`.
- `GET /recommendations?user_id=demo-user&type=similar` returns HTTP 200; an empty recommendation cache is expected for a fresh database.
- `POST /profile/summary` through Recommendation exercises the public LLM adapter path.
- Internal private service flow from the User Profile container: seed demo catalog, add Dune, read the same user's library from PostgreSQL, run worker endpoints, verify non-empty unread recommendations, and verify Open Library discovery.

Azure frontend troubleshooting:

- If sign-in works but adding a book shows `KeyError: 'book'`, the frontend expected a newer User Profile response shape. Rebuild and redeploy all images with the same tag; the frontend now handles unexpected add-book payloads gracefully instead of crashing.
- If the demo scenario reports `POST /catalog/seed/demo -> HTTP 405: {"detail":"Method Not Allowed"}`, the deployed Book Catalog image is missing the seed endpoint or the route is stale. The frontend now falls back to seeding through `POST /books`, but the correct fix is still to redeploy the latest `book-catalog` image.
- If `POST /me/books` fails because Book Catalog returns `[]`, check for internal service URLs beginning with `http://`. Azure Container Apps redirects those calls to `https://`; Python `requests` follows a 301 by converting POST to GET. The Bicep and live stage now use `https://...internal...` URLs.
- If adding a book shows success but the reading list is empty, check that `DATABASE_URL=secretref:database-url` exists on `book-catalog`, `user-profile`, `embedding-worker`, and `recommendation`. Without it, each service falls back to `/tmp/app_state.json`, so the deployed microservices do not share state.
- If Open Library search returns HTTP 502 with a read timeout, confirm `OPEN_LIBRARY_TIMEOUT_SECONDS=25` is present on Book Catalog. Azure egress can be slower than local Docker.
- After redeploying, hard-refresh the browser because Streamlit can keep stale frontend state after an exception.

Container App scale settings:

- `embedding-worker`: min `1`, max `2`
- `recommendation`: min `1`, max `2`
- request/response services and frontend: min `0`, max `2`

Azure OpenAI was not configured in the first deployment because no endpoint/deployment settings were provided. The cloud `llm-service` therefore ran deterministic mode behind the same `/v1/embed` and `/v1/generate` API. To enable Azure OpenAI, redeploy with:

```bash
DEPLOY_APPS=true DEPLOY_POSTGRES=true POSTGRES_ADMIN_PASSWORD='<same-or-rotated-password>' \
AZURE_OPENAI_ENDPOINT='https://<name>.openai.azure.com' \
AZURE_OPENAI_API_KEY='<key>' \
AZURE_OPENAI_EMBED_DEPLOYMENT='<embedding-deployment>' \
AZURE_OPENAI_CHAT_DEPLOYMENT='<chat-deployment>' \
scripts/deploy_azure.sh book-ai-library-stage-rg spaincentral book-ai-library azure-20260616-1
```

To stop Azure costs for this stage environment, delete the resource group:

```bash
az group delete --name book-ai-library-stage-rg
```

Validate Bicep:

```bash
az bicep build --file infra/bicep/main.bicep
```

Preview infra without apps/PostgreSQL:

```bash
DEPLOY_APPS=false DEPLOY_POSTGRES=false \
  scripts/azure_what_if.sh book-ai-library-stage-rg spaincentral book-ai-library local-test
```

Preview with PostgreSQL:

```bash
DEPLOY_APPS=false DEPLOY_POSTGRES=true POSTGRES_ADMIN_PASSWORD='<strong-password>' \
  scripts/azure_what_if.sh book-ai-library-stage-rg spaincentral book-ai-library local-test
```

Allowed Azure for Students regions observed for this subscription:

- `italynorth`
- `switzerlandnorth`
- `swedencentral`
- `spaincentral`
- `germanywestcentral`

Use `spaincentral` unless there is a reason not to.

## Remaining Work

Keep the project lean and presentation-ready:

1. Add more REST contract tests for idempotency and conditional requests.
2. Add persisted worker failure history or DLQ-style diagnostics.
3. Harden Azure auth: managed identity/RBAC instead of connection strings and ACR admin credentials.
4. Add Azure live smoke tests after a manual deployment.
5. Replace local demo password handling with Azure AD B2C, Entra External ID, or another real identity provider before treating this as production software.
6. Split broad Python dependencies if image size becomes a grading concern.
