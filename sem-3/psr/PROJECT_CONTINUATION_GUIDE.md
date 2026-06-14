# Book AI Library - Continuation Guide

This document explains how to continue the project from the current POC state. It is written as a teaching handoff: what to do, why it matters, and how to avoid common mistakes.

For the current engineering review, known weaknesses, and recommended priority order, read [PROJECT_REVIEW_AND_PRIORITIES.md](./PROJECT_REVIEW_AND_PRIORITIES.md) before starting new implementation work. For the single canonical continuation prompt, read [NEXT_PROMPT.md](./NEXT_PROMPT.md).

## 1. Where The Project Is Now

You have a working local POC of a cloud-native service-oriented application:

- Frontend: Streamlit.
- Backend services: User Profile, Book Catalog, Embedding Worker, Recommendation, LLM Service.
- Data: PostgreSQL with normalized tables and pgvector support in Docker Compose.
- Async communication: event boundary is preserved through `BookCreated`, `BookEmbedded`, and `UserBookAdded`.
- LLM path: local Ollama through the `llm-service` container, with deterministic mode for tests. The default local generation model is `gemma4:e2b`, and app calls use `OLLAMA_THINK=false` so Gemma 4 returns final text through `/api/generate` instead of spending the token budget on hidden thinking.
- Standalone LLM path: `docker-compose.llm.yml` can run only Ollama and LLM Service for quick prompt testing.
- AMD GPU path: `docker-compose.amd.yml` and `docker-compose.llm.amd.yml` switch Ollama to the ROCm image and pass AMD GPU devices into the container.
- Local demo catalog: `POST /catalog/seed/demo` gives the recommender a stable offline corpus when Open Library is slow or unavailable.
- Recommendation filtering: cached reads and recomputation exclude books already in the user's reading list by exact ID and logical identity (`isbn`, `openlibrary_key`, normalized `title + author`). The API returns a `filter_summary`, so the frontend can show how many cached items were hidden. This matters because demo/manual/Open Library sources can create different database IDs for the same book.
- Azure path: Bicep, GitHub Actions, Azure Service Bus adapter, and Azure OpenAI adapter are prepared. Bicep builds locally, Azure what-if passes in `spaincentral`, and the adapters have mocked tests.

The correct mental model is:

```text
Local Docker Compose = staging / development environment.
Azure Container Apps = target deployment environment.
Azure OpenAI and Service Bus = production-like cloud dependencies.
```

Do not skip the local staging step. It is cheaper, faster, and safer than debugging first-time architecture mistakes in Azure.

## 2. The Most Important Design Rule

Preserve the async boundary.

When a user adds a book, the user request must finish quickly. Embeddings and recommendations happen later:

```text
POST /me/books
  -> Book Catalog creates/enriches book
  -> BookCreated event
  -> Embedding Worker computes vector
  -> BookEmbedded event
  -> Recommendation Service recomputes cached recommendations
```

Why this matters:

- The UI is responsive.
- LLM latency does not block user requests.
- Recommendation reads stay simple and fast.
- The examiner can clearly see the asynchronous communication requirement.

Do not move LLM calls into `GET /recommendations`. That would make the architecture worse and weaken the project.

## 3. What You Should Do Next

### Step 1 - Keep Local Docker Compose Green

Run:

```bash
pytest -q
docker compose up -d --build
.venv/bin/python scripts/local_app_smoke.py
```

Why:

- `pytest` checks service logic and contracts.
- `local_app_smoke.py` checks that the running web-facing Docker stack works through published ports.
- This is your staging gate before Azure.

If this fails, fix local first. Do not deploy broken local behavior to Azure.

### Step 2 - Use The Standalone LLM Stack When Debugging Prompts

Run:

```bash
docker compose -f docker-compose.llm.yml up -d --build
./scripts/llm_smoke.py
./scripts/llm_prompt.py "Recommend one short science fiction book"
```

You can also open:

- <http://127.0.0.1:8005/>
- <http://127.0.0.1:8005/docs>

Why:

- It isolates the LLM from the rest of the app.
- Ollama models stay in the persistent `ollama-models` Docker volume.
- You hit `llm-service` on port `8005`, not Ollama directly.
- Ollama remains internal-only, so host port `11434` can still be used by another local Ollama install.
- Docker defaults to `LLM_PROVIDER=ollama-with-fallback`: it tries Ollama first and logs/falls back to a deterministic template only if Ollama fails, times out, or returns empty text.

Gemma 4 model choice:

- Use `gemma4:e2b` by default for the local POC. It is the practical choice for an AMD RX 6600-class 8 GB GPU.
- Use `OLLAMA_GENERATE_MODEL=gemma4:e4b` only if the local machine has enough GPU memory.
- Use `OLLAMA_GENERATE_MODEL=gemma4:26b` only as an explicit MoE experiment. It has about 4B active parameters, but the full package is much larger than the edge models.
- Do not set `gemma4:31b` as the default for this project; it is a workstation-class dense model.
- Keep `OLLAMA_THINK=false` for normal app demos. The Ollama CLI may show thinking output, but `/api/generate` can return an empty `response` when thinking consumes the whole `num_predict` budget.

For AMD GPU acceleration, first verify the host devices:

```bash
ls -l /dev/kfd /dev/dri
```

Then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.amd.yml up -d --build
```

or for only the LLM stack:

```bash
docker compose -f docker-compose.llm.yml -f docker-compose.llm.amd.yml up -d --build
```

If `/dev/kfd` or `/dev/dri` is missing, Docker cannot use the AMD GPU yet. Fix ROCm/driver/container permissions before treating GPU acceleration as verified.

### Step 3 - Use The Demo Catalog For Reliable Local Testing

When using the web UI, click `Seed demo catalog` before expecting recommendations.

Why:

- A recommender needs unread candidate books to recommend.
- Open Library may be slow or temporarily unavailable.
- The demo seed publishes the same `BookCreated` events, so it still tests the async embedding pipeline.

For presentation, use the stronger one-click path:

```text
System flow -> Run demo scenario
```

That button seeds the demo catalog, adds Dune to the reading list, runs the embedding and recommendation workers, and reads cached recommendations. It is the fastest way to show the microservice flow without relying on background timing.

### Step 4 - Understand The Azure Region Policy

The Azure for Students subscription is restricted by policy. The allowed deployment regions are:

- `italynorth`
- `switzerlandnorth`
- `swedencentral`
- `spaincentral`
- `germanywestcentral`

Use `spaincentral` unless you have a reason to choose another allowed region.

Why:

- `westeurope` and `polandcentral` were rejected by policy.
- Guessing regions wastes time.
- A successful `what-if` in an allowed region proves that ARM accepts the template shape before you spend money on resources.

### Step 5 - Validate Bicep Before Deploying

Azure CLI is available in the current environment.

Local commands:

```bash
az bicep build --file infra/bicep/main.bicep
```

Then run a what-if deployment:

```bash
az deployment group what-if \
  --resource-group book-ai-library-stage-rg \
  --template-file infra/bicep/main.bicep \
  --parameters location=spaincentral appName=book-ai-library deployApps=false deployPostgres=false
```

Why:

- `az bicep build` catches template syntax problems.
- `what-if` shows what Azure would create before it creates anything.
- You learn the cloud deployment shape without spending money unexpectedly.

### Step 6 - Deploy Infrastructure First, Apps Second

The current deployment model intentionally separates infrastructure from application containers.

First deployment:

```bash
DEPLOY_APPS=false DEPLOY_POSTGRES=false \
  ./scripts/deploy_azure.sh book-ai-library-stage-rg spaincentral book-ai-library local-test
```

Current script behavior:

1. Creates a resource group.
2. Deploys ACR, Log Analytics, Container Apps environment, Service Bus skeleton, and optionally PostgreSQL.
3. If `DEPLOY_APPS=true`, builds and pushes Docker images.
4. If `DEPLOY_APPS=true`, deploys Container Apps.

Before trusting this as production, improve:

- Managed identities instead of ACR admin credentials.
- Key Vault or Container Apps secrets for sensitive values.
- Real Azure OpenAI deployment names.
- Azure Service Bus authentication using managed identity/RBAC instead of connection strings.

## 4. Do You Need To Work In The Azure Browser Portal?

Short answer: no, not for normal development.

You can do almost everything from the command line:

- Create resource groups.
- Deploy Bicep.
- Build and push images.
- Update Container Apps.
- View logs.
- Run what-if deployments.

The browser portal is useful for:

- First-time checking whether resources exist.
- Looking at logs visually.
- Finding exact names of Azure OpenAI deployments.
- Checking quota/region availability.
- Debugging permissions.

Recommended workflow:

```text
Use CLI and Bicep for repeatable changes.
Use the portal only to inspect and verify.
```

Why:

- CLI/Bicep changes are reproducible.
- Portal-only changes are easy to forget and hard to explain in a university submission.
- Infrastructure as Code is one of the requirements, so the final answer should show Bicep, not screenshots of manual clicks.

## 5. Can The Assistant Configure Azure For You?

Yes, but only if your machine/session has safe Azure access configured.

Good options:

1. You run `az login` locally and then ask the assistant to run Azure CLI commands.
2. You create GitHub Actions OIDC credentials and let CI deploy through the manual deployment gate.
3. You provide non-secret resource names and let the assistant prepare commands/Bicep, while you run the final deployment.

Do not paste personal passwords, Azure portal passwords, or long-lived secrets into chat.

If CLI access is used, the safer pattern is:

```bash
az login
az account set --subscription <subscription-id>
az account show
```

Then the assistant can run commands like:

```bash
az bicep build --file infra/bicep/main.bicep
az deployment group what-if ...
```

For GitHub Actions, use OIDC/federated credentials rather than stored Azure passwords. That is more professional and matches modern DevOps practice.

## 6. Should You Start Building In Azure Now?

Not yet as the primary development loop.

You should continue in this order:

```text
1. Local Python tests
2. Local Docker Compose staging
3. CI Docker Compose smoke
4. Azure infrastructure what-if
5. Manual Azure deployment gate
6. Real Azure smoke test
```

Why:

- Local Docker Compose is cheaper and faster.
- You can reset local volumes easily.
- Azure errors often involve permissions, DNS, quotas, and billing, which distract from application logic.
- A stable Compose stack gives you confidence that Azure problems are deployment problems, not application problems.

Use Azure when:

- Local Compose smoke passes.
- Bicep builds.
- You use one of the allowed subscription regions.
- You have decided how to handle secrets.
- You are ready to pay for running resources.

## 7. Can Most Services Run In Azure While The LLM Runs Locally?

Technically yes, but it is usually the wrong architecture for this project.

To make Azure services call a local PC, the local LLM service must be reachable from Azure. That means one of:

- Public IP and port forwarding.
- VPN.
- Tailscale or similar private network.
- Cloudflare Tunnel or ngrok.
- A self-hosted VM that runs Ollama.

Problems with cloud-to-local LLM:

- Your laptop must stay online.
- Home network latency makes embedding slow.
- Security is harder.
- Azure Container Apps cannot call `localhost` on your PC.
- A public tunnel can expose an LLM endpoint accidentally.
- It weakens the "cloud SaaS AI service" requirement if the final architecture depends on your laptop.

Better options:

1. For local development: run everything in Docker Compose, including Ollama.
2. For Azure POC deployment: use Azure OpenAI through `LLM_PROVIDER=azure-openai`.
3. For cost-saving experiments: run the full backend locally and only use Azure for infrastructure validation.
4. For self-hosted cloud LLM: run Ollama on an Azure VM or GPU provider, then point `LLM_SERVICE_URL` or `OLLAMA_BASE_URL` to it.

Professional recommendation:

```text
Use local Ollama only in local Docker Compose.
Use Azure OpenAI for the Azure deployment.
Keep LLM Service as the adapter so the rest of the app does not care which provider is used.
```

## 8. What Should The Final Azure Architecture Be?

Target POC architecture:

```text
Frontend                  -> Azure Container Apps
User Profile Service      -> Azure Container Apps
Book Catalog Service      -> Azure Container Apps
Embedding Worker          -> Azure Container Apps
Recommendation Service    -> Azure Container Apps
LLM Service               -> Azure Container Apps
PostgreSQL + pgvector     -> Azure Database for PostgreSQL Flexible Server
Events                    -> Azure Service Bus topics/subscriptions
Images                    -> Azure Container Registry
Logs                      -> Log Analytics
Embeddings/generation     -> Azure OpenAI
Infrastructure            -> Bicep
CI/CD                     -> GitHub Actions with manual Azure gate
```

This matches the university requirements cleanly:

- Microservices: separate services.
- Async communication: Service Bus.
- SaaS/cloud services: Azure OpenAI, PostgreSQL, Service Bus, ACR.
- Serverless/Kubernetes: Azure Container Apps.
- Frontend: Streamlit.
- IaC: Bicep.
- CI/CD: GitHub Actions.
- Architecture diagram: existing `index.html`.

## 9. What Is Still Weak In The Project?

Be honest about these points in a submission or defense:

- Azure adapters are tested with mocks and Bicep what-if, but not yet proven through live deployed Container Apps.
- Bicep includes PostgreSQL Flexible Server, but still needs managed identities and stronger secret handling.
- Endpoint-level FastAPI tests now have a stable real-process path in `tests/test_http_process_endpoints.py`. It starts Uvicorn services on localhost ports and calls endpoints over HTTP. The old `tests/test_http_endpoints.py` in-process ASGI harness remains quarantined and skipped unless `RUN_EXPERIMENTAL_HTTP_TESTS=1`.
- Failure-path tests now cover Open Library timeout propagation, Book Catalog 502 mapping, Ollama generation fallback, PostgreSQL connection failure, and an embedding-worker LLM dependency failure through the real-process HTTP harness. Azure Service Bus live failure paths and broader endpoint-level errors still need coverage.
- pgvector ANN indexes should be added after the final embedding dimension is known.
- The UI now has a topology-style System flow tab, service health, model config, event backlog/timestamps, worker run/failure/latency state, runtime smoke checks, a one-click demo scenario, explicit recommendation refresh/process controls, side-by-side recommendation mode comparison, recommendation filtering feedback, empty states, and addable recommendation cards. Browser-level regression is wired into CI through a dedicated Playwright/Chromium job.
- The local app smoke now fails if recommendations overlap the user's reading list by ID or logical book identity.
- Local Gemma 4 AMD GPU execution still needs a live smoke run on a host where `/dev/kfd` and `/dev/dri` are visible to Docker.
- The latest main Compose run served `gemma4:e2b` through `llm-service` with `provider=ollama:gemma4:e2b` for direct generation, but Ollama logs showed CPU execution in this sandbox because the AMD ROCm devices were not visible.

This is not a failure. It is a normal POC state. The important thing is that the project already documents the gap and has a clear path to close it.

## 10. Concrete Next Work Package

Do this as the next focused implementation pass:

1. Keep the acceptance loop short: host pytest, Docker pytest, isolated Compose smoke, local app smoke.
2. Extend browser/screenshot regression from the System flow tab to the Recommendations tab and one-click demo path.
3. Polish the System flow tab further: clearer fallback/model status and cleaner visual hierarchy.
4. Expand the real-process HTTP endpoint tests with more failure paths and edge cases.
5. If AMD ROCm devices are present, run the AMD compose override and confirm Ollama uses the GPU path.
6. Only after local testing is smooth, decide whether to deploy infrastructure-only resources in `spaincentral`.
7. Add pgvector ANN indexes after the embedding dimension is final.

Suggested test commands after the pass:

```bash
pytest -q
scripts/docker_pytest.sh
PYTHON_BIN=.venv/bin/python scripts/compose_smoke.sh
docker compose up -d --build
.venv/bin/python scripts/local_app_smoke.py
docker compose -f docker-compose.llm.yml config
docker compose -f docker-compose.llm.yml -f docker-compose.llm.amd.yml config
.venv/bin/python scripts/llm_smoke.py
az bicep build --file infra/bicep/main.bicep
DEPLOY_APPS=false DEPLOY_POSTGRES=true POSTGRES_ADMIN_PASSWORD='<strong-password>' \
  scripts/azure_what_if.sh book-ai-library-stage-rg spaincentral book-ai-library local-test
```

Then, only when those pass, try:

```bash
az deployment group what-if \
  --resource-group <resource-group> \
  --template-file infra/bicep/main.bicep \
  --parameters appName=book-ai-library deployApps=false
```

## 11. Decision Table

| Question | Professional answer |
| --- | --- |
| Do I need Azure Portal in browser? | No. Use it for inspection. Use CLI/Bicep for real changes. |
| Can I work from command line? | Yes. That is the preferred DevOps workflow. |
| Can the assistant configure Azure? | Yes, if `az login` or GitHub OIDC is configured. Do not share passwords or long-lived secrets. |
| Should I deploy now? | Not while local app behavior is still being refined. Infrastructure-only deployment is reasonable later. |
| Should I continue in Docker? | Yes. Treat Docker Compose as staging. |
| Can Azure services call local Ollama? | Technically yes through tunnel/VPN/public endpoint, but it is fragile and not recommended for the final POC. |
| Best LLM setup for local dev? | Ollama inside Docker Compose, defaulting to `gemma4:e2b` and `OLLAMA_THINK=false` for local GPU practicality and stable API responses. |
| Best LLM setup for Azure? | Azure OpenAI behind the existing LLM Service adapter. |
| What must not change? | Do not remove the async recommendation boundary. |
| Why can owned books still appear if filtering only by ID? | The same book can exist as manual, demo, and Open Library records with different IDs. Filter by ISBN/Open Library key/title-author identity too. |

## 12. Current Best Work Focus

The best next work is not Azure. The best next work is making the local demo presentation-proof.

Work in this order:

1. Extend Streamlit browser regression to Recommendations and the one-click demo path.
2. Continue UI polish on the System flow and recommendation states.
3. More HTTP/browser tests using the real-process harness; keep the skipped in-process tests quarantined until the runtime issue is solved.
4. Event observability refinements: persisted failure history, DLQ-style diagnostics, and clearer fallback/model status.
5. Only then return to Azure deployment.

Why:

- The app already satisfies the architectural shape.
- A reviewer will judge the POC by whether the local flow is understandable and reliable.
- Azure deployment should prove portability, not hide local product issues.

## 13. Next Prompt

There is only one continuation prompt for the project: [NEXT_PROMPT.md](./NEXT_PROMPT.md).
