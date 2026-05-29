# Book AI Library - Continuation Guide

This document explains how to continue the project from the current POC state. It is written as a teaching handoff: what to do, why it matters, and how to avoid common mistakes.

## 1. Where The Project Is Now

You have a working local POC of a cloud-native service-oriented application:

- Frontend: Streamlit.
- Backend services: User Profile, Book Catalog, Embedding Worker, Recommendation, LLM Service.
- Data: PostgreSQL with normalized tables and pgvector support in Docker Compose.
- Async communication: event boundary is preserved through `BookCreated`, `BookEmbedded`, and `UserBookAdded`.
- LLM path: local Ollama through the `llm-service` container, with deterministic mode for tests.
- Azure path: Bicep, GitHub Actions, Azure Service Bus adapter, and Azure OpenAI adapter are prepared, but not fully validated against real Azure resources.

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
PYTHON_BIN=.venv/bin/python scripts/compose_smoke.sh
```

Why:

- `pytest` checks service logic and contracts.
- `compose_smoke.sh` checks that the services work together as containers.
- This is your staging gate before Azure.

If this fails, fix local first. Do not deploy broken local behavior to Azure.

### Step 2 - Add The Missing Production Dependency For Service Bus

The code has an Azure Service Bus adapter in `shared/events.py`, but the Python dependency is not installed yet.

Add a runtime dependency:

```text
azure-servicebus
```

Prefer doing this carefully:

1. Add it to `requirements.txt`.
2. Add tests that mock the Azure client.
3. Rebuild containers.
4. Re-run local tests.

Why:

- The local event adapter works now.
- Azure deployment will need the real Azure client package.
- You want failures to happen in CI, not during a live deployment.

### Step 3 - Validate Bicep Before Deploying

Install Azure CLI locally or use GitHub Actions.

Local commands:

```bash
az login
az bicep build --file infra/bicep/main.bicep
```

Then run a what-if deployment:

```bash
az deployment group what-if \
  --resource-group <resource-group> \
  --template-file infra/bicep/main.bicep \
  --parameters appName=book-ai-library deployApps=false
```

Why:

- `az bicep build` catches template syntax problems.
- `what-if` shows what Azure would create before it creates anything.
- You learn the cloud deployment shape without spending money unexpectedly.

### Step 4 - Deploy Infrastructure First, Apps Second

The current deployment model intentionally separates infrastructure from application containers.

First deployment:

```bash
./scripts/deploy_azure.sh <resource-group> <location> <app-name> <tag>
```

Current script behavior:

1. Creates a resource group.
2. Deploys ACR, Log Analytics, Container Apps environment, and Service Bus skeleton.
3. Builds and pushes Docker images.
4. Deploys Container Apps.

Before trusting this as production, improve:

- PostgreSQL Flexible Server resource in Bicep.
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
- You know your Azure region supports the needed services.
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

- Azure adapters are prepared but not yet proven against real Azure resources.
- Bicep still needs PostgreSQL Flexible Server, managed identities, and stronger secret handling.
- Endpoint-level FastAPI tests are postponed because `TestClient` hung in the current runtime.
- Failure-path tests are still missing for Open Library, Ollama, Azure Service Bus, and PostgreSQL.
- pgvector ANN indexes should be added after the final embedding dimension is known.

This is not a failure. It is a normal POC state. The important thing is that the project already documents the gap and has a clear path to close it.

## 10. Concrete Next Work Package

Do this as the next focused implementation pass:

1. Add `azure-servicebus` dependency.
2. Add mocked tests for Azure Service Bus publish/pull behavior.
3. Add mocked tests for Azure OpenAI embed/generate behavior.
4. Add Bicep validation to local docs and CI.
5. Add PostgreSQL Flexible Server to Bicep.
6. Add Container Apps secrets for database URL, Service Bus, and Azure OpenAI.
7. Add a manual Azure smoke test script.

Suggested test commands after the pass:

```bash
pytest -q
PYTHON_BIN=.venv/bin/python scripts/compose_smoke.sh
az bicep build --file infra/bicep/main.bicep
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
| Should I deploy now? | Not before Bicep builds, Service Bus dependency is packaged, and local Compose smoke passes. |
| Should I continue in Docker? | Yes. Treat Docker Compose as staging. |
| Can Azure services call local Ollama? | Technically yes through tunnel/VPN/public endpoint, but it is fragile and not recommended for the final POC. |
| Best LLM setup for local dev? | Ollama inside Docker Compose. |
| Best LLM setup for Azure? | Azure OpenAI behind the existing LLM Service adapter. |
| What must not change? | Do not remove the async recommendation boundary. |

## 12. Prompt For The Next Session

Use this prompt when continuing:

```text
You are a senior software developer and DevOps engineer. Continue the Book AI Library POC. Read README.md and PROJECT_CONTINUATION_GUIDE.md first. Keep both documents up to date. The current app passes local tests and Docker Compose smoke; services use granular repositories, event contracts are enforced, and Azure-ready Service Bus/Azure OpenAI adapter paths exist. Do not deploy automatically. Next, package and test the Azure Service Bus dependency, add mocked Azure Service Bus and Azure OpenAI adapter tests, validate Bicep with az bicep build/what-if when Azure CLI is available, add PostgreSQL Flexible Server and stronger secret handling to Bicep, and preserve the async recommendation boundary.
```
