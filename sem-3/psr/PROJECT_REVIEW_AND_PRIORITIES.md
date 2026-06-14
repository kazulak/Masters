# Book AI Library - Review And Priority Plan

Date: 2026-06-14

This review evaluates the current POC against the university requirements and against the standard expected from a presentable microservice/cloud project. The short version: the local POC is technically real and much stronger than a mock architecture document, but it is not yet a polished demo and the Azure path still has cloud-readiness gaps.

## Executive Verdict

The project is in a good POC state, not a finished submission state.

What is already strong:

- The system has real microservices, not only folders pretending to be services.
- The async boundary is preserved: adding a book does not synchronously compute AI recommendations.
- Local Docker Compose has PostgreSQL, pgvector, Ollama, FastAPI services, and Streamlit.
- The LLM adapter, event bus abstraction, and repository layer give the project a reasonable evolution path.
- Documentation is honest about what is local, what is Azure-targeted, and what still needs hardening.

What is still weak:

- The user-facing demo is not yet smooth enough to confidently present without explanation.
- Azure Service Bus behavior is not equivalent to the local event store.
- Azure IaC uses connection strings, ACR admin credentials, and public PostgreSQL access.
- CI proves local behavior, but not enough browser/user workflows or Azure live behavior.
- Recommendation quality is still simple vector similarity plus heuristics, not a rich recommender.

## Requirements Status

| Requirement | Current status | Review |
| --- | --- | --- |
| Microservices, min. 3 nodes | Met locally | Six deployable services exist: frontend, user-profile, book-catalog, embedding-worker, recommendation, llm-service. |
| Async communication | Met locally, partial for Azure | Local event adapter works. Azure Service Bus adapter exists but does not yet preserve the same event envelope/status behavior. |
| SaaS cloud service | Target prepared | Azure OpenAI, Service Bus, PostgreSQL, ACR are represented in Bicep/adapters, but the full live deployed app has not been exercised. |
| Serverless/Kubernetes-like architecture | Target prepared | Azure Container Apps Bicep exists; local Compose is the staging environment. |
| Minimal frontend | Met | Streamlit exists, but needs demo polish and browser-level regression checks. |
| Infrastructure as Code | Partial | Bicep builds and what-if passes, but security and deployment completeness are not production-grade. |
| CI/CD | Partial | Tests, Docker builds, smoke checks, and a manual Azure gate exist. Missing lint, browser tests, and live Azure smoke after deploy. |
| Architecture diagram | Met | `index.html` and documentation cover the architecture. Add the live System flow UI only as a demo aid, not as the official diagram. |

## Priority Findings

### P0 - Fix Before Treating Azure As A Real Stage

1. Azure Service Bus adapter is not behaviorally equivalent to the local event store.

Evidence: [shared/events.py](/home/tom/repos/Masters/sem-3/psr/shared/events.py:63) publishes only `event["payload"]`, while the local path stores an event envelope with `id`, `topic`, `type`, `created_at`, and `delivered_to`. [shared/events.py](/home/tom/repos/Masters/sem-3/psr/shared/events.py:102) reconstructs a partial envelope with empty `created_at` and synthetic delivery state.

Impact: The same worker code can run locally and in Azure, but observability and event history semantics change. The System flow/status UI cannot honestly show Azure backlog/timestamps the same way it shows local events. This weakens the async requirement when demonstrated in the cloud.

Required fix: Send the full event envelope as JSON, keep `subject` as the event type, preserve `created_at`, and introduce explicit retry/dead-letter handling. Add tests that publish/pull the same contract through the Azure adapter mock.

2. Azure IaC currently uses insecure/simple POC auth patterns.

Evidence: [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:52) reads the Service Bus root management connection string. [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:65) enables ACR admin credentials. [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:124) creates PostgreSQL with public network access, and [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:169) allows Azure services through the firewall. Container Apps receive ACR password, Service Bus connection string, database URL, and Azure OpenAI key as secrets at [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:211).

Impact: This is acceptable for a student POC if clearly labeled, but it should not be described as production-ready. It is also the next likely source of deployment bugs.

Required fix: Move toward managed identity, ACR pull via RBAC, Service Bus RBAC, Key Vault or Container Apps secret references, and restricted PostgreSQL networking. Keep the current path only as `poc-simple`.

3. Azure deployment may not run the same app behavior as local Docker.

Evidence: Container Apps set `EVENT_BUS_PROVIDER=azure-service-bus` at [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:282), but app state still defaults to `/tmp/app_state.json` at [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:257) when PostgreSQL is not deployed. `LLM_PROVIDER` is deterministic unless Azure OpenAI settings are provided at [infra/bicep/main.bicep](/home/tom/repos/Masters/sem-3/psr/infra/bicep/main.bicep:279).

Impact: An infrastructure-only deploy can pass but not represent the real recommendation system. A deployed app without PostgreSQL and Azure OpenAI would be a weaker demo than the local Docker stack.

Required fix: Define two explicit Azure stages: `infra-only` for what-if/provisioning and `app-stage` for an actual demo. The `app-stage` should require PostgreSQL, migrations, Service Bus, seeded catalog, and either Azure OpenAI or a documented deterministic fallback.

### P1 - Fix Before Final Presentation

4. Worker failures are currently too easy to miss.

Evidence: [services/embedding-worker/app/main.py](/home/tom/repos/Masters/sem-3/psr/services/embedding-worker/app/main.py:47) catches all exceptions and ignores them. [services/recommendation/app/main.py](/home/tom/repos/Masters/sem-3/psr/services/recommendation/app/main.py:103) does the same.

Impact: If Ollama times out, Open Library is slow, PostgreSQL drops a connection, or an event payload is bad, the app can now show the latest in-memory worker error. Persisted failure history and DLQ-style diagnostics are still missing.

Required fix: Keep the new `last_error`, `last_success_at`, run count, failure count, and duration fields in `/status`; next add persisted failure history and DLQ-style diagnostics. For Azure, rely on Service Bus retry/dead-letter behavior instead of swallowing errors.

5. HTTP endpoint tests have a stable baseline but need broader coverage.

Evidence: [tests/test_http_process_endpoints.py](/home/tom/repos/Masters/sem-3/psr/tests/test_http_process_endpoints.py:1) starts real Uvicorn processes and calls HTTP endpoints. [tests/test_http_endpoints.py](/home/tom/repos/Masters/sem-3/psr/tests/test_http_endpoints.py:15) remains skipped because in-process HTTP transports hang in the current runtime.

Impact: The biggest endpoint-testing risk is reduced, the happy-path multi-service flow is covered, and one real-process LLM dependency failure path is covered. More failure-path HTTP coverage is still needed.

Required fix: Keep the real-process harness and expand it with failure paths and edge cases. Keep the skipped in-process harness quarantined until the runtime issue is understood.

6. The frontend is useful but not yet presentation-grade.

Evidence: The Streamlit app has the right tabs and controls, but the System flow is a status dashboard rather than an interactive architecture explorer. Recommendation refresh is still a manual mental model: users must understand seeding, waiting, processing updates, and cached async recomputation.

Impact: A reviewer may conclude the backend is unreliable if the UI does not clearly show "event queued -> embedding done -> recommendation recomputed." This is a demo risk, not just a design preference.

Required fix: Continue polishing the System flow page and Recommendations page. It now has live topology data, worker latency/failure counters, a one-click "Run demo scenario" button, a presentation-focused recommendation view, and CI-enabled browser regression for System flow, but still needs clearer fallback visualization and broader browser coverage.

7. Recommendation quality is still POC-level.

Evidence: [services/recommendation/app/main.py](/home/tom/repos/Masters/sem-3/psr/services/recommendation/app/main.py:26) scores candidates using vector similarity plus small genre/mood bonuses, and [services/recommendation/app/main.py](/home/tom/repos/Masters/sem-3/psr/services/recommendation/app/main.py:54) uses templated explanations.

Impact: The architecture is valid, but the "AI recommendation" story is thin if judged by output quality. It can still pass the distributed systems requirements, but it will not feel impressive.

Required fix: Keep LLM calls off the hot path, but generate richer explanations during async recomputation. Store them in `recommendations.explanations`. Add a deterministic fallback for tests.

8. PostgreSQL access is not pooled.

Evidence: [shared/repositories.py](/home/tom/repos/Masters/sem-3/psr/shared/repositories.py:29) opens a new psycopg connection per repository operation.

Impact: Fine for a small local POC, but fragile under concurrent requests and bad for Container Apps cold starts. It can also make errors look random during demos.

Required fix: Add a small psycopg connection pool per process. Keep transaction boundaries explicit for multi-step operations such as adding a book to a reading list and publishing an event.

9. Runtime schema creation still competes with migrations.

Evidence: [shared/storage.py](/home/tom/repos/Masters/sem-3/psr/shared/storage.py:64) creates PostgreSQL schema at runtime. The official schema also exists in [infra/db/001_init.sql](/home/tom/repos/Masters/sem-3/psr/infra/db/001_init.sql).

Impact: The database can drift depending on whether it was created by Compose SQL, runtime helpers, or future Azure migration steps.

Required fix: Make migrations authoritative. Use one migration path locally and in Azure. Keep runtime checks limited to "is schema present?" diagnostics.

### P2 - Improve After The Demo Is Stable

10. CI/CD lacks lint, format, type checks, and broader browser coverage.

Evidence: [.github/workflows/ci.yml](/home/tom/repos/Masters/sem-3/psr/.github/workflows/ci.yml:45) runs compile/tests, PostgreSQL integration, Docker pytest, Compose smoke, a Streamlit browser screenshot job, and Bicep build, but no ruff/format/type checks.

Impact: Code quality can degrade during rapid iteration. System flow screenshot coverage now exists, but recommendation/demo browser flows can still regress without a targeted browser test.

Required fix: Add `ruff check`, `ruff format --check`, optional `mypy` for shared/service modules, and extend Playwright screenshot checks to the Recommendations tab and one-click demo path.

11. CI manual Azure gate default region is now corrected, but Azure remains manual-only.

Evidence: [.github/workflows/ci.yml](/home/tom/repos/Masters/sem-3/psr/.github/workflows/ci.yml:18) defaults to `spaincentral`, which matches the allowed Azure for Students region policy.

Impact: Region-policy friction is reduced. Deployment should still remain manual until Azure smoke tests and identity hardening are complete.

Required fix: Add post-deployment Azure smoke tests before treating the manual gate as a real release pipeline.

12. Docker images are not hardened.

Evidence: The service Dockerfiles install the broad project requirements and run as root.

Impact: Acceptable for a local POC, but not ideal for a cloud demo. Images are larger than necessary and have a larger dependency/security surface.

Required fix: Split service dependencies, run as a non-root user, add healthchecks, and keep deterministic/LLM/test dependencies out of services that do not need them.

13. PostgreSQL vector search lacks an ANN index.

Evidence: The schema has pgvector support, but the ANN index is still a documented next step.

Impact: Fine at current demo scale. It will degrade as the seeded catalog grows.

Required fix: After the embedding dimension is fixed for the chosen provider, add an IVFFlat or HNSW index migration and test cosine ordering still works.

14. Observability is basic.

Evidence: Services expose `/health` and some `/status` data, but there are no correlation IDs, traces, or structured request/event logs.

Impact: Debugging multi-service flows is slower than it should be.

Required fix: Add a correlation ID to user actions and propagate it through events. Show that ID in the System flow UI and logs.

## Recommended Work Order

### Phase 1 - Make The Local Demo Convincing

Goal: A reviewer can open the frontend, click one button, and see the architecture working.

Done in the current local pass:

1. Added a "Run demo scenario" button in Streamlit.
2. Kept the topology graph live-data driven: frontend, user-profile, book-catalog, event bus, embedding-worker, llm-service, recommendation, PostgreSQL.
3. Displayed per-node health, counts, model config, and last event timestamps.
4. Displayed recommendation filtering feedback via `filter_summary`.
5. Added stable real-process HTTP endpoint tests.
6. Added Streamlit browser screenshot coverage in `tests/test_streamlit_browser.py` and wired it into CI with Playwright/Chromium.
7. Added worker latency/failure counters to System flow.
8. Improved Recommendations with mode comparison, cache/filtering metrics, empty states, and addable cards.
9. Added a real-process HTTP failure-path test for embedding-worker behavior when LLM Service is unreachable.

Remaining tasks:

1. Extend screenshot/browser regression to the Recommendations tab and one-click demo flow.
2. Expand real-process HTTP tests with more failure paths and edge cases.
3. Polish fallback visualization in the System flow tab.

Why first: This gives the fastest improvement in presentation quality and catches most local usability bugs.

### Phase 2 - Make Async Processing Trustworthy

Goal: Failed events are visible, testable, and recoverable.

Tasks:

1. Persist worker failure history instead of keeping only in-memory latest error.
2. Add DLQ-style event diagnostics to the local event adapter and status UI.
3. Add more failure-path tests for invalid event payloads and worker retry behavior.
4. Make Azure Service Bus publish/pull preserve the full event envelope.
5. Add mocked DLQ/retry tests.

Why second: The async boundary is one of the core project requirements. It must be demonstrably reliable.

### Phase 3 - Make Azure Stage Honest

Goal: Azure is a real stage, not only an infrastructure sketch.

Tasks:

1. Keep `infra-only` deployment separate from `app-stage`.
2. Require PostgreSQL for `app-stage`; do not rely on `/tmp/app_state.json`.
3. Add a migration job or deployment script step.
4. Add Azure smoke tests after deployment.
5. Replace connection-string auth with managed identity where possible.

Why third: Azure debugging is slower and may cost money. Deploy only after local behavior is stable.

### Phase 4 - Improve Recommendation Quality

Goal: Recommendations feel meaningfully AI-assisted while keeping the hot path fast.

Tasks:

1. Keep `GET /recommendations` cache-only.
2. During async recomputation, call LLM Service to generate explanations.
3. Store explanations in PostgreSQL.
4. Add deterministic explanation generation for tests.
5. Add better candidate diversity controls.

Why fourth: This improves the user-facing story without weakening the microservice architecture.

## What Not To Do Yet

- Do not rewrite the frontend to React before the Streamlit demo flow is clear.
- Do not deploy paid Azure resources automatically from CI.
- Do not put LLM calls into `GET /recommendations`.
- Do not make Gemma model size the main task. Larger models will not fix architecture or UX problems.
- Do not call the Azure deployment production-ready until identity, networking, migrations, and live smoke tests are solved.

## Next Prompt

There is only one continuation prompt for the project: [NEXT_PROMPT.md](./NEXT_PROMPT.md).
