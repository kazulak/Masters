# Book AI Library - Architecture POC Handoff

## Context

This repository contains a university-submission proof-of-concept for **Book AI Library**: a cloud-native SOA application on Microsoft Azure. It includes both the architecture document and a runnable local implementation of the service flow.

The static architecture artifact is [index.html](./index.html). The local application uses FastAPI services, a Streamlit frontend, a JSON-backed local state store, and a lightweight local pub/sub adapter that mirrors the Azure Service Bus event flow.

## Scope Implemented

The POC covers the 8 project requirements and maps each one to a concrete Azure component:

1. Microservices: User Profile, Book Catalog, Embedding, Recommendation, LLM Service, Frontend.
2. Asynchronous communication: Azure Service Bus topics for `BookCreated`, `BookEmbedded`, and `UserBookAdded`.
3. Cloud SaaS services: Azure OpenAI, Azure Database for PostgreSQL, Azure Service Bus, Azure Container Registry.
4. Serverless / Kubernetes: Azure Container Apps.
5. Minimal frontend: Streamlit.
6. Infrastructure as Code: Bicep modules.
7. CI/CD: GitHub Actions.
8. Architecture diagrams: SVG component diagram, Mermaid ERD, Mermaid sequence diagrams.

## Design Decisions

- The architecture is intentionally POC-first but production-extensible.
- Azure Container Apps is used as the managed serverless Kubernetes layer to avoid cluster operations and reduce student-credit cost.
- The LLM Service is isolated as an infrastructure adapter so business services do not depend directly on Azure OpenAI or Ollama SDKs.
- Recommendations are precomputed asynchronously; recommendation reads do not call the LLM on the hot path.
- PostgreSQL with pgvector is used instead of a dedicated vector database because the expected POC scale is below 200k books.
- `book_embeddings.model_version` is included to mitigate embedding model and vector dimension lock-in.

## Application Structure

- [services/user-profile](./services/user-profile): FastAPI user profile and reading-list service.
- [services/book-catalog](./services/book-catalog): FastAPI book metadata service and `BookCreated` publisher.
- [services/embedding-worker](./services/embedding-worker): FastAPI-hosted async worker that consumes `BookCreated`, calls LLM Service, writes embeddings, and publishes `BookEmbedded`.
- [services/recommendation](./services/recommendation): FastAPI recommendation service and async consumer for `BookEmbedded` and `UserBookAdded`.
- [services/llm-service](./services/llm-service): FastAPI local LLM adapter with deterministic embeddings and template generation.
- [frontend/streamlit](./frontend/streamlit): Minimal Streamlit UI.
- [shared](./shared): Shared config, JSON storage, pub/sub, and text/vector utilities.
- [scripts/run_local.sh](./scripts/run_local.sh): Starts all services and Streamlit locally.
- [scripts/smoke_test.py](./scripts/smoke_test.py): End-to-end smoke test for add-book to recommendation.
- [infra/bicep](./infra/bicep): Starter Bicep resources for ACR and Log Analytics.
- [index.html](./index.html): Complete static architecture document.

## How To View

Open [index.html](./index.html) in a browser. Network access is only needed for the Mermaid CDN used to render the ERD and sequence diagrams. The component diagram is inline SVG and works offline.

## How To Run Locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
./scripts/start_local.sh
```

Then open:

- Streamlit frontend: <http://127.0.0.1:8501>
- User Profile API: <http://127.0.0.1:8001/docs>
- Book Catalog API: <http://127.0.0.1:8002/docs>
- Embedding Worker API: <http://127.0.0.1:8003/docs>
- Recommendation API: <http://127.0.0.1:8004/docs>
- LLM Service API: <http://127.0.0.1:8005/docs>

The local state file defaults to `.local/app_state.json`.

Run tests:

```bash
pytest -q
```

Run an end-to-end smoke test while services are running:

```bash
python scripts/smoke_test.py
```

Stop the local app:

```bash
./scripts/stop_local.sh
```

## Next Worker Notes

- Start with API contracts, event models, and database migrations before implementation.
- Build the LLM Service on day 2-3, before business services, because Azure OpenAI quota, latency, and deployment names are the highest-risk external dependency.
- Keep Service Bus usage behind a local `MessageBus` abstraction so Azure Service Bus can later be swapped for Kafka/Event Hubs without changing business code.
- Do not remove the async boundary in the recommendation flow; it is the key architectural decision that keeps read latency predictable.
- Replace the JSON-backed store with PostgreSQL migrations and schema-level roles before any shared deployment.
- Replace the local pub/sub adapter in [shared/events.py](./shared/events.py) with Azure Service Bus SDK calls behind the same `publish` and `pull` semantics.

Token usage: total=143,447 input=107,313 (+ 3,465,856 cached) output=36,134 (reasoning 4,395)
To continue this session, run codex resume 019e423e-e06f-7b70-867c-aeee13a78657

run codex resume 019e423e-e06f-7b70-867c-aeee13a78657
