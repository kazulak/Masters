# Next Prompt

Use this as the single canonical continuation prompt. Do not keep separate prompt copies in README, the continuation guide, or the review document.

```text
You are a senior software developer and DevOps engineer. Continue the Book AI Library POC. Read README.md, PROJECT_CONTINUATION_GUIDE.md, PROJECT_REVIEW_AND_PRIORITIES.md, and NEXT_PROMPT.md first. Keep documents up to date. Focus locally before Azure unless explicitly asked to deploy.

Current architecture: the app is a local-first microservice POC. Synchronous calls are REST/HTTP between Streamlit and FastAPI services; asynchronous work is handled through BookCreated, BookEmbedded, and UserBookAdded events. The backend supports multiple users through X-User-Id, while the Streamlit frontend is a demo client with a selectable demo user.

Current local status: Docker Compose runs frontend, user-profile, book-catalog, embedding-worker, recommendation, llm-service, PostgreSQL/pgvector, and Ollama. The async recommendation boundary must stay intact: GET /recommendations must remain cache-only and must not call the LLM. Recommendations filter owned books by exact ID and logical identity, return filter_summary, and local_app_smoke.py verifies no recommendation overlaps the reading list.

Recent UX state: Streamlit has a topology-style System flow tab, one-click Run demo scenario, service health, model config, event backlog/timestamps, worker latency/failure counters, runtime smoke checks, recommendation filtering feedback, selectable demo user, and recommendation cards that can be added to the reading list. Recommendation modes are now intentionally different: similar, widen, and mood use different scoring and explanations.

Current test state: host pytest passes with 49 passed/5 skipped. Docker pytest is isolated through scripts/docker_pytest.sh and passes with 50 passed/4 skipped. Compose smoke uses project book-ai-ci, builds the frontend, and uses alternate host ports. Real-process HTTP endpoint tests live in tests/test_http_process_endpoints.py and cover multi-service flows plus an embedding-worker LLM dependency failure path. Streamlit browser screenshot coverage lives in tests/test_streamlit_browser.py; CI now has a dedicated streamlit-browser job that starts the deterministic Compose stack and installs Playwright Chromium.

Next priorities: keep making the frontend genuinely presentation-grade, not just functional. Extend screenshot/browser regression to the Recommendations tab and one-click demo path, improve System flow fallback/model visualization, add more real-process failure-path HTTP tests, and add persisted worker failure history or DLQ-style event diagnostics. Preserve the async recommendation boundary and do not deploy paid Azure resources automatically.
```
