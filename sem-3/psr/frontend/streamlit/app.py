from __future__ import annotations

import html
import os
from datetime import datetime, timezone

import requests
import streamlit as st

USER_PROFILE_URL = os.getenv("USER_PROFILE_URL", "http://127.0.0.1:8001")
BOOK_CATALOG_URL = os.getenv("BOOK_CATALOG_URL", "http://127.0.0.1:8002")
EMBEDDING_WORKER_URL = os.getenv("EMBEDDING_WORKER_URL", "http://127.0.0.1:8003")
RECOMMENDATION_URL = os.getenv("RECOMMENDATION_URL", "http://127.0.0.1:8004")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8005")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "demo-user")

DEMO_BOOK = {
    "title": "Dune",
    "author": "Frank Herbert",
    "genres": ["science fiction", "adventure"],
    "description": "A desert planet, political intrigue, ecology, prophecy, and power.",
    "rating": 5,
}

st.set_page_config(page_title="Book AI Library", page_icon="book", layout="wide")
st.title("Book AI Library")
st.sidebar.header("Demo user")
st.sidebar.text_input("User ID", value=DEFAULT_USER_ID, key="active_user_id")

RECOMMENDATION_MODES = {
    "similar": {
        "label": "Similar",
        "caption": "Closest vector matches to your reading list.",
        "accent": "#2563eb",
    },
    "widen": {
        "label": "Widen",
        "caption": "Prefer good matches from genres you do not already read.",
        "accent": "#0f766e",
    },
    "mood": {
        "label": "Mood",
        "caption": "Bias toward the mood stored in your user profile.",
        "accent": "#7c3aed",
    },
}


def _inject_presentation_css() -> None:
    st.markdown(
        """
        <style>
          .demo-hero {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 16px 18px;
            background: #f9fafb;
            margin: 6px 0 16px;
          }
          .demo-hero h3 {
            margin: 0 0 6px;
            color: #111827;
            font-size: 19px;
            letter-spacing: 0;
          }
          .demo-hero p {
            margin: 0;
            color: #4b5563;
            font-size: 14px;
            line-height: 1.45;
          }
          .mode-panel {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
            min-height: 104px;
          }
          .mode-panel.active {
            border-width: 2px;
            background: #f9fafb;
          }
          .mode-title {
            font-weight: 800;
            color: #111827;
            font-size: 14px;
          }
          .mode-caption {
            color: #4b5563;
            font-size: 12px;
            line-height: 1.35;
            margin-top: 5px;
          }
          .mode-count {
            font-weight: 800;
            color: #111827;
            font-size: 22px;
            margin-top: 8px;
          }
          .rec-card {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 14px;
            background: #ffffff;
            margin-bottom: 12px;
          }
          .rec-card-title {
            color: #111827;
            font-weight: 800;
            font-size: 17px;
            line-height: 1.25;
          }
          .rec-card-author {
            color: #4b5563;
            font-size: 13px;
            margin-top: 3px;
          }
          .rec-pill {
            display: inline-block;
            margin: 8px 6px 0 0;
            padding: 3px 8px;
            border-radius: 999px;
            background: #f3f4f6;
            color: #374151;
            font-size: 11px;
            font-weight: 700;
          }
          .rec-explanation {
            color: #374151;
            font-size: 13px;
            line-height: 1.45;
            margin-top: 10px;
          }
          .empty-state {
            border: 1px dashed #9ca3af;
            border-radius: 8px;
            padding: 18px;
            background: #f9fafb;
          }
          .empty-state h4 {
            margin: 0 0 7px;
            color: #111827;
            font-size: 16px;
          }
          .empty-state p {
            margin: 0;
            color: #4b5563;
            font-size: 13px;
            line-height: 1.45;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _user_id() -> str:
    return st.session_state.get("active_user_id") or DEFAULT_USER_ID


def _headers() -> dict[str, str]:
    return {"X-User-Id": _user_id()}


def _json_or_none(method: str, url: str, **kwargs) -> tuple[dict | list | None, str | None]:
    try:
        response = requests.request(method, url, **kwargs)
        if not response.ok:
            return None, response.text
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _age_seconds(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))


def _worker_summary(status_payload: dict) -> dict:
    worker = status_payload.get("worker", {}) if isinstance(status_payload, dict) else {}
    return {
        "last_success_at": worker.get("last_success_at"),
        "last_error": worker.get("last_error"),
        "last_error_at": worker.get("last_error_at"),
        "last_duration_ms": worker.get("last_duration_ms"),
        "total_runs": worker.get("total_runs", 0),
        "total_failures": worker.get("total_failures", 0),
        "consecutive_failures": worker.get("consecutive_failures", 0),
        "last_result": worker.get("last_result") or {},
    }


def _health() -> dict[str, bool]:
    endpoints = {
        "Frontend": None,
        "User Profile": USER_PROFILE_URL,
        "Book Catalog": BOOK_CATALOG_URL,
        "Embedding Worker": EMBEDDING_WORKER_URL,
        "Recommendation": RECOMMENDATION_URL,
        "LLM Service": LLM_SERVICE_URL,
    }
    status = {}
    for name, base_url in endpoints.items():
        if base_url is None:
            status[name] = True
            continue
        _, error = _json_or_none("GET", f"{base_url}/health", timeout=3)
        status[name] = error is None
    return status


def _process_async_once() -> dict[str, dict | str]:
    result: dict[str, dict | str] = {}
    payload, error = _json_or_none("POST", f"{EMBEDDING_WORKER_URL}/work", timeout=60)
    result["embedding"] = payload if error is None else error
    payload, error = _json_or_none("POST", f"{RECOMMENDATION_URL}/work", timeout=30)
    result["recommendation"] = payload if error is None else error
    return result


def _recommendations(rec_type: str) -> tuple[dict | None, str | None]:
    payload, error = _json_or_none(
        "GET",
        f"{RECOMMENDATION_URL}/recommendations",
        params={"user_id": _user_id(), "type": rec_type},
        timeout=10,
    )
    return payload if isinstance(payload, dict) else None, error


def _reading_list() -> tuple[list[dict], str | None]:
    payload, error = _json_or_none("GET", f"{USER_PROFILE_URL}/me/books", headers=_headers(), timeout=10)
    if error:
        return [], error
    return (payload or {}).get("books", []) if isinstance(payload, dict) else [], None


def _seed_demo_catalog() -> tuple[dict | None, str | None]:
    payload, error = _json_or_none("POST", f"{BOOK_CATALOG_URL}/catalog/seed/demo", timeout=20)
    return payload if isinstance(payload, dict) else None, error


def _add_demo_book() -> tuple[dict | None, str | None]:
    payload, error = _json_or_none(
        "POST",
        f"{USER_PROFILE_URL}/me/books",
        json=DEMO_BOOK,
        headers=_headers(),
        timeout=20,
    )
    return payload if isinstance(payload, dict) else None, error


def _run_demo_scenario() -> dict:
    steps: list[dict[str, str]] = []

    seed, seed_error = _seed_demo_catalog()
    steps.append(
        {
            "step": "Seed candidate catalog",
            "status": "pass" if seed_error is None else "fail",
            "detail": f"{(seed or {}).get('total_catalog_size', '-')} catalog books" if seed_error is None else seed_error,
        }
    )
    if seed_error:
        return {"ok": False, "steps": steps}

    added, add_error = _add_demo_book()
    added_book = (added or {}).get("book", {})
    steps.append(
        {
            "step": "Add user book",
            "status": "pass" if add_error is None else "fail",
            "detail": added_book.get("title", add_error or "-"),
        }
    )
    if add_error:
        return {"ok": False, "steps": steps}

    pipeline_result: dict[str, dict | str] = {}
    for _ in range(3):
        pipeline_result = _process_async_once()
    async_ok = isinstance(pipeline_result.get("embedding"), dict) and isinstance(pipeline_result.get("recommendation"), dict)
    steps.append(
        {
            "step": "Process async pipeline",
            "status": "pass" if async_ok else "fail",
            "detail": str(pipeline_result),
        }
    )

    recs, rec_error = _recommendations("similar")
    rec_count = len((recs or {}).get("books", []))
    filter_summary = (recs or {}).get("filter_summary", {})
    steps.append(
        {
            "step": "Read cached recommendations",
            "status": "pass" if rec_error is None and rec_count > 0 else "fail",
            "detail": (
                f"{rec_count} shown, {filter_summary.get('owned_filtered_count', 0)} owned filtered"
                if rec_error is None
                else rec_error
            ),
        }
    )

    return {
        "ok": all(step["status"] == "pass" for step in steps),
        "steps": steps,
        "added_book": added_book,
        "recommendations": recs or {},
        "pipeline_result": pipeline_result,
    }


def _flow_snapshot() -> dict:
    catalog, _ = _json_or_none("GET", f"{BOOK_CATALOG_URL}/books", timeout=10)
    reading, _ = _json_or_none("GET", f"{USER_PROFILE_URL}/me/books", headers=_headers(), timeout=10)
    models, _ = _json_or_none("GET", f"{LLM_SERVICE_URL}/v1/models", timeout=10)
    embedding_status, _ = _json_or_none("GET", f"{EMBEDDING_WORKER_URL}/status", timeout=10)
    recommendation_status, _ = _json_or_none("GET", f"{RECOMMENDATION_URL}/status", timeout=10)
    rec_counts = {}
    rec_filters = {}
    for rec_type in ("similar", "widen", "mood"):
        payload, _ = _recommendations(rec_type)
        rec_counts[rec_type] = len((payload or {}).get("books", []))
        rec_filters[rec_type] = (payload or {}).get("filter_summary", {})
    event_backlog = []
    seen_backlog = set()
    for payload in (embedding_status, recommendation_status):
        if isinstance(payload, dict):
            for row in payload.get("event_backlog", []):
                key = (row.get("topic"), row.get("subscriber"), tuple(row.get("event_types", [])))
                if key in seen_backlog:
                    continue
                seen_backlog.add(key)
                event_backlog.append(row)
    return {
        "catalog_count": len(catalog or []),
        "reading_count": len((reading or {}).get("books", [])),
        "rec_counts": rec_counts,
        "rec_filters": rec_filters,
        "models": models or {},
        "event_backlog": event_backlog,
        "embedding_status": embedding_status or {},
        "recommendation_status": recommendation_status or {},
    }


def _run_smoke_checks() -> list[dict[str, str]]:
    checks = []
    for name, ok in _health().items():
        checks.append({"check": f"{name} health", "status": "pass" if ok else "fail", "detail": "online" if ok else "offline"})

    seed, seed_error = _seed_demo_catalog()
    checks.append(
        {
            "check": "Demo catalog seed",
            "status": "pass" if seed_error is None else "fail",
            "detail": f"{(seed or {}).get('total_catalog_size', '-')} books" if seed_error is None else seed_error,
        }
    )

    async_result = _process_async_once()
    async_ok = isinstance(async_result.get("embedding"), dict) and isinstance(async_result.get("recommendation"), dict)
    checks.append({"check": "Async workers", "status": "pass" if async_ok else "fail", "detail": str(async_result)})

    recs, rec_error = _recommendations("similar")
    rec_count = len((recs or {}).get("books", []))
    checks.append(
        {
            "check": "Recommendation read",
            "status": "pass" if rec_error is None and rec_count > 0 else "fail",
            "detail": f"{rec_count} books" if rec_error is None else rec_error,
        }
    )

    models, model_error = _json_or_none("GET", f"{LLM_SERVICE_URL}/v1/models", timeout=10)
    checks.append(
        {
            "check": "LLM model config",
            "status": "pass" if model_error is None else "fail",
            "detail": (models or {}).get("ollama_generate_model", model_error or "-"),
        }
    )
    return checks


def _render_flow(status: dict[str, bool], snapshot: dict) -> None:
    pending_by_subscriber = {
        row.get("subscriber"): row.get("pending", 0)
        for row in snapshot.get("event_backlog", [])
    }
    last_by_subscriber = {
        row.get("subscriber"): row.get("last_delivered_at") or row.get("last_event_at") or "-"
        for row in snapshot.get("event_backlog", [])
    }
    catalog_count = snapshot.get("catalog_count", 0)
    reading_count = snapshot.get("reading_count", 0)
    rec_total = sum(snapshot.get("rec_counts", {}).values())
    embedding_pending = pending_by_subscriber.get("embedding-worker", 0)
    recommendation_pending = pending_by_subscriber.get("recommendation-service", 0)
    llm_model = snapshot.get("models", {}).get("ollama_generate_model", "-")
    embedding_worker = _worker_summary(snapshot.get("embedding_status", {}))
    recommendation_worker = _worker_summary(snapshot.get("recommendation_status", {}))
    llm_provider = snapshot.get("models", {}).get("provider", "-")

    def node(
        name: str,
        detail: str,
        grid_class: str,
        status_name: str | None = None,
        kind: str = "svc",
        metric: str | None = None,
    ) -> str:
        state = "online" if status_name is None or status.get(status_name) else "offline"
        return (
            f"<div class='topology-node {kind} {state} {grid_class}'>"
            f"<div class='node-title'>{html.escape(name)}</div>"
            f"<div class='node-detail'>{html.escape(detail)}</div>"
            + (f"<div class='node-metric'>{html.escape(metric)}</div>" if metric else "")
            +
            f"<div class='node-state'>{'online' if state == 'online' else 'offline'}</div>"
            "</div>"
        )

    st.markdown(
        """
        <style>
          .topology {
            position: relative;
            display: grid;
            grid-template-columns: 1fr 1.15fr 1.15fr 1.15fr 1.05fr;
            grid-template-rows: auto auto auto;
            gap: 16px 18px;
            min-width: 920px;
            overflow-x: auto;
            padding: 14px 4px 22px;
          }
          .topology::before {
            content: "";
            position: absolute;
            left: 11%;
            right: 11%;
            top: 40%;
            border-top: 2px solid #9ca3af;
            z-index: 0;
          }
          .topology::after {
            content: "";
            position: absolute;
            left: 46%;
            right: 21%;
            top: 62%;
            border-top: 2px dashed #b45309;
            z-index: 0;
          }
          .topology-node {
            position: relative;
            z-index: 1;
            min-height: 88px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 12px;
            background: #f9fafb;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
          }
          .topology-node.online { border-color: #0f766e; background: #ecfdf5; }
          .topology-node.offline { border-color: #b91c1c; background: #fef2f2; }
          .topology-node.data { border-color: #2563eb; background: #eff6ff; }
          .topology-node.bus { border-color: #b45309; background: #fffbeb; }
          .node-title { font-weight: 700; color: #111827; font-size: 14px; }
          .node-detail { color: #4b5563; font-size: 12px; line-height: 1.35; margin-top: 5px; }
          .node-metric {
            margin-top: 8px;
            color: #111827;
            font-size: 12px;
            font-weight: 700;
          }
          .node-state {
            display: inline-block;
            margin-top: 10px;
            padding: 2px 7px;
            border-radius: 999px;
            background: rgba(255,255,255,.72);
            color: #374151;
            font-size: 11px;
            font-weight: 700;
          }
          .topology-label {
            position: relative;
            z-index: 2;
            align-self: center;
            justify-self: center;
            padding: 3px 8px;
            border-radius: 999px;
            background: #ffffff;
            color: #4b5563;
            border: 1px solid #e5e7eb;
            font-size: 11px;
            font-weight: 700;
          }
          .topology-label.async { color: #92400e; border-color: #fcd34d; background: #fffbeb; }
          .frontend { grid-column: 1; grid-row: 1 / span 2; align-self: center; }
          .profile { grid-column: 2; grid-row: 1; }
          .catalog { grid-column: 2; grid-row: 2; }
          .recommend { grid-column: 2; grid-row: 3; }
          .rest-label { grid-column: 1 / span 2; grid-row: 1; transform: translateY(72px); }
          .bus { grid-column: 3; grid-row: 2; }
          .worker { grid-column: 4; grid-row: 2; }
          .llm { grid-column: 4; grid-row: 1; }
          .postgres { grid-column: 5; grid-row: 2; }
          .openlib { grid-column: 3; grid-row: 1; }
          .async-label { grid-column: 3 / span 2; grid-row: 3; transform: translateY(-74px); }
          .read-label { grid-column: 2 / span 4; grid-row: 3; transform: translateY(-8px); }
          @media (max-width: 980px) {
            .topology { min-width: 860px; }
          }
        </style>
        <div class="topology">
        """
        + node("Frontend", "Streamlit demo UI. Calls backend APIs; no business logic.", "frontend", "Frontend", metric=f"user: {_user_id()}")
        + "<div class='topology-label rest-label'>REST</div>"
        + node("User Profile", "Owns user profile and reading list. Publishes UserBookAdded.", "profile", "User Profile", metric=f"{reading_count} owned books")
        + node("Book Catalog", "Owns metadata. Enriches/searches Open Library. Publishes BookCreated.", "catalog", "Book Catalog", metric=f"{catalog_count} catalog books")
        + node(
            "Recommendation",
            "Serves cached recommendation reads. No LLM on hot path.",
            "recommend",
            "Recommendation",
            metric=f"{rec_total} shown | {recommendation_worker['last_duration_ms'] or '-'} ms",
        )
        + node("Open Library", "External metadata source for discovery and enrichment.", "openlib", None, "data")
        + node("Service Bus", "Local event adapter now; Azure Service Bus target. topics: books, users.", "bus", None, "bus", metric=f"{embedding_pending + recommendation_pending} pending deliveries")
        + node(
            "Embedding Worker",
            "Consumes BookCreated, calls LLM embed, writes vectors, publishes BookEmbedded.",
            "worker",
            "Embedding Worker",
            metric=f"{embedding_worker['last_duration_ms'] or '-'} ms | last: {last_by_subscriber.get('embedding-worker', '-')}",
        )
        + node("LLM Service", "Adapter for Ollama/Gemma 4 locally and Azure OpenAI in cloud.", "llm", "LLM Service", metric=f"{llm_provider} | {llm_model}")
        + node("PostgreSQL + pgvector", "Normalized tables, embeddings, cached recommendations, event log.", "postgres", None, "data")
        + "<div class='topology-label async async-label'>async events</div>"
        + "<div class='topology-label read-label'>cached recommendation read + vector writes</div>"
        + "</div>",
        unsafe_allow_html=True,
    )


tab_discover, tab_add, tab_list, tab_recs, tab_flow = st.tabs(
    ["Discover", "Add book", "Reading list", "Recommendations", "System flow"]
)

with tab_discover:
    query = st.text_input("Open Library search", value="Ursula Le Guin")
    col_search, col_demo_seed, col_open_seed = st.columns([1, 1, 1])
    with col_search:
        search_clicked = st.button("Search Open Library")
    with col_demo_seed:
        demo_seed_clicked = st.button("Seed demo catalog")
    with col_open_seed:
        open_seed_clicked = st.button("Seed Open Library")

    if demo_seed_clicked:
        response = requests.post(f"{BOOK_CATALOG_URL}/catalog/seed/demo", timeout=15)
        if response.ok:
            result = response.json()
            st.success(
                f"Demo catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                f"{result['total_catalog_size']} books total."
            )
        else:
            st.error(response.text)

    if open_seed_clicked:
        response = requests.post(
            f"{BOOK_CATALOG_URL}/catalog/seed/openlibrary",
            json={"queries": ["science fiction", "fantasy", "mystery", "historical fiction"], "limit_per_query": 8},
            timeout=30,
        )
        if response.ok:
            result = response.json()
            st.success(
                f"Catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                f"{result['total_catalog_size']} books total."
            )
        else:
            fallback = requests.post(f"{BOOK_CATALOG_URL}/catalog/seed/demo", timeout=15)
            if fallback.ok:
                result = fallback.json()
                st.warning("Open Library is unavailable or slow. Seeded the local demo catalog instead.")
                st.success(
                    f"Demo catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                    f"{result['total_catalog_size']} books total."
                )
            else:
                st.error(response.text)

    if search_clicked:
        response = requests.get(
            f"{BOOK_CATALOG_URL}/external/openlibrary/search",
            params={"query": query, "limit": 8},
            timeout=15,
        )
        if not response.ok:
            st.error(response.text)
        else:
            st.session_state["open_library_results"] = response.json()

    for index, book in enumerate(st.session_state.get("open_library_results", [])):
        with st.container(border=True):
            cols = st.columns([1, 4])
            if book.get("cover_url"):
                cols[0].image(book["cover_url"], width=90)
            cols[1].write(f"**{book['title']}** by {book.get('author', 'Unknown')}")
            cols[1].caption(", ".join(book.get("genres", [])[:5]))
            if cols[1].button("Add to reading list", key=f"add-open-library-{index}"):
                response = requests.post(
                    f"{USER_PROFILE_URL}/me/books",
                    json=book | {"rating": 4},
                    headers=_headers(),
                    timeout=15,
                )
                if response.ok:
                    st.session_state["last_added_book"] = response.json()["book"]["title"]
                    st.success(f"Added {st.session_state['last_added_book']}.")
                else:
                    st.error(response.text)

with tab_add:
    with st.form("add-book"):
        title = st.text_input("Title", value="Dune")
        author = st.text_input("Author", value="Frank Herbert")
        genres = st.text_input("Genres", value="science fiction, adventure")
        description = st.text_area(
            "Description",
            value="A desert planet, political intrigue, ecology, prophecy, and power.",
        )
        rating = st.slider("Rating", min_value=1, max_value=5, value=5)
        submitted = st.form_submit_button("Add to reading list")

    if submitted:
        payload = {
            "title": title,
            "author": author,
            "genres": [item.strip() for item in genres.split(",") if item.strip()],
            "description": description,
            "rating": rating,
        }
        response = requests.post(f"{USER_PROFILE_URL}/me/books", json=payload, headers=_headers(), timeout=15)
        if response.ok:
            st.session_state["last_added_book"] = response.json()["book"]["title"]
            st.success(f"Added {st.session_state['last_added_book']}. Recommendations update asynchronously.")
        else:
            st.error(response.text)

with tab_list:
    response = requests.get(f"{USER_PROFILE_URL}/me/books", headers=_headers(), timeout=10)
    if not response.ok:
        st.error(response.text)
    else:
        books = response.json()["books"]
        if not books:
            st.info("No books in the reading list yet.")
        for book in books:
            st.write(f"**{book['title']}** by {book.get('author', 'Unknown')} · rating: {book.get('rating') or '-'}")
            st.caption(", ".join(book.get("genres", [])))

with tab_recs:
    _inject_presentation_css()
    st.markdown(
        """
        <div class="demo-hero">
          <h3>Cached recommendations from the async pipeline</h3>
          <p>
            Add or discover books, process the background workers, then compare three recommendation strategies.
            This page reads only cached recommendations; it never calls the LLM on the hot path.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1.25, 1, 1, 1])
    with controls[0]:
        rec_type = st.radio(
            "Recommendation mode",
            list(RECOMMENDATION_MODES),
            format_func=lambda key: RECOMMENDATION_MODES[key]["label"],
            horizontal=True,
        )
    with controls[1]:
        refresh_clicked = st.button("Refresh cache read", use_container_width=True)
    with controls[2]:
        process_clicked = st.button("Process async updates", use_container_width=True)
    with controls[3]:
        demo_clicked = st.button("Run demo scenario", type="primary", use_container_width=True)

    if rec_type is None:
        rec_type = "similar"

    if process_clicked:
        st.session_state["last_pipeline_result"] = _process_async_once()
    if demo_clicked:
        st.session_state["last_demo_result"] = _run_demo_scenario()
    if refresh_clicked:
        st.rerun()

    if "last_pipeline_result" in st.session_state:
        st.caption(f"Last async pass: {st.session_state['last_pipeline_result']}")
    if "last_demo_result" in st.session_state:
        demo_result = st.session_state["last_demo_result"]
        st.dataframe(demo_result["steps"], hide_index=True, use_container_width=True)
        if demo_result.get("ok"):
            st.success("Demo scenario completed. Recommendations below are cached reads after async processing.")

    mode_payloads = {}
    mode_errors = {}
    for mode in RECOMMENDATION_MODES:
        mode_payloads[mode], mode_errors[mode] = _recommendations(mode)

    comparison_cols = st.columns(3)
    for column, mode in zip(comparison_cols, RECOMMENDATION_MODES, strict=False):
        payload_for_mode = mode_payloads.get(mode) or {}
        summary = payload_for_mode.get("filter_summary", {})
        count = len(payload_for_mode.get("books", []))
        active_class = " active" if mode == rec_type else ""
        accent = RECOMMENDATION_MODES[mode]["accent"]
        column.markdown(
            f"""
            <div class="mode-panel{active_class}" style="border-color:{accent}">
              <div class="mode-title">{html.escape(RECOMMENDATION_MODES[mode]["label"])}</div>
              <div class="mode-caption">{html.escape(RECOMMENDATION_MODES[mode]["caption"])}</div>
              <div class="mode-count">{count} shown</div>
              <div class="mode-caption">{summary.get("owned_filtered_count", 0)} owned hidden</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    payload, error = _recommendations(rec_type)
    if error:
        st.error(error)
    else:
        books = payload["books"] if payload else []
        filter_summary = (payload or {}).get("filter_summary", {})
        status_cols = st.columns(4)
        status_cols[0].metric("Mode", RECOMMENDATION_MODES[rec_type]["label"])
        status_cols[1].metric("Shown", len(books))
        status_cols[2].metric("Owned hidden", filter_summary.get("owned_filtered_count", 0))
        status_cols[3].metric("Resolved cached IDs", filter_summary.get("resolved_count", 0))
        if payload and payload.get("computed_at"):
            st.caption(f"Computed at: {payload['computed_at']}")
        st.caption("Cache-only read: owned books are filtered by ID, ISBN/Open Library key, and normalized title plus author.")
        if not books:
            st.markdown(
                """
                <div class="empty-state">
                  <h4>No cached recommendations for this mode yet</h4>
                  <p>
                    Run the demo scenario for a reliable presentation path, or add a book and press
                    Process async updates. The recommender only serves rows computed by the background pipeline.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for index, book in enumerate(books, start=1):
            card_cols = st.columns([1, 5, 1.2])
            with card_cols[0]:
                if book.get("cover_url"):
                    st.image(book["cover_url"], width=94)
                else:
                    st.markdown(f"### #{index}")
            with card_cols[1]:
                genres = book.get("genres", [])[:4]
                pills = "".join(f"<span class='rec-pill'>{html.escape(genre)}</span>" for genre in genres)
                explanation = payload["explanations"].get(book["id"], "Recommended from your reading profile.")
                st.markdown(
                    f"""
                    <div class="rec-card">
                      <div class="rec-card-title">{html.escape(book["title"])}</div>
                      <div class="rec-card-author">by {html.escape(book.get("author") or "Unknown")}</div>
                      <div>{pills or "<span class='rec-pill'>No genres</span>"}</div>
                      <div class="rec-explanation">{html.escape(explanation)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with card_cols[2]:
                st.write("")
                st.write("")
                if st.button("Add", key=f"add-rec-{rec_type}-{book['id']}", use_container_width=True):
                    add_payload = {
                        key: book.get(key)
                        for key in (
                            "title",
                            "author",
                            "isbn",
                            "description",
                            "genres",
                            "published_year",
                            "source",
                            "openlibrary_key",
                            "cover_url",
                        )
                    }
                    add_payload["rating"] = 4
                    response = requests.post(
                        f"{USER_PROFILE_URL}/me/books",
                        json=add_payload,
                        headers=_headers(),
                        timeout=15,
                    )
                    if response.ok:
                        st.success(f"Added {book['title']} to {_user_id()}.")
                        st.rerun()
                    else:
                        st.error(response.text)

with tab_flow:
    status = _health()
    snapshot = _flow_snapshot()
    hero_cols = st.columns([1, 1, 2])
    if hero_cols[0].button("Run demo scenario", type="primary"):
        st.session_state["last_demo_result"] = _run_demo_scenario()
        st.rerun()
    if hero_cols[1].button("Refresh flow"):
        st.rerun()
    hero_cols[2].caption("Demo scenario seeds the catalog, adds Dune to the reading list, runs the async workers, then reads cached recommendations.")

    if "last_demo_result" in st.session_state:
        demo_result = st.session_state["last_demo_result"]
        if demo_result.get("ok"):
            st.success("Demo scenario completed end-to-end.")
        else:
            st.error("Demo scenario stopped before completion.")
        st.dataframe(demo_result["steps"], hide_index=True, use_container_width=True)

    _render_flow(status, snapshot)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Catalog", snapshot["catalog_count"])
    metric_cols[1].metric("Reading list", snapshot["reading_count"])
    metric_cols[2].metric("Similar", snapshot["rec_counts"]["similar"])
    metric_cols[3].metric("Widen", snapshot["rec_counts"]["widen"])
    metric_cols[4].metric("Mood", snapshot["rec_counts"]["mood"])

    status_cols = st.columns(6)
    for column, (name, ok) in zip(status_cols, status.items(), strict=False):
        column.metric(name, "online" if ok else "offline")

    model_cols = st.columns(3)
    model_cols[0].metric("LLM provider", snapshot["models"].get("provider", "-"))
    model_cols[1].metric("Generation model", snapshot["models"].get("ollama_generate_model", "-"))
    model_cols[2].metric("Embedding model", snapshot["models"].get("ollama_embed_model", "-"))

    st.subheader("Async worker observability")
    worker_rows = []
    for name, payload in (
        ("Embedding Worker", snapshot["embedding_status"]),
        ("Recommendation", snapshot["recommendation_status"]),
    ):
        worker = _worker_summary(payload)
        worker_rows.append(
            {
                "worker": name,
                "runs": worker["total_runs"],
                "failures": worker["total_failures"],
                "consecutive failures": worker["consecutive_failures"],
                "last duration ms": worker["last_duration_ms"] if worker["last_duration_ms"] is not None else "-",
                "last success": worker["last_success_at"] or "-",
                "last error": worker["last_error"] or "-",
                "last result": worker["last_result"],
            }
        )
    st.dataframe(worker_rows, hide_index=True, use_container_width=True)

    backlog_rows = snapshot["event_backlog"]
    if backlog_rows:
        st.subheader("Async event backlog")
        def backlog_display_row(row: dict) -> dict:
            pending_age = _age_seconds(row.get("oldest_pending_at"))
            return {
                "topic": row["topic"],
                "subscriber": row["subscriber"],
                "events": ", ".join(row["event_types"]),
                "total": row["total"],
                "pending": row["pending"],
                "delivered": row["delivered"],
                "last event": row["last_event_at"] or "-",
                "last delivered": row["last_delivered_at"] or "-",
                "oldest pending": row.get("oldest_pending_at") or "-",
                "pending age s": pending_age if pending_age is not None else "-",
            }

        st.dataframe(
            [backlog_display_row(row) for row in backlog_rows],
            hide_index=True,
            use_container_width=True,
        )

    action_cols = st.columns([1, 2])
    if action_cols[0].button("Run async pass"):
        st.session_state["last_pipeline_result"] = _process_async_once()
        st.rerun()
    if action_cols[1].button("Run smoke checks"):
        st.session_state["last_smoke_checks"] = _run_smoke_checks()

    if "last_smoke_checks" in st.session_state:
        st.subheader("Runtime smoke checks")
        st.caption("These are live demo checks against the running services. Full pytest and Docker checks run in CI or from the terminal.")
        st.dataframe(st.session_state["last_smoke_checks"], hide_index=True, use_container_width=True)

    prompt = st.text_input("LLM prompt", value="Recommend one short science fiction book.")
    if st.button("Send prompt"):
        payload, error = _json_or_none(
            "POST",
            f"{LLM_SERVICE_URL}/v1/generate",
            json={"prompt": prompt},
            timeout=300,
        )
        if error:
            st.error(error)
        else:
            st.write(payload["text"])
            st.caption(payload["provider"])
