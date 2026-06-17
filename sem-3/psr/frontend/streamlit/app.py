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

EXPLORE_LISTS = {
    "Meaning and classics": [
        {
            "title": "The Brothers Karamazov",
            "author": "Fyodor Dostoevsky",
            "genres": ["classic", "philosophy", "literary"],
            "description": "A moral and psychological novel about faith, doubt, family, and responsibility.",
            "source": "curated-demo",
        },
        {
            "title": "Crime and Punishment",
            "author": "Fyodor Dostoevsky",
            "genres": ["classic", "psychology", "literary"],
            "description": "A study of guilt, pride, suffering, and moral consequence.",
            "source": "curated-demo",
        },
        {
            "title": "Man's Search for Meaning",
            "author": "Viktor E. Frankl",
            "genres": ["psychology", "memoir", "philosophy"],
            "description": "A concise account of finding meaning under extreme suffering.",
            "source": "curated-demo",
        },
    ],
    "Short science fiction": [
        {
            "title": "The Left Hand of Darkness",
            "author": "Ursula K. Le Guin",
            "genres": ["science fiction", "literary"],
            "description": "A humane science-fiction novel about culture, gender, diplomacy, and trust.",
            "source": "curated-demo",
        },
        {
            "title": "The Dispossessed",
            "author": "Ursula K. Le Guin",
            "genres": ["science fiction", "political"],
            "description": "A political and philosophical story about two societies and one physicist crossing between them.",
            "source": "curated-demo",
        },
        {
            "title": "Foundation",
            "author": "Isaac Asimov",
            "genres": ["science fiction", "classic"],
            "description": "A compact classic about empire, probability, institutions, and long-range history.",
            "source": "curated-demo",
        },
    ],
    "Modern fantasy foundations": [
        {
            "title": "The Hobbit",
            "author": "J.R.R. Tolkien",
            "genres": ["fantasy", "adventure"],
            "description": "A clear entry point into quest fantasy, courage, home, and wonder.",
            "source": "curated-demo",
        },
        {
            "title": "A Wizard of Earthsea",
            "author": "Ursula K. Le Guin",
            "genres": ["fantasy", "coming of age"],
            "description": "A precise, mythic fantasy about power, naming, pride, and balance.",
            "source": "curated-demo",
        },
        {
            "title": "The Name of the Wind",
            "author": "Patrick Rothfuss",
            "genres": ["fantasy", "adventure"],
            "description": "A lyrical fantasy about talent, mythmaking, memory, and ambition.",
            "source": "curated-demo",
        },
    ],
}

st.set_page_config(page_title="Book AI Library", page_icon="book", layout="wide")
st.title("Book AI Library")
if "active_user_id" not in st.session_state:
    st.session_state["active_user_id"] = DEFAULT_USER_ID

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
          .llm-status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            margin: 10px 0 16px;
          }
          .llm-status-card {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
            min-height: 110px;
          }
          .llm-status-card.ok { border-color: #0f766e; background: #ecfdf5; }
          .llm-status-card.warn { border-color: #b45309; background: #fffbeb; }
          .llm-status-card.info { border-color: #2563eb; background: #eff6ff; }
          .llm-status-title {
            color: #111827;
            font-size: 13px;
            font-weight: 800;
          }
          .llm-status-value {
            color: #111827;
            font-size: 17px;
            font-weight: 800;
            line-height: 1.25;
            margin-top: 7px;
          }
          .llm-status-detail {
            color: #4b5563;
            font-size: 12px;
            line-height: 1.35;
            margin-top: 7px;
          }
          @media (max-width: 900px) {
            .llm-status-grid { grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _user_id() -> str:
    return st.session_state.get("active_user_id") or DEFAULT_USER_ID


def _headers() -> dict[str, str]:
    return {"X-User-Id": _user_id()}


def _profile() -> tuple[dict | None, str | None]:
    payload, error = _json_or_none("GET", f"{USER_PROFILE_URL}/me", headers=_headers(), timeout=10)
    return payload if isinstance(payload, dict) else None, error


def _save_profile(
    user_id: str,
    email: str,
    display_name: str,
    mood: str,
    genres: list[str],
    extra_preferences: dict | None = None,
) -> tuple[dict | None, str | None]:
    preferences = {"mood": mood, "genres": genres}
    if extra_preferences:
        preferences.update(extra_preferences)
    payload, error = _json_or_none(
        "POST",
        f"{USER_PROFILE_URL}/me",
        headers={"X-User-Id": user_id},
        json={
            "id": user_id,
            "email": email,
            "display_name": display_name,
            "preferences": preferences,
        },
        timeout=10,
    )
    return payload if isinstance(payload, dict) else None, error


def _signin(email: str, password: str) -> tuple[dict | None, str | None]:
    normalized_email = email.strip().lower()
    payload, error = _json_or_none(
        "POST",
        f"{USER_PROFILE_URL}/auth/signin",
        json={"email": normalized_email, "password": password},
        timeout=10,
    )
    if _method_not_allowed(error):
        payload, error = _json_or_none(
            "GET",
            f"{USER_PROFILE_URL}/me",
            headers={"X-User-Id": normalized_email},
            timeout=10,
        )
    return payload if isinstance(payload, dict) else None, error


def _signup(
    email: str,
    password: str,
    display_name: str,
    mood: str,
    genres: list[str],
) -> tuple[dict | None, str | None]:
    normalized_email = email.strip().lower()
    payload, error = _json_or_none(
        "POST",
        f"{USER_PROFILE_URL}/auth/signup",
        json={
            "email": normalized_email,
            "password": password,
            "display_name": display_name,
            "mood": mood,
            "genres": genres,
        },
        timeout=10,
    )
    if _method_not_allowed(error):
        payload, error = _json_or_none(
            "POST",
            f"{USER_PROFILE_URL}/me",
            headers={"X-User-Id": normalized_email},
            json={
                "id": normalized_email,
                "email": normalized_email,
                "display_name": display_name,
                "preferences": {"mood": mood, "genres": genres, "_password": password},
            },
            timeout=10,
        )
    return payload if isinstance(payload, dict) else None, error


def _json_or_none(method: str, url: str, **kwargs) -> tuple[dict | list | None, str | None]:
    try:
        response = requests.request(method, url, **kwargs)
        if not response.ok:
            return None, f"{method} {url} -> HTTP {response.status_code}: {response.text}"
        try:
            return response.json(), None
        except ValueError as exc:
            return None, f"{method} {url} -> invalid JSON response: {response.text[:300]}"
    except requests.RequestException as exc:
        return None, str(exc)


def _method_not_allowed(error: str | None) -> bool:
    return bool(error and ("Method Not Allowed" in error or "HTTP 405" in error))


def _demo_seed_books() -> list[dict]:
    books: list[dict] = [DEMO_BOOK]
    for shelf_books in EXPLORE_LISTS.values():
        books.extend(shelf_books)
    unique: dict[tuple[str, str], dict] = {}
    for book in books:
        key = (book.get("title", "").strip().lower(), book.get("author", "").strip().lower())
        if key[0]:
            unique[key] = book
    return list(unique.values())


def _extract_book(payload: dict | list | None) -> dict | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("book")
    if isinstance(nested, dict) and nested.get("title"):
        return nested
    if payload.get("title"):
        return payload
    return None


def _create_catalog_book(book: dict) -> tuple[dict | None, str | None]:
    payload, error = _json_or_none(
        "POST",
        f"{BOOK_CATALOG_URL}/books",
        json={
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
            if book.get(key) is not None
        },
        timeout=15,
    )
    return _extract_book(payload), error


def _add_book_to_reading_list(book: dict, rating: int = 4) -> tuple[dict | None, str | None]:
    payload, error = _json_or_none(
        "POST",
        f"{USER_PROFILE_URL}/me/books",
        json=book | {"rating": rating},
        headers=_headers(),
        timeout=20,
    )
    if error:
        return None, error
    added_book = _extract_book(payload)
    if not added_book:
        return None, f"Unexpected User Profile response from POST /me/books: {payload}"
    return added_book, None


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


def _llm_runtime_cards(models: dict) -> str:
    provider = str(models.get("provider", "-"))
    generate_model = str(models.get("ollama_generate_model", "-"))
    embed_model = str(models.get("ollama_embed_model", "-"))
    timeout = str(models.get("ollama_timeout_seconds", "-"))
    think = str(models.get("ollama_think", "-"))

    if provider == "ollama-with-fallback":
        provider_state = "warn"
        provider_value = "Ollama first, fallback enabled"
        provider_detail = "The app tries local Ollama/Gemma 4 first and uses deterministic output only if Ollama fails, times out, or returns empty text."
    elif provider == "ollama":
        provider_state = "ok"
        provider_value = "Ollama required"
        provider_detail = "Local Ollama must answer. Failures surface as service errors instead of falling back."
    elif provider == "azure-openai":
        provider_state = "info"
        provider_value = "Azure OpenAI"
        provider_detail = "Cloud model path behind the same LLM Service adapter. No business service calls Azure OpenAI directly."
    else:
        provider_state = "info"
        provider_value = "Deterministic local"
        provider_detail = "Fast test/demo mode. It proves service wiring without requiring a downloaded model."

    return f"""
    <div class="llm-status-grid">
      <div class="llm-status-card {provider_state}">
        <div class="llm-status-title">LLM runtime</div>
        <div class="llm-status-value">{html.escape(provider_value)}</div>
        <div class="llm-status-detail">{html.escape(provider_detail)}</div>
      </div>
      <div class="llm-status-card info">
        <div class="llm-status-title">Generation path</div>
        <div class="llm-status-value">{html.escape(generate_model)}</div>
        <div class="llm-status-detail">think={html.escape(think)} · timeout={html.escape(timeout)}s · public API is llm-service, not Ollama.</div>
      </div>
      <div class="llm-status-card info">
        <div class="llm-status-title">Embedding path</div>
        <div class="llm-status-value">{html.escape(embed_model)}</div>
        <div class="llm-status-detail">Embedding Worker calls /v1/embed asynchronously after BookCreated events.</div>
      </div>
    </div>
    """


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
    payload, error = _json_or_none("POST", f"{RECOMMENDATION_URL}/recompute/{_user_id()}", timeout=60)
    result["user_recompute"] = payload if error is None else error
    return result


def _recommendations(rec_type: str) -> tuple[dict | None, str | None]:
    payload, error = _json_or_none(
        "GET",
        f"{RECOMMENDATION_URL}/recommendations",
        params={"user_id": _user_id(), "type": rec_type},
        timeout=10,
    )
    return payload if isinstance(payload, dict) else None, error


def _ask_recommendations(
    prompt: str,
    rec_type: str,
    limit: int,
    allow_outside_candidates: bool = True,
) -> tuple[dict | None, str | None]:
    payload, error = _json_or_none(
        "POST",
        f"{RECOMMENDATION_URL}/recommendations/ask",
        json={
            "user_id": _user_id(),
            "prompt": prompt,
            "type": rec_type,
            "limit": limit,
            "allow_outside_candidates": allow_outside_candidates,
        },
        timeout=180,
    )
    return payload if isinstance(payload, dict) else None, error


def _profile_summary() -> tuple[dict | None, str | None]:
    payload, error = _json_or_none(
        "POST",
        f"{RECOMMENDATION_URL}/profile/summary",
        json={"user_id": _user_id(), "limit": 5},
        timeout=180,
    )
    return payload if isinstance(payload, dict) else None, error


def _reading_list() -> tuple[list[dict], str | None]:
    payload, error = _json_or_none("GET", f"{USER_PROFILE_URL}/me/books", headers=_headers(), timeout=10)
    if error:
        return [], error
    return (payload or {}).get("books", []) if isinstance(payload, dict) else [], None


def _seed_demo_catalog() -> tuple[dict | None, str | None]:
    payload, error = _json_or_none("POST", f"{BOOK_CATALOG_URL}/catalog/seed/demo", timeout=20)
    if _method_not_allowed(error):
        imported = 0
        failures: list[str] = []
        for book in _demo_seed_books():
            created, create_error = _create_catalog_book(book)
            if created:
                imported += 1
            elif create_error:
                failures.append(f"{book.get('title', 'Untitled')}: {create_error}")
        catalog, _ = _json_or_none("GET", f"{BOOK_CATALOG_URL}/books", timeout=10)
        if failures and imported == 0:
            return None, "Demo seed endpoint is unavailable and fallback POST /books failed: " + "; ".join(failures[:3])
        return {
            "imported": imported,
            "existing": 0,
            "total_catalog_size": len(catalog or []),
            "source": "frontend-fallback",
            "warning": "Book Catalog seed endpoint returned 405, so the frontend seeded books through POST /books.",
        }, None
    return payload if isinstance(payload, dict) else None, error


def _add_demo_book() -> tuple[dict | None, str | None]:
    book, error = _add_book_to_reading_list(DEMO_BOOK, DEMO_BOOK["rating"])
    return {"book": book} if book else None, error


def _run_demo_scenario() -> dict:
    steps: list[dict[str, str]] = []

    def log(
        number: int,
        actor: str,
        target: str,
        operation: str,
        channel: str,
        status: str,
        response: str,
        note: str,
    ) -> None:
        steps.append(
            {
                "#": str(number),
                "actor": actor,
                "target": target,
                "operation": operation,
                "channel": channel,
                "status": status,
                "response": response,
                "why it matters": note,
            }
        )

    seed, seed_error = _seed_demo_catalog()
    log(
        1,
        "Frontend",
        "Book Catalog",
        "POST /catalog/seed/demo",
        "REST",
        "pass" if seed_error is None else "fail",
        f"{(seed or {}).get('total_catalog_size', '-')} catalog books" if seed_error is None else seed_error or "-",
        "Creates unread candidate books so the vector engine has something to rank.",
    )
    if seed_error:
        return {"ok": False, "steps": steps}

    added, add_error = _add_demo_book()
    added_book = (added or {}).get("book", {})
    log(
        2,
        "Frontend",
        "User Profile",
        "POST /me/books",
        "REST",
        "pass" if add_error is None else "fail",
        added_book.get("title", add_error or "-"),
        "Persists the user's library entry and asks Book Catalog to deduplicate/create metadata.",
    )
    if add_error:
        return {"ok": False, "steps": steps}

    log(
        3,
        "User Profile",
        "Book Catalog",
        "POST /books",
        "REST",
        "pass",
        f"book_id={added_book.get('id', '-')}",
        "Book Catalog owns book metadata and publishes BookCreated only when a new catalog row appears.",
    )
    log(
        4,
        "User Profile",
        "Event Bus",
        "UserBookAdded",
        "async event",
        "pass",
        f"user_id={_user_id()}",
        "The user's request is finished before recommendation recomputation happens.",
    )

    pipeline_result: dict[str, dict | str] = {}
    for _ in range(3):
        pipeline_result = _process_async_once()
    async_ok = isinstance(pipeline_result.get("embedding"), dict) and isinstance(pipeline_result.get("recommendation"), dict)
    embedding_result = pipeline_result.get("embedding", {})
    recommendation_result = pipeline_result.get("user_recompute") or pipeline_result.get("recommendation", {})
    log(
        5,
        "Embedding Worker",
        "Event Bus",
        "pull BookCreated",
        "async event",
        "pass" if async_ok else "fail",
        str(embedding_result),
        "The worker consumes catalog events after the user already received a response.",
    )
    log(
        6,
        "Embedding Worker",
        "LLM Service",
        "POST /v1/embed",
        "REST",
        "pass" if async_ok else "fail",
        "embedding vector stored" if async_ok else str(embedding_result),
        "LLM Service hides Ollama/Azure OpenAI behind one stable internal API.",
    )
    log(
        7,
        "Embedding Worker",
        "Event Bus",
        "BookEmbedded",
        "async event",
        "pass" if async_ok else "fail",
        "published after vector write" if async_ok else str(embedding_result),
        "Recommendation recomputation starts from a vector-ready event, not from the user request.",
    )
    log(
        8,
        "Recommendation",
        "PostgreSQL + pgvector",
        "recompute similar/widen/mood",
        "async worker",
        "pass" if async_ok else "fail",
        str(recommendation_result),
        "The service prepares suggestion rows so normal recommendation reads stay fast.",
    )

    recs, rec_error = _recommendations("similar")
    rec_count = len((recs or {}).get("books", []))
    filter_summary = (recs or {}).get("filter_summary", {})
    log(
        9,
        "Frontend",
        "Recommendation",
        "GET /recommendations",
        "REST instant read",
        "pass" if rec_error is None and rec_count > 0 else "fail",
        (
            f"{rec_count} shown, {filter_summary.get('owned_filtered_count', 0)} owned filtered"
            if rec_error is None
            else rec_error or "-"
        ),
        "This read does not call the LLM; it returns suggestions prepared by the background service.",
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
    catalog_count = snapshot.get("catalog_count", 0)
    reading_count = snapshot.get("reading_count", 0)
    rec_total = sum(snapshot.get("rec_counts", {}).values())
    embedding_pending = pending_by_subscriber.get("embedding-worker", 0)
    recommendation_pending = pending_by_subscriber.get("recommendation-service", 0)
    llm_model = snapshot.get("models", {}).get("ollama_generate_model", "-")
    embedding_worker = _worker_summary(snapshot.get("embedding_status", {}))
    llm_provider = snapshot.get("models", {}).get("provider", "-")

    def node_status(name: str) -> str:
        return "online" if status.get(name) else "offline"

    def component_card(title: str, subtitle: str, facts: list[str], kind: str = "service") -> None:
        labels = {
            "frontend": "Frontend",
            "service": "Container App service",
            "worker": "Async worker",
            "event": "Async messaging",
            "data": "Azure SaaS / data",
        }
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(f"{labels[kind]} · {subtitle}")
            for fact in facts:
                st.write(fact)

    st.markdown("**Microsoft Azure target / local Docker runtime**")
    st.caption(
        "The same boxes run locally in Docker Compose and are deployable as Azure Container Apps. "
        "Solid rows below are REST calls; event rows are asynchronous messages."
    )

    with st.container(border=True):
        st.markdown("**Azure Container Apps boundary**")
        st.caption("Independently deployable containers; only frontend and recommendation are public in Azure.")

        row = st.columns(5)
        with row[0]:
            component_card(
                "Frontend - Streamlit",
                node_status("Frontend"),
                [f"user: `{_user_id()}`", "calls backend over REST"],
                "frontend",
            )
        with row[1]:
            component_card(
                "User Profile",
                node_status("User Profile"),
                [f"{reading_count} owned books", "accounts + reading list"],
            )
        with row[2]:
            component_card(
                "Book Catalog",
                node_status("Book Catalog"),
                [f"{catalog_count} catalog books", "metadata + Open Library lookup"],
            )
        with row[3]:
            component_card(
                "Recommendation",
                node_status("Recommendation"),
                [f"{rec_total} visible suggestions", "similar / widen / mood"],
            )
        with row[4]:
            component_card(
                "LLM Service",
                node_status("LLM Service"),
                [f"provider: `{llm_provider}`", f"model: `{llm_model}`"],
            )

        row = st.columns([1.2, 1.2, 1, 1])
        with row[0]:
            component_card(
                "Event Bus",
                "books + users topics",
                [
                    f"{embedding_pending + recommendation_pending} pending deliveries",
                    "BookCreated / BookEmbedded / UserBookAdded",
                ],
                "event",
            )
        with row[1]:
            component_card(
                "Embedding Worker",
                node_status("Embedding Worker"),
                [
                    f"last duration: `{embedding_worker['last_duration_ms'] or '-'} ms`",
                    "writes vectors after events",
                ],
                "worker",
            )
        with row[2]:
            component_card(
                "PostgreSQL + pgvector",
                "managed data layer",
                ["users, books, vectors", "recommendations, events"],
                "data",
            )
        with row[3]:
            component_card(
                "External AI / APIs",
                "adapter boundary",
                ["Open Library metadata", "Ollama, Azure OpenAI, or deterministic LLM"],
                "data",
            )

    st.markdown("**Runtime flow**")
    st.dataframe(
        [
            {
                "from": "Frontend",
                "to": "User Profile",
                "channel": "REST",
                "what happens": "sign in, create account, profile, reading list",
            },
            {
                "from": "User Profile",
                "to": "Book Catalog",
                "channel": "REST",
                "what happens": "deduplicate or create book metadata",
            },
            {
                "from": "Book Catalog",
                "to": "Event Bus",
                "channel": "async event",
                "what happens": "BookCreated is published",
            },
            {
                "from": "User Profile",
                "to": "Event Bus",
                "channel": "async event",
                "what happens": "UserBookAdded is published",
            },
            {
                "from": "Event Bus",
                "to": "Embedding Worker",
                "channel": "async event",
                "what happens": "worker consumes BookCreated after the user response",
            },
            {
                "from": "Embedding Worker",
                "to": "LLM Service",
                "channel": "REST",
                "what happens": "POST /v1/embed generates an embedding",
            },
            {
                "from": "Embedding Worker",
                "to": "Event Bus",
                "channel": "async event",
                "what happens": "BookEmbedded is published after vector write",
            },
            {
                "from": "Event Bus",
                "to": "Recommendation",
                "channel": "async event",
                "what happens": "recommendations are recomputed in the background",
            },
            {
                "from": "Frontend",
                "to": "Recommendation",
                "channel": "REST",
                "what happens": "fast suggestions read, or explicit Ask AI generation",
            },
        ],
        hide_index=True,
        use_container_width=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Catalog", catalog_count)
    metric_cols[0].caption("Books known to Book Catalog and available as candidates.")
    metric_cols[1].metric("Reading list", reading_count)
    metric_cols[1].caption("Books owned by the signed-in user; hidden from suggestions.")
    metric_cols[2].metric("Prepared suggestions", rec_total)
    metric_cols[2].caption("Visible suggestions across similar, widen, and mood.")
    metric_cols[3].metric("Event backlog", embedding_pending + recommendation_pending)
    metric_cols[3].caption("Pending async deliveries for the workers.")

    st.caption(
        "Read this from top to bottom: Streamlit calls FastAPI services over REST. "
        "Book/user changes publish async events. Workers consume those events, write PostgreSQL/pgvector state, "
        "and Recommendation serves prepared suggestions. The LLM is only called by explicit commands or async workers, not by instant reads."
    )


def _render_demo_trace(demo_result: dict) -> None:
    if demo_result.get("ok"):
        st.success("Demo scenario completed end-to-end.")
    else:
        st.error("Demo scenario stopped before completion.")

    rows = []
    for step in demo_result.get("steps", []):
        rows.append(
            {
                "#": step.get("#", "-"),
                "actor -> target": f"{step.get('actor', '-') } -> {step.get('target', '-')}",
                "operation": step.get("operation", step.get("step", "-")),
                "channel": step.get("channel", "-"),
                "status": step.get("status", "-"),
                "response": step.get("response", step.get("detail", "-")),
                "why it matters": step.get("why it matters", step.get("step", "-")),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)

    recommendations = (demo_result.get("recommendations") or {}).get("books", [])
    if recommendations:
        st.caption(
            "Demo produced prepared suggestions: "
            + ", ".join(book.get("title", "Untitled") for book in recommendations[:5])
        )


def _render_login() -> None:
    _inject_presentation_css()
    st.markdown(
        """
        <div class="demo-hero">
          <h3>Book AI Library</h3>
          <p>
            Sign in to continue with a saved library, or create a local account for the demo.
            User profiles and reading lists are persisted in PostgreSQL when Docker Compose is running.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    auth_mode = st.radio("Account action", ["Sign in", "Create account"], horizontal=True)

    if auth_mode == "Sign in":
        with st.form("signin-form"):
            email = st.text_input("Email", value="demo@example.edu", key="signin-email")
            password = st.text_input("Password", type="password", key="signin-password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            profile, error = _signin(email.strip(), password)
            if error:
                st.error(error)
                return
            st.session_state["active_user_id"] = profile["id"]
            st.session_state["logged_in_user"] = profile["id"]
            st.session_state["profile"] = profile
            st.rerun()

    else:
        with st.form("signup-form"):
            display_name = st.text_input("Display name", value="Demo User")
            email = st.text_input("Email", value="demo@example.edu", key="signup-email")
            password = st.text_input("Password", type="password", key="signup-password")
            mood = st.selectbox("Current reading mood", ["curious", "adventurous", "reflective", "comfort", "dark"])
            genres_text = st.text_input("Favourite genres", value="science fiction, philosophy")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            genres = [item.strip() for item in genres_text.split(",") if item.strip()]
            profile, error = _signup(email.strip(), password, display_name.strip() or email.strip(), mood, genres)
            if error:
                st.error(error)
                return
            st.session_state["active_user_id"] = profile["id"]
            st.session_state["logged_in_user"] = profile["id"]
            st.session_state["profile"] = profile
            st.rerun()


if "logged_in_user" not in st.session_state:
    _render_login()
    st.stop()

st.sidebar.header("Account")
profile, profile_error = _profile()
if profile_error:
    st.sidebar.error(profile_error)
else:
    st.session_state["profile"] = profile
    st.sidebar.write(profile.get("display_name", _user_id()) if profile else _user_id())
    st.sidebar.caption(_user_id())
if st.sidebar.button("Sign out"):
    st.session_state.pop("logged_in_user", None)
    st.rerun()


tab_home, tab_explore, tab_discover, tab_add, tab_list, tab_recs, tab_flow = st.tabs(
    ["Home", "Explore", "Discover", "Add book", "Reading list", "Recommendations", "Architecture"]
)

with tab_home:
    _inject_presentation_css()
    st.markdown(
        """
        <div class="demo-hero">
          <h3>Your reading profile</h3>
          <p>
            The profile text is generated on request through the LLM Service from your persisted library and current prepared recommendation candidates.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    profile = st.session_state.get("profile") or {}
    preferences = profile.get("preferences") or {}
    home_cols = st.columns([1, 1, 1])
    home_cols[0].metric("User", profile.get("display_name", _user_id()))
    books, list_error = _reading_list()
    home_cols[1].metric("Books saved", len(books))
    home_cols[2].metric("Mood", preferences.get("mood", "-"))
    saved_summary = preferences.get("profile_summary")
    if saved_summary:
        st.write(saved_summary.get("text", ""))
        st.caption(
            f"Saved summary · provider: {saved_summary.get('provider', '-')}"
            + (f" · generated: {saved_summary.get('generated_at')}" if saved_summary.get("generated_at") else "")
        )
    else:
        st.info("No saved AI profile summary yet. The page loaded from PostgreSQL only; press the button below to call the LLM.")

    if st.button("Regenerate profile summary with LLM", type="primary"):
        summary_payload, summary_error = _profile_summary()
        if summary_error:
            st.warning(summary_error)
        elif summary_payload:
            next_profile, save_error = _save_profile(
                _user_id(),
                profile.get("email", f"{_user_id()}@example.edu"),
                profile.get("display_name", _user_id()),
                preferences.get("mood", "curious"),
                preferences.get("genres", []),
                extra_preferences={
                    key: value
                    for key, value in preferences.items()
                    if key not in {"mood", "genres", "profile_summary"}
                }
                | {
                    "profile_summary": {
                        "text": summary_payload.get("summary", ""),
                        "provider": summary_payload.get("provider", "-"),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
            if save_error:
                st.error(save_error)
            else:
                st.session_state["profile"] = next_profile
                st.rerun()
    if list_error:
        st.error(list_error)
    elif not books:
        st.info("Your library is empty. Use Explore or Add book to save the first title, then run async updates.")

with tab_explore:
    _inject_presentation_css()
    st.markdown(
        """
        <div class="demo-hero">
          <h3>Explore reading shelves</h3>
          <p>
            Curated demo shelves give the catalog enough meaningful candidates for recommendations without waiting on Open Library.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    shelf_name = st.radio("Shelf", list(EXPLORE_LISTS), horizontal=True)
    for index, book in enumerate(EXPLORE_LISTS[shelf_name]):
        with st.container(border=True):
            cols = st.columns([4, 1])
            cols[0].write(f"**{book['title']}** by {book['author']}")
            cols[0].caption(", ".join(book["genres"]))
            cols[0].write(book["description"])
            if cols[1].button("Add", key=f"explore-{shelf_name}-{index}", use_container_width=True):
                added_book, error = _add_book_to_reading_list(book, rating=4)
                if added_book:
                    st.success(f"Added {added_book['title']}. Run async updates to refresh recommendations.")
                else:
                    st.error(error)

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
        result, error = _seed_demo_catalog()
        if result:
            if result.get("warning"):
                st.warning(result["warning"])
            st.success(
                f"Demo catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                f"{result['total_catalog_size']} books total."
            )
        else:
            st.error(error)

    if open_seed_clicked:
        result, error = _json_or_none(
            "POST",
            f"{BOOK_CATALOG_URL}/catalog/seed/openlibrary",
            json={"queries": ["science fiction", "fantasy", "mystery", "historical fiction"], "limit_per_query": 8},
            timeout=30,
        )
        if isinstance(result, dict):
            st.success(
                f"Catalog seeded: {result.get('imported', 0)} imported, {result.get('existing', 0)} already present, "
                f"{result.get('total_catalog_size', '-')} books total."
            )
        else:
            result, fallback_error = _seed_demo_catalog()
            if result:
                st.warning("Open Library is unavailable or slow. Seeded the local demo catalog instead.")
                if result.get("warning"):
                    st.caption(result["warning"])
                st.success(
                    f"Demo catalog seeded: {result['imported']} imported, {result['existing']} already present, "
                    f"{result['total_catalog_size']} books total."
                )
            else:
                st.error(fallback_error or error)

    if search_clicked:
        payload, error = _json_or_none(
            "GET",
            f"{BOOK_CATALOG_URL}/external/openlibrary/search",
            params={"query": query, "limit": 8},
            timeout=15,
        )
        if error:
            st.error(error)
        else:
            st.session_state["open_library_results"] = payload if isinstance(payload, list) else []

    for index, book in enumerate(st.session_state.get("open_library_results", [])):
        with st.container(border=True):
            cols = st.columns([1, 4])
            if book.get("cover_url"):
                cols[0].image(book["cover_url"], width=90)
            cols[1].write(f"**{book['title']}** by {book.get('author', 'Unknown')}")
            cols[1].caption(", ".join(book.get("genres", [])[:5]))
            if cols[1].button("Add to reading list", key=f"add-open-library-{index}"):
                added_book, error = _add_book_to_reading_list(book, rating=4)
                if added_book:
                    st.session_state["last_added_book"] = added_book["title"]
                    st.success(f"Added {st.session_state['last_added_book']}.")
                else:
                    st.error(error)

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
        added_book, error = _add_book_to_reading_list(payload, rating=rating)
        if added_book:
            st.session_state["last_added_book"] = added_book["title"]
            st.success(f"Added {st.session_state['last_added_book']}. Recommendations update asynchronously.")
        else:
            st.error(error)

with tab_list:
    books, error = _reading_list()
    if error:
        st.error(error)
    else:
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
          <h3>Book suggestions</h3>
          <p>
            First choose a fast similarity mode. Then, when you want a more personal answer,
            press Ask AI to let the LLM reason over your library and the selected suggestions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profile = st.session_state.get("profile") or {}
    preferences = profile.get("preferences") or {}
    with st.expander("Personalization", expanded=False):
        st.caption("These settings are saved to your profile. They are used when you press Ask AI.")
        recommendation_instructions = st.text_area(
            "Your reading preferences for the AI",
            value=preferences.get(
                "recommendation_instructions",
                "Prefer books with strong ideas, clear narrative, and avoid recommending books already in my library.",
            ),
            height=90,
        )
        if st.button("Save personalization"):
            next_profile, save_error = _save_profile(
                _user_id(),
                profile.get("email", f"{_user_id()}@example.edu"),
                profile.get("display_name", _user_id()),
                preferences.get("mood", "curious"),
                preferences.get("genres", []),
                extra_preferences={
                    key: value
                    for key, value in preferences.items()
                    if key not in {"mood", "genres", "recommendation_instructions"}
                }
                | {"recommendation_instructions": recommendation_instructions.strip()},
            )
            if save_error:
                st.error(save_error)
            else:
                st.session_state["profile"] = next_profile
                st.success("Personalization saved.")

    st.subheader("1. Fast suggestions")
    st.caption(
        "These suggestions are prepared by the background recommendation service after books are added. "
        "They use vector similarity and mode-specific scoring, so the page loads quickly and does not call the LLM."
    )
    controls = st.columns([1.35, 1, 1, 1])
    with controls[0]:
        rec_type = st.radio(
            "Mode",
            list(RECOMMENDATION_MODES),
            format_func=lambda key: RECOMMENDATION_MODES[key]["label"],
            horizontal=True,
        )
    with controls[1]:
        display_limit = st.slider("Books to show", min_value=1, max_value=10, value=5)
    with controls[2]:
        refresh_clicked = st.button("Refresh", use_container_width=True)
    with controls[3]:
        process_clicked = st.button("Update suggestions", use_container_width=True)

    if rec_type is None:
        rec_type = "similar"

    if process_clicked:
        st.session_state["last_pipeline_result"] = _process_async_once()
    if refresh_clicked:
        st.rerun()

    if "last_pipeline_result" in st.session_state:
        st.caption(f"Last background update: {st.session_state['last_pipeline_result']}")

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
              <div class="mode-caption">{summary.get("owned_filtered_count", 0)} already in your library hidden</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("2. Ask AI to refine the suggestions")
    st.caption(
        "This is the only action on this page that calls the LLM. The AI sees your library, saved personalization, "
        "your one-time request below, and the selected similarity suggestions."
    )
    ask_cols = st.columns([3, 1])
    with ask_cols[0]:
        ai_prompt = st.text_area(
            "One-time request for this answer",
            value="I want something thoughtful but not too long, with strong ideas and a clear story.",
            height=90,
        )
        allow_outside = st.checkbox(
            "Allow the AI to suggest outside the vector candidate list",
            value=True,
            help="Vector similarity grounds the prompt, but the LLM may propose a better outside book if it explains why.",
        )
    with ask_cols[1]:
        st.write("")
        st.write("")
        ask_clicked = st.button("Ask AI", type="primary", use_container_width=True)
    if ask_clicked:
        st.session_state["last_ai_recommendation"] = _ask_recommendations(ai_prompt, rec_type, display_limit, allow_outside)
    if "last_ai_recommendation" in st.session_state:
        ai_payload, ai_error = st.session_state["last_ai_recommendation"]
        if ai_error:
            st.error(ai_error)
        elif ai_payload:
            st.write(ai_payload.get("answer", "The AI response did not include an answer field."))
            st.caption(
                f"Provider: {ai_payload.get('provider', '-')} · engine: {ai_payload.get('engine', '-')} · "
                f"outside picks allowed: {ai_payload.get('allow_outside_candidates', False)}"
            )
            if ai_payload.get("user_instructions"):
                st.caption(f"Saved personalization used: {ai_payload.get('user_instructions')}")
            with st.expander("Suggestions passed to the AI", expanded=False):
                for item in ai_payload.get("candidates", []):
                    book = item.get("book", {})
                    st.write(f"**{book.get('title', 'Untitled')}** by {book.get('author', 'Unknown')}")
                    st.caption(
                        f"score={item.get('score', 0):.2f} · {item.get('reason', '')} · "
                        + ", ".join(book.get("genres", []))
                    )

    st.divider()
    st.subheader("Selected suggestions")
    payload, error = _recommendations(rec_type)
    if error:
        st.error(error)
    else:
        books = ((payload or {}).get("books", []))[:display_limit]
        filter_summary = (payload or {}).get("filter_summary", {})
        status_cols = st.columns(3)
        status_cols[0].metric("Mode", RECOMMENDATION_MODES[rec_type]["label"])
        status_cols[1].metric("Shown", len(books))
        status_cols[2].metric("Already in library hidden", filter_summary.get("owned_filtered_count", 0))
        if payload and payload.get("computed_at"):
            st.caption(f"Last prepared at: {payload.get('computed_at')}")
        st.caption("Books already in your library are filtered by ID, ISBN/Open Library key, and normalized title plus author.")
        if not books:
            st.markdown(
                """
                <div class="empty-state">
                  <h4>No suggestions for this mode yet</h4>
                  <p>
                    Add a book from Explore or Discover, then press Update suggestions.
                    The recommendation service prepares suggestions in the background.
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
                explanation = (payload or {}).get("explanations", {}).get(
                    book["id"],
                    "Recommended from your reading profile.",
                )
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
                    added_book, error = _add_book_to_reading_list(add_payload, rating=4)
                    if added_book:
                        st.success(f"Added {added_book['title']} to {_user_id()}.")
                        st.rerun()
                    else:
                        st.error(error)

with tab_flow:
    _inject_presentation_css()
    status = _health()
    snapshot = _flow_snapshot()
    st.subheader("Live microservice architecture")
    st.caption(
        "This view mirrors the static architecture diagram with live health and event metrics. "
        "Frontend-to-service traffic is REST; embeddings and recommendation recomputation are asynchronous events. "
        "The LLM is hidden behind one LLM Service adapter so local Ollama, deterministic Azure demo mode, "
        "Azure OpenAI, or a remote llm-service URL can be swapped without changing business services."
    )
    hero_cols = st.columns([1, 1, 2])
    if hero_cols[0].button("Run demo scenario", type="primary"):
        st.session_state["last_demo_result"] = _run_demo_scenario()
        st.rerun()
    if hero_cols[1].button("Refresh flow"):
        st.rerun()
    hero_cols[2].caption(
        "Demo scenario logs the real request/event path: catalog seed, book add, async embedding, "
        "recommendation recompute, and instant suggestion read."
    )

    _render_flow(status, snapshot)

    if "last_demo_result" in st.session_state:
        st.subheader("Demo scenario trace")
        st.caption("This is a live execution log, not a scripted animation. Each row shows which microservice acted and what response it returned.")
        _render_demo_trace(st.session_state["last_demo_result"])

    st.subheader("Runtime evidence")
    st.caption("These details support the diagram: health proves each node is alive, worker state proves the async path ran, and backlog shows whether events are pending.")
    health_rows = [{"component": name, "status": "online" if ok else "offline"} for name, ok in status.items()]
    st.dataframe(health_rows, hide_index=True, use_container_width=True)

    st.subheader("LLM route")
    llm_cols = st.columns(3)
    llm_cols[0].metric("Provider", snapshot["models"].get("provider", "-"))
    llm_cols[0].caption("Which backend `llm-service` is using right now.")
    llm_cols[1].metric("Generation model", snapshot["models"].get("ollama_generate_model", "-"))
    llm_cols[1].caption("Used for profile summaries and Ask AI when Ollama is active.")
    llm_cols[2].metric("Embedding model", snapshot["models"].get("ollama_embed_model", "-"))
    llm_cols[2].caption("Used by Embedding Worker after BookCreated events.")

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
            st.write((payload or {}).get("text", "The LLM response did not include text."))
            st.caption((payload or {}).get("provider", "-"))
