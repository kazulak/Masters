from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from shared.config import DEFAULT_USER_ID, LLM_SERVICE_URL
from shared.events import pull
from shared.repositories import (
    books_by_ids,
    candidate_books_by_vector,
    event_backlog_summary,
    get_user,
    get_user_recommendation,
    list_user_books,
    list_user_ids_with_reading_list,
    upsert_recommendation,
)
from shared.text import book_identity_keys

RECOMMENDATION_TYPES = ("similar", "widen", "mood")
MOOD_KEYWORDS = {
    "curious": {"science", "history", "philosophy", "mystery", "literary"},
    "adventurous": {"adventure", "fantasy", "quest", "exploration", "travel"},
    "reflective": {"literary", "philosophy", "memoir", "history", "classic"},
    "comfort": {"fantasy", "humor", "romance", "cozy", "children"},
    "dark": {"horror", "dystopian", "crime", "mystery", "thriller"},
}
WORKER_STATE = {
    "last_success_at": None,
    "last_error": None,
    "last_error_at": None,
    "last_duration_ms": None,
    "last_result": None,
    "total_runs": 0,
    "total_failures": 0,
    "consecutive_failures": 0,
}


class AskRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    prompt: str = Field(min_length=3, max_length=800)
    type: str = Field(default="similar", pattern="^(similar|widen|mood)$")
    limit: int = Field(default=5, ge=1, le=10)
    allow_outside_candidates: bool = True


class ProfileSummaryRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    limit: int = Field(default=5, ge=1, le=10)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_success(result: dict[str, int], duration_ms: int) -> None:
    WORKER_STATE.update(
        {
            "last_success_at": _utc_now(),
            "last_error": None,
            "last_error_at": None,
            "last_duration_ms": duration_ms,
            "last_result": result,
            "total_runs": WORKER_STATE["total_runs"] + 1,
            "consecutive_failures": 0,
        }
    )


def _record_failure(exc: Exception, duration_ms: int) -> None:
    WORKER_STATE.update(
        {
            "last_error": f"{type(exc).__name__}: {exc}",
            "last_error_at": _utc_now(),
            "last_duration_ms": duration_ms,
            "total_runs": WORKER_STATE["total_runs"] + 1,
            "total_failures": WORKER_STATE["total_failures"] + 1,
            "consecutive_failures": WORKER_STATE["consecutive_failures"] + 1,
        }
    )


def _book_terms(book: dict) -> set[str]:
    text = " ".join(
        [
            str(book.get("title", "")),
            str(book.get("author", "")),
            str(book.get("description", "")),
            " ".join(book.get("genres", [])),
        ]
    ).lower()
    return {
        token.strip(".,:;!?()[]{}\"'")
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}\"'")) >= 4
    }


def _score_for_mode(base_score: float, book: dict, rec_type: str, read_genres: set[str], mood: str) -> tuple[float, str]:
    genres = {genre.lower() for genre in book.get("genres", [])}
    overlap = genres.intersection({genre.lower() for genre in read_genres})
    new_genres = genres.difference({genre.lower() for genre in read_genres})

    if rec_type == "similar":
        score = base_score + (0.08 * min(len(overlap), 3))
        reason = "closest vector match"
        if overlap:
            reason = f"shares {', '.join(sorted(overlap)[:2])}"
        return score, reason

    if rec_type == "widen":
        novelty_bonus = 0.22 if new_genres else 0
        overlap_penalty = 0.14 * min(len(overlap), 2)
        score = (base_score * 0.55) + novelty_bonus - overlap_penalty + (0.03 * min(len(new_genres), 3))
        reason = "adds a new direction"
        if new_genres:
            reason = f"adds {', '.join(sorted(new_genres)[:2])}"
        return score, reason

    mood_terms = MOOD_KEYWORDS.get(mood.lower(), MOOD_KEYWORDS["curious"])
    book_terms = _book_terms(book)
    mood_hits = mood_terms.intersection(book_terms).union(mood_terms.intersection(genres))
    score = (base_score * 0.60) + (0.16 * min(len(mood_hits), 3))
    reason = f"fits a {mood} mood"
    if mood_hits:
        reason = f"matches {mood} via {', '.join(sorted(mood_hits)[:2])}"
    return score, reason


def _score_candidates(user_id: str, rec_type: str) -> list[dict]:
    owned_books = list_user_books(user_id)
    read_genres = {
        genre
        for book in owned_books
        for genre in book.get("genres", [])
    }
    mood = (get_user(user_id) or {}).get("preferences", {}).get("mood", "curious")

    scored = []
    for item in candidate_books_by_vector(user_id, limit=50):
        book = item["book"]
        score, reason = _score_for_mode(item["score"], book, rec_type, read_genres, mood)
        scored.append({"book": book, "score": score, "reason": reason})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:10]


class RecommendationEngine:
    name = "base"

    def recommend(self, user_id: str, rec_type: str, limit: int = 10) -> list[dict]:
        raise NotImplementedError


class VectorSimilarityEngine(RecommendationEngine):
    name = "vector-similarity"

    def recommend(self, user_id: str, rec_type: str, limit: int = 10) -> list[dict]:
        return _score_candidates(user_id, rec_type)[:limit]


def _engine() -> RecommendationEngine:
    engine_name = os.getenv("RECOMMENDATION_ENGINE", "vector-similarity")
    if engine_name == "vector-similarity":
        return VectorSimilarityEngine()
    raise HTTPException(status_code=500, detail=f"Unknown recommendation engine: {engine_name}")


def recompute_user(user_id: str) -> dict[str, int]:
    computed = 0
    engine = _engine()
    for rec_type in RECOMMENDATION_TYPES:
        candidates = engine.recommend(user_id, rec_type, limit=10)
        book_ids = [item["book"]["id"] for item in candidates]
        explanations = {
            item["book"]["id"]: (
                f"{item['book']['title']} is a {rec_type} pick from {engine.name}: {item['reason']}. "
                f"Score {item['score']:.2f}; generated asynchronously from cached embeddings."
            )
            for item in candidates
        }
        upsert_recommendation(user_id, rec_type, book_ids, explanations)
        computed += 1
    return {"computed": computed}


def _filter_owned_books(user_id: str, row: dict) -> tuple[list[dict], dict[str, str], dict[str, int]]:
    owned_books = list_user_books(user_id)
    owned_ids = {book["id"] for book in owned_books}
    owned_keys = {
        key
        for book in owned_books
        for key in book_identity_keys(book)
    }
    books = books_by_ids(row["book_ids"])
    unread_books = [
        book
        for book in books
        if book["id"] not in owned_ids and not book_identity_keys(book).intersection(owned_keys)
    ]
    explanations = row.get("explanations", {})
    filtered = len(books) - len(unread_books)
    return unread_books, {
        book["id"]: explanations[book["id"]]
        for book in unread_books
        if book["id"] in explanations
    }, {
        "cached_count": len(row.get("book_ids", [])),
        "resolved_count": len(books),
        "owned_filtered_count": filtered,
    }


def _cached_recommendation_payload(user_id: str, rec_type: str) -> dict:
    row = get_user_recommendation(user_id, rec_type) or {
        "user_id": user_id,
        "type": rec_type,
        "book_ids": [],
        "explanations": {},
        "computed_at": None,
    }
    books, explanations, filter_summary = _filter_owned_books(user_id, row)
    return {
        "user_id": user_id,
        "type": rec_type,
        "books": books,
        "explanations": explanations,
        "computed_at": row["computed_at"],
        "filter_summary": filter_summary,
    }


def _fallback_candidates(user_id: str, rec_type: str, limit: int) -> list[dict]:
    payload = _cached_recommendation_payload(user_id, rec_type)
    if payload["books"]:
        return payload["books"][:limit]
    return [item["book"] for item in _engine().recommend(user_id, rec_type, limit=limit)]


def _engine_candidates(user_id: str, rec_type: str, limit: int) -> tuple[str, list[dict]]:
    engine = _engine()
    candidates = engine.recommend(user_id, rec_type, limit=max(limit, 10))
    if not candidates:
        cached = _cached_recommendation_payload(user_id, rec_type)
        explanations = cached.get("explanations", {})
        candidates = [
            {
                "book": book,
                "score": 0.0,
                "reason": explanations.get(book["id"], "cached candidate awaiting fresh engine score"),
            }
            for book in cached.get("books", [])
        ]
    return engine.name, candidates[:limit]


def _book_line(book: dict, index: int) -> str:
    genres = ", ".join(book.get("genres", [])[:4]) or "unknown genres"
    description = str(book.get("description", "")).strip()
    if len(description) > 240:
        description = description[:237].rstrip() + "..."
    return (
        f"{index}. {book.get('title', 'Untitled')} by {book.get('author', 'Unknown')} "
        f"({genres}). {description}"
    )


def _candidate_line(candidate: dict, index: int) -> str:
    book = candidate["book"]
    base = _book_line(book, index)
    return f"{base} Engine score={candidate['score']:.2f}; reason={candidate['reason']}."


def _call_llm(prompt: str, context: dict | None = None, timeout: int = 120) -> dict:
    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/v1/generate",
            json={"prompt": prompt, "context": context or {}},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM Service unavailable: {exc}") from exc


def _user_context(user_id: str) -> tuple[dict, list[dict]]:
    user = get_user(user_id) or {"id": user_id, "display_name": user_id, "preferences": {}}
    return user, list_user_books(user_id)


def _process_once(limit: int = 10) -> dict[str, int]:
    events = []
    events.extend(pull("books", "recommendation-service", {"BookEmbedded"}, limit=limit))
    events.extend(pull("users", "recommendation-service", {"UserBookAdded"}, limit=limit))

    affected_users = set()
    for event in events:
        if event["type"] == "UserBookAdded":
            affected_users.add(event["payload"]["user_id"])
        elif event["type"] == "BookEmbedded":
            affected_users.update(list_user_ids_with_reading_list())

    for user_id in affected_users:
        recompute_user(user_id)

    return {"events": len(events), "users": len(affected_users)}


def process_once(limit: int = 10) -> dict[str, int]:
    started = perf_counter()
    try:
        result = _process_once(limit)
    except Exception as exc:
        _record_failure(exc, int((perf_counter() - started) * 1000))
        raise
    _record_success(result, int((perf_counter() - started) * 1000))
    return result


async def _worker_loop() -> None:
    while True:
        try:
            process_once()
        except Exception as exc:
            print(f"recommendation background pass failed: {exc}", flush=True)
            pass
        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(_worker_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Book AI Library - Recommendation Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "recommendation"}


@app.get("/status")
def status() -> dict:
    backlog = [
        row
        for row in event_backlog_summary()
        if row["subscriber"] == "recommendation-service"
    ]
    return {
        "status": "ok",
        "service": "recommendation",
        "worker": WORKER_STATE,
        "event_backlog": backlog,
    }


@app.post("/work")
def work() -> dict[str, int]:
    return process_once()


@app.post("/recompute/{user_id}")
def recompute(user_id: str) -> dict[str, int]:
    return recompute_user(user_id)


@app.get("/recommendations")
def get_recommendations(
    user_id: str = Query(default=DEFAULT_USER_ID),
    type: str = Query(default="similar", pattern="^(similar|widen|mood)$"),
) -> dict:
    return _cached_recommendation_payload(user_id, type)


@app.post("/recommendations/ask")
def ask_for_recommendations(payload: AskRequest) -> dict:
    user, owned_books = _user_context(payload.user_id)
    preferences = user.get("preferences", {}) or {}
    user_instructions = str(preferences.get("recommendation_instructions", "")).strip()
    engine_name, candidate_rows = _engine_candidates(payload.user_id, payload.type, payload.limit)
    candidate_books = [item["book"] for item in candidate_rows]
    if not candidate_books and not owned_books:
        raise HTTPException(
            status_code=409,
            detail="No reading history or candidate recommendations are available yet. Add a book first.",
        )

    owned_lines = "\n".join(_book_line(book, index) for index, book in enumerate(owned_books[:8], start=1))
    candidate_lines = "\n".join(_candidate_line(item, index) for index, item in enumerate(candidate_rows, start=1))
    outside_policy = (
        "You may recommend a book outside the candidate list if it is clearly better for the user request. "
        "If you do, label it as an outside pick and explain why the engine candidates were not enough."
        if payload.allow_outside_candidates
        else "Only recommend books from the candidate list."
    )
    prompt = f"""
You are the recommendation reasoning layer for Book AI Library.
The recommendation system has two stages:
1. A replaceable recommendation engine proposes grounded candidate books.
2. You make the final recommendation using the user's request, profile, instructions, reading list, and candidates.

User request:
{payload.prompt}

User profile:
- id: {payload.user_id}
- display name: {user.get("display_name", payload.user_id)}
- preferences: {user.get("preferences", {})}
- persistent user instructions: {user_instructions or "None provided."}

Books already in the user's library:
{owned_lines or "No books yet."}

Candidate books from engine "{engine_name}" using mode "{payload.type}":
{candidate_lines or "No grounded candidates are available yet."}

Policy:
{outside_policy}

Pick the best 1-3 books for the user's request.
For each pick, explain briefly whether it came from the engine candidates or is an outside pick, and why it fits the user's request and reading history.
Finish with one concise next-step sentence.
""".strip()
    llm_payload = _call_llm(
        prompt,
        context={
            "title": candidate_books[0].get("title", "a strong outside recommendation") if candidate_books else "a strong outside recommendation",
            "reason": payload.prompt,
        },
    )
    return {
        "user_id": payload.user_id,
        "prompt": payload.prompt,
        "type": payload.type,
        "engine": engine_name,
        "allow_outside_candidates": payload.allow_outside_candidates,
        "user_instructions": user_instructions,
        "provider": llm_payload.get("provider", "unknown"),
        "answer": llm_payload.get("text", ""),
        "books": candidate_books,
        "candidates": candidate_rows,
        "source": "llm-over-engine-candidates",
        "hot_path": "POST /recommendations/ask intentionally calls LLM Service over engine candidates; GET /recommendations remains cache-only.",
    }


@app.post("/profile/summary")
def profile_summary(payload: ProfileSummaryRequest) -> dict:
    user, owned_books = _user_context(payload.user_id)
    candidates = _fallback_candidates(payload.user_id, "similar", payload.limit)
    owned_lines = "\n".join(_book_line(book, index) for index, book in enumerate(owned_books[:10], start=1))
    candidate_lines = "\n".join(_book_line(book, index) for index, book in enumerate(candidates[:5], start=1))
    prompt = f"""
Write a short, warm reading profile for this Book AI Library user.

User:
- id: {payload.user_id}
- display name: {user.get("display_name", payload.user_id)}
- preferences: {user.get("preferences", {})}

Library:
{owned_lines or "The library is empty."}

Possible next books:
{candidate_lines or "No computed recommendations yet."}

In 3-5 sentences: describe the user's apparent taste, mention one pattern in their library, and finish with one suggested next book from the possible next books if available.
""".strip()
    context_title = candidates[0].get("title", "the first book in your recommendation list") if candidates else "your first saved book"
    llm_payload = _call_llm(
        prompt,
        context={
            "title": context_title,
            "reason": "it matches the user's saved library",
        },
    )
    return {
        "user_id": payload.user_id,
        "provider": llm_payload.get("provider", "unknown"),
        "summary": llm_payload.get("text", ""),
        "books_considered": len(owned_books),
        "candidate_count": len(candidates),
    }
