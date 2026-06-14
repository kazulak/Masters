from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI, Query

from shared.config import DEFAULT_USER_ID
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


def recompute_user(user_id: str) -> dict[str, int]:
    computed = 0
    for rec_type in RECOMMENDATION_TYPES:
        candidates = _score_candidates(user_id, rec_type)
        book_ids = [item["book"]["id"] for item in candidates]
        explanations = {
            item["book"]["id"]: (
                f"{item['book']['title']} is a {rec_type} pick: {item['reason']}. "
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
    row = get_user_recommendation(user_id, type) or {
        "user_id": user_id,
        "type": type,
        "book_ids": [],
        "explanations": {},
        "computed_at": None,
    }
    books, explanations, filter_summary = _filter_owned_books(user_id, row)
    return {
        "user_id": user_id,
        "type": type,
        "books": books,
        "explanations": explanations,
        "computed_at": row["computed_at"],
        "filter_summary": filter_summary,
    }
