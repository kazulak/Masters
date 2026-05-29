from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from shared.config import DEFAULT_USER_ID
from shared.events import pull
from shared.repositories import (
    books_by_ids,
    candidate_books_by_vector,
    get_user,
    get_user_recommendation,
    list_user_books,
    list_user_ids_with_reading_list,
    upsert_recommendation,
)

RECOMMENDATION_TYPES = ("similar", "widen", "mood")


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
        score = item["score"]
        genres = set(book.get("genres", []))
        if rec_type == "widen" and genres and not genres.intersection(read_genres):
            score += 0.18
        if rec_type == "mood" and mood.lower() in " ".join(genres).lower():
            score += 0.12
        scored.append({"book": book, "score": score})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:10]


def recompute_user(user_id: str) -> dict[str, int]:
    computed = 0
    for rec_type in RECOMMENDATION_TYPES:
        candidates = _score_candidates(user_id, rec_type)
        book_ids = [item["book"]["id"] for item in candidates]
        explanations = {
            item["book"]["id"]: f"{item['book']['title']} matches your {rec_type} recommendation profile."
            for item in candidates
        }
        upsert_recommendation(user_id, rec_type, book_ids, explanations)
        computed += 1
    return {"computed": computed}


def process_once(limit: int = 10) -> dict[str, int]:
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


async def _worker_loop() -> None:
    while True:
        try:
            process_once()
        except Exception:
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
    books = books_by_ids(row["book_ids"])
    return {
        "user_id": user_id,
        "type": type,
        "books": books,
        "explanations": row["explanations"],
        "computed_at": row["computed_at"],
    }
