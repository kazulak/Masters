from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, Query

from shared.config import DEFAULT_USER_ID
from shared.events import pull
from shared.storage import read_state, update_state
from shared.text import average, cosine

RECOMMENDATION_TYPES = ("similar", "widen", "mood")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _book_embeddings(state: dict) -> dict[str, list[float]]:
    result = {}
    for row in state["book_embeddings"].values():
        result[row["book_id"]] = row["embedding"]
    return result


def _score_candidates(state: dict, user_id: str, rec_type: str) -> list[dict]:
    embeddings = _book_embeddings(state)
    user_rows = [row for row in state["reading_list"].values() if row["user_id"] == user_id]
    owned_ids = {row["book_id"] for row in user_rows}
    seed_vectors = [embeddings[book_id] for book_id in owned_ids if book_id in embeddings]
    if not seed_vectors:
        return []

    profile = average(seed_vectors)
    read_genres = {
        genre
        for book_id in owned_ids
        for genre in state["books"].get(book_id, {}).get("genres", [])
    }
    mood = state["users"].get(user_id, {}).get("preferences", {}).get("mood", "curious")

    scored = []
    for book_id, vector in embeddings.items():
        if book_id in owned_ids or book_id not in state["books"]:
            continue
        book = state["books"][book_id]
        score = cosine(profile, vector)
        genres = set(book.get("genres", []))
        if rec_type == "widen" and genres and not genres.intersection(read_genres):
            score += 0.18
        if rec_type == "mood" and mood.lower() in " ".join(genres).lower():
            score += 0.12
        scored.append({"book": book, "score": score})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:10]


def recompute_user(user_id: str) -> dict[str, int]:
    state = read_state()
    computed = 0
    for rec_type in RECOMMENDATION_TYPES:
        candidates = _score_candidates(state, user_id, rec_type)
        book_ids = [item["book"]["id"] for item in candidates]
        explanations = {
            item["book"]["id"]: f"{item['book']['title']} matches your {rec_type} recommendation profile."
            for item in candidates
        }

        def mutate(state: dict, rec_type: str = rec_type) -> dict:
            row_id = f"{user_id}:{rec_type}"
            row = {
                "id": row_id,
                "user_id": user_id,
                "type": rec_type,
                "book_ids": book_ids,
                "explanations": explanations,
                "computed_at": _now(),
            }
            state["recommendations"][row_id] = row
            return row

        update_state(mutate)
        computed += 1
    return {"computed": computed}


def process_once(limit: int = 10) -> dict[str, int]:
    events = []
    events.extend(pull("books", "recommendation-service", {"BookEmbedded"}, limit=limit))
    events.extend(pull("users", "recommendation-service", {"UserBookAdded"}, limit=limit))

    state = read_state()
    affected_users = set()
    for event in events:
        if event["type"] == "UserBookAdded":
            affected_users.add(event["payload"]["user_id"])
        elif event["type"] == "BookEmbedded":
            affected_users.update(row["user_id"] for row in state["reading_list"].values())

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
    state = read_state()
    row = state["recommendations"].get(f"{user_id}:{type}") or {
        "user_id": user_id,
        "type": type,
        "book_ids": [],
        "explanations": {},
        "computed_at": None,
    }
    books = [state["books"][book_id] for book_id in row["book_ids"] if book_id in state["books"]]
    return {
        "user_id": user_id,
        "type": type,
        "books": books,
        "explanations": row["explanations"],
        "computed_at": row["computed_at"],
    }
