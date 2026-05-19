from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from uuid import uuid4

import requests
from fastapi import FastAPI

from shared.config import LLM_SERVICE_URL
from shared.events import publish, pull
from shared.storage import read_state, update_state
from shared.text import book_text

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_once(limit: int = 10) -> dict[str, int]:
    events = pull("books", "embedding-worker", {"BookCreated"}, limit=limit)
    processed = 0
    skipped = 0

    for event in events:
        book_id = event["payload"]["book_id"]
        state = read_state()
        book = state["books"].get(book_id)
        if not book:
            skipped += 1
            continue

        response = requests.post(
            f"{LLM_SERVICE_URL}/v1/embed",
            json={"text": book_text(book)},
            timeout=15,
        )
        response.raise_for_status()
        embedding_payload = response.json()

        def mutate(state: dict) -> dict:
            for existing in state["book_embeddings"].values():
                if (
                    existing["book_id"] == book_id
                    and existing["model_version"] == embedding_payload["model_version"]
                ):
                    return existing
            row_id = str(uuid4())
            row = {
                "id": row_id,
                "book_id": book_id,
                "embedding": embedding_payload["embedding"],
                "model_version": embedding_payload["model_version"],
                "created_at": _now(),
            }
            state["book_embeddings"][row_id] = row
            return row

        row = update_state(mutate)
        publish("books", "BookEmbedded", {"book_id": book_id, "embedding_id": row["id"]})
        processed += 1

    return {"processed": processed, "skipped": skipped}


async def _worker_loop() -> None:
    while True:
        try:
            process_once()
        except Exception:
            # The endpoint /work exposes failures during local verification.
            pass
        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(_worker_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Book AI Library - Embedding Worker", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "embedding-worker"}


@app.post("/work")
def work() -> dict[str, int]:
    return process_once()
