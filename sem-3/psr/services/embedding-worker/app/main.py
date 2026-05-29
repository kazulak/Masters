from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import requests
from fastapi import FastAPI

from shared.config import LLM_SERVICE_URL
from shared.events import publish, pull
from shared.repositories import get_book, upsert_book_embedding
from shared.text import book_text


def process_once(limit: int = 10) -> dict[str, int]:
    events = pull("books", "embedding-worker", {"BookCreated"}, limit=limit)
    processed = 0
    skipped = 0

    for event in events:
        book_id = event["payload"]["book_id"]
        book = get_book(book_id)
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

        row = upsert_book_embedding(
            book_id,
            embedding_payload["embedding"],
            embedding_payload["model_version"],
        )
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
