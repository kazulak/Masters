from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter

import requests
from fastapi import FastAPI

from shared.config import LLM_SERVICE_URL
from shared.events import publish, pull
from shared.repositories import event_backlog_summary, get_book, upsert_book_embedding
from shared.text import book_text

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


def _process_once(limit: int = 10) -> dict[str, int]:
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
            print(f"embedding-worker background pass failed: {exc}", flush=True)
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


@app.get("/status")
def status() -> dict:
    backlog = [
        row
        for row in event_backlog_summary()
        if row["subscriber"] == "embedding-worker"
    ]
    return {
        "status": "ok",
        "service": "embedding-worker",
        "worker": WORKER_STATE,
        "event_backlog": backlog,
    }


@app.post("/work")
def work() -> dict[str, int]:
    return process_once()
