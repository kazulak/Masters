from __future__ import annotations

import json
import threading
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from shared import config

_LOCK = threading.Lock()
_PG_LOCK_ID = 4204242


def _empty_state() -> dict[str, Any]:
    return {
        "users": {},
        "books": {},
        "reading_list": {},
        "book_embeddings": {},
        "recommendations": {},
        "events": [],
    }


def _database_url() -> str | None:
    return config.env("DATABASE_URL", "") or None


def _connect():
    database_url = _database_url()
    if not database_url:
        return None
    return psycopg.connect(database_url)


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(_empty_state(), indent=2), encoding="utf-8")


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _parse_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(item) for item in text.split(",")]


def _ensure_postgres(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id text PRIMARY KEY,
                email text UNIQUE NOT NULL,
                display_name text NOT NULL,
                preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id text PRIMARY KEY,
                isbn text,
                title text NOT NULL,
                author text NOT NULL,
                description text NOT NULL DEFAULT '',
                genres text[] NOT NULL DEFAULT '{}',
                published_year integer,
                created_at timestamptz NOT NULL DEFAULT now(),
                source text NOT NULL DEFAULT 'manual',
                openlibrary_key text,
                cover_url text,
                dedupe_key text UNIQUE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reading_list (
                id text PRIMARY KEY,
                user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                book_id text NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                read_at timestamptz NOT NULL DEFAULT now(),
                rating smallint CHECK (rating BETWEEN 1 AND 5),
                UNIQUE (user_id, book_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS book_embeddings (
                id text PRIMARY KEY,
                book_id text NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                embedding vector NOT NULL,
                model_version text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE (book_id, model_version)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                id text PRIMARY KEY,
                user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                type text NOT NULL CHECK (type IN ('similar', 'widen', 'mood')),
                book_ids text[] NOT NULL DEFAULT '{}',
                explanations jsonb NOT NULL DEFAULT '{}'::jsonb,
                computed_at timestamptz,
                UNIQUE (user_id, type)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id text PRIMARY KEY,
                topic text NOT NULL,
                type text NOT NULL,
                payload jsonb NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS event_deliveries (
                event_id text NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                subscriber text NOT NULL,
                delivered_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (event_id, subscriber)
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_topic_type ON events(topic, type, created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reading_list_user ON reading_list(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_book_embeddings_book ON book_embeddings(book_id)")


def _lock_postgres(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (_PG_LOCK_ID,))


def _read_postgres(conn) -> dict[str, Any]:
    _lock_postgres(conn)
    _ensure_postgres(conn)
    state = _empty_state()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, email, display_name, preferences, created_at FROM users ORDER BY created_at")
        for row in cursor.fetchall():
            state["users"][row[0]] = {
                "id": row[0],
                "email": row[1],
                "display_name": row[2],
                "preferences": row[3],
                "created_at": row[4].isoformat(),
            }

        cursor.execute(
            """
            SELECT id, isbn, title, author, description, genres, published_year, created_at,
                   source, openlibrary_key, cover_url, dedupe_key
            FROM books
            ORDER BY created_at
            """
        )
        for row in cursor.fetchall():
            state["books"][row[0]] = {
                "id": row[0],
                "isbn": row[1],
                "title": row[2],
                "author": row[3],
                "description": row[4],
                "genres": row[5] or [],
                "published_year": row[6],
                "created_at": row[7].isoformat(),
                "source": row[8],
                "openlibrary_key": row[9],
                "cover_url": row[10],
                "dedupe_key": row[11],
            }

        cursor.execute("SELECT id, user_id, book_id, read_at, rating FROM reading_list ORDER BY read_at")
        for row in cursor.fetchall():
            state["reading_list"][row[0]] = {
                "id": row[0],
                "user_id": row[1],
                "book_id": row[2],
                "read_at": row[3].isoformat(),
                "rating": row[4],
            }

        cursor.execute("SELECT id, book_id, embedding::text, model_version, created_at FROM book_embeddings ORDER BY created_at")
        for row in cursor.fetchall():
            state["book_embeddings"][row[0]] = {
                "id": row[0],
                "book_id": row[1],
                "embedding": _parse_vector(row[2]),
                "model_version": row[3],
                "created_at": row[4].isoformat(),
            }

        cursor.execute("SELECT id, user_id, type, book_ids, explanations, computed_at FROM recommendations")
        for row in cursor.fetchall():
            state["recommendations"][row[0]] = {
                "id": row[0],
                "user_id": row[1],
                "type": row[2],
                "book_ids": row[3] or [],
                "explanations": row[4],
                "computed_at": row[5].isoformat() if row[5] else None,
            }

        cursor.execute(
            """
            SELECT e.id, e.topic, e.type, e.payload, e.created_at,
                   COALESCE(array_agg(d.subscriber) FILTER (WHERE d.subscriber IS NOT NULL), '{}')
            FROM events e
            LEFT JOIN event_deliveries d ON d.event_id = e.id
            GROUP BY e.id
            ORDER BY e.created_at
            """
        )
        for row in cursor.fetchall():
            state["events"].append(
                {
                    "id": row[0],
                    "topic": row[1],
                    "type": row[2],
                    "payload": row[3],
                    "created_at": row[4].isoformat(),
                    "delivered_to": row[5] or [],
                }
            )
    return state


def _write_postgres(conn, state: dict[str, Any]) -> None:
    _lock_postgres(conn)
    _ensure_postgres(conn)
    with conn.cursor() as cursor:
        cursor.execute(
            """
            TRUNCATE event_deliveries, events, recommendations, book_embeddings,
                     reading_list, books, users
            RESTART IDENTITY CASCADE
            """
        )
        for user in state["users"].values():
            cursor.execute(
                """
                INSERT INTO users (id, email, display_name, preferences, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user["id"],
                    user.get("email", f"{user['id']}@example.edu"),
                    user.get("display_name", user["id"]),
                    Jsonb(user.get("preferences", {})),
                    user.get("created_at"),
                ),
            )

        for book in state["books"].values():
            cursor.execute(
                """
                INSERT INTO books (
                    id, isbn, title, author, description, genres, published_year, created_at,
                    source, openlibrary_key, cover_url, dedupe_key
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    book["id"],
                    book.get("isbn"),
                    book["title"],
                    book.get("author", "Unknown"),
                    book.get("description", ""),
                    book.get("genres", []),
                    book.get("published_year"),
                    book.get("created_at"),
                    book.get("source", "manual"),
                    book.get("openlibrary_key"),
                    book.get("cover_url"),
                    book.get("dedupe_key"),
                ),
            )

        for row in state["reading_list"].values():
            if row["user_id"] not in state["users"] or row["book_id"] not in state["books"]:
                continue
            cursor.execute(
                """
                INSERT INTO reading_list (id, user_id, book_id, read_at, rating)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, book_id) DO UPDATE
                SET rating = EXCLUDED.rating, read_at = EXCLUDED.read_at
                """,
                (row["id"], row["user_id"], row["book_id"], row.get("read_at"), row.get("rating")),
            )

        for row in state["book_embeddings"].values():
            if row["book_id"] not in state["books"]:
                continue
            cursor.execute(
                """
                INSERT INTO book_embeddings (id, book_id, embedding, model_version, created_at)
                VALUES (%s, %s, %s::vector, %s, %s)
                ON CONFLICT (book_id, model_version) DO UPDATE
                SET embedding = EXCLUDED.embedding, created_at = EXCLUDED.created_at
                """,
                (
                    row["id"],
                    row["book_id"],
                    _vector_literal(row["embedding"]),
                    row["model_version"],
                    row.get("created_at"),
                ),
            )

        for row in state["recommendations"].values():
            if row["user_id"] not in state["users"]:
                continue
            cursor.execute(
                """
                INSERT INTO recommendations (id, user_id, type, book_ids, explanations, computed_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, type) DO UPDATE
                SET book_ids = EXCLUDED.book_ids,
                    explanations = EXCLUDED.explanations,
                    computed_at = EXCLUDED.computed_at
                """,
                (
                    row["id"],
                    row["user_id"],
                    row["type"],
                    row.get("book_ids", []),
                    Jsonb(row.get("explanations", {})),
                    row.get("computed_at"),
                ),
            )

        for event in state["events"]:
            cursor.execute(
                """
                INSERT INTO events (id, topic, type, payload, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (event["id"], event["topic"], event["type"], Jsonb(event["payload"]), event.get("created_at")),
            )
            for subscriber in event.get("delivered_to", []):
                cursor.execute(
                    """
                    INSERT INTO event_deliveries (event_id, subscriber)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (event["id"], subscriber),
                )


def read_state() -> dict[str, Any]:
    conn = _connect()
    if conn is not None:
        with _LOCK, conn:
            return _read_postgres(conn)

    path = config.state_file()
    with _LOCK:
        _ensure_file(path)
        return json.loads(path.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    conn = _connect()
    if conn is not None:
        with _LOCK, conn:
            _write_postgres(conn, state)
        return

    path = config.state_file()
    with _LOCK:
        _ensure_file(path)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def update_state(mutator: Callable[[dict[str, Any]], Any]) -> Any:
    conn = _connect()
    if conn is not None:
        with _LOCK, conn:
            state = _read_postgres(conn)
            result = mutator(state)
            _write_postgres(conn, state)
            return deepcopy(result)

    path = config.state_file()
    with _LOCK:
        _ensure_file(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        result = mutator(state)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        return deepcopy(result)


def reset_state() -> None:
    write_state(_empty_state())
