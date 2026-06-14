from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from shared import config
from shared.storage import _ensure_postgres, _parse_vector, _vector_literal, read_state, update_state
from shared.text import average, book_identity_keys, cosine, normalize


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def database_url() -> str | None:
    return config.env("DATABASE_URL", "") or None


def is_postgres_enabled() -> bool:
    return database_url() is not None


def _connect():
    url = database_url()
    if not url:
        return None
    return psycopg.connect(url)


def _book_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "isbn": row[1],
        "title": row[2],
        "author": row[3],
        "description": row[4],
        "genres": row[5] or [],
        "published_year": row[6],
        "created_at": row[7].isoformat() if hasattr(row[7], "isoformat") else row[7],
        "source": row[8],
        "openlibrary_key": row[9],
        "cover_url": row[10],
        "dedupe_key": row[11],
    }


def _user_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "email": row[1],
        "display_name": row[2],
        "preferences": row[3] or {},
        "created_at": row[4].isoformat() if hasattr(row[4], "isoformat") else row[4],
    }


def _reading_row_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "user_id": row[1],
        "book_id": row[2],
        "read_at": row[3].isoformat() if hasattr(row[3], "isoformat") else row[3],
        "rating": row[4],
    }


def list_books(query: str = "") -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        books = list(read_state()["books"].values())
        if not query:
            return books
        needle = normalize(query)
        return [
            book
            for book in books
            if needle in normalize(book["title"]) or needle in normalize(book.get("author", ""))
        ]

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            if query:
                needle = f"%{normalize(query)}%"
                cursor.execute(
                    """
                    SELECT id, isbn, title, author, description, genres, published_year, created_at,
                           source, openlibrary_key, cover_url, dedupe_key
                    FROM books
                    WHERE lower(title) LIKE %s OR lower(author) LIKE %s
                    ORDER BY created_at
                    """,
                    (needle, needle),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, isbn, title, author, description, genres, published_year, created_at,
                           source, openlibrary_key, cover_url, dedupe_key
                    FROM books
                    ORDER BY created_at
                    """
                )
            return [_book_from_row(row) for row in cursor.fetchall()]


def count_books() -> int:
    conn = _connect()
    if conn is None:
        return len(read_state()["books"])
    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM books")
            return int(cursor.fetchone()[0])


def get_book(book_id: str) -> dict[str, Any] | None:
    conn = _connect()
    if conn is None:
        return read_state()["books"].get(book_id)
    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, isbn, title, author, description, genres, published_year, created_at,
                       source, openlibrary_key, cover_url, dedupe_key
                FROM books
                WHERE id = %s
                """,
                (book_id,),
            )
            row = cursor.fetchone()
            return _book_from_row(row) if row else None


def upsert_book(payload: dict[str, Any], dedupe_key: str) -> dict[str, Any]:
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            for existing in state["books"].values():
                if existing.get("dedupe_key") == dedupe_key:
                    return {"book": existing, "created": False}
            book = payload | {"id": str(uuid4()), "created_at": utc_now(), "dedupe_key": dedupe_key}
            state["books"][book["id"]] = book
            return {"book": book, "created": True}

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, isbn, title, author, description, genres, published_year, created_at,
                       source, openlibrary_key, cover_url, dedupe_key
                FROM books
                WHERE dedupe_key = %s
                """,
                (dedupe_key,),
            )
            existing = cursor.fetchone()
            if existing:
                return {"book": _book_from_row(existing), "created": False}

            book_id = str(uuid4())
            cursor.execute(
                """
                INSERT INTO books (
                    id, isbn, title, author, description, genres, published_year, created_at,
                    source, openlibrary_key, cover_url, dedupe_key
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, isbn, title, author, description, genres, published_year, created_at,
                          source, openlibrary_key, cover_url, dedupe_key
                """,
                (
                    book_id,
                    payload.get("isbn"),
                    payload["title"],
                    payload.get("author", "Unknown"),
                    payload.get("description", ""),
                    payload.get("genres", []),
                    payload.get("published_year"),
                    utc_now(),
                    payload.get("source", "manual"),
                    payload.get("openlibrary_key"),
                    payload.get("cover_url"),
                    dedupe_key,
                ),
            )
            return {"book": _book_from_row(cursor.fetchone()), "created": True}


def get_or_create_user(user_id: str, defaults: dict[str, Any]) -> dict[str, Any]:
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            return state["users"].setdefault(user_id, defaults | {"id": user_id, "created_at": utc_now()})

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (id, email, display_name, preferences, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    user_id,
                    defaults.get("email", f"{user_id}@example.edu"),
                    defaults.get("display_name", user_id),
                    Jsonb(defaults.get("preferences", {})),
                    utc_now(),
                ),
            )
            cursor.execute(
                "SELECT id, email, display_name, preferences, created_at FROM users WHERE id = %s",
                (user_id,),
            )
            return _user_from_row(cursor.fetchone())


def get_user(user_id: str) -> dict[str, Any] | None:
    conn = _connect()
    if conn is None:
        return read_state()["users"].get(user_id)
    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, email, display_name, preferences, created_at FROM users WHERE id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
            return _user_from_row(row) if row else None


def upsert_user(profile: dict[str, Any]) -> dict[str, Any]:
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            existing = state["users"].get(profile["id"], {})
            row = profile | {"created_at": existing.get("created_at", utc_now())}
            state["users"][profile["id"]] = row
            return row

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (id, email, display_name, preferences, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET email = EXCLUDED.email,
                    display_name = EXCLUDED.display_name,
                    preferences = EXCLUDED.preferences
                RETURNING id, email, display_name, preferences, created_at
                """,
                (
                    profile["id"],
                    profile.get("email", f"{profile['id']}@example.edu"),
                    profile.get("display_name", profile["id"]),
                    Jsonb(profile.get("preferences", {})),
                    utc_now(),
                ),
            )
            return _user_from_row(cursor.fetchone())


def upsert_reading_list_entry(user_id: str, book_id: str, rating: int | None) -> dict[str, Any]:
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            for row in state["reading_list"].values():
                if row["user_id"] == user_id and row["book_id"] == book_id:
                    row["rating"] = rating
                    return row
            row = {
                "id": str(uuid4()),
                "user_id": user_id,
                "book_id": book_id,
                "read_at": utc_now(),
                "rating": rating,
            }
            state["reading_list"][row["id"]] = row
            return row

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            row_id = str(uuid4())
            cursor.execute(
                """
                INSERT INTO reading_list (id, user_id, book_id, read_at, rating)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, book_id) DO UPDATE
                SET rating = EXCLUDED.rating
                RETURNING id, user_id, book_id, read_at, rating
                """,
                (row_id, user_id, book_id, utc_now(), rating),
            )
            return _reading_row_from_row(cursor.fetchone())


def list_user_books(user_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        state = read_state()
        rows = [row for row in state["reading_list"].values() if row["user_id"] == user_id]
        return [
            state["books"][row["book_id"]] | {"rating": row.get("rating")}
            for row in rows
            if row["book_id"] in state["books"]
        ]

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT b.id, b.isbn, b.title, b.author, b.description, b.genres, b.published_year,
                       b.created_at, b.source, b.openlibrary_key, b.cover_url, b.dedupe_key, rl.rating
                FROM reading_list rl
                JOIN books b ON b.id = rl.book_id
                WHERE rl.user_id = %s
                ORDER BY rl.read_at
                """,
                (user_id,),
            )
            return [_book_from_row(row[:12]) | {"rating": row[12]} for row in cursor.fetchall()]


def list_user_ids_with_reading_list() -> list[str]:
    conn = _connect()
    if conn is None:
        return sorted({row["user_id"] for row in read_state()["reading_list"].values()})
    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT user_id FROM reading_list ORDER BY user_id")
            return [row[0] for row in cursor.fetchall()]


def upsert_book_embedding(book_id: str, embedding: list[float], model_version: str) -> dict[str, Any]:
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            for existing in state["book_embeddings"].values():
                if existing["book_id"] == book_id and existing["model_version"] == model_version:
                    return existing
            row = {
                "id": str(uuid4()),
                "book_id": book_id,
                "embedding": embedding,
                "model_version": model_version,
                "created_at": utc_now(),
            }
            state["book_embeddings"][row["id"]] = row
            return row

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO book_embeddings (id, book_id, embedding, model_version, created_at)
                VALUES (%s, %s, %s::vector, %s, %s)
                ON CONFLICT (book_id, model_version) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at
                RETURNING id, book_id, embedding::text, model_version, created_at
                """,
                (str(uuid4()), book_id, _vector_literal(embedding), model_version, utc_now()),
            )
            row = cursor.fetchone()
            return {
                "id": row[0],
                "book_id": row[1],
                "embedding": _parse_vector(row[2]),
                "model_version": row[3],
                "created_at": row[4].isoformat(),
            }


def user_profile_vector(user_id: str) -> list[float]:
    conn = _connect()
    if conn is None:
        state = read_state()
        embeddings = {row["book_id"]: row["embedding"] for row in state["book_embeddings"].values()}
        owned_ids = [row["book_id"] for row in state["reading_list"].values() if row["user_id"] == user_id]
        return average([embeddings[book_id] for book_id in owned_ids if book_id in embeddings])

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT e.embedding::text
                FROM reading_list rl
                JOIN book_embeddings e ON e.book_id = rl.book_id
                WHERE rl.user_id = %s
                """,
                (user_id,),
            )
            return average([_parse_vector(row[0]) for row in cursor.fetchall()])


def pgvector_candidate_books(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    profile = user_profile_vector(user_id)
    if not profile:
        return []
    owned_keys = {
        key
        for book in list_user_books(user_id)
        for key in book_identity_keys(book)
    }
    conn = _connect()
    if conn is None:
        return []
    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT b.id, b.isbn, b.title, b.author, b.description, b.genres, b.published_year,
                       b.created_at, b.source, b.openlibrary_key, b.cover_url, b.dedupe_key,
                       e.embedding::text,
                       1 - (e.embedding <=> %s::vector) AS score
                FROM book_embeddings e
                JOIN books b ON b.id = e.book_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM reading_list rl
                    WHERE rl.user_id = %s AND rl.book_id = b.id
                )
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (_vector_literal(profile), user_id, _vector_literal(profile), max(limit * 4, limit)),
            )
            rows = cursor.fetchall()

    candidates = [
        {"book": _book_from_row(row[:12]), "embedding": _parse_vector(row[12]), "score": float(row[13] or 0.0)}
        for row in rows
    ]
    return [
        item
        for item in candidates
        if not book_identity_keys(item["book"]).intersection(owned_keys)
    ][:limit]


def memory_candidate_books(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    state = read_state()
    embeddings = {row["book_id"]: row["embedding"] for row in state["book_embeddings"].values()}
    owned_ids = {row["book_id"] for row in state["reading_list"].values() if row["user_id"] == user_id}
    owned_keys = {
        key
        for book in list_user_books(user_id)
        for key in book_identity_keys(book)
    }
    profile = user_profile_vector(user_id)
    if not profile:
        return []

    scored = []
    for book_id, vector in embeddings.items():
        if book_id in owned_ids or book_id not in state["books"]:
            continue
        book = state["books"][book_id]
        if book_identity_keys(book).intersection(owned_keys):
            continue
        scored.append({"book": book, "embedding": vector, "score": cosine(profile, vector)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


def candidate_books_by_vector(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    if is_postgres_enabled():
        return pgvector_candidate_books(user_id, limit=limit)
    return memory_candidate_books(user_id, limit=limit)


def get_user_recommendation(user_id: str, rec_type: str) -> dict[str, Any] | None:
    conn = _connect()
    if conn is None:
        return read_state()["recommendations"].get(f"{user_id}:{rec_type}")
    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, user_id, type, book_ids, explanations, computed_at FROM recommendations WHERE user_id = %s AND type = %s",
                (user_id, rec_type),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "user_id": row[1],
                "type": row[2],
                "book_ids": row[3] or [],
                "explanations": row[4] or {},
                "computed_at": row[5].isoformat() if row[5] else None,
            }


def upsert_recommendation(user_id: str, rec_type: str, book_ids: list[str], explanations: dict[str, str]) -> dict[str, Any]:
    row = {
        "id": f"{user_id}:{rec_type}",
        "user_id": user_id,
        "type": rec_type,
        "book_ids": book_ids,
        "explanations": explanations,
        "computed_at": utc_now(),
    }
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            state["recommendations"][row["id"]] = row
            return row

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO recommendations (id, user_id, type, book_ids, explanations, computed_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, type) DO UPDATE
                SET book_ids = EXCLUDED.book_ids,
                    explanations = EXCLUDED.explanations,
                    computed_at = EXCLUDED.computed_at
                RETURNING id, user_id, type, book_ids, explanations, computed_at
                """,
                (row["id"], user_id, rec_type, book_ids, Jsonb(explanations), row["computed_at"]),
            )
            saved = cursor.fetchone()
            return {
                "id": saved[0],
                "user_id": saved[1],
                "type": saved[2],
                "book_ids": saved[3] or [],
                "explanations": saved[4] or {},
                "computed_at": saved[5].isoformat() if saved[5] else None,
            }


def books_by_ids(book_ids: list[str]) -> list[dict[str, Any]]:
    if not book_ids:
        return []
    conn = _connect()
    if conn is None:
        state = read_state()
        return [state["books"][book_id] for book_id in book_ids if book_id in state["books"]]
    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, isbn, title, author, description, genres, published_year, created_at,
                       source, openlibrary_key, cover_url, dedupe_key
                FROM books
                WHERE id = ANY(%s)
                """,
                (book_ids,),
            )
            by_id = {_book_from_row(row)["id"]: _book_from_row(row) for row in cursor.fetchall()}
            return [by_id[book_id] for book_id in book_ids if book_id in by_id]


def publish_event(topic: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    event = {
        "id": str(uuid4()),
        "topic": topic,
        "type": event_type,
        "payload": payload,
        "created_at": utc_now(),
        "delivered_to": [],
    }
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            state["events"].append(event)
            return event

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO events (id, topic, type, payload, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (event["id"], topic, event_type, Jsonb(payload), event["created_at"]),
            )
    return event


def pull_events(
    topic: str,
    subscriber: str,
    event_types: set[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        def mutate(state: dict[str, Any]) -> list[dict[str, Any]]:
            selected = []
            for event in state["events"]:
                if len(selected) >= limit:
                    break
                if event["topic"] != topic:
                    continue
                if event_types and event["type"] not in event_types:
                    continue
                if subscriber in event["delivered_to"]:
                    continue
                event["delivered_to"].append(subscriber)
                selected.append(event)
            return selected

        return update_state(mutate)

    with conn:
        _ensure_postgres(conn)
        with conn.cursor() as cursor:
            params: list[Any] = [topic, subscriber]
            type_filter = ""
            if event_types:
                type_filter = "AND e.type = ANY(%s)"
                params.append(sorted(event_types))
            params.append(limit)
            cursor.execute(
                f"""
                SELECT e.id, e.topic, e.type, e.payload, e.created_at
                FROM events e
                WHERE e.topic = %s
                  AND NOT EXISTS (
                    SELECT 1 FROM event_deliveries d
                    WHERE d.event_id = e.id AND d.subscriber = %s
                  )
                  {type_filter}
                ORDER BY e.created_at
                LIMIT %s
                FOR UPDATE SKIP LOCKED
                """,
                params,
            )
            rows = cursor.fetchall()
            for row in rows:
                cursor.execute(
                    """
                    INSERT INTO event_deliveries (event_id, subscriber)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (row[0], subscriber),
                )
            return [
                {
                    "id": row[0],
                    "topic": row[1],
                    "type": row[2],
                    "payload": row[3],
                    "created_at": row[4].isoformat(),
                    "delivered_to": [subscriber],
                }
                for row in rows
            ]


EVENT_SUBSCRIPTIONS = (
    {"topic": "books", "subscriber": "embedding-worker", "event_types": {"BookCreated"}},
    {"topic": "books", "subscriber": "recommendation-service", "event_types": {"BookEmbedded"}},
    {"topic": "users", "subscriber": "recommendation-service", "event_types": {"UserBookAdded"}},
)


def event_backlog_summary() -> list[dict[str, Any]]:
    conn = _connect()
    if conn is None:
        events = read_state()["events"]
        rows = []
        for subscription in EVENT_SUBSCRIPTIONS:
            matching = [
                event
                for event in events
                if event["topic"] == subscription["topic"] and event["type"] in subscription["event_types"]
            ]
            pending = [event for event in matching if subscription["subscriber"] not in event.get("delivered_to", [])]
            delivered = [event for event in matching if subscription["subscriber"] in event.get("delivered_to", [])]
            rows.append(
                {
                    "topic": subscription["topic"],
                    "subscriber": subscription["subscriber"],
                    "event_types": sorted(subscription["event_types"]),
                    "total": len(matching),
                    "pending": len(pending),
                    "delivered": len(delivered),
                    "last_event_at": matching[-1]["created_at"] if matching else None,
                    "last_delivered_at": delivered[-1]["created_at"] if delivered else None,
                    "oldest_pending_at": pending[0]["created_at"] if pending else None,
                }
            )
        return rows

    with conn:
        _ensure_postgres(conn)
        rows = []
        with conn.cursor() as cursor:
            for subscription in EVENT_SUBSCRIPTIONS:
                params = (
                    subscription["topic"],
                    sorted(subscription["event_types"]),
                    subscription["subscriber"],
                )
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(*) FILTER (WHERE d.event_id IS NULL) AS pending,
                        COUNT(d.event_id) AS delivered,
                        MAX(e.created_at) AS last_event_at,
                        MAX(d.delivered_at) AS last_delivered_at,
                        MIN(e.created_at) FILTER (WHERE d.event_id IS NULL) AS oldest_pending_at
                    FROM events e
                    LEFT JOIN event_deliveries d
                      ON d.event_id = e.id AND d.subscriber = %s
                    WHERE e.topic = %s AND e.type = ANY(%s)
                    """,
                    (subscription["subscriber"], params[0], params[1]),
                )
                row = cursor.fetchone()
                rows.append(
                    {
                        "topic": subscription["topic"],
                        "subscriber": subscription["subscriber"],
                        "event_types": sorted(subscription["event_types"]),
                        "total": int(row[0] or 0),
                        "pending": int(row[1] or 0),
                        "delivered": int(row[2] or 0),
                        "last_event_at": row[3].isoformat() if row[3] else None,
                        "last_delivered_at": row[4].isoformat() if row[4] else None,
                        "oldest_pending_at": row[5].isoformat() if row[5] else None,
                    }
                )
        return rows
