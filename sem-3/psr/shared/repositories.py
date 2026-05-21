from __future__ import annotations

from typing import Any

import psycopg

from shared import config
from shared.storage import _ensure_postgres, _parse_vector, _vector_literal, read_state
from shared.text import average, cosine


def database_url() -> str | None:
    return config.env("DATABASE_URL", "") or None


def is_postgres_enabled() -> bool:
    return database_url() is not None


def _connect():
    url = database_url()
    if not url:
        return None
    return psycopg.connect(url)


def user_profile_vector(user_id: str) -> list[float]:
    state = read_state()
    embeddings = {row["book_id"]: row["embedding"] for row in state["book_embeddings"].values()}
    owned_ids = [row["book_id"] for row in state["reading_list"].values() if row["user_id"] == user_id]
    vectors = [embeddings[book_id] for book_id in owned_ids if book_id in embeddings]
    return average(vectors)


def pgvector_candidate_books(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    profile = user_profile_vector(user_id)
    if not profile:
        return []
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
                (_vector_literal(profile), user_id, _vector_literal(profile), limit),
            )
            rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "book": {
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
                },
                "embedding": _parse_vector(row[12]),
                "score": float(row[13] or 0.0),
            }
        )
    return results


def memory_candidate_books(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    state = read_state()
    embeddings = {row["book_id"]: row["embedding"] for row in state["book_embeddings"].values()}
    owned_ids = {row["book_id"] for row in state["reading_list"].values() if row["user_id"] == user_id}
    profile = user_profile_vector(user_id)
    if not profile:
        return []

    scored = []
    for book_id, vector in embeddings.items():
        if book_id in owned_ids or book_id not in state["books"]:
            continue
        scored.append({"book": state["books"][book_id], "embedding": vector, "score": cosine(profile, vector)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


def candidate_books_by_vector(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    if is_postgres_enabled():
        return pgvector_candidate_books(user_id, limit=limit)
    return memory_candidate_books(user_id, limit=limit)
