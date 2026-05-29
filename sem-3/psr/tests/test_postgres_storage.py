from __future__ import annotations

import os

import pytest

from shared.repositories import (
    get_or_create_user,
    get_user_recommendation,
    list_user_books,
    upsert_book,
    upsert_book_embedding,
    upsert_reading_list_entry,
    upsert_recommendation,
)
from shared.storage import reset_state
from shared.text import deterministic_embedding


pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to run PostgreSQL-backed integration tests",
)


def test_postgres_granular_repositories_round_trip(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    reset_state()

    user = get_or_create_user(
        "u1",
        {
            "email": "u1@example.edu",
            "display_name": "User One",
            "preferences": {"mood": "curious"},
        },
    )
    book = upsert_book(
        {
            "isbn": None,
            "title": "Dune",
            "author": "Frank Herbert",
            "description": "desert ecology",
            "genres": ["science fiction"],
            "published_year": 1965,
            "source": "manual",
            "openlibrary_key": None,
            "cover_url": None,
        },
        "title:dune|author:frank herbert",
    )["book"]
    reading_row = upsert_reading_list_entry("u1", book["id"], 5)
    embedding = upsert_book_embedding(book["id"], deterministic_embedding("desert ecology", 8), "test-8")
    recommendation = upsert_recommendation("u1", "similar", [book["id"]], {book["id"]: "because"})

    assert user["email"] == "u1@example.edu"
    assert book["title"] == "Dune"
    assert reading_row["rating"] == 5
    assert len(embedding["embedding"]) == 8
    assert list_user_books("u1")[0]["title"] == "Dune"
    assert get_user_recommendation("u1", "similar") == recommendation
