from __future__ import annotations

import os

import pytest

from shared.storage import read_state, reset_state, update_state
from shared.text import deterministic_embedding


pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to run PostgreSQL-backed integration tests",
)


def test_postgres_normalized_storage_round_trip(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    reset_state()

    def seed(state: dict) -> None:
        state["users"]["u1"] = {
            "id": "u1",
            "email": "u1@example.edu",
            "display_name": "User One",
            "preferences": {"mood": "curious"},
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        state["books"]["b1"] = {
            "id": "b1",
            "isbn": None,
            "title": "Dune",
            "author": "Frank Herbert",
            "description": "desert ecology",
            "genres": ["science fiction"],
            "published_year": 1965,
            "created_at": "2026-01-01T00:00:00+00:00",
            "source": "manual",
            "openlibrary_key": None,
            "cover_url": None,
            "dedupe_key": "title:dune|author:frank herbert",
        }
        state["reading_list"]["r1"] = {
            "id": "r1",
            "user_id": "u1",
            "book_id": "b1",
            "read_at": "2026-01-02T00:00:00+00:00",
            "rating": 5,
        }
        state["book_embeddings"]["e1"] = {
            "id": "e1",
            "book_id": "b1",
            "embedding": deterministic_embedding("desert ecology", 8),
            "model_version": "test-8",
            "created_at": "2026-01-03T00:00:00+00:00",
        }

    update_state(seed)
    state = read_state()

    assert state["users"]["u1"]["email"] == "u1@example.edu"
    assert state["books"]["b1"]["title"] == "Dune"
    assert state["reading_list"]["r1"]["rating"] == 5
    assert len(state["book_embeddings"]["e1"]["embedding"]) == 8
