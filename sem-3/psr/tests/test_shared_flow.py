from __future__ import annotations

import importlib.util
from pathlib import Path

from shared.events import publish, pull
from shared.storage import read_state, reset_state, update_state
from shared.text import deterministic_embedding


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_pull_is_per_subscriber(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()

    event = publish("books", "BookCreated", {"book_id": "b1"})

    first = pull("books", "embedding-worker", {"BookCreated"})
    second = pull("books", "recommendation-service", {"BookCreated"})
    third = pull("books", "embedding-worker", {"BookCreated"})

    assert first[0]["id"] == event["id"]
    assert second[0]["id"] == event["id"]
    assert third == []


def test_recommendation_recompute_uses_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()

    def seed(state: dict) -> None:
        state["users"]["u1"] = {
            "id": "u1",
            "email": "u1@example.edu",
            "display_name": "User",
            "preferences": {"mood": "curious", "genres": ["science fiction"]},
            "created_at": "now",
        }
        state["books"]["b1"] = {
            "id": "b1",
            "title": "Dune",
            "author": "Frank Herbert",
            "genres": ["science fiction"],
            "description": "desert politics ecology",
            "created_at": "now",
        }
        state["books"]["b2"] = {
            "id": "b2",
            "title": "Foundation",
            "author": "Isaac Asimov",
            "genres": ["science fiction"],
            "description": "empire psychohistory strategy",
            "created_at": "now",
        }
        state["reading_list"]["r1"] = {
            "id": "r1",
            "user_id": "u1",
            "book_id": "b1",
            "read_at": "now",
            "rating": 5,
        }
        for book_id in ("b1", "b2"):
            state["book_embeddings"][f"e-{book_id}"] = {
                "id": f"e-{book_id}",
                "book_id": book_id,
                "embedding": deterministic_embedding(state["books"][book_id]["description"], 16),
                "model_version": "test",
                "created_at": "now",
            }

    update_state(seed)
    recommendation = load_module("recommendation_main", "services/recommendation/app/main.py")

    result = recommendation.recompute_user("u1")
    state = read_state()

    assert result == {"computed": 3}
    assert state["recommendations"]["u1:similar"]["book_ids"] == ["b2"]


def test_recommendation_response_filters_stale_owned_books(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()

    def seed(state: dict) -> None:
        state["users"]["u1"] = {
            "id": "u1",
            "email": "u1@example.edu",
            "display_name": "User",
            "preferences": {"mood": "curious", "genres": ["science fiction"]},
            "created_at": "now",
        }
        state["books"]["b1"] = {
            "id": "b1",
            "title": "Dune",
            "author": "Frank Herbert",
            "genres": ["science fiction"],
            "description": "desert politics ecology",
            "created_at": "now",
        }
        state["books"]["b2"] = {
            "id": "b2",
            "title": "Foundation",
            "author": "Isaac Asimov",
            "genres": ["science fiction"],
            "description": "empire psychohistory strategy",
            "created_at": "now",
        }
        state["reading_list"]["r1"] = {
            "id": "r1",
            "user_id": "u1",
            "book_id": "b1",
            "read_at": "now",
            "rating": 5,
        }
        state["recommendations"]["u1:similar"] = {
            "id": "u1:similar",
            "user_id": "u1",
            "type": "similar",
            "book_ids": ["b1", "b2"],
            "explanations": {"b1": "stale owned book", "b2": "fresh candidate"},
            "computed_at": "now",
        }

    update_state(seed)
    recommendation = load_module("recommendation_main_stale", "services/recommendation/app/main.py")

    response = recommendation.get_recommendations(user_id="u1", type="similar")

    assert [book["id"] for book in response["books"]] == ["b2"]
    assert response["explanations"] == {"b2": "fresh candidate"}
