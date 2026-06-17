from __future__ import annotations

import importlib.util
from pathlib import Path

from shared.events import publish, pull
from shared.repositories import event_backlog_summary
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


def test_event_backlog_summary_tracks_pending_and_delivered(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()

    publish("books", "BookCreated", {"book_id": "b1"})
    before = {
        (row["topic"], row["subscriber"], tuple(row["event_types"])): row
        for row in event_backlog_summary()
    }

    assert before[("books", "embedding-worker", ("BookCreated",))]["pending"] == 1
    assert before[("books", "embedding-worker", ("BookCreated",))]["oldest_pending_at"]

    pull("books", "embedding-worker", {"BookCreated"})
    after = {
        (row["topic"], row["subscriber"], tuple(row["event_types"])): row
        for row in event_backlog_summary()
    }

    assert after[("books", "embedding-worker", ("BookCreated",))]["pending"] == 0
    assert after[("books", "embedding-worker", ("BookCreated",))]["delivered"] == 1
    assert after[("books", "embedding-worker", ("BookCreated",))]["oldest_pending_at"] is None


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


def test_user_profile_signup_signin_and_public_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()
    user_profile = load_module("user_profile_main_auth", "services/user-profile/app/main.py")

    created = user_profile.signup(
        user_profile.SignUpRequest(
            email="Reader@Example.edu",
            password="secret",
            display_name="Reader",
            mood="reflective",
            genres=["classic"],
        )
    )
    signed_in = user_profile.signin(user_profile.SignInRequest(email="reader@example.edu", password="secret"))

    assert created.id == "reader@example.edu"
    assert signed_in.id == "reader@example.edu"
    assert signed_in.preferences["mood"] == "reflective"
    assert "_password" not in signed_in.preferences


def test_user_profile_default_profiles_use_unique_emails(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()
    user_profile = load_module("user_profile_main_defaults", "services/user-profile/app/main.py")

    first = user_profile.get_me("alice")
    second = user_profile.get_me("bob@example.edu")

    assert first.email == "alice@example.edu"
    assert second.email == "bob@example.edu"
    assert first.email != second.email


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
        state["books"]["b3"] = {
            "id": "b3",
            "title": "Dune",
            "author": "Frank Herbert",
            "genres": ["science fiction"],
            "description": "duplicate catalog copy",
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
            "book_ids": ["b1", "b3", "b2"],
            "explanations": {
                "b1": "stale owned book",
                "b3": "duplicate owned book",
                "b2": "fresh candidate",
            },
            "computed_at": "now",
        }

    update_state(seed)
    recommendation = load_module("recommendation_main_stale", "services/recommendation/app/main.py")

    response = recommendation.get_recommendations(user_id="u1", type="similar")

    assert [book["id"] for book in response["books"]] == ["b2"]
    assert response["explanations"] == {"b2": "fresh candidate"}


def test_recommendation_recompute_excludes_duplicate_owned_books(monkeypatch, tmp_path):
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
        books = {
            "owned": {
                "id": "owned",
                "title": "Dune",
                "author": "Frank Herbert",
                "genres": ["science fiction"],
                "description": "desert politics ecology",
                "created_at": "now",
            },
            "duplicate": {
                "id": "duplicate",
                "title": "Dune",
                "author": "Frank Herbert",
                "genres": ["science fiction"],
                "description": "same book from another catalog source",
                "created_at": "now",
            },
            "candidate": {
                "id": "candidate",
                "title": "Foundation",
                "author": "Isaac Asimov",
                "genres": ["science fiction"],
                "description": "empire psychohistory strategy",
                "created_at": "now",
            },
        }
        state["books"].update(books)
        state["reading_list"]["r1"] = {
            "id": "r1",
            "user_id": "u1",
            "book_id": "owned",
            "read_at": "now",
            "rating": 5,
        }
        for book_id, book in books.items():
            state["book_embeddings"][f"e-{book_id}"] = {
                "id": f"e-{book_id}",
                "book_id": book_id,
                "embedding": deterministic_embedding(book["description"], 16),
                "model_version": "test",
                "created_at": "now",
            }

    update_state(seed)
    recommendation = load_module("recommendation_main_duplicate_recompute", "services/recommendation/app/main.py")

    recommendation.recompute_user("u1")
    response = recommendation.get_recommendations(user_id="u1", type="similar")

    assert [book["id"] for book in response["books"]] == ["candidate"]


def test_recommendation_modes_use_distinct_scoring_and_explanations(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()

    def seed(state: dict) -> None:
        state["users"]["u1"] = {
            "id": "u1",
            "email": "u1@example.edu",
            "display_name": "User",
            "preferences": {"mood": "adventurous", "genres": ["science fiction"]},
            "created_at": "now",
        }
        books = {
            "owned": {
                "id": "owned",
                "title": "Dune",
                "author": "Frank Herbert",
                "genres": ["science fiction"],
                "description": "desert politics ecology",
                "created_at": "now",
            },
            "similar": {
                "id": "similar",
                "title": "Foundation",
                "author": "Isaac Asimov",
                "genres": ["science fiction"],
                "description": "empire psychohistory strategy",
                "created_at": "now",
            },
            "widen": {
                "id": "widen",
                "title": "The Name of the Rose",
                "author": "Umberto Eco",
                "genres": ["historical fiction", "mystery"],
                "description": "monastery mystery philosophy books",
                "created_at": "now",
            },
            "mood": {
                "id": "mood",
                "title": "The Hobbit",
                "author": "J. R. R. Tolkien",
                "genres": ["fantasy", "adventure"],
                "description": "quest adventure travel dragon",
                "created_at": "now",
            },
        }
        state["books"].update(books)
        state["reading_list"]["r1"] = {
            "id": "r1",
            "user_id": "u1",
            "book_id": "owned",
            "read_at": "now",
            "rating": 5,
        }
        embeddings = {
            "owned": [1.0, 0.0, 0.0, 0.0],
            "similar": [0.99, 0.01, 0.0, 0.0],
            "widen": [0.75, 0.66, 0.0, 0.0],
            "mood": [0.72, 0.69, 0.0, 0.0],
        }
        for book_id, embedding in embeddings.items():
            state["book_embeddings"][f"e-{book_id}"] = {
                "id": f"e-{book_id}",
                "book_id": book_id,
                "embedding": embedding,
                "model_version": "test",
                "created_at": "now",
            }

    update_state(seed)
    recommendation = load_module("recommendation_main_distinct_modes", "services/recommendation/app/main.py")

    for rec_type in ("similar", "widen", "mood"):
        recommendation.recompute_user("u1")

    similar = recommendation.get_recommendations(user_id="u1", type="similar")
    widen = recommendation.get_recommendations(user_id="u1", type="widen")
    mood = recommendation.get_recommendations(user_id="u1", type="mood")

    assert similar["books"][0]["id"] == "similar"
    assert widen["books"][0]["id"] == "widen"
    assert mood["books"][0]["id"] == "mood"
    assert "similar pick" in similar["explanations"]["similar"]
    assert "widen pick" in widen["explanations"]["widen"]
    assert "mood pick" in mood["explanations"]["mood"]


def test_ai_recommendation_prompt_uses_llm_over_cached_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()

    def seed(state: dict) -> None:
        state["users"]["u1"] = {
            "id": "u1",
            "email": "u1@example.edu",
            "display_name": "User",
            "preferences": {
                "mood": "reflective",
                "genres": ["philosophy"],
                "recommendation_instructions": "Prefer psychologically serious books.",
            },
            "created_at": "now",
        }
        state["books"]["owned"] = {
            "id": "owned",
            "title": "Dune",
            "author": "Frank Herbert",
            "genres": ["science fiction"],
            "description": "desert politics ecology",
            "created_at": "now",
        }
        state["books"]["candidate"] = {
            "id": "candidate",
            "title": "Man's Search for Meaning",
            "author": "Viktor E. Frankl",
            "genres": ["psychology", "philosophy"],
            "description": "meaning under suffering",
            "created_at": "now",
        }
        state["reading_list"]["r1"] = {
            "id": "r1",
            "user_id": "u1",
            "book_id": "owned",
            "read_at": "now",
            "rating": 5,
        }
        state["recommendations"]["u1:similar"] = {
            "id": "u1:similar",
            "user_id": "u1",
            "type": "similar",
            "book_ids": ["candidate"],
            "explanations": {"candidate": "fresh candidate"},
            "computed_at": "now",
        }

    update_state(seed)
    recommendation = load_module("recommendation_ai_prompt", "services/recommendation/app/main.py")

    def fake_llm(prompt: str, context: dict | None = None, timeout: int = 120) -> dict:
        assert "I want something meaningful" in prompt
        assert "Man's Search for Meaning" in prompt
        assert "Prefer psychologically serious books." in prompt
        assert "outside pick" in prompt
        return {"provider": "test-llm", "text": "Pick Man's Search for Meaning."}

    monkeypatch.setattr(recommendation, "_call_llm", fake_llm)

    response = recommendation.ask_for_recommendations(
        recommendation.AskRequest(user_id="u1", prompt="I want something meaningful", type="similar")
    )

    assert response["provider"] == "test-llm"
    assert response["answer"] == "Pick Man's Search for Meaning."
    assert [book["id"] for book in response["books"]] == ["candidate"]
    assert response["source"] == "llm-over-engine-candidates"
    assert response["engine"] == "vector-similarity"
    assert response["allow_outside_candidates"] is True
    assert response["user_instructions"] == "Prefer psychologically serious books."


def test_embedding_worker_status_records_success_and_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()
    embedding = load_module("embedding_worker_observability", "services/embedding-worker/app/main.py")

    success = embedding.process_once()
    status = embedding.status()

    assert success == {"processed": 0, "skipped": 0}
    assert status["worker"]["total_runs"] == 1
    assert status["worker"]["total_failures"] == 0
    assert status["worker"]["last_success_at"]
    assert status["worker"]["last_error"] is None

    def fail_pull(*args, **kwargs):
        raise RuntimeError("event store unavailable")

    monkeypatch.setattr(embedding, "pull", fail_pull)

    try:
        embedding.process_once()
    except RuntimeError:
        pass
    else:
        raise AssertionError("process_once should re-raise worker failures")

    status = embedding.status()
    assert status["worker"]["total_runs"] == 2
    assert status["worker"]["total_failures"] == 1
    assert status["worker"]["consecutive_failures"] == 1
    assert "event store unavailable" in status["worker"]["last_error"]


def test_recommendation_worker_status_records_success_and_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    reset_state()
    recommendation = load_module("recommendation_observability", "services/recommendation/app/main.py")

    success = recommendation.process_once()
    status = recommendation.status()

    assert success == {"events": 0, "users": 0}
    assert status["worker"]["total_runs"] == 1
    assert status["worker"]["total_failures"] == 0
    assert status["worker"]["last_success_at"]
    assert status["worker"]["last_error"] is None

    def fail_pull(*args, **kwargs):
        raise RuntimeError("event bus unavailable")

    monkeypatch.setattr(recommendation, "pull", fail_pull)

    try:
        recommendation.process_once()
    except RuntimeError:
        pass
    else:
        raise AssertionError("process_once should re-raise worker failures")

    status = recommendation.status()
    assert status["worker"]["total_runs"] == 2
    assert status["worker"]["total_failures"] == 1
    assert status["worker"]["consecutive_failures"] == 1
    assert "event bus unavailable" in status["worker"]["last_error"]
