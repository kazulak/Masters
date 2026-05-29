from __future__ import annotations

import pytest

from shared.event_contracts import EventContractError, validate_event_payload
from shared.events import publish, pull
from shared.storage import reset_state


def test_event_payload_contracts_accept_known_events() -> None:
    assert validate_event_payload("books", "BookCreated", {"book_id": "b1"}) == {"book_id": "b1"}
    assert validate_event_payload(
        "books",
        "BookEmbedded",
        {"book_id": "b1", "embedding_id": "e1"},
    ) == {"book_id": "b1", "embedding_id": "e1"}
    assert validate_event_payload(
        "users",
        "UserBookAdded",
        {"user_id": "u1", "book_id": "b1"},
    ) == {"user_id": "u1", "book_id": "b1"}


def test_event_payload_contracts_reject_unknown_or_extra_fields() -> None:
    with pytest.raises(EventContractError):
        validate_event_payload("books", "BookDeleted", {"book_id": "b1"})

    with pytest.raises(EventContractError):
        validate_event_payload("books", "BookCreated", {"book_id": "b1", "extra": "nope"})


def test_publish_validates_contract_before_persisting(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("EVENT_BUS_PROVIDER", "local")
    reset_state()

    with pytest.raises(EventContractError):
        publish("books", "BookCreated", {"wrong": "payload"})

    publish("books", "BookCreated", {"book_id": "b1"})
    assert pull("books", "worker", {"BookCreated"})[0]["payload"] == {"book_id": "b1"}
