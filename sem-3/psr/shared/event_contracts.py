from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class EventContractError(ValueError):
    pass


class BookCreatedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str = Field(min_length=1)


class BookEmbeddedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str = Field(min_length=1)
    embedding_id: str = Field(min_length=1)


class UserBookAddedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)


_CONTRACTS: dict[tuple[str, str], type[BaseModel]] = {
    ("books", "BookCreated"): BookCreatedPayload,
    ("books", "BookEmbedded"): BookEmbeddedPayload,
    ("users", "UserBookAdded"): UserBookAddedPayload,
}


def validate_event_payload(topic: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    contract = _CONTRACTS.get((topic, event_type))
    if contract is None:
        raise EventContractError(f"Unknown event contract: {topic}/{event_type}")
    try:
        return contract(**payload).model_dump()
    except ValidationError as exc:
        raise EventContractError(f"Invalid event payload for {topic}/{event_type}: {exc}") from exc
