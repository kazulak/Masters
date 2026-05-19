from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from shared.storage import update_state


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def publish(topic: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    event = {
        "id": str(uuid4()),
        "topic": topic,
        "type": event_type,
        "payload": payload,
        "created_at": utc_now(),
        "delivered_to": [],
    }

    def mutate(state: dict[str, Any]) -> dict[str, Any]:
        state["events"].append(event)
        return event

    return update_state(mutate)


def pull(
    topic: str,
    subscriber: str,
    event_types: set[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
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
