from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from shared import config
from shared.event_contracts import validate_event_payload
from shared.repositories import publish_event, pull_events


def _azure_not_configured() -> RuntimeError:
    return RuntimeError(
        "EVENT_BUS_PROVIDER=azure-service-bus requires azure-servicebus and "
        "AZURE_SERVICE_BUS_CONNECTION_STRING. Local runs should keep EVENT_BUS_PROVIDER=local."
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_event(topic: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "topic": topic,
        "type": event_type,
        "payload": payload,
        "created_at": _utc_now(),
        "delivered_to": [],
    }


def _message_payload(message: Any) -> dict[str, Any]:
    body = getattr(message, "body", None)
    if body is None:
        text = str(message)
    elif isinstance(body, bytes | bytearray):
        text = body.decode("utf-8")
    elif isinstance(body, str):
        text = body
    else:
        chunks = []
        for chunk in body:
            if isinstance(chunk, bytes | bytearray):
                chunks.append(chunk.decode("utf-8"))
            else:
                chunks.append(str(chunk))
        text = "".join(chunks)
    return json.loads(text)


def _publish_azure_service_bus(topic: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from azure.servicebus import ServiceBusClient, ServiceBusMessage
    except ImportError as exc:
        raise _azure_not_configured() from exc

    connection_string = config.env("AZURE_SERVICE_BUS_CONNECTION_STRING", "")
    if not connection_string:
        raise _azure_not_configured()
    event = _new_event(topic, event_type, payload)
    with ServiceBusClient.from_connection_string(connection_string) as client:
        sender = client.get_topic_sender(topic_name=topic)
        with sender:
            sender.send_messages(
                ServiceBusMessage(
                    json.dumps(event["payload"]),
                    subject=event_type,
                    message_id=event["id"],
                    content_type="application/json",
                )
            )
    return event


def _pull_azure_service_bus(
    topic: str,
    subscriber: str,
    event_types: set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        from azure.servicebus import ServiceBusClient
    except ImportError as exc:
        raise _azure_not_configured() from exc

    connection_string = config.env("AZURE_SERVICE_BUS_CONNECTION_STRING", "")
    if not connection_string:
        raise _azure_not_configured()
    with ServiceBusClient.from_connection_string(connection_string) as client:
        receiver = client.get_subscription_receiver(topic_name=topic, subscription_name=subscriber)
        events = []
        with receiver:
            for message in receiver.receive_messages(max_message_count=limit, max_wait_time=2):
                if event_types and message.subject not in event_types:
                    receiver.abandon_message(message)
                    continue
                payload = _message_payload(message)
                validate_event_payload(topic, message.subject, payload)
                events.append(
                    {
                        "id": message.message_id,
                        "topic": topic,
                        "type": message.subject,
                        "payload": payload,
                        "created_at": "",
                        "delivered_to": [subscriber],
                    }
                )
                receiver.complete_message(message)
        return events


def publish(topic: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = validate_event_payload(topic, event_type, payload)
    if config.env("EVENT_BUS_PROVIDER", "local") == "azure-service-bus":
        return _publish_azure_service_bus(topic, event_type, payload)
    return publish_event(topic, event_type, payload)


def pull(
    topic: str,
    subscriber: str,
    event_types: set[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if config.env("EVENT_BUS_PROVIDER", "local") == "azure-service-bus":
        return _pull_azure_service_bus(topic, subscriber, event_types, limit)
    return pull_events(topic, subscriber, event_types, limit)
