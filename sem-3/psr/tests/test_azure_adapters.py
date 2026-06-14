from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from shared.events import publish, pull


ROOT = Path(__file__).resolve().parents[1]


class FakeServiceBusMessage:
    def __init__(self, body: str, subject: str, message_id: str, content_type: str | None = None) -> None:
        self.body = body
        self.subject = subject
        self.message_id = message_id
        self.content_type = content_type


class FakeReceivedMessage:
    def __init__(self, body: str, subject: str, message_id: str) -> None:
        self.body = body
        self.subject = subject
        self.message_id = message_id


class FakeSender:
    sent: list[FakeServiceBusMessage] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def send_messages(self, message: FakeServiceBusMessage) -> None:
        self.sent.append(message)


class FakeReceiver:
    messages: list[FakeReceivedMessage] = []
    completed: list[str] = []
    abandoned: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def receive_messages(self, max_message_count: int, max_wait_time: int) -> list[FakeReceivedMessage]:
        return self.messages[:max_message_count]

    def complete_message(self, message: FakeReceivedMessage) -> None:
        self.completed.append(message.message_id)

    def abandon_message(self, message: FakeReceivedMessage) -> None:
        self.abandoned.append(message.message_id)


class FakeServiceBusClient:
    connection_string: str | None = None

    @classmethod
    def from_connection_string(cls, connection_string: str):
        cls.connection_string = connection_string
        return cls()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get_topic_sender(self, topic_name: str) -> FakeSender:
        assert topic_name == "books"
        return FakeSender()

    def get_subscription_receiver(self, topic_name: str, subscription_name: str) -> FakeReceiver:
        assert topic_name == "books"
        assert subscription_name == "recommendation-service"
        return FakeReceiver()


@pytest.fixture
def fake_servicebus(monkeypatch):
    FakeSender.sent = []
    FakeReceiver.messages = []
    FakeReceiver.completed = []
    FakeReceiver.abandoned = []
    FakeServiceBusClient.connection_string = None

    azure_module = types.ModuleType("azure")
    servicebus_module = types.ModuleType("azure.servicebus")
    servicebus_module.ServiceBusClient = FakeServiceBusClient
    servicebus_module.ServiceBusMessage = FakeServiceBusMessage

    monkeypatch.setitem(sys.modules, "azure", azure_module)
    monkeypatch.setitem(sys.modules, "azure.servicebus", servicebus_module)


def test_azure_service_bus_publish_uses_contract_and_topic_sender(monkeypatch, fake_servicebus) -> None:
    monkeypatch.setenv("EVENT_BUS_PROVIDER", "azure-service-bus")
    monkeypatch.setenv("AZURE_SERVICE_BUS_CONNECTION_STRING", "Endpoint=sb://example/")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    event = publish("books", "BookCreated", {"book_id": "b1"})

    assert event["type"] == "BookCreated"
    assert event["payload"] == {"book_id": "b1"}
    assert FakeServiceBusClient.connection_string == "Endpoint=sb://example/"
    assert len(FakeSender.sent) == 1
    assert FakeSender.sent[0].subject == "BookCreated"
    assert FakeSender.sent[0].body == '{"book_id": "b1"}'
    assert FakeSender.sent[0].content_type == "application/json"


def test_azure_service_bus_pull_completes_matching_messages_and_abandons_filtered(monkeypatch, fake_servicebus) -> None:
    monkeypatch.setenv("EVENT_BUS_PROVIDER", "azure-service-bus")
    monkeypatch.setenv("AZURE_SERVICE_BUS_CONNECTION_STRING", "Endpoint=sb://example/")
    FakeReceiver.messages = [
        FakeReceivedMessage('{"book_id": "b1", "embedding_id": "e1"}', "BookEmbedded", "m1"),
        FakeReceivedMessage('{"book_id": "b2"}', "BookCreated", "m2"),
    ]

    events = pull("books", "recommendation-service", {"BookEmbedded"}, limit=10)

    assert events == [
        {
            "id": "m1",
            "topic": "books",
            "type": "BookEmbedded",
            "payload": {"book_id": "b1", "embedding_id": "e1"},
            "created_at": "",
            "delivered_to": ["recommendation-service"],
        }
    ]
    assert FakeReceiver.completed == ["m1"]
    assert FakeReceiver.abandoned == ["m2"]


def load_llm_module(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure-openai")
    spec = importlib.util.spec_from_file_location("llm_main_for_azure_test", ROOT / "services/llm-service/app/main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_azure_openai_embedding_request_is_mapped(monkeypatch) -> None:
    module = load_llm_module(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://book-ai.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "embed-small")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"embedding": [0.1, 0.2]}]}

    def fake_post(url, headers, json, timeout):
        assert url == (
            "https://book-ai.openai.azure.com/openai/deployments/embed-small/"
            "embeddings?api-version=2024-02-01"
        )
        assert headers["api-key"] == "secret"
        assert json == {"input": "Dune"}
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.embed(module.EmbedRequest(text="Dune"))

    assert result.embedding == [0.1, 0.2]
    assert result.dimensions == 2
    assert result.model_version == "azure-openai:embed-small:2"


def test_azure_openai_generate_request_is_mapped(monkeypatch) -> None:
    module = load_llm_module(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://book-ai.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-mini")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "Read Dune next."}}]}

    def fake_post(url, headers, json, timeout):
        assert url == (
            "https://book-ai.openai.azure.com/openai/deployments/gpt-mini/"
            "chat/completions?api-version=2024-02-01"
        )
        assert headers["api-key"] == "secret"
        assert json["messages"] == [{"role": "user", "content": "Recommend a book."}]
        assert json["temperature"] == 0.2
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.generate(module.GenerateRequest(prompt="Recommend a book."))

    assert result.text == "Read Dune next."
    assert result.provider == "azure-openai:gpt-mini"
