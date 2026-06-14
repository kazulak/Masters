from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import pytest


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EXPERIMENTAL_HTTP_TESTS") != "1",
    reason="In-process FastAPI HTTP transports hang in the current local runtime; use container smoke for default CI.",
)


def load_llm_module(monkeypatch, provider: str):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    spec = importlib.util.spec_from_file_location("llm_main_http_for_test", ROOT / "services/llm-service/app/main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_json(app, method: str, path: str, *, query: dict | None = None, payload: dict | None = None) -> tuple[int, dict]:
    body = json.dumps(payload or {}).encode("utf-8") if payload is not None else b""
    headers = [(b"host", b"testserver")]
    if payload is not None:
        headers.append((b"content-type", b"application/json"))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": urlencode(query or {}).encode("utf-8"),
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def call() -> list[dict]:
        messages: list[dict] = []
        request_sent = False

        async def receive() -> dict:
            nonlocal request_sent
            if request_sent:
                return {"type": "http.disconnect"}
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict) -> None:
            messages.append(message)

        await app(scope, receive, send)
        return messages

    messages = asyncio.run(call())
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body.decode("utf-8"))


def test_llm_health_and_models_http(monkeypatch):
    module = load_llm_module(monkeypatch, "deterministic")

    health_status, health = request_json(module.app, "GET", "/health")
    models_status, models = request_json(module.app, "GET", "/v1/models")

    assert health_status == 200
    assert health["provider"] == "deterministic"
    assert models_status == 200
    assert models["ollama_generate_model"] == "gemma4:e2b"


def test_llm_generate_http_get_and_post(monkeypatch):
    module = load_llm_module(monkeypatch, "deterministic")

    get_status, get_response = request_json(module.app, "GET", "/v1/generate", query={"prompt": "Recommend a book."})
    post_status, post_response = request_json(
        module.app,
        "POST",
        "/v1/generate",
        payload={"text": "Recommend a book.", "context": {"title": "Dune"}},
    )

    assert get_status == 200
    assert get_response["provider"] == "local-template"
    assert post_status == 200
    assert post_response["text"].startswith("Recommended Dune")


def test_llm_generate_http_requires_prompt_or_text(monkeypatch):
    module = load_llm_module(monkeypatch, "deterministic")

    status, payload = request_json(module.app, "POST", "/v1/generate", payload={})

    assert status == 422
    assert payload["detail"]
