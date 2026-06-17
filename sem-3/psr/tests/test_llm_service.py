from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[1]


def load_llm_module(monkeypatch, provider: str):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    spec = importlib.util.spec_from_file_location("llm_main_for_test", ROOT / "services/llm-service/app/main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ollama_embedding_response_is_mapped(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embeddings": [[0.1, 0.2, 0.3]]}

    def fake_post(url, json, timeout):
        assert url.endswith("/api/embed")
        assert json["input"] == "Dune"
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.embed(module.EmbedRequest(text="Dune"))

    assert result.embedding == [0.1, 0.2, 0.3]
    assert result.dimensions == 3
    assert result.model_version.startswith("ollama:")


def test_ollama_generate_response_is_mapped(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": "A concise recommendation."}

    def fake_post(url, json, timeout):
        assert url.endswith("/api/generate")
        assert json["model"] == "gemma4:e2b"
        assert json["prompt"] == "Explain why Dune is good."
        assert json["stream"] is False
        assert json["think"] is False
        assert json["options"]["num_predict"] == 512
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setenv("OLLAMA_GENERATE_MODEL", "gemma4:e2b")

    result = module.generate(module.GenerateRequest(prompt="Explain why Dune is good."))

    assert result.text == "A concise recommendation."
    assert result.provider == "ollama:gemma4:e2b"


def test_ollama_generate_timeout_returns_504(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama")

    def fake_post(*args, **kwargs):
        raise module.requests.Timeout("slow model")

    monkeypatch.setattr(module.requests, "post", fake_post)

    with pytest.raises(HTTPException) as exc:
        module.generate(module.GenerateRequest(prompt="Recommend a book."))

    assert exc.value.status_code == 504
    assert "timed out" in exc.value.detail


def test_ollama_generate_empty_response_returns_502(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": ""}

    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(HTTPException) as exc:
        module.generate(module.GenerateRequest(prompt="Recommend a book."))

    assert exc.value.status_code == 502
    assert "empty response" in exc.value.detail


def test_ollama_auto_pull_recovers_from_missing_generate_model(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama")
    monkeypatch.setenv("OLLAMA_AUTO_PULL", "true")
    monkeypatch.setenv("OLLAMA_GENERATE_MODEL", "qwen3:0.6b")

    calls: list[tuple[str, dict]] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise module.requests.HTTPError(self.text)

        def json(self) -> dict:
            return self._payload

    def fake_post(url, json, timeout):
        calls.append((url, json))
        if url.endswith("/api/generate") and len(calls) == 1:
            return FakeResponse(400, {}, 'model "qwen3:0.6b" not found')
        if url.endswith("/api/pull"):
            assert json == {"model": "qwen3:0.6b", "stream": False}
            return FakeResponse(200, {"status": "success"})
        if url.endswith("/api/generate"):
            return FakeResponse(200, {"response": "A tiny POC answer."})
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.generate(module.GenerateRequest(prompt="Recommend a book."))

    assert result.provider == "ollama:qwen3:0.6b"
    assert result.text == "A tiny POC answer."
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == ["generate", "pull", "generate"]


def test_llm_root_and_models_are_browser_friendly(monkeypatch):
    module = load_llm_module(monkeypatch, "deterministic")

    root = module.root()
    models = module.models()

    assert root["service"] == "llm-service"
    assert root["docs"] == "/docs"
    assert "/v1/generate?prompt=" in root["prompt_examples"]["GET"]
    assert models["provider"] == "deterministic"
    assert models["ollama_num_predict"] == "512"
    assert models["ollama_think"] == "false"


def test_generate_accepts_text_alias_in_post_body(monkeypatch):
    module = load_llm_module(monkeypatch, "deterministic")

    result = module.generate(module.GenerateRequest(text="Recommend a book."))

    assert result.provider == "local-template"
    assert result.text.startswith("Recommended")


def test_get_helpers_delegate_to_existing_llm_paths(monkeypatch):
    module = load_llm_module(monkeypatch, "deterministic")

    embedding = module.embed_get("Dune")
    generated = module.generate_get("Recommend a book.")

    assert embedding.dimensions > 0
    assert generated.provider == "local-template"


def test_ollama_with_fallback_uses_deterministic_when_ollama_fails(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama-with-fallback")

    def fake_post(*args, **kwargs):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.embed(module.EmbedRequest(text="Dune"))

    assert result.model_version.startswith("local-hash-")
    assert result.dimensions > 0


def test_ollama_with_fallback_generate_uses_template_when_ollama_fails(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama-with-fallback")

    def fake_post(*args, **kwargs):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.generate(module.GenerateRequest(prompt="Recommend a book.", context={"title": "Dune"}))

    assert result.provider == "local-template"
    assert result.text.startswith("Recommended Dune")


def test_ollama_with_fallback_generate_uses_template_when_ollama_is_empty(monkeypatch):
    module = load_llm_module(monkeypatch, "ollama-with-fallback")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": ""}

    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: FakeResponse())

    result = module.generate(module.GenerateRequest(prompt="Recommend a book.", context={"title": "Dune"}))

    assert result.provider == "local-template"
    assert result.text.startswith("Recommended Dune")
