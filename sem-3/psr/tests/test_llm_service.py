from __future__ import annotations

import importlib.util
from pathlib import Path


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
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setenv("OLLAMA_GENERATE_MODEL", "gemma4:e2b")

    result = module.generate(module.GenerateRequest(prompt="Explain why Dune is good."))

    assert result.text == "A concise recommendation."
    assert result.provider == "ollama:gemma4:e2b"


def test_llm_root_and_models_are_browser_friendly(monkeypatch):
    module = load_llm_module(monkeypatch, "deterministic")

    root = module.root()
    models = module.models()

    assert root["service"] == "llm-service"
    assert root["docs"] == "/docs"
    assert "/v1/generate?prompt=" in root["prompt_examples"]["GET"]
    assert models["provider"] == "deterministic"


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
