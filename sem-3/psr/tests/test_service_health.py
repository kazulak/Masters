from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_user_profile_health():
    module = load_module("user_profile_health", "services/user-profile/app/main.py")
    assert module.health()["service"] == "user-profile"


def test_book_catalog_health():
    module = load_module("book_catalog_health", "services/book-catalog/app/main.py")
    assert module.health()["service"] == "book-catalog"


def test_embedding_worker_health():
    module = load_module("embedding_worker_health", "services/embedding-worker/app/main.py")
    assert module.health()["service"] == "embedding-worker"


def test_recommendation_health():
    module = load_module("recommendation_health", "services/recommendation/app/main.py")
    assert module.health()["service"] == "recommendation"


def test_llm_service_health(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    module = load_module("llm_service_health", "services/llm-service/app/main.py")
    payload = module.health()
    assert payload["service"] == "llm-service"
    assert payload["provider"] == "deterministic"
