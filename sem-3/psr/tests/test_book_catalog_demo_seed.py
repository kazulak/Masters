from __future__ import annotations

import importlib.util
from pathlib import Path

from shared.storage import read_state, reset_state


ROOT = Path(__file__).resolve().parents[1]


def load_book_catalog_module():
    spec = importlib.util.spec_from_file_location("book_catalog_demo_seed", ROOT / "services/book-catalog/app/main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_seed_imports_books_and_is_idempotent(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APP_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_state()
    module = load_book_catalog_module()

    first = module.seed_demo_catalog()
    second = module.seed_demo_catalog()
    state = read_state()

    assert first.source == "demo"
    assert first.imported >= 10
    assert second.imported == 0
    assert second.existing == first.imported
    assert len(state["events"]) == first.imported
    assert all(event["type"] == "BookCreated" for event in state["events"])
