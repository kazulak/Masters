from __future__ import annotations

import importlib.util
from pathlib import Path

import psycopg
import pytest
import requests
from fastapi import HTTPException

from shared.open_library import search_open_library
from shared.repositories import list_books

ROOT = Path(__file__).resolve().parents[1]


def load_book_catalog_module():
    spec = importlib.util.spec_from_file_location("book_catalog_failure_paths", ROOT / "services/book-catalog/app/main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_open_library_timeout_is_propagated(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        raise requests.Timeout("open library timed out")

    monkeypatch.setattr("shared.open_library.requests.get", fake_get)

    with pytest.raises(requests.Timeout):
        search_open_library("earthsea")


def test_book_catalog_external_search_maps_open_library_failure_to_502(monkeypatch) -> None:
    module = load_book_catalog_module()

    def fail_search(*args, **kwargs):
        raise requests.Timeout("open library timed out")

    monkeypatch.setattr(module, "search_open_library", fail_search)

    with pytest.raises(HTTPException) as exc:
        module.external_search("earthsea")

    assert exc.value.status_code == 502
    assert "Open Library unavailable" in exc.value.detail


def test_postgres_connection_failure_is_not_silently_downgraded(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://book_ai:wrong@127.0.0.1:1/book_ai_library?connect_timeout=1",
    )

    with pytest.raises(psycopg.OperationalError):
        list_books()
