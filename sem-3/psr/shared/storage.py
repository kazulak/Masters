from __future__ import annotations

import json
import threading
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from shared.config import state_file

_LOCK = threading.Lock()


def _empty_state() -> dict[str, Any]:
    return {
        "users": {},
        "books": {},
        "reading_list": {},
        "book_embeddings": {},
        "recommendations": {},
        "events": [],
    }


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(_empty_state(), indent=2), encoding="utf-8")


def read_state() -> dict[str, Any]:
    path = state_file()
    with _LOCK:
        _ensure_file(path)
        return json.loads(path.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    path = state_file()
    with _LOCK:
        _ensure_file(path)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def update_state(mutator: Callable[[dict[str, Any]], Any]) -> Any:
    path = state_file()
    with _LOCK:
        _ensure_file(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        result = mutator(state)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        return deepcopy(result)


def reset_state() -> None:
    write_state(_empty_state())
