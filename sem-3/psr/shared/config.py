from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def state_file() -> Path:
    return Path(env("APP_STATE_FILE", ".local/app_state.json"))


def embedding_dim() -> int:
    return int(env("EMBEDDING_DIM", "64"))


BOOK_CATALOG_URL = env("BOOK_CATALOG_URL", "http://127.0.0.1:8002")
LLM_SERVICE_URL = env("LLM_SERVICE_URL", "http://127.0.0.1:8005")
USER_PROFILE_URL = env("USER_PROFILE_URL", "http://127.0.0.1:8001")
RECOMMENDATION_URL = env("RECOMMENDATION_URL", "http://127.0.0.1:8004")
DEFAULT_USER_ID = env("DEFAULT_USER_ID", "demo-user")
