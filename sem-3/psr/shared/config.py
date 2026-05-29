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
OPEN_LIBRARY_BASE_URL = env("OPEN_LIBRARY_BASE_URL", "https://openlibrary.org")
OPEN_LIBRARY_TIMEOUT_SECONDS = float(env("OPEN_LIBRARY_TIMEOUT_SECONDS", "8"))
DATABASE_URL = os.getenv("DATABASE_URL")
LLM_PROVIDER = env("LLM_PROVIDER", "deterministic")
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = env("OLLAMA_EMBED_MODEL", "embeddinggemma")
OLLAMA_GENERATE_MODEL = env("OLLAMA_GENERATE_MODEL", "gemma3:1b")
OLLAMA_TIMEOUT_SECONDS = float(env("OLLAMA_TIMEOUT_SECONDS", "60"))
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = env("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_EMBED_DEPLOYMENT = env("AZURE_OPENAI_EMBED_DEPLOYMENT", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
