from __future__ import annotations

import time
import os

import requests


def env_url(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


URLS = [
    f"{env_url('USER_PROFILE_URL', 'http://127.0.0.1:8001')}/health",
    f"{env_url('BOOK_CATALOG_URL', 'http://127.0.0.1:8002')}/health",
    f"{env_url('EMBEDDING_WORKER_URL', 'http://127.0.0.1:8003')}/health",
    f"{env_url('RECOMMENDATION_URL', 'http://127.0.0.1:8004')}/health",
    f"{env_url('LLM_SERVICE_URL', 'http://127.0.0.1:8005')}/health",
]


def main() -> None:
    deadline = time.time() + 20
    pending = set(URLS)
    while pending and time.time() < deadline:
        for url in list(pending):
            try:
                response = requests.get(url, timeout=2)
                if response.ok:
                    pending.remove(url)
            except requests.RequestException:
                pass
        time.sleep(0.25)
    if pending:
        raise SystemExit(f"services did not become healthy: {sorted(pending)}")


if __name__ == "__main__":
    main()
