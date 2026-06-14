#!/usr/bin/env python3
from __future__ import annotations

import os
import time

import requests


def env_url(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


def require_ok(response: requests.Response) -> requests.Response:
    response.raise_for_status()
    return response


def main() -> None:
    frontend_url = env_url("FRONTEND_URL", "http://127.0.0.1:8501")
    user_url = env_url("USER_PROFILE_URL", "http://127.0.0.1:8001")
    catalog_url = env_url("BOOK_CATALOG_URL", "http://127.0.0.1:8002")
    embedding_url = env_url("EMBEDDING_WORKER_URL", "http://127.0.0.1:8003")
    recommendation_url = env_url("RECOMMENDATION_URL", "http://127.0.0.1:8004")
    llm_url = env_url("LLM_SERVICE_URL", "http://127.0.0.1:8005")
    user_id = os.getenv("DEFAULT_USER_ID", "demo-user")
    headers = {"X-User-Id": user_id}

    require_ok(requests.get(frontend_url, timeout=10))
    for url in (user_url, catalog_url, embedding_url, recommendation_url, llm_url):
        require_ok(requests.get(f"{url}/health", timeout=10))

    seed = require_ok(requests.post(f"{catalog_url}/catalog/seed/demo", timeout=20)).json()
    if seed["total_catalog_size"] < 10:
        raise SystemExit(f"demo catalog too small: {seed}")

    added = require_ok(
        requests.post(
            f"{user_url}/me/books",
            json={
                "title": "Dune",
                "author": "Frank Herbert",
                "genres": ["science fiction", "adventure"],
                "description": "A desert planet, political intrigue, ecology, prophecy, and power.",
                "rating": 5,
            },
            headers=headers,
            timeout=20,
        )
    ).json()
    added_book_id = added["book"]["id"]

    deadline = time.time() + int(os.getenv("LOCAL_APP_SMOKE_TIMEOUT_SECONDS", "90"))
    last_payload = None
    while time.time() < deadline:
        requests.post(f"{embedding_url}/work", timeout=60).raise_for_status()
        requests.post(f"{recommendation_url}/work", timeout=30).raise_for_status()
        response = require_ok(
            requests.get(
                f"{recommendation_url}/recommendations",
                params={"user_id": user_id, "type": "similar"},
                timeout=10,
            )
        )
        last_payload = response.json()
        if last_payload["books"]:
            recommended_ids = {book["id"] for book in last_payload["books"]}
            if added_book_id in recommended_ids:
                raise SystemExit(f"recommendations included the just-added book {added_book_id}: {last_payload}")
            print(
                "ok: local app "
                f"demo_total={seed['total_catalog_size']} "
                f"recommendations={len(last_payload['books'])}"
            )
            return
        time.sleep(2)

    raise SystemExit(f"recommendations were not produced before timeout: {last_payload}")


if __name__ == "__main__":
    main()
