#!/usr/bin/env python3
from __future__ import annotations

import time
import os

import requests


def env_url(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


def main() -> None:
    user_url = env_url("USER_PROFILE_URL", "http://127.0.0.1:8001")
    catalog_url = env_url("BOOK_CATALOG_URL", "http://127.0.0.1:8002")
    embedding_url = env_url("EMBEDDING_WORKER_URL", "http://127.0.0.1:8003")
    recommendation_url = env_url("RECOMMENDATION_URL", "http://127.0.0.1:8004")
    user_id = os.getenv("DEFAULT_USER_ID", "demo-user")
    headers = {"X-User-Id": user_id}

    reading_books = [
        {
            "title": "Dune",
            "author": "Frank Herbert",
            "genres": ["science fiction", "adventure"],
            "description": "Desert planet politics and prophecy.",
            "rating": 5,
        },
        {
            "title": "The Left Hand of Darkness",
            "author": "Ursula K. Le Guin",
            "genres": ["science fiction", "social"],
            "description": "An envoy explores culture, identity, and politics.",
            "rating": 4,
        },
    ]

    for book in reading_books:
        response = requests.post(f"{user_url}/me/books", json=book, headers=headers, timeout=15)
        response.raise_for_status()

    response = requests.post(
        f"{catalog_url}/books",
        json={
            "title": "Foundation",
            "author": "Isaac Asimov",
            "genres": ["science fiction", "strategy"],
            "description": "A long-range plan to reduce a galactic dark age.",
        },
        timeout=15,
    )
    response.raise_for_status()

    deadline = time.time() + 20
    while time.time() < deadline:
        requests.post(f"{embedding_url}/work", timeout=15).raise_for_status()
        requests.post(f"{recommendation_url}/work", timeout=15).raise_for_status()
        response = requests.get(
            f"{recommendation_url}/recommendations",
            params={"user_id": user_id, "type": "similar"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if payload["books"]:
            print(f"ok: {len(payload['books'])} recommendations returned")
            return
        time.sleep(2)

    raise SystemExit("recommendations were not produced before timeout")


if __name__ == "__main__":
    main()
