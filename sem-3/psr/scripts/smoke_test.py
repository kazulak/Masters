#!/usr/bin/env python3
from __future__ import annotations

import time

import requests


def main() -> None:
    user_url = "http://127.0.0.1:8001"
    catalog_url = "http://127.0.0.1:8002"
    recommendation_url = "http://127.0.0.1:8004"
    headers = {"X-User-Id": "demo-user"}

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
        response = requests.get(
            f"{recommendation_url}/recommendations",
            params={"user_id": "demo-user", "type": "similar"},
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
