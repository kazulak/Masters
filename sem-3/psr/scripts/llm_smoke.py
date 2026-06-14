#!/usr/bin/env python3
from __future__ import annotations

import os

import requests


def main() -> None:
    base_url = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8005").rstrip("/")

    health = requests.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()

    embed = requests.post(f"{base_url}/v1/embed", json={"text": "Dune"}, timeout=300)
    embed.raise_for_status()
    embed_payload = embed.json()
    if not embed_payload.get("embedding"):
        raise SystemExit("LLM embedding response did not contain a vector")

    generate = requests.post(
        f"{base_url}/v1/generate",
        json={"prompt": "Recommend one concise science fiction book."},
        timeout=300,
    )
    generate.raise_for_status()
    text = generate.json().get("text", "").strip()
    if not text:
        raise SystemExit("LLM generate response was empty")

    print(
        "ok: llm-service "
        f"provider={health.json().get('provider')} "
        f"embedding_dims={embed_payload.get('dimensions')} "
        f"generated_chars={len(text)}"
    )


if __name__ == "__main__":
    main()
