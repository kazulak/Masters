#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import requests


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: scripts/llm_prompt.py "your prompt"')

    base_url = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8005").rstrip("/")
    prompt = " ".join(sys.argv[1:])
    response = requests.post(f"{base_url}/v1/generate", json={"prompt": prompt}, timeout=120)
    response.raise_for_status()
    print(response.json()["text"])


if __name__ == "__main__":
    main()
