from __future__ import annotations

import time

import requests

URLS = [
    "http://127.0.0.1:8001/health",
    "http://127.0.0.1:8002/health",
    "http://127.0.0.1:8003/health",
    "http://127.0.0.1:8004/health",
    "http://127.0.0.1:8005/health",
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
