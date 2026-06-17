from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
SERVICE_APPS = {
    "user-profile": ROOT / "services/user-profile/app",
    "book-catalog": ROOT / "services/book-catalog/app",
    "embedding-worker": ROOT / "services/embedding-worker/app",
    "recommendation": ROOT / "services/recommendation/app",
    "llm-service": ROOT / "services/llm-service/app",
}


def free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError as exc:
        pytest.skip(f"local socket binding is blocked in this environment: {exc}")


@contextmanager
def uvicorn_service(
    service_name: str,
    tmp_path: Path,
    extra_env: dict[str, str] | None = None,
    app_state_file: Path | None = None,
) -> Iterator[str]:
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "APP_STATE_FILE": str(app_state_file or tmp_path / f"{service_name}.json"),
            "LLM_PROVIDER": "deterministic",
            "EVENT_BUS_PROVIDER": "local",
            "DATABASE_URL": "",
        }
    )
    if extra_env:
        env.update(extra_env)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--app-dir",
            str(SERVICE_APPS[service_name]),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=1)
                pytest.fail(f"{service_name} exited early\nstdout:\n{stdout}\nstderr:\n{stderr}")
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.ok:
                    break
                last_error = response.text
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(0.2)
        else:
            pytest.fail(f"{service_name} did not become healthy: {last_error}")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("service_name", sorted(SERVICE_APPS))
def test_service_health_over_real_http(service_name: str, tmp_path: Path) -> None:
    with uvicorn_service(service_name, tmp_path) as base_url:
        response = requests.get(f"{base_url}/health", timeout=3)

    assert response.status_code == 200
    assert response.json()["service"] == service_name


def test_llm_generate_real_http_get_and_post(tmp_path: Path) -> None:
    with uvicorn_service("llm-service", tmp_path) as base_url:
        get_response = requests.get(
            f"{base_url}/v1/generate",
            params={"prompt": "Recommend one book."},
            timeout=3,
        )
        post_response = requests.post(
            f"{base_url}/v1/generate",
            json={"text": "Recommend one book.", "context": {"title": "Dune"}},
            timeout=3,
        )

    assert get_response.status_code == 200
    assert get_response.json()["provider"] == "local-template"
    assert post_response.status_code == 200
    assert post_response.json()["text"].startswith("Recommended Dune")


def test_book_catalog_seed_and_list_real_http(tmp_path: Path) -> None:
    with uvicorn_service("book-catalog", tmp_path) as base_url:
        seed_response = requests.post(f"{base_url}/catalog/seed/demo", timeout=5)
        list_response = requests.get(f"{base_url}/books", timeout=3)
        book_id = list_response.json()[0]["id"]
        book_response = requests.get(f"{base_url}/books/{book_id}", timeout=3)
        cached_response = requests.get(
            f"{base_url}/books/{book_id}",
            headers={"If-None-Match": book_response.headers["ETag"]},
            timeout=3,
        )

    assert seed_response.status_code == 200
    assert seed_response.json()["total_catalog_size"] >= 10
    assert list_response.status_code == 200
    assert len(list_response.json()) == seed_response.json()["total_catalog_size"]
    assert book_response.status_code == 200
    assert book_response.headers["ETag"]
    assert cached_response.status_code == 304


def test_user_profile_add_book_calls_book_catalog_over_real_http(tmp_path: Path) -> None:
    state_file = tmp_path / "shared-state.json"
    with uvicorn_service("book-catalog", tmp_path, app_state_file=state_file) as catalog_url:
        with uvicorn_service(
            "user-profile",
            tmp_path,
            extra_env={"BOOK_CATALOG_URL": catalog_url},
            app_state_file=state_file,
        ) as user_url:
            add_response = requests.post(
                f"{user_url}/me/books",
                json={
                    "title": "Dune",
                    "author": "Frank Herbert",
                    "genres": ["science fiction"],
                    "description": "desert politics ecology",
                    "rating": 5,
                },
                headers={"X-User-Id": "http-user"},
                timeout=5,
            )
            reading_response = requests.get(
                f"{user_url}/me/books",
                headers={"X-User-Id": "http-user"},
                timeout=3,
            )
            catalog_response = requests.get(f"{catalog_url}/books", timeout=3)

    assert add_response.status_code == 200
    assert add_response.json()["book"]["title"] == "Dune"
    assert reading_response.status_code == 200
    assert [book["title"] for book in reading_response.json()["books"]] == ["Dune"]
    assert catalog_response.status_code == 200
    assert any(book["title"] == "Dune" for book in catalog_response.json())


def test_recommendation_pipeline_over_real_http_processes_async_boundary(tmp_path: Path) -> None:
    state_file = tmp_path / "pipeline-state.json"
    headers = {"X-User-Id": "http-pipeline-user"}
    with uvicorn_service("llm-service", tmp_path, app_state_file=state_file) as llm_url:
        with uvicorn_service("book-catalog", tmp_path, app_state_file=state_file) as catalog_url:
            with uvicorn_service(
                "user-profile",
                tmp_path,
                extra_env={"BOOK_CATALOG_URL": catalog_url},
                app_state_file=state_file,
            ) as user_url:
                with uvicorn_service(
                    "embedding-worker",
                    tmp_path,
                    extra_env={"LLM_SERVICE_URL": llm_url},
                    app_state_file=state_file,
                ) as embedding_url:
                    with uvicorn_service(
                        "recommendation",
                        tmp_path,
                        extra_env={"LLM_SERVICE_URL": llm_url},
                        app_state_file=state_file,
                    ) as recommendation_url:
                        seed_response = requests.post(f"{catalog_url}/catalog/seed/demo", timeout=5)
                        add_response = requests.post(
                            f"{user_url}/me/books",
                            json={
                                "title": "Dune",
                                "author": "Frank Herbert",
                                "genres": ["science fiction", "adventure"],
                                "description": "A desert planet, political intrigue, ecology, prophecy, and power.",
                                "rating": 5,
                            },
                            headers=headers,
                            timeout=5,
                        )
                        for _ in range(5):
                            requests.post(f"{embedding_url}/work", timeout=10).raise_for_status()
                            requests.post(f"{recommendation_url}/work", timeout=10).raise_for_status()
                        rec_response = requests.get(
                            f"{recommendation_url}/recommendations",
                            params={"user_id": "http-pipeline-user", "type": "similar"},
                            timeout=5,
                        )
                        ask_response = requests.post(
                            f"{recommendation_url}/recommendations/ask",
                            json={
                                "user_id": "http-pipeline-user",
                                "type": "similar",
                                "prompt": "I want a thoughtful science fiction recommendation.",
                            },
                            timeout=10,
                        )
                        summary_response = requests.post(
                            f"{recommendation_url}/profile/summary",
                            json={"user_id": "http-pipeline-user"},
                            timeout=10,
                        )
                        embedding_status = requests.get(f"{embedding_url}/status", timeout=3)
                        recommendation_status = requests.get(f"{recommendation_url}/status", timeout=3)

    assert seed_response.status_code == 200
    assert add_response.status_code == 200
    assert rec_response.status_code == 200
    payload = rec_response.json()
    assert payload["filter_summary"]["resolved_count"] >= len(payload["books"])
    assert "owned_filtered_count" in payload["filter_summary"]
    assert payload["books"]
    assert ask_response.status_code == 200
    ask_payload = ask_response.json()
    assert ask_payload["provider"] == "local-template"
    assert ask_payload["source"] == "llm-over-engine-candidates"
    assert ask_payload["engine"] == "vector-similarity"
    assert ask_payload["allow_outside_candidates"] is True
    assert ask_payload["books"]
    assert ask_payload["candidates"]
    assert summary_response.status_code == 200
    assert summary_response.json()["summary"]
    assert embedding_status.json()["worker"]["last_success_at"]
    assert recommendation_status.json()["worker"]["last_success_at"]


def test_embedding_worker_reports_llm_dependency_failure_over_real_http(tmp_path: Path) -> None:
    state_file = tmp_path / "llm-failure-state.json"
    with uvicorn_service("book-catalog", tmp_path, app_state_file=state_file) as catalog_url:
        seed_response = requests.post(f"{catalog_url}/catalog/seed/demo", timeout=5)
        with uvicorn_service(
            "embedding-worker",
            tmp_path,
            extra_env={"LLM_SERVICE_URL": "http://127.0.0.1:9"},
            app_state_file=state_file,
        ) as embedding_url:
            work_response = requests.post(f"{embedding_url}/work", timeout=10)
            status_response = requests.get(f"{embedding_url}/status", timeout=3)

    assert seed_response.status_code == 200
    assert work_response.status_code in {200, 500}
    status_payload = status_response.json()
    assert status_payload["worker"]["total_failures"] >= 1
    assert status_payload["worker"]["last_error"]
