from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
import requests


def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL", "http://127.0.0.1:8501")


def _require_frontend() -> str:
    frontend_url = _frontend_url()
    try:
        requests.get(frontend_url, timeout=3).raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Streamlit frontend is not running at {frontend_url}: {exc}")
    return frontend_url


def _browser_page(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    return browser, page


def _login_if_needed(page) -> None:
    if page.get_by_text("Book AI Library").is_visible(timeout=3_000):
        page.get_by_label("Create account").check(timeout=10_000)
        suffix = uuid4().hex[:8]
        page.get_by_label("Display name").fill(f"Browser User {suffix}")
        page.get_by_label("Email").last.fill(f"browser-{suffix}@example.edu")
        page.get_by_label("Password").last.fill("demo")
        page.get_by_role("button", name="Create account").click(timeout=10_000)
        page.get_by_text("Your reading profile").wait_for(timeout=30_000)


def test_streamlit_system_flow_screenshot(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    frontend_url = _require_frontend()

    screenshot_path = tmp_path / "streamlit-system-flow.png"
    with playwright.sync_playwright() as p:
        try:
            browser, page = _browser_page(p)
        except Exception as exc:
            pytest.skip(f"Playwright Chromium is not installed or cannot launch: {exc}")
        try:
            page.goto(frontend_url, wait_until="networkidle", timeout=30_000)
            _login_if_needed(page)
            page.get_by_text("Architecture").click(timeout=10_000)
            page.get_by_text("Run demo scenario").wait_for(timeout=10_000)
            page.get_by_text("Live microservice architecture").wait_for(timeout=10_000)
            page.get_by_text("LLM route").wait_for(timeout=10_000)
            page.screenshot(path=screenshot_path, full_page=True)
        finally:
            browser.close()

    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 20_000


def test_streamlit_recommendations_demo_screenshot(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    frontend_url = _require_frontend()

    screenshot_path = tmp_path / "streamlit-recommendations-demo.png"
    with playwright.sync_playwright() as p:
        try:
            browser, page = _browser_page(p)
        except Exception as exc:
            pytest.skip(f"Playwright Chromium is not installed or cannot launch: {exc}")
        try:
            page.goto(frontend_url, wait_until="networkidle", timeout=30_000)
            _login_if_needed(page)
            page.get_by_text("Architecture").click(timeout=10_000)
            page.get_by_role("button", name="Run demo scenario").click(timeout=10_000)
            page.get_by_text("Demo scenario completed").wait_for(timeout=60_000)
            page.get_by_text("Recommendations").click(timeout=10_000)
            page.get_by_text("Book suggestions").wait_for(timeout=10_000)
            page.get_by_text("Similar").wait_for(timeout=10_000)
            page.get_by_text("Widen").wait_for(timeout=10_000)
            page.get_by_text("Mood").wait_for(timeout=10_000)
            page.get_by_text("Already in library hidden").wait_for(timeout=10_000)
            page.screenshot(path=screenshot_path, full_page=True)
        finally:
            browser.close()

    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 20_000
