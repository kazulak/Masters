from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests


def test_streamlit_system_flow_screenshot(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    frontend_url = os.getenv("FRONTEND_URL", "http://127.0.0.1:8501")
    try:
        requests.get(frontend_url, timeout=3).raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Streamlit frontend is not running at {frontend_url}: {exc}")

    screenshot_path = tmp_path / "streamlit-system-flow.png"
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"Playwright Chromium is not installed or cannot launch: {exc}")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(frontend_url, wait_until="networkidle", timeout=30_000)
            page.get_by_text("System flow").click(timeout=10_000)
            page.get_by_text("Run demo scenario").wait_for(timeout=10_000)
            page.screenshot(path=screenshot_path, full_page=True)
        finally:
            browser.close()

    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 20_000
