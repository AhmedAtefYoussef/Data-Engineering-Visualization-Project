"""Real-browser checks (playwright + chromium + gunicorn): search, reset, Generate-only updates and keyboard access.

Skipped automatically when playwright, a chromium binary or gunicorn is missing."""
from __future__ import annotations

import pytest

pytest.skip("These tests cover the previous website in a browser, which read pipeline artifacts. "
            "The current site is tested in tests/test_site.py.", allow_module_level=True)  # superseded

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tests._art import ROOT, get_art

playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
if shutil.which("gunicorn") is None and subprocess.run([sys.executable, "-c", "import gunicorn"], capture_output=True).returncode:
    pytest.skip("gunicorn is not available", allow_module_level=True)

CHROMIUM_CANDIDATES = [os.environ.get("CHROMIUM_PATH", ""), "/opt/pw-browsers/chromium", shutil.which("chromium") or "", shutil.which("chromium-browser") or ""]
DEFAULT_YEARS = ("2016", "2025")


@pytest.fixture(scope="module")
def server():
    art = get_art()
    if not art.available:
        pytest.skip("processed artifacts are not available")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "NYC_PROCESSED_DIR": str(art.dir), "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-m", "gunicorn", "app.app:server", "--workers", "1", "--threads", "4", "--preload", "--timeout", "120",
                             "-b", f"127.0.0.1:{port}"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/"
    try:
        for _ in range(120):
            if proc.poll() is not None:
                pytest.skip("gunicorn did not start")
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.skip("gunicorn did not become ready")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as pw:
        last = None
        for path in CHROMIUM_CANDIDATES:
            if path and not Path(path).exists():
                continue
            try:
                br = pw.chromium.launch(executable_path=path or None, args=["--no-sandbox"])
                break
            except Exception as exc:                      # no usable chromium here
                last = exc
        else:
            pytest.skip(f"chromium is not available ({last})")
        yield br
        br.close()


@pytest.fixture()
def page(server, browser):
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.set_default_timeout(30000)
    pg.goto(server)
    pg.click("#tab-explore")
    pg.wait_for_selector("#g-rq1 .js-plotly-plot")
    pg.wait_for_function("document.querySelector('#kpis .kpi')")
    yield pg
    pg.close()


def _text(page, sel: str) -> str:
    return " ".join(page.inner_text(sel).split())


def _years(page) -> tuple[str, str]:
    handles = page.locator("#f-years [role=slider]")
    return handles.nth(0).get_attribute("aria-valuenow"), handles.nth(1).get_attribute("aria-valuenow")


def _figure_data(page) -> str:
    return page.evaluate("JSON.stringify(Array.from(document.querySelectorAll('.js-plotly-plot')).map(g => g.data))")


def _is_default(page) -> bool:
    return (_text(page, "#f-borough").replace("×", "").strip() == "All" and _years(page) == DEFAULT_YEARS
            and _text(page, "#f-construction").replace("×", "").strip() == "All")


def test_search_sets_controls_reset_restores_and_generate_updates(page):
    before = _figure_data(page)
    page.fill("#search-text", "Brooklyn 2022 new buildings")
    page.click("#search-apply")
    page.wait_for_function("document.querySelector('#f-years [role=slider]').getAttribute('aria-valuenow') === '2022'")
    assert "Brooklyn" in _text(page, "#f-borough") and "New Construction" in _text(page, "#f-construction")
    assert _years(page) == ("2022", "2022")
    assert "Filters set. Press Generate report to update the charts." in _text(page, "#rail-hint")
    assert "Borough: Brooklyn" in _text(page, "#search-chips")
    page.wait_for_timeout(1500)
    assert _figure_data(page) == before, "Apply search must not rebuild the report"

    page.click("#reset-filters")
    page.wait_for_function("document.querySelector('#f-years [role=slider]').getAttribute('aria-valuenow') === '2016'")
    assert _is_default(page) and _text(page, "#search-chips") == "" and page.input_value("#search-text") == ""
    assert "Filters set. Press Generate report" in _text(page, "#rail-hint")
    page.wait_for_timeout(1500)
    assert _figure_data(page) == before, "Reset must not rebuild the report"

    # A dropdown change alone leaves the figures untouched until Generate report is pressed.
    page.click("#f-borough")
    page.locator("#f-borough .Select-menu-outer").get_by_text("Bronx", exact=True).click()
    page.wait_for_timeout(1500)
    assert _figure_data(page) == before
    page.click("#generate")
    page.wait_for_function("document.querySelector('#report-status')?.innerText.includes('Bronx')")
    page.wait_for_timeout(1500)
    assert _figure_data(page) != before


def test_enter_in_search_box_acts_like_apply(page):
    page.fill("#search-text", "Queens since 2020")
    page.press("#search-text", "Enter")
    page.wait_for_function("document.querySelector('#f-years [role=slider]').getAttribute('aria-valuenow') === '2020'")
    assert "Queens" in _text(page, "#f-borough")


def test_empty_state_reset_button_works(page):
    page.fill("#search-text", "fatal 2016 staten island")
    page.press("#search-text", "Enter")
    page.wait_for_function("document.querySelector('#f-years [role=slider]').getAttribute('aria-valuenow') === '2016'")
    page.click("#f-construction")
    page.locator("#f-construction .Select-menu-outer").get_by_text("Other", exact=True).click()
    page.click("#generate")
    try:
        page.wait_for_selector("#report-status button", timeout=15000)
    except Exception:
        pytest.skip("these filters did not empty the fixture, so there is no empty-state button to test")
    page.click("#report-status button")
    page.wait_for_function("document.querySelector('#f-years [role=slider]').getAttribute('aria-valuenow') === '2016' && "
                           "document.querySelectorAll('#f-years [role=slider]')[1].getAttribute('aria-valuenow') === '2025'")
    assert _is_default(page)


def test_skip_link_and_tab_keyboard_navigation(server, browser):
    pg = browser.new_page()
    pg.goto(server)
    pg.wait_for_selector("#tab-story")
    pg.keyboard.press("Tab")
    assert pg.evaluate("document.activeElement.textContent") == "Skip to content"
    assert pg.evaluate("document.activeElement.getBoundingClientRect().top") >= 0
    pg.keyboard.press("Enter")
    assert pg.evaluate("location.hash") == "#main"
    # roving tabindex: only the selected tab is tabbable
    assert [pg.get_attribute(f"#tab-{t}", "tabindex") for t in ("story", "explore", "method")] == ["0", "-1", "-1"]
    pg.focus("#tab-story")
    pg.keyboard.press("ArrowRight")
    pg.wait_for_function("document.activeElement.id === 'tab-explore' && document.getElementById('tab-explore').getAttribute('aria-selected') === 'true'")
    assert [pg.get_attribute(f"#tab-{t}", "tabindex") for t in ("story", "explore", "method")] == ["-1", "0", "-1"]
    pg.keyboard.press("End")
    pg.wait_for_function("document.activeElement.id === 'tab-method' && document.getElementById('tab-method').getAttribute('aria-selected') === 'true'")
    pg.keyboard.press("ArrowRight")                                     # wraps to the first tab
    pg.wait_for_function("document.activeElement.id === 'tab-story'")
    pg.keyboard.press("Home")
    assert pg.evaluate("document.activeElement.id") == "tab-story"
    pg.keyboard.press("ArrowLeft")                                      # wraps to the last tab
    pg.wait_for_function("document.activeElement.id === 'tab-method'")
    outline = pg.evaluate("getComputedStyle(document.activeElement).outlineStyle + ' ' + getComputedStyle(document.activeElement).outlineWidth")
    assert outline.startswith("solid") and outline.endswith("2px")
    pg.close()
