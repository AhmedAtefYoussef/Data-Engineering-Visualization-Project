"""App smoke tests: layout, callbacks as plain functions, search parser, missing-data import."""
from __future__ import annotations

import pytest

pytest.skip("These tests cover the previous website, which read pipeline artifacts. "
            "The current site is tested in tests/test_site.py.", allow_module_level=True)  # superseded

import importlib
import os
import subprocess
import sys

import plotly.graph_objects as go
import pytest

from tests._art import ROOT, get_art  # noqa: E402  (must precede config import: sets NYC_PROCESSED_DIR)
import config
from dash import no_update

from app.callbacks import HINT_SET, apply_search, build_report, filters_from_inputs
from app.search import parse_query

@pytest.fixture(scope="module")
def art():
    return get_art()


EXTREME = dict(boroughs=["Staten Island"], years=(2016, 2016), construction=["Other"], factor=["Vehicle/Road Defect"], severity=["Fatal"], hours=["Night"], ped_cyc=True)


@pytest.mark.parametrize("text,expect", [
    ("Brooklyn 2022 new buildings", dict(boroughs=["Brooklyn"], years=(2022, 2022), construction=["New Construction"])),
    ("pedestrian crashes in Queens since 2020", dict(boroughs=["Queens"], years=(2020, config.Y_END), ped_cyc=True)),
    ("fatal 2019-2022 manhattan", dict(boroughs=["Manhattan"], years=(2019, 2022), severity=["Fatal"])),
    ("SI preservation before 2021", dict(boroughs=["Staten Island"], years=(config.Y_START, 2020), construction=["Preservation"])),
    ("speeding at night in the bronx", dict(boroughs=["Bronx"], factor=["Speed/Aggressive"], hours=["Night"])),
])
def test_parse_query(text, expect):
    assert parse_query(text)["filters"] == expect


def test_parse_gibberish():
    q = parse_query("qwerty zzzz")
    assert q["filters"] == {} and q["unmatched"] == ["qwerty", "zzzz"] and q["tokens"] == []
    assert parse_query("")["filters"] == {}


def test_layout_builds(art):
    from app.layout import build_layout
    lay = build_layout(art)
    assert lay is not None and "g-rq10" in str(lay)


@pytest.mark.parametrize("filters", [{}, EXTREME, dict(boroughs=["Bronx"], years=(2020, 2022))])
def test_report_callback(art, filters):
    figs, caps, kpi, table, status = build_report(art, filters_from_inputs(*[filters.get(k) for k in ("boroughs", "years", "construction", "factor", "severity", "hours", "ped_cyc")]))
    assert len(figs) == 10 and all(isinstance(f, go.Figure) for f in figs)
    assert len(caps) == 10 and all(c.endswith("This is an association, not proof of cause.") for c in caps)
    for f in figs:
        f.to_plotly_json()


def test_apply_search_sets_controls_and_hint_only():
    out = apply_search("Brooklyn 2022 new buildings")
    assert out[0] == ["Brooklyn"] and out[1] == [2022, 2022] and out[2] == ["New Construction"]
    assert len(out) == 9 and out[-1] == HINT_SET == "Filters set. Press Generate report to update the charts."
    assert len(out[7]) == 3                       # three understood-token chips


def test_apply_search_nothing_understood_keeps_controls():
    out = apply_search("qwerty")
    assert all(v is no_update for v in out[:7]) and "unchanged" in out[-1]


def test_import_with_missing_dir(tmp_path):
    env = {**os.environ, "NYC_PROCESSED_DIR": str(tmp_path / "nope"), "PYTHONPATH": str(ROOT)}
    code = "import app.app as a; l=a.app.layout(); assert 'Processed data not found' in str(l); print('ok')"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_dark_theme_and_reset(art):
    from analysis.rq import RQS
    from app.callbacks import reset_values
    from app.charts import TEMPLATES, apply_theme
    fig = RQS[2]["fig"](art, {}, 20)
    dark = apply_theme(fig, True)
    assert dark.layout.template == TEMPLATES[True] and fig.layout.template == TEMPLATES[False]
    assert dark.layout.meta["note"] == fig.layout.meta["note"]
    out = reset_values()
    assert out[-1] == HINT_SET and out[0] == [] and len(out) == 9


def test_ten_questions_only():
    from analysis.rq import RQS
    assert len(RQS) == config.N_RQ == 10


def test_report_callback_has_no_search_or_reset_inputs(art):
    """Search and Reset only set controls: the report callback listens to Generate report and the theme, nothing else."""
    import dash
    from app.callbacks import register
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register(app, art)
    report = next(c for k, c in app.callback_map.items() if "g-rq1.figure" in k)
    ids = [i["id"] for i in report["inputs"]]
    assert ids == ["generate", "theme"], ids
    controls = next(c for k, c in app.callback_map.items() if "rail-hint.children" in k)
    assert {"search-apply", "search-text", "reset-filters"} <= {i["id"] for i in controls["inputs"]}


def test_layout_has_skip_link_roving_tabs_and_banners(art):
    from app.layout import build_layout
    text = str(build_layout(art))
    assert "Skip to content" in text and "tabIndex=-1" in text and "rail-hint" in text
    assert "reset-empty" not in text and "search-trigger" not in text
    assert "Partial download" not in text            # complete run: no partial banner


def test_partial_banner_is_separate_from_fixture_banner(art):
    import copy
    from app.layout import build_layout
    other = copy.copy(art)
    other.__dict__["meta"] = {**art.meta, "partial": True, "max_rows": 20000, "fixture": False}
    text = str(build_layout(other))
    assert "Partial download (20,000 rows): code check only, not findings." in text and "Fixture data" not in text
    other.__dict__["meta"] = {**art.meta, "partial": False, "fixture": True}
    text = str(build_layout(other))
    assert "Fixture data" in text and "Partial download" not in text
