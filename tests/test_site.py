"""Tests for the website: the search parser, the data layer, the report and the Dash callbacks.

    python -m pytest tests/test_site.py -q

The callback tests post to Dash's own endpoint through the Flask test client, with the same
payload a browser sends, so they exercise routing, callbacks and JSON serialisation together.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from app import data
from app.callbacks import CONTROL_IDS, control_values
from app.filters import BOROUGHS, FACTOR_GROUPS, HOUR_BANDS, YEAR_END, YEAR_START, from_dict, make_filters
from app.report import CHART_KEYS, build_report
from app.search import BOROUGH_WORDS, CONSTRUCTION_WORDS, FACTOR_WORDS, HOUR_WORDS, SIZE_WORDS, parse_query


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def store():
    return data.load()


# ---------------------------------------------------------------------------------------------
# Search parser (pure functions, no data)
# ---------------------------------------------------------------------------------------------
def test_assignment_example_query():
    parsed = parse_query("Brooklyn 2022 new construction")
    assert parsed.values == {"year_from": 2022, "year_to": 2022, "boroughs": ["Brooklyn"],
                             "construction": ["New Construction"]}
    assert parsed.unmatched == []


def test_second_example_query():
    parsed = parse_query("bronx pedestrians night 2019-2021")
    assert parsed.values == {"year_from": 2019, "year_to": 2021, "boroughs": ["Bronx"],
                             "hours": ["Night"], "harm": "ped"}
    assert [label for _, label, _ in parsed.understood] == ["Years", "Borough", "Time of day", "Who was harmed"]


@pytest.mark.parametrize("query, expected", [
    ("staten island fatal since 2021", {"boroughs": ["Staten Island"], "harm": "fatal", "year_from": 2021,
                                        "year_to": YEAR_END}),
    ("queens and the bronx cyclists speeding", {"boroughs": ["Bronx", "Queens"], "harm": "cyc",
                                                "factors": ["Speed or aggressive driving"]}),
    ("pedestrians and cyclists before 2020", {"harm": "pedcyc", "year_from": YEAR_START, "year_to": 2019}),
    ("manhattan large buildings near housing", {"boroughs": ["Manhattan"], "sizes": ["50-149", "150+"],
                                               "scope": "near"}),
    ("preservation evening distracted driving", {"construction": ["Preservation"], "hours": ["Evening"],
                                                 "factors": ["Distraction"]}),
    ("all crashes 2016 to 2018", {"harm": "all", "year_from": 2016, "year_to": 2018}),
])
def test_queries(query, expected):
    assert parse_query(query).values == expected


def test_corrected_typo_is_shown_in_the_understood_chip():
    parsed = parse_query("Brookyln 2022")
    assert ("brookyln", "Borough", "Brooklyn (from brookyln)") in parsed.understood
    clamped = parse_query("bronx 1999-2030")
    assert "data covers 2016 to 2025" in clamped.understood[0][2]


def test_unknown_words_are_reported_and_change_nothing():
    parsed = parse_query("brooklyn unicorns 2022")
    assert parsed.unmatched == ["unicorns"]
    assert parsed.values == {"year_from": 2022, "year_to": 2022, "boroughs": ["Brooklyn"]}
    nothing = parse_query("zzz qqq")
    assert nothing.values == {} and nothing.understood == [] and nothing.unmatched == ["zzz", "qqq"]


def test_year_outside_the_data_sets_nothing():
    assert "year_from" not in parse_query("brooklyn 1999").values
    assert parse_query("").is_empty and parse_query(None).values == {}


def test_parser_vocabulary_maps_to_real_filter_values():
    assert set(BOROUGH_WORDS.values()) == set(BOROUGHS)
    assert set(FACTOR_WORDS.values()) <= set(FACTOR_GROUPS)
    assert set(HOUR_WORDS.values()) <= set(HOUR_BANDS)
    assert set(CONSTRUCTION_WORDS.values()) == {"New Construction", "Preservation"}
    assert all(make_filters(sizes=list(bands)).sizes == bands for bands in SIZE_WORDS.values())


# ---------------------------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------------------------
def test_filters_are_forgiving_and_round_trip():
    filters = make_filters(boroughs=["Brooklyn", "Atlantis"], year_from=2030, year_to="2019", harm="nonsense")
    assert filters.boroughs == ("Brooklyn",) and (filters.year_from, filters.year_to) == (2019, YEAR_END)
    assert filters.harm == "injury"
    assert from_dict(filters.to_dict()) == filters
    assert make_filters(boroughs=["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]).boroughs == ()
    assert control_values(make_filters()) == [[], YEAR_START, YEAR_END, [], [], [], [], "injury", "city"]


# ---------------------------------------------------------------------------------------------
# Data layer: the exported tables agree with each other and with the notebook's totals
# ---------------------------------------------------------------------------------------------
def test_cubes_agree_on_totals(store):
    rows = store.meta["rows"]
    assert int(store.crash.counts["n_all"].sum()) == rows["crashes"]
    assert int(store.hour.counts["n_all"].sum()) == rows["crashes"]
    assert int(store.crash.counts["n_injury"].sum()) == rows["injury_crashes"]
    assert int(store.grid.counts["n_all"].sum()) == rows["crashes_located"]
    assert int(store.zips.counts["n_injury"].sum()) == int(store.zip_year["injury_crashes"].sum())


def test_default_selection_matches_the_notebook(store):
    filters = make_filters()
    numbers = data.headline_numbers(store, filters)
    assert numbers["crashes"] == store.meta["rows"]["injury_crashes"]
    assert int(data.weekday_hour(store, filters).sum()) == numbers["crashes"]
    assert int(data.factor_mix(store, filters).sum()) == numbers["crashes"]
    yearly = data.crash_trend(store, filters).sum(axis=1) + 0
    assert int(yearly.sum()) == numbers["crashes"] - numbers["crashes_unknown_borough"]

    questions, rq7 = store.results["questions"], store.results["detail"]["rq7"]
    new_only = make_filters(construction=["New Construction"])
    # RQ7 is headlined on new construction; with both types the site gives the notebook's pooled row.
    for selection, expected in [(data.site_selection(store, new_only), questions["RQ7"]),
                                (data.site_selection(store, filters), rq7["pooled"])]:
        assert selection["excess"] == pytest.approx(expected["estimate"], abs=1e-6)
        assert selection["low"] == pytest.approx(expected["low"], abs=1e-6)
        assert selection["high"] == pytest.approx(expected["high"], abs=1e-6)
        assert selection["n"] == expected["n"]
    did = data.event_study(store, new_only)["treated"]["New Construction"]["did"]
    assert did["percent_change"] == pytest.approx(questions["RQ3"]["estimate"], abs=1e-9)
    assert did["percent_low"] == pytest.approx(questions["RQ3"]["low"], abs=1e-9)
    assert did["percent_high"] == pytest.approx(questions["RQ3"]["high"], abs=1e-9)
    both = data.event_study(store, filters)
    assert both["difference"]["percent_change"] == pytest.approx(questions["RQ5"]["estimate"], abs=1e-9)


def test_filters_narrow_every_table_consistently(store):
    filters = make_filters(boroughs=["Brooklyn"], year_from=2022, year_to=2022, harm="ped", hours=["Night"])
    numbers = data.headline_numbers(store, filters)
    assert 0 < numbers["crashes"] < store.meta["rows"]["injury_crashes"]
    assert int(data.weekday_hour(store, filters).sum()) == numbers["crashes"]
    assert int(data.weekday_hour(store, filters)[:, 6:].sum()) == 0          # night is 00:00 to 05:59
    trend = data.crash_trend(store, filters)
    assert list(trend.columns) == ["Brooklyn"] and len(trend) == 12 and trend.attrs["monthly"]
    near = data.headline_numbers(store, make_filters(boroughs=["Brooklyn"], scope="near"))
    assert near["crashes"] == near["crashes_near"] <= near["crashes_anywhere"]


# ---------------------------------------------------------------------------------------------
# Report: every selection yields eight figures, including selections with no data
# ---------------------------------------------------------------------------------------------
SELECTIONS = {
    "default": {},
    "brooklyn 2022 new construction": {"boroughs": ["Brooklyn"], "year_from": 2022, "year_to": 2022,
                                       "construction": ["New Construction"]},
    "bronx pedestrians night": {"boroughs": ["Bronx"], "year_from": 2019, "year_to": 2021, "harm": "ped",
                                "hours": ["Night"]},
    "all crashes": {"harm": "all"},
    "near large new buildings": {"scope": "near", "construction": ["New Construction"], "sizes": ["150+"],
                                 "harm": "cyc"},
    "empty": {"boroughs": ["Staten Island"], "year_from": 2020, "year_to": 2020, "factors": ["Vehicle defect"],
              "hours": ["Night"], "harm": "fatal", "scope": "near", "sizes": ["150+"]},
}


@pytest.mark.parametrize("name", SELECTIONS)
def test_report_builds_for_every_selection(store, name):
    report = build_report(store, make_filters(**SELECTIONS[name]))
    assert list(report["panels"]) == CHART_KEYS
    for key, panel in report["panels"].items():
        assert isinstance(panel.figure, go.Figure), key
        assert panel.scope and panel.note, key
        json.dumps(panel.figure.to_plotly_json(), default=_plain)            # serialisable


def test_empty_selection_shows_empty_states_not_broken_charts(store):
    report = build_report(store, make_filters(**SELECTIONS["empty"]))
    assert report["numbers"]["crashes"] == 0
    for key in ["trend", "factors", "heatmap", "scatter", "event", "selection"]:
        figure = report["panels"][key].figure
        assert len(figure.data) == 0 and figure.layout.annotations, key
    assert "No crashes match" in str(report["header"])


def test_all_crashes_carries_the_reporting_caution(store):
    report = build_report(store, make_filters(harm="all"))
    assert "cannot be compared across 2020" in str(report["header"])
    assert "cannot be compared across 2020" not in str(build_report(store, make_filters())["header"])


# ---------------------------------------------------------------------------------------------
# Story: the text is generated from the exported results, so it must quote them
# ---------------------------------------------------------------------------------------------
def _text(component) -> str:
    """All strings inside a Dash component tree, joined."""
    if component is None:
        return ""
    if isinstance(component, (str, int, float)):
        return str(component)
    if isinstance(component, (list, tuple)):
        return " ".join(_text(child) for child in component)
    return _text(getattr(component, "children", None))


def test_story_quotes_the_exported_results(store):
    from app import story
    questions = store.results["questions"]
    assert len(questions) == 10 and all(item["sentence"] for item in questions.values())
    hero = _text(story.hero(store))
    assert story.signed(questions["RQ3"]["estimate"]) in hero
    assert f"{50 + questions['RQ7']['estimate']:.0f}th" in hero and f"{questions['RQ7']['n']:,}" in hero
    findings = _text(story.findings(store))
    for item in questions.values():
        assert item["sentence"].split("; the interval includes zero")[0] in findings
    assert "not counted after adjusting for several tests" in findings   # RQ9
    page = " ".join(_text(part(store)) for part in
                    [story.hero, story.findings, story.limits, story.impact, story.data_and_method, story.footer])
    for banned in ["—", "causes", "caused by", "leads to", "because of the housing"]:
        assert banned not in page, banned


def test_robustness_table_matches_the_story(store):
    table = store.robustness
    assert (table["kind"] == "central").sum() == 1
    central = table[table["kind"] == "central"].iloc[0]
    assert central["percent_change"] == pytest.approx(store.results["questions"]["RQ3"]["estimate"], abs=1e-9)
    assert set(store.results["detail"]["unscreened_variants"]) <= set(table["variant"])


def test_site_imports_nothing_from_the_old_code():
    """The deployed site must run from app/ and app/data/ alone."""
    code = ("import sys; import app.app; "
            "bad = [name for name in sys.modules if name.split('.')[0] in {'pipeline', 'analysis', 'config'}]; "
            "print(bad)")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-800:]
    assert result.stdout.strip().splitlines()[-1] == "[]"


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


# ---------------------------------------------------------------------------------------------
# Dash callbacks over HTTP (Flask test client)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from app.app import server
    return server.test_client()


def _callback(client, trigger: str, values: dict):
    """Post one callback the way the browser does and return the 'response' part of the reply."""
    dependencies = client.get("/_dash-dependencies").get_json()
    spec = next(item for item in dependencies if any(f"{entry['id']}.{entry['property']}" == trigger
                                                     for entry in item["inputs"]))

    def with_values(entries):
        return [{"id": entry["id"], "property": entry["property"],
                 "value": values.get(f"{entry['id']}.{entry['property']}")} for entry in entries]

    outputs = [{"id": part.rsplit(".", 1)[0], "property": part.rsplit(".", 1)[1]}
               for part in spec["output"].strip(".").split("...")]
    payload = {"output": spec["output"], "outputs": outputs if len(outputs) > 1 else outputs[0],
               "inputs": with_values(spec["inputs"]), "state": with_values(spec["state"]),
               "changedPropIds": [trigger]}
    reply = client.post("/_dash-update-component", json=payload)
    assert reply.status_code == 200, reply.get_data(as_text=True)[:500]
    return reply.get_json()["response"]


def _control_state(filters) -> dict:
    return {f"{control_id}.value": value for control_id, value in zip(CONTROL_IDS, control_values(filters))}


def test_page_and_grid_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and b"design-tokens" in page.data
    layout = client.get("/_dash-layout").get_data(as_text=True)
    assert "Generate report" in layout and "Do injury crashes rise" in layout
    grid = client.get("/data/crash-grid.geojson", headers={"Accept-Encoding": "gzip"})
    assert grid.status_code == 200 and grid.headers["Content-Encoding"] == "gzip"


def test_search_callback_fills_the_dropdowns(client):
    response = _callback(client, "search.value", {"search.value": "Brooklyn 2022 new construction"})
    assert response["f-borough"]["value"] == ["Brooklyn"]
    assert (response["f-year-from"]["value"], response["f-year-to"]["value"]) == (2022, 2022)
    assert response["f-construction"]["value"] == ["New Construction"]
    assert "Understood" in json.dumps(response["search-feedback"])


@pytest.mark.parametrize("name", SELECTIONS)
def test_generate_report_callback_updates_every_chart(client, name):
    filters = make_filters(**SELECTIONS[name])
    response = _callback(client, "generate.n_clicks", {"generate.n_clicks": 1, **_control_state(filters)})
    for key in CHART_KEYS:
        assert "layout" in response[f"fig-{key}"]["figure"], key
        assert response[f"scope-{key}"]["children"] and response[f"note-{key}"]["children"], key
    assert from_dict(response["report-filters"]["data"]) == filters
    assert response["report-tiles"]["children"]


def test_enter_in_the_search_box_generates_from_the_query(client):
    stale_controls = _control_state(make_filters())                 # dropdowns have not caught up yet
    response = _callback(client, "search.n_submit", {"search.n_submit": 1, **stale_controls,
                                                     "search.value": "bronx pedestrians night 2019-2021"})
    shown = from_dict(response["report-filters"]["data"])
    assert shown == make_filters(boroughs=["Bronx"], year_from=2019, year_to=2021, hours=["Night"], harm="ped")
    assert "From your search" in json.dumps(response["report-header"])


# =============================================================================================
# Added coverage: a wider query list, conservation across filters, every chart builder on its own,
# and the wiring of the Generate report button.
# =============================================================================================
from app import charts                                           # noqa: E402

QUERY_TABLE = [
    # (query, values the parser must set, words it must report as not understood)
    ("BROOKLYN 2022 NEW BUILDINGS", {"boroughs": ["Brooklyn"], "year_from": 2022, "year_to": 2022,
                                     "construction": ["New Construction"]}, []),
    ("BrOoKlYn   2022   New   Buildings", {"boroughs": ["Brooklyn"], "year_from": 2022, "year_to": 2022,
                                          "construction": ["New Construction"]}, []),
    ("Queens 2018-2020", {"boroughs": ["Queens"], "year_from": 2018, "year_to": 2020}, []),
    ("queens 2018–2020", {"boroughs": ["Queens"], "year_from": 2018, "year_to": 2020}, []),
    ("queens 2020-2018", {"boroughs": ["Queens"], "year_from": 2018, "year_to": 2020}, []),
    ("between 2017 and 2019 manhattan", {"boroughs": ["Manhattan"], "year_from": 2017, "year_to": 2019}, []),
    ("bronx through 2018", {"boroughs": ["Bronx"], "year_from": YEAR_START, "year_to": 2018}, []),
    ("after 2022 brooklyn", {"boroughs": ["Brooklyn"], "year_from": 2023, "year_to": YEAR_END}, []),
    ("SI cyclists", {"boroughs": ["Staten Island"], "harm": "cyc"}, []),
    ("Brooklyn, 2022; new buildings!", {"boroughs": ["Brooklyn"], "year_from": 2022, "year_to": 2022,
                                        "construction": ["New Construction"]}, []),
    ("brooklyn and queens", {"boroughs": ["Brooklyn", "Queens"]}, []),
    ("bronx 1999-2030", {"boroughs": ["Bronx"], "year_from": YEAR_START, "year_to": YEAR_END}, []),
    ("manhattan towers fatal", {"boroughs": ["Manhattan"], "sizes": ["150+"], "harm": "fatal"}, []),
    ("pedestrian error brooklyn", {"boroughs": ["Brooklyn"], "factors": ["Pedestrian or cyclist error"]}, []),
    ("new york city", {}, []),
    # a close misspelling of a borough or construction word is corrected and shown as such
    ("Brookyln 2022", {"boroughs": ["Brooklyn"], "year_from": 2022, "year_to": 2022}, []),
    ("manhatan pedestrians", {"boroughs": ["Manhattan"], "harm": "ped"}, []),
    ("brooklin new construciton", {"boroughs": ["Brooklyn"], "construction": ["New Construction"]}, []),
    # two separate years make a range
    ("2020 2018 bronx", {"boroughs": ["Bronx"], "year_from": 2018, "year_to": 2020}, []),
    # nothing usable
    ("", {}, []),
    ("   ", {}, []),
    ("???", {}, []),
    ("asdf qwerty 12345", {}, ["asdf", "qwerty", "12345"]),
    ("brooklyn 20222", {"boroughs": ["Brooklyn"]}, ["20222"]),
]


@pytest.mark.parametrize("query, values, unmatched", QUERY_TABLE, ids=[repr(row[0])[:40] for row in QUERY_TABLE])
def test_query_table(query, values, unmatched):
    parsed = parse_query(query)
    assert parsed.values == values
    assert parsed.unmatched == unmatched
    filters = parsed.filters()                                # always convertible to a valid selection
    assert YEAR_START <= filters.year_from <= filters.year_to <= YEAR_END


def test_parser_never_raises_on_odd_input():
    for text in [None, "\u0000", "a" * 5000, "2022" * 50, "\U0001F4A5 brooklyn", "(((", "\\", "[a-z", "+",
                 "150+ 150+ 150+"]:
        parsed = parse_query(text)
        assert isinstance(parsed.values, dict)
        parsed.filters()


# --- conservation and impossible combinations ----------------------------------------------------
def test_borough_partition_conserves_totals(store):
    """Crashes of the five boroughs plus the unknown-borough rows add up to the citywide count."""
    whole = data.headline_numbers(store, make_filters())
    per_borough = sum(data.headline_numbers(store, make_filters(boroughs=[b]))["crashes"] for b in BOROUGHS)
    assert per_borough + whole["crashes_unknown_borough"] == whole["crashes"]


def test_year_partition_conserves_totals(store):
    whole = data.headline_numbers(store, make_filters(harm="all"))["crashes"]
    split = sum(data.headline_numbers(store, make_filters(harm="all", year_from=y, year_to=y))["crashes"]
                for y in range(YEAR_START, YEAR_END + 1))
    assert split == whole


def test_hour_band_partition_conserves_totals(store):
    whole = int(data.weekday_hour(store, make_filters()).sum())
    parts = sum(int(data.weekday_hour(store, make_filters(hours=[band])).sum()) for band in HOUR_BANDS)
    assert parts == whole


def test_factor_partition_conserves_totals(store):
    whole = int(data.factor_mix(store, make_filters()).sum())
    parts = sum(int(data.factor_mix(store, make_filters(factors=[group])).sum()) for group in FACTOR_GROUPS)
    assert parts == whole


def test_stricter_filters_never_add_crashes(store):
    wide = data.headline_numbers(store, make_filters(harm="all"))["crashes"]
    injury = data.headline_numbers(store, make_filters(harm="injury"))["crashes"]
    fatal = data.headline_numbers(store, make_filters(harm="fatal"))["crashes"]
    near = data.headline_numbers(store, make_filters(harm="all", scope="near"))["crashes"]
    assert wide >= injury >= fatal >= 0 and wide >= near >= 0


def test_housing_filters_partition_the_buildings(store):
    both = data.headline_numbers(store, make_filters())["buildings"]
    kinds = sum(data.headline_numbers(store, make_filters(construction=[kind]))["buildings"]
                for kind in ["New Construction", "Preservation"])
    assert kinds == both
    assert len(data.map_sites(store, make_filters())) <= both


IMPOSSIBLE = make_filters(boroughs=["Staten Island"], year_from=2020, year_to=2020, factors=["Vehicle defect"],
                          hours=["Night"], harm="fatal", scope="near", sizes=["150+"],
                          construction=["Preservation"])


def test_impossible_combination_gives_empty_frames_without_raising(store):
    numbers = data.headline_numbers(store, IMPOSSIBLE)
    assert numbers["crashes"] == 0 and numbers["buildings"] == 0
    assert data.map_cells(store, IMPOSSIBLE).empty and data.map_sites(store, IMPOSSIBLE).empty
    trend = data.crash_trend(store, IMPOSSIBLE)
    assert int(trend.to_numpy().sum()) == 0 and list(trend.columns) == ["Staten Island"]
    assert int(data.factor_mix(store, IMPOSSIBLE).sum()) == 0
    assert int(data.weekday_hour(store, IMPOSSIBLE).sum()) == 0
    tables = data.completions_by_year(store, IMPOSSIBLE)
    assert all(int(table.to_numpy().sum()) == 0 for table in tables.values())
    assert data.zip_relation(store, IMPOSSIBLE)["table"].empty
    assert data.event_study(store, IMPOSSIBLE) is not None
    assert data.site_selection(store, IMPOSSIBLE) is not None


# --- every chart builder, fed from the data layer ------------------------------------------------
FILTERS_FOR_CHARTS = {
    "normal": make_filters(),
    "one borough one year": make_filters(boroughs=["Staten Island"], year_from=2021, year_to=2021),
    "empty": IMPOSSIBLE,
}


def _assert_valid(figure, label):
    assert isinstance(figure, go.Figure), label
    json.dumps(figure.to_plotly_json(), default=_plain)
    figure.to_dict()                                           # plotly validates the whole tree here
    assert figure.layout.template is not None, label


def _builders(store, filters):
    yield "map", charts.map_figure(data.map_cells(store, filters), data.map_sites(store, filters),
                                   filters.harm_noun, "/data/crash-grid.geojson", bool(filters.boroughs))
    yield "trend", charts.trend_figure(data.crash_trend(store, filters), filters.harm_noun, filters.harm == "all")
    yield "factors", charts.factor_figure(data.factor_mix(store, filters))
    yield "heatmap", charts.heatmap_figure(data.weekday_hour(store, filters), filters.harm_noun)
    yield "completions", charts.completions_figure(data.completions_by_year(store, filters))
    yield "scatter", charts.scatter_figure(data.zip_relation(store, filters)["table"], filters.harm_noun)
    study, selection = data.event_study(store, filters), data.site_selection(store, filters)
    # report.py swaps in empty_figure when these say "empty"; the builders need a populated dict
    yield "event", charts.empty_figure("n/a") if study.get("empty") else charts.event_figure(study)
    yield "selection", charts.empty_figure("n/a") if selection.get("empty") else charts.selection_figure(selection)
    yield "break", charts.reporting_break_figure(store.city_year)
    yield "robustness", charts.robustness_figure(store.robustness)
    yield "empty", charts.empty_figure("nothing here")


@pytest.mark.parametrize("name", FILTERS_FOR_CHARTS)
def test_every_chart_builder_returns_a_valid_figure(store, name):
    seen = []
    for label, figure in _builders(store, FILTERS_FOR_CHARTS[name]):
        _assert_valid(figure, f"{name}/{label}")
        seen.append(label)
    assert len(seen) == 11


def test_chart_types_cover_the_assignment_list(store):
    """The brief asks for bars, lines, a heatmap, a map and a scatter plot."""
    kinds = {type(trace).__name__ for _, figure in _builders(store, make_filters()) for trace in figure.data}
    assert {"Bar", "Scatter", "Heatmap", "Choroplethmap", "Scattermap"} <= kinds


def test_a_chart_shows_the_two_domains_together(store):
    """Required: one visualisation relating housing to transport (map overlay and ZIP scatter)."""
    report = build_report(store, make_filters())
    map_traces = {type(trace).__name__ for trace in report["panels"]["map"].figure.data}
    assert {"Choroplethmap", "Scattermap"} <= map_traces                  # crashes under housing sites
    assert len(report["panels"]["scatter"].figure.data) >= 1


# --- the Generate report button -----------------------------------------------------------------
def _callbacks(client):
    return client.get("/_dash-dependencies").get_json()


def test_generate_button_drives_every_chart_and_the_numbers(client):
    spec = next(item for item in _callbacks(client)
                if "generate.n_clicks" in {f"{i['id']}.{i['property']}" for i in item["inputs"]})
    outputs = spec["output"]
    for key in CHART_KEYS:
        assert f"fig-{key}.figure" in outputs, key
        assert f"scope-{key}.children" in outputs and f"note-{key}.children" in outputs, key
    assert "report-tiles.children" in outputs and "report-header.children" in outputs
    # the dropdowns are State, not Input: changing a dropdown must not redraw charts by itself
    inputs = {f"{i['id']}.{i['property']}" for i in spec["inputs"]}
    states = {f"{s['id']}.{s['property']}" for s in spec["state"]}
    assert inputs == {"generate.n_clicks", "search.n_submit"}
    assert {f"{control}.value" for control in CONTROL_IDS} <= states and "search.value" in states


def test_no_other_callback_writes_to_the_charts(client):
    for item in _callbacks(client):
        if "generate.n_clicks" in {f"{i['id']}.{i['property']}" for i in item["inputs"]}:
            continue
        assert "fig-" not in item["output"], item["output"]


def test_changing_a_dropdown_only_flags_the_report_as_stale(client):
    filters = make_filters(boroughs=["Queens"])
    response = _callback(client, "f-borough.value", {**_control_state(filters),
                                                     "report-filters.data": make_filters().to_dict()})
    assert "Press Generate report" in json.dumps(response["stale-hint"])
    assert "is-stale" in response["generate"]["className"]


def test_generate_with_an_impossible_selection_still_answers(client):
    response = _callback(client, "generate.n_clicks", {"generate.n_clicks": 1, **_control_state(IMPOSSIBLE)})
    for key in CHART_KEYS:
        assert "layout" in response[f"fig-{key}"]["figure"], key
    assert "No crashes match" in json.dumps(response["report-header"])


def test_reset_restores_defaults(client):
    response = _callback(client, "reset.n_clicks", {"reset.n_clicks": 1, "search.value": "bronx"})
    assert [response[control_id]["value"] for control_id in CONTROL_IDS] == control_values(make_filters())


def test_search_with_nothing_understood_leaves_the_dropdowns_alone(client):
    response = _callback(client, "search.value", {"search.value": "zzz qqq"})
    assert response["search-feedback"]["children"]
    assert not any(control_id in response for control_id in CONTROL_IDS)
