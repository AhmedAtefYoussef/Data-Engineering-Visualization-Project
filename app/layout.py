"""Page structure: masthead, hero, the explorer (search, filters, Generate report, charts) and the story.

The layout holds no data logic. Chart boxes exist from the first paint with fixed heights, so
a new report replaces figures in place and nothing on the page moves.
"""
from __future__ import annotations

import logging

from dash import dcc, html

from app import story
from app.data import Store
from app.filters import (BOROUGHS, CONSTRUCTION_TYPES, DEFAULT_HARM, DEFAULT_SCOPE, FACTOR_GROUPS, HARM,
                         HOUR_BAND_RANGES, HOUR_BANDS, SCOPE, SIZE_BANDS, YEAR_END, YEAR_START, YEARS)
from app.report import CHART_KEYS
from app.theme import GRAPH_CONFIG

log = logging.getLogger(__name__)

SEARCH_HINT = "Typing fills the filters below and shows here what was understood."
EXAMPLE_QUERIES = ["Brooklyn 2022 new construction", "bronx pedestrians night 2019-2021",
                   "queens cyclists speeding since 2021", "manhattan large buildings near housing"]

# key -> (title, chart type shown as a small label, CSS size class, which datasets it uses)
CHART_CARDS = {
    "map": ("Housing sites over crash density", "Map", "card-wide", "Both datasets"),
    "selection": ("How crash-heavy the streets were at housing sites before construction", "Bar chart", "", "Both datasets"),
    "trend": ("Crashes over time by borough", "Line chart", "", "Crashes"),
    "completions": ("Affordable units completed per year", "Bar chart", "", "Housing"),
    "factors": ("Contributing factors named in police reports", "Bar chart", "", "Crashes"),
    "heatmap": ("When crashes happen", "Heatmap", "", "Crashes"),
    "scatter": ("Housing against crashes, one dot per ZIP code", "Scatter plot", "", "Both datasets"),
    "event": ("Before and after completion", "Line chart with intervals", "card-wide", "Both datasets"),
}
# Wheel zoom stays off on the map so that scrolling the page never gets trapped inside it;
# the map zooms with its buttons, a double click or a pinch.
MAP_CONFIG = {**GRAPH_CONFIG, "scrollZoom": False}


def search_hint() -> list:
    """What the feedback line says while the search box is empty (it keeps the line's height reserved)."""
    return [html.P(SEARCH_HINT, className="feedback-next")]


def _options(values: list[str], labels: dict | None = None) -> list[dict]:
    return [{"label": (labels or {}).get(value, value), "value": value} for value in values]


OPTION_HEIGHT_PX = 44            # one menu row: a comfortable click and touch target
MENU_ROWS_VISIBLE = 7            # rows shown before the menu scrolls
MULTI_NOTE = "choose one or more"


def _field(label: str, control_id: str, control, help_text: str | None = None, wide: bool = False,
           note: str | None = None) -> html.Div:
    """A labelled control. The label is tied to the control so a click or a screen reader finds it.

    `note` is a short qualifier shown beside the label, used to say that a filter takes several values.
    """
    label_children = [label, html.Span(note, className="field-note")] if note else label
    children = [html.Label(label_children, htmlFor=control_id, className="field-label", id=f"{control_id}-label"),
                control]
    if help_text:
        children.append(html.P(help_text, className="field-help"))
    # role="group" names the control for assistive technology even where the script in
    # assets/site.js has not run, because a <label> cannot reach the dropdown's inner combobox.
    return html.Div(children, className="field field-wide" if wide else "field", role="group",
                    **{"aria-labelledby": f"{control_id}-label"})


def _dropdown(control_id: str, options: list[dict], value=None, multi: bool = True, placeholder: str = "All",
              searchable: bool = False):
    """One filter dropdown.

    A multi-value menu stays open after a pick, so several values take one visit. Typing to narrow
    the list is switched on only where the list is long, because on a phone it opens the keyboard.
    """
    return dcc.Dropdown(id=control_id, options=options, value=value if value is not None else ([] if multi else None),
                        multi=multi, clearable=multi, searchable=searchable, placeholder=placeholder,
                        closeOnSelect=not multi, optionHeight=OPTION_HEIGHT_PX,
                        maxHeight=OPTION_HEIGHT_PX * MENU_ROWS_VISIBLE, className="dropdown", persistence=False)


def controls() -> html.Div:
    """Search box, the nine dropdowns and the Generate report button."""
    hour_labels = {band: f"{band} ({HOUR_BAND_RANGES[band]})" for band in HOUR_BANDS}
    size_labels = {band: f"{band} units" for band in SIZE_BANDS}
    harm_labels = {key: item["label"] for key, item in HARM.items()}
    search = html.Div(className="search", children=[
        html.Label("Search", htmlFor="search", className="field-label"),
        html.Div(className="search-box", children=[
            dcc.Input(id="search", type="search", debounce=False, value="", className="search-input",
                      placeholder="Type a place, a period and what you want to see", autoComplete="off",
                      spellCheck=False, n_submit=0),
            # The clear button and the key hint share one row on the right of the field, so the gap
            # between them is set in one place. assets/site.js empties the field when the button is pressed.
            html.Div(className="search-tools", children=[
                html.Button("\u00d7", id="search-clear", className="search-clear", type="button", n_clicks=0,
                            title="Clear the search", **{"aria-label": "Clear the search"}),
                html.Span(["Press ", html.Kbd("Enter"), " to generate"], className="search-key",
                          **{"aria-hidden": "true"}),
            ]),
        ]),
        html.P(["Try ", *[html.Button(text, id=f"example-{position}", className="example",
                                      n_clicks=0, type="button")
                          for position, text in enumerate(EXAMPLE_QUERIES)]], className="examples"),
        html.Div(search_hint(), id="search-feedback", className="search-feedback", role="status",
                 **{"aria-live": "polite"}),
    ])
    filters = html.Div(className="fields", children=[
        _field("Borough", "f-borough", _dropdown("f-borough", _options(BOROUGHS), placeholder="All five"), note=MULTI_NOTE),
        _field("From year", "f-year-from", _dropdown("f-year-from", _options(YEARS), YEAR_START, multi=False)),
        _field("To year", "f-year-to", _dropdown("f-year-to", _options(YEARS), YEAR_END, multi=False)),
        _field("Construction type", "f-construction",
               _dropdown("f-construction", _options(CONSTRUCTION_TYPES), placeholder="Both types"), note=MULTI_NOTE),
        _field("Building size", "f-size", _dropdown("f-size", _options(SIZE_BANDS, size_labels), placeholder="Any size"),
               note=MULTI_NOTE),
        _field("Contributing factor", "f-factor",
               _dropdown("f-factor", _options(FACTOR_GROUPS), placeholder="Any factor", searchable=True),
               note=MULTI_NOTE),
        _field("Time of day", "f-hour", _dropdown("f-hour", _options(HOUR_BANDS, hour_labels), placeholder="Any time"),
               note=MULTI_NOTE),
        _field("Who was harmed", "f-harm", _dropdown("f-harm", _options(list(HARM), harm_labels), DEFAULT_HARM, multi=False)),
        # The option labels are kept short enough to fit the control; the qualifier sits under it.
        _field("Crash location", "f-scope", _dropdown("f-scope", _options(list(SCOPE), SCOPE), DEFAULT_SCOPE, multi=False),
               help_text="Near housing: the housing filters apply to each crash's nearest site."),
    ])
    actions = html.Div(className="actions", children=[
        html.Button("Generate report", id="generate", n_clicks=0, className="button button-primary",
                    type="button"),
        html.Button("Reset filters", id="reset", n_clicks=0, className="button button-quiet", type="button"),
        html.P(id="stale-hint", className="stale-hint", role="status", **{"aria-live": "polite"}),
    ])
    return html.Div([search, html.P("or set the filters yourself", className="divider"), filters, actions],
                    className="controls", role="group", **{"aria-label": "Report filters"})


def graph_box(graph_id: str, config: dict, size_class: str = "") -> html.Div:
    """A graph inside a box whose height is set in CSS, so a new figure never changes the page height."""
    return html.Div(dcc.Graph(id=graph_id, config=config, responsive=True, style={"height": "100%"}),
                    className=f"graph {size_class}".strip())


def chart_card(key: str) -> html.Article:
    """One chart with its title, the filters and n behind it, and a note on how to read it."""
    title, kind, size_class, datasets = CHART_CARDS[key]
    is_map = key == "map"
    return html.Article(className=f"card {size_class}".strip(), **{"aria-labelledby": f"title-{key}"}, children=[
        html.Header(className="card-head", children=[
            html.H3(title, id=f"title-{key}"),
            html.P(f"{kind} · {datasets}", className="card-kind"),
        ]),
        html.P(id=f"scope-{key}", className="card-scope"),
        graph_box(f"fig-{key}", config=MAP_CONFIG if is_map else GRAPH_CONFIG,
                  size_class="graph-map" if is_map else ""),
        html.P(id=f"note-{key}", className="chart-note"),
    ])


def explorer() -> html.Section:
    return html.Section(id="explore", className="band band-explore", children=[
        html.H2("Build your own report"),
        html.P("Type a query or set the filters, then press Generate report. The summary numbers and charts "
               "below are redrawn for that selection; each chart says which filters it does not use.",
               className="section-lead"),
        controls(),
        dcc.Store(id="report-filters"),
        html.Div(id="report", className="report", children=[
            html.H3("Report", className="visually-hidden"),
            html.Div(id="report-header", className="report-header", role="status", **{"aria-live": "polite"}),
            html.Div(id="report-tiles", className="tiles"),
            html.Div([chart_card(key) for key in CHART_KEYS], className="cards"),
        ]),
        html.P(["Injury crashes are the default outcome. A crash counts when at least one person was injured or "
                "killed. Total crash reports cannot be compared across 2020, as ",
                html.A("the reporting chart", href="#data"), " shows."], className="aside explore-foot"),
    ])


def build_layout(store: Store) -> html.Div:
    """The whole page."""
    return html.Div(className="page", children=[
        html.A("Skip to the report", href="#explore", className="skip-link"),
        html.Header(className="masthead", children=[
            html.A("NYC housing and street safety", href="#top", className="brand"),
            html.Nav([html.A("Explore", href="#explore"), html.A("Findings", href="#findings"),
                      html.A("Limits", href="#limits"), html.A("For the city", href="#impact"),
                      html.A("Data", href="#data")], className="nav", **{"aria-label": "Sections"}),
        ]),
        html.Main(id="top", children=[
            story.hero(store),
            explorer(),
            story.findings(store),
            story.limits(store),
            story.impact(store),
            story.data_and_method(store),
        ]),
        story.footer(store),
    ])


def missing_data_layout(message: str) -> html.Div:
    """Shown when app/data/ has not been built. The message and the build command go to the server log only."""
    log.error("%s Build the data with: python tools/build_app_data.py", message)
    return html.Div(className="page", children=[html.Main(html.Section(className="hero", children=[
        html.H1("This page is not ready yet"),
        html.P("The data behind this page has not been loaded. Please try again later.", className="lead"),
    ]))])
