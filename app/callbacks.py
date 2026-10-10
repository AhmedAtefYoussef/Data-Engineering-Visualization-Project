"""The three interactions of the page.

1. Search: typing a query fills the dropdowns and says what was understood.
2. Generate report: one callback redraws every number and chart from the dropdowns, or from
   the query when Enter is pressed in the search box.
3. A hint that appears when the controls no longer match the report on screen.
"""
from __future__ import annotations

import logging

from dash import Dash, Input, Output, State, ctx, html, no_update

from app.data import Store
from app.filters import Filters, from_dict, make_filters
from app.layout import EXAMPLE_QUERIES, search_hint
from app.report import CHART_KEYS, build_report
from app.search import ParsedQuery, parse_query

log = logging.getLogger(__name__)

CONTROL_IDS = ["f-borough", "f-year-from", "f-year-to", "f-construction", "f-size", "f-factor", "f-hour",
               "f-harm", "f-scope"]
GENERATE_LABEL, GENERATING_LABEL = "Generate report", "Generating…"


def control_values(filters: Filters) -> list:
    """The nine dropdown values for a selection, in CONTROL_IDS order."""
    return [list(filters.boroughs), filters.year_from, filters.year_to, list(filters.construction),
            list(filters.sizes), list(filters.factors), list(filters.hours), filters.harm, filters.scope]


def filters_from_controls(values: list, source: str = "controls") -> Filters:
    """Inverse of `control_values`."""
    boroughs, year_from, year_to, construction, sizes, factors, hours, harm, scope = values
    return make_filters(boroughs, year_from, year_to, construction, sizes, factors, hours, harm, scope, source)


def search_feedback(parsed: ParsedQuery) -> list:
    """What the parser understood, as labelled chips, and the words it could not place."""
    if parsed.is_empty:
        return search_hint()
    if not parsed.understood:
        return [html.P(["Nothing in ", html.Q(parsed.text.strip()), " matched a filter, so the filters are "
                        "unchanged. Try a borough, a year or range, new construction or preservation, a road "
                        "user, a time of day or a factor such as speeding."], className="feedback-miss")]
    children = [html.Span("Understood", className="feedback-lead"),
                html.Ul([html.Li([html.Span(label, className="chip-label"), value], className="chip chip-found",
                                 title=f"from “{typed}”")
                         for typed, label, value in parsed.understood], className="chips")]
    if parsed.unmatched:
        children.append(html.P(["Not used: ", ", ".join(parsed.unmatched), "."], className="feedback-unused"))
    children.append(html.P("Filters not named in the query were set back to their defaults. "
                           "Press Enter or Generate report.", className="feedback-next"))
    return children


def figure_outputs() -> list[Output]:
    """Three outputs per chart, in CHART_KEYS order."""
    outputs = []
    for key in CHART_KEYS:
        outputs += [Output(f"fig-{key}", "figure"), Output(f"scope-{key}", "children"),
                    Output(f"note-{key}", "children")]
    return outputs


def report_outputs(store: Store, filters: Filters, query: str | None) -> list:
    """Values for the generate callback: header, tiles, three per chart, the stored filters."""
    report = build_report(store, filters, query)
    values = [report["header"], report["tiles"]]
    for key in CHART_KEYS:
        panel = report["panels"][key]
        values += [panel.figure, panel.scope, panel.note]
    return values + [filters.to_dict()]


def register(app: Dash, store: Store) -> None:
    """Attach the callbacks to the app."""

    @app.callback(
        [Output(control_id, "value") for control_id in CONTROL_IDS],
        Output("search-feedback", "children"),
        Input("search", "value"), Input("reset", "n_clicks"),
        prevent_initial_call=True)
    def apply_search(text, _reset):
        """Fill the dropdowns from the query as it is typed; Reset puts every control back."""
        if ctx.triggered_id == "reset":
            return control_values(make_filters()) + [search_hint()]
        parsed = parse_query(text)
        if not parsed.understood:
            return [no_update] * len(CONTROL_IDS) + [search_feedback(parsed)]
        return control_values(parsed.filters()) + [search_feedback(parsed)]

    @app.callback(
        Output("search", "value"),
        [Input(f"example-{position}", "n_clicks") for position in range(len(EXAMPLE_QUERIES))],
        Input("reset", "n_clicks"),
        prevent_initial_call=True)
    def fill_search(*_clicks):
        """An example button writes its query into the search box; Reset clears the box."""
        if ctx.triggered_id == "reset":
            return ""
        return EXAMPLE_QUERIES[int(ctx.triggered_id.rsplit("-", 1)[1])]

    @app.callback(
        Output("report-header", "children"), Output("report-tiles", "children"), *figure_outputs(),
        Output("report-filters", "data"),
        Input("generate", "n_clicks"), Input("search", "n_submit"),
        [State(control_id, "value") for control_id in CONTROL_IDS], State("search", "value"),
        running=[(Output("generate", "disabled"), True, False),
                 (Output("generate", "children"), GENERATING_LABEL, GENERATE_LABEL),
                 (Output("report", "className"), "report is-loading", "report")])
    def generate(_clicks, _submits, *state):
        """Redraw the whole report. Runs once on page load with the default filters."""
        *values, text = state
        filters, query = filters_from_controls(values), None
        parsed = parse_query(text)
        if ctx.triggered_id == "search" and parsed.understood:
            # Enter in the search box: use the query itself, so a fast typist never gets a report
            # built from dropdowns that have not caught up yet.
            filters, query = parsed.filters(), parsed.text.strip()
        elif parsed.understood and parsed.filters() == filters:
            query = parsed.text.strip()          # the dropdowns still show exactly what the query set
        return report_outputs(store, filters, query)

    @app.callback(
        Output("stale-hint", "children"), Output("generate", "className"),
        [Input(control_id, "value") for control_id in CONTROL_IDS], Input("report-filters", "data"))
    def flag_stale(*inputs):
        """Tell the reader when the dropdowns differ from the report on screen."""
        *values, shown = inputs
        if shown is None or filters_from_controls(values) == from_dict(shown):
            return "", "button button-primary"
        return "Filters changed. Press Generate report to update the charts.", "button button-primary is-stale"
