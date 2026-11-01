"""Try to break the result: change the comparison and watch the estimate respond. Ids here start with `sc-`.

There is no Generate button: every choice asks /api/scenario at once. The page is drawn here and driven by
assets/js/45-tools.js.
"""
from __future__ import annotations

import json

from dash import Dash, dcc, html
from flask import jsonify, request

from app import components, scenarios
from app.data import Store

CACHE_SECONDS = 300
CONTROLS = [
    ("sites", "Which sites", "The kind of completed building being tested."),
    ("borough", "Borough", "Applies to the tested and the comparison sites."),
    ("size", "Building size, in units", "Applies to the tested sites."),
    ("group", "Comparison group", "Which sites stand in for no building."),
    ("cluster", "Leave out the largest cluster", "Sites within 250 m of each other are resampled together."),
    ("cohort", "Leave out a cohort", "Drop the sites completed in one year."),
]


def _control(key: str, label: str, help_text: str, store: Store) -> html.Div:
    """A named radio group. The radios stay native, so arrow keys and screen readers work as usual."""
    choices = [{"label": text, "value": value} for value, text in scenarios.options(store)[key]]
    group = html.Div(className="segmented-field", children=[
        html.Span(label, className="field-label", id=f"sc-{key}-label"),
        dcc.RadioItems(id=f"sc-{key}", options=choices, value=scenarios.DEFAULT_SPEC[key], inline=True,
                       className="segmented", labelClassName="segment")])
    return html.Div(className="sc-control", role="radiogroup", **{"aria-labelledby": f"sc-{key}-label"},
                    children=[group, html.P(help_text, className="field-help")])


def _button(button_id: str, text: str, **extra) -> html.Button:
    return html.Button(text, id=button_id, type="button", className="button button-secondary", **extra)


def layout(store: Store) -> html.Section:
    reference = scenarios.reference(store)
    published = reference["published"]
    lead = (f"The published result compares {published['n_tested_sites']:,} new buildings with {published['n_comparison_sites']:,} "
            "sites not yet finished. Change who is compared and the answer is recomputed from the records at once. "
            "If a small change makes it jump, it was never solid.")
    options = {key: [value for value, _ in choices] for key, choices in scenarios.options(store).items()}
    return components.section("lab", "Try to break the result", lead, [
        html.Div(id="sc-root", className="sc-root", **{"data-reference": json.dumps(reference), "data-options": json.dumps(options)}, children=[
            html.Div(className="sc-readout lift", children=[
                html.P(id="sc-value", className="sc-value", children=f"{published['estimate']:+.1f}%".replace("-", "−")),
                html.P(id="sc-range", className="sc-range"),
                html.P(id="sc-tag", className="sc-tag"),
            ]),
            html.Div(id="sc-axis", className="sc-axis lift"),
            html.P(id="sc-sentence", className="sc-sentence lift"),
            html.Div(id="sc-empty", className="sc-empty", hidden=True),
            html.P(id="sc-status", className="visually-hidden", role="status", **{"aria-live": "polite"}),
            components.surface([_control(key, label, help_text, store) for key, label, help_text in CONTROLS],
                               class_name="sc-controls lift"),
            html.Div(className="sc-actions lift", children=[
                _button("sc-back", "Back one step", disabled=True), _button("sc-reset", "Reset", disabled=True),
                _button("sc-copy", "Copy link")]),
            html.Div(className="sc-lower", children=[
                html.Div(className="sc-tally lift", children=[
                    html.P(id="sc-count", className="sc-count"),
                    html.P(id="sc-count-note", className="sc-count-note"),
                ]),
                html.Div(className="sc-history-wrap lift", children=[
                    html.H3("The last 8 scenarios", className="sc-history-title"),
                    html.Ol(id="sc-history", className="sc-history"),
                ]),
            ]),
        ]),
    ], "plum", "chapter-lab")


def register(app: Dash, store: Store) -> None:
    def scenario():
        response = jsonify(scenarios.run(store, request.args.to_dict()))
        response.headers["Cache-Control"] = f"public, max-age={CACHE_SECONDS}"
        return response

    if "api_scenario" not in app.server.view_functions:
        app.server.add_url_rule("/api/scenario", "api_scenario", scenario)
