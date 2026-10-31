"""The whole page, top to bottom. One continuous page: each part is a module in app/sections/ with
`layout(store)` and `register(app, store)`, and owns its id prefix."""
from __future__ import annotations

from dash import Dash, html

from app import story
from app.data import Store
from app.sections import block, explorer, findings, lab, method, overview

SECTIONS = [overview, block, lab, explorer, findings, method]
NAV = [("story", "The story"), ("block", "Your block"), ("lab", "Test it"), ("explore", "Explore"),
       ("findings", "Findings"), ("method", "Method")]


def _nav_link(target: str, label: str) -> html.A:
    """Two stacked copies of the label: on hover the first rolls away and the second rolls in."""
    return html.A(html.Span([html.Span(label, className="roll-a"), html.Span(label, className="roll-b", **{"aria-hidden": "true"})],
                            className="roll"), href=f"#{target}", className="nav-link")


def _masthead() -> html.Header:
    return html.Header(className="masthead", children=[
        html.A([html.Span(className="brand-mark", **{"aria-hidden": "true"}), "Housing & Streets, NYC"],
               href="#top", className="brand"),
        html.Nav([_nav_link(*item) for item in NAV], className="nav", **{"aria-label": "Sections"}),
    ])


def _facts(store: Store) -> html.Div:
    """Real figures for the browser's animations, so no script types a number."""
    rows = store.meta["rows"]
    return html.Div(id="facts", hidden=True, **{"data-crashes": f"{rows['crashes']:,}", "data-housing": f"{rows['housing']:,}",
                                                "data-sites": f"{len(store.sites):,}"})


def build_page(store: Store) -> html.Div:
    return html.Div(className="page", id="top", children=[
        html.A("Skip to the report", href="#explore", className="skip-link"),
        html.Div(id="thread", className="thread", **{"aria-hidden": "true"}),
        _facts(store),
        _masthead(),
        html.Main([section.layout(store) for section in SECTIONS]),
        story.footer(store),
    ])


def register(app: Dash, store: Store) -> None:
    for section in SECTIONS:
        section.register(app, store)
