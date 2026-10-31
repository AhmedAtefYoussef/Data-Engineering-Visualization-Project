"""The opening: the question over a live city where every point of light is a real building.

The canvas is one stage shared with the story's chapters: assets/js/20-stage.js. Ids here start with `ov-`.
"""
from __future__ import annotations

import gzip

from dash import Dash, html
from flask import Response, request

from app import components, story_data
from app.data import Store
from app.filters import YEAR_END, YEAR_START, buffer_m
from app.stats import NEUTRAL_PERCENTILE
from app.story_text import signed

ONE_YEAR_S = 31_536_000


def _hero(store: Store) -> html.Div:
    rows, questions = store.meta["rows"], store.results["questions"]
    after, where = questions["RQ3"], questions["RQ7"]
    percentile = NEUTRAL_PERCENTILE + where["estimate"]
    return html.Div(className="hero", children=[
        html.H1("Do injury crashes rise around affordable housing once it opens?", **{"data-kinetic": "words"}),
        html.P(f"Each point of light is one of {len(store.sites):,} city-financed buildings, set against "
               f"{rows['crashes']:,} police crash reports from {YEAR_START} to {YEAR_END}.",
               className="hero-lead", **{"data-kinetic": "lines"}),
        html.Div(className="hero-answers", children=[
            components.metric(f"{signed(after['estimate'])}%", f"change in injury crashes within {buffer_m()} m after a new "
                              "building opens, against sites not yet finished",
                              f"95% interval {signed(after['low'])}% to {signed(after['high'])}%, {after['n']:,} sites. "
                              "The records cannot tell this from no change.", "is-crash", count_to=after["estimate"]),
            components.metric(f"{percentile:.0f}th", "percentile: how crash-heavy the average new site's streets "
                              "already were, before construction",
                              f"A typical street point in its borough is the {NEUTRAL_PERCENTILE}th. The sites were "
                              "chosen where crashes already happen.", "is-housing", count_to=round(percentile)),
        ]),
        html.Div(className="hero-actions", children=[
            html.A("Follow the story", href="#story", className="button button-primary magnetic"),
            html.A("Find your block", href="#block", className="button button-secondary magnetic"),
        ]),
    ])


def layout(store: Store) -> html.Section:
    return html.Section(id="story", className="story-root", **{"data-tint": "midnight"}, children=[
        html.Div(className="stage", id="ov-stage", **{"data-url": story_data.versioned_url(store)},
                 children=html.Canvas(id="ov-canvas", className="stage-canvas", role="img",
                                      **{"aria-label": f"A map of New York City drawn from {len(store.sites):,} points of light, "
                                                       "one for each located affordable-housing building."})),
        html.Div(className="story-steps", children=[_hero(store)]),
    ])


def register(app: Dash, store: Store) -> None:
    def story_json() -> Response:
        payload, compressed, digest = story_data.encoded(store)
        accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "")
        response = Response(compressed if accepts_gzip else payload, mimetype="application/json")
        if accepts_gzip:
            response.headers["Content-Encoding"] = "gzip"
        response.headers["Cache-Control"] = f"public, max-age={ONE_YEAR_S}, immutable" if request.args.get("v") == digest else "no-cache"
        response.headers["Vary"] = "Accept-Encoding"
        response.set_etag(digest)
        return response.make_conditional(request)

    assert gzip is not None
    app.server.add_url_rule(story_data.STORY_URL, "story_json", story_json)
