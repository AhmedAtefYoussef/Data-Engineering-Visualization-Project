"""The overview: the answer first, then the guided story. Ids here start with `ov-`."""
from __future__ import annotations

from dash import Dash, html

from app import components, story
from app.data import Store


def layout(store: Store) -> html.Div:
    return components.view([story.hero(store)], "view-overview")


def register(app: Dash, store: Store) -> None:
    return None
