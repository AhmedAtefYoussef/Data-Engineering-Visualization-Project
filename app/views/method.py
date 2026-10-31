"""Data and method: limits, what this means for the city, and how the tables were made. Ids here start with `md-`."""
from __future__ import annotations

from dash import Dash, html

from app import components, story
from app.data import Store


def layout(store: Store) -> html.Div:
    return components.view([story.limits(store), story.impact(store), story.data_and_method(store)], "view-method")


def register(app: Dash, store: Store) -> None:
    return None
