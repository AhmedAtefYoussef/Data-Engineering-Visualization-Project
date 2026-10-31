"""Test the result: change the comparison and watch the estimate respond. Ids here start with `sc-`."""
from __future__ import annotations

from dash import Dash, html

from app import components
from app.data import Store


def layout(store: Store) -> html.Div:
    return components.view([components.page_header("Test the result", "Change the comparison and see what holds.")])


def register(app: Dash, store: Store) -> None:
    return None
