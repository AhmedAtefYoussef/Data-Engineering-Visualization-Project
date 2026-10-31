"""Test it: change the comparison and watch the estimate respond. Ids here start with `sc-`."""
from __future__ import annotations

from dash import Dash, html

from app import components
from app.data import Store


def layout(store: Store) -> html.Section:
    return components.section("lab", "Try to break the result", "Change the comparison and see what holds.", [], "plum")


def register(app: Dash, store: Store) -> None:
    return None
