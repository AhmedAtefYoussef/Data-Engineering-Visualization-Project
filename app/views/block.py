"""Your block: find a building or ZIP code and see its record against its borough. Ids here start with `bk-`."""
from __future__ import annotations

from dash import Dash, html

from app import components
from app.data import Store


def layout(store: Store) -> html.Div:
    return components.view([components.page_header("Your block", "Find a building or a ZIP code.")])


def register(app: Dash, store: Store) -> None:
    return None
