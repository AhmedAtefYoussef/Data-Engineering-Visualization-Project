"""Explore: the search box, the dropdowns and Generate report, redrawing every chart for one selection."""
from __future__ import annotations

from dash import Dash, html

from app import callbacks, layout as explorer_layout
from app.data import Store


def layout(store: Store) -> html.Section:
    return explorer_layout.explorer()


def register(app: Dash, store: Store) -> None:
    callbacks.register(app, store)
