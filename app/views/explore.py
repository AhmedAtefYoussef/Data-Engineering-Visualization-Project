"""Explore: search, dropdowns and Generate report, redrawing every chart for one selection."""
from __future__ import annotations

from dash import Dash, html

from app import callbacks, components, layout as explorer_layout
from app.data import Store


def layout(store: Store) -> html.Div:
    return components.view([explorer_layout.explorer()], "view-explore")


def register(app: Dash, store: Store) -> None:
    callbacks.register(app, store)
