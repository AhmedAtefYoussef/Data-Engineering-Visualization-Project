"""Findings: the ten research questions, each with its estimate, interval, n and owner. Ids here start with `fd-`."""
from __future__ import annotations

from dash import Dash, html

from app import components, story
from app.data import Store


def layout(store: Store) -> html.Div:
    return components.view([story.findings(store)], "view-findings")


def register(app: Dash, store: Store) -> None:
    return None
