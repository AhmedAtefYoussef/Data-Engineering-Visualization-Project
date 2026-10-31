"""The site's routes, in one place. Each view is a module with `layout(store)` and `register(app, store)`."""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from app.views import block, explore, findings, method, overview, scenarios


@dataclass(frozen=True)
class Route:
    path: str
    label: str
    icon: str          # CSS class suffix of the rail icon
    title: str         # browser tab title
    view: ModuleType


ROUTES = [
    Route("/", "Overview", "overview", "Overview", overview),
    Route("/block", "Your block", "block", "Your block", block),
    Route("/scenarios", "Test it", "scenarios", "Test the result", scenarios),
    Route("/explore", "Explore", "explore", "Explore the records", explore),
    Route("/findings", "Findings", "findings", "Ten questions", findings),
    Route("/method", "Method", "method", "Data and method", method),
]
_BY_PATH = {route.path: route for route in ROUTES}


def resolve(pathname: str | None) -> Route | None:
    """The route for a path, ignoring a trailing slash. None when no route matches."""
    if not isinstance(pathname, str):
        return ROUTES[0]
    return _BY_PATH.get(pathname.rstrip("/") or "/")
