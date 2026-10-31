"""The frame around every view: the rail, the phone tab bar, the routed content and the footer."""
from __future__ import annotations

from dash import Dash, Input, Output, clientside_callback, dcc, html

from app import components, routes, story
from app.data import Store

ROUTED_ID = "page-content"


def _brand() -> dcc.Link:
    return dcc.Link([html.Span(className="brand-mark", **{"aria-hidden": "true"}),
                     html.Span(["NYC housing", html.Br(), "and street safety"], className="brand-name")],
                    href="/", className="brand", title="Overview")


def _nav_link(route: routes.Route) -> dcc.Link:
    return dcc.Link([html.Span(className=f"nav-icon icon-{route.icon}", **{"aria-hidden": "true"}),
                     html.Span(route.label)], href=route.path, className="nav-link")


def rail(store: Store) -> html.Aside:
    """Labelled navigation on wide screens; on a phone it becomes a bar with scrolling tabs."""
    meta = store.meta
    return html.Aside(className="rail", children=[
        _brand(),
        html.Nav([_nav_link(route) for route in routes.ROUTES], className="nav", **{"aria-label": "Pages"}),
        html.Div(className="rail-foot", children=[
            html.P(f"{meta['rows']['housing']:,} buildings"),
            html.P(f"{meta['rows']['crashes']:,} crash reports"),
            html.P(f"Records through {meta['constants']['window_end'] if 'window_end' in meta['constants'] else 2025}"),
        ]),
    ])


def build_shell(store: Store) -> html.Div:
    return html.Div(className="app", children=[
        dcc.Location(id="url", refresh="callback-nav"),
        dcc.Store(id="route-sink"),
        html.A("Skip to the content", href="#main", className="skip-link"),
        rail(store),
        html.Div(className="stage", children=[
            html.Main(id="main", tabIndex=-1, children=html.Div(components.view_skeleton(), id=ROUTED_ID)),
            story.footer(store),
        ]),
    ])


def _not_found() -> html.Div:
    return components.view([
        components.page_header("That page is not here",
                               "The address may be out of date. The overview is the best place to start."),
        dcc.Link("Go to the overview", href="/", className="button button-primary"),
    ])


def validation_layout(store: Store) -> html.Div:
    """Every component of every view, so Dash can check all callbacks against what can appear."""
    return html.Div([build_shell(store), *[route.view.layout(store) for route in routes.ROUTES]])


def register(app: Dash, store: Store) -> None:
    @app.callback(Output(ROUTED_ID, "children"), Input("url", "pathname"))
    def render(pathname):
        """Draw the view for the address. An unknown address gets a page that leads back."""
        route = routes.resolve(pathname)
        return route.view.layout(store) if route else _not_found()

    # Mark the current link, name the tab and start each new page from the top.
    titles = {route.path: route.title for route in routes.ROUTES}
    clientside_callback(
        "function (pathname) { return window.NYC.onRoute(pathname, " + repr(titles) + "); }",
        Output("route-sink", "data"), Input("url", "pathname"))

    for route in routes.ROUTES:
        route.view.register(app, store)
