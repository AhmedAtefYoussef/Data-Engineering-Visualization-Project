"""The routed shell: every address draws its view, and every callback target exists in some view."""
from __future__ import annotations

import json

import pytest

from app import routes, shell
from app.data import load


@pytest.fixture(scope="module")
def store():
    return load()


@pytest.fixture(scope="module")
def client():
    from app.app import server
    return server.test_client()


def _render(client, pathname: str) -> str:
    body = {"output": f"{shell.ROUTED_ID}.children", "outputs": {"id": shell.ROUTED_ID, "property": "children"},
            "inputs": [{"id": "url", "property": "pathname", "value": pathname}], "changedPropIds": ["url.pathname"]}
    response = client.post("/_dash-update-component", json=body)
    assert response.status_code == 200, pathname
    return json.dumps(response.get_json())


@pytest.mark.parametrize("route", routes.ROUTES, ids=lambda route: route.path)
def test_each_route_serves_the_page_and_draws_its_view(client, route):
    assert client.get(route.path).status_code == 200      # a shared link opens directly
    assert f'view-{route.icon}' in _render(client, route.path) or "page-header" in _render(client, route.path)


def test_a_trailing_slash_is_the_same_route():
    assert routes.resolve("/explore/") is routes.resolve("/explore")


@pytest.mark.parametrize("value", [None, 7, ["/"], {"a": 1}])
def test_a_non_string_path_falls_back_to_the_overview(value):
    assert routes.resolve(value) is routes.ROUTES[0]


def test_an_unknown_address_leads_back(client):
    rendered = _render(client, "/nowhere")
    assert "not here" in rendered and '"href": "/"' in rendered


def test_core_routes_hold_their_key_content(client):
    assert "Generate report" in _render(client, "/explore")
    assert "Do injury crashes rise" in _render(client, "/")


def test_ids_are_unique_across_all_views(store):
    seen: list[str] = []

    def walk(component) -> None:
        if hasattr(component, "id"):
            seen.append(component.id if isinstance(component.id, str) else json.dumps(component.id))
        children = getattr(component, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if child is not None and not isinstance(child, (str, int, float)):
                walk(child)

    walk(shell.validation_layout(store))
    assert len(seen) == len(set(seen)), sorted({item for item in seen if seen.count(item) > 1})
