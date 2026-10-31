"""Shared building blocks for the views. Pure functions returning Dash components; no data logic, no callbacks."""
from __future__ import annotations

from dash import dcc, html

from app.theme import GRAPH_CONFIG


def view(children: list, class_name: str = "") -> html.Div:
    """The root of one page. Its entrance is one short fade, set in CSS."""
    return html.Div(children, className=f"view {class_name}".strip())


def page_header(title: str, lead: str | list | None = None, actions: list | None = None) -> html.Header:
    """The title row: an h1 that names the page, one sentence about the data, real actions on the right."""
    return html.Header(className="page-header", children=[
        html.Div([html.H1(title), html.P(lead, className="page-lead") if lead else None], className="page-title"),
        html.Div(actions, className="page-actions") if actions else None,
    ])


def panel(title: str | None, children: list, *, note: str | list | None = None, aside: str | list | None = None,
          class_name: str = "", panel_id: str | None = None) -> html.Section:
    """A titled surface. `aside` sits right of the title (period, n); `note` is how to read it, under the content."""
    head = html.Header([html.H2(title), html.Div(aside, className="panel-aside") if aside else None],
                       className="panel-head") if title else None
    extra = {"id": panel_id} if panel_id else {}
    return html.Section([head, *children, html.P(note, className="panel-note") if note else None],
                        className=f"panel {class_name}".strip(), **extra)


def metric(value: str, label: str, detail: str | None = None, tone: str = "", count_to: float | None = None) -> html.Div:
    """One figure with its label. `count_to` makes assets/js/00-core.js count up to it, ending on `value`'s text."""
    extra = {"data-countup": str(count_to), "data-final": value} if count_to is not None else {}
    return html.Div(className="metric", children=[
        html.P(label, className="metric-label"),
        html.P(value, className=f"metric-value {tone}".strip(), **extra),
        html.P(detail, className="metric-detail") if detail else None,
    ])


def metric_strip(metrics: list, strip_id: str | None = None) -> html.Div:
    """Figures in one container split by hairlines. Never a tile per figure."""
    extra = {"id": strip_id} if strip_id else {}
    return html.Div(metrics, className="metric-strip", style={"--metric-count": len(metrics)}, **extra)


def interval_mark(low: float, estimate: float, high: float, reach: float | None = None, label: str | None = None) -> html.Div:
    """One interval against zero, drawn in place. It is read by the text beside it unless a `label` is given."""
    reach = reach or max(abs(low), abs(high), abs(estimate)) * 1.15 or 1.0

    def position(value: float) -> float:
        return min(100.0, max(0.0, 50 + 50 * value / reach))

    attrs = {"role": "img", "aria-label": label} if label else {"aria-hidden": "true"}
    return html.Div(className="interval", **attrs, children=[
        html.Span(className="interval-zero"),
        html.Span(className="interval-bar", style={"left": f"{position(low):.2f}%",
                                                   "right": f"{100 - position(high):.2f}%"}),
        html.Span(className="interval-dot", style={"left": f"{position(estimate):.2f}%"}),
    ])


def graph(graph_id: str, figure=None, *, size_class: str = "", config: dict | None = None) -> html.Div:
    """A chart in a box whose height is set in CSS, so a new figure never moves the page."""
    props = {"figure": figure} if figure is not None else {}
    return html.Div(dcc.Graph(id=graph_id, config=config or GRAPH_CONFIG, responsive=True,
                              style={"height": "100%"}, **props),
                    className=f"graph {size_class}".strip())


def skeleton(lines: int = 3, class_name: str = "") -> html.Div:
    """A loading placeholder of the final shape. Announced by the container, so it is hidden from readers."""
    return html.Div([html.Span(className="skeleton-line") for _ in range(lines)],
                    className=f"skeleton {class_name}".strip(), **{"aria-hidden": "true"})


def view_skeleton() -> html.Div:
    """What shows between the first paint and the first view."""
    return html.Div([skeleton(2, "skeleton-title"), skeleton(4, "skeleton-block")], className="view",
                    role="status", **{"aria-label": "Loading"})


def segmented(control_id: str, options: list[dict], value, label: str) -> html.Div:
    """A row of mutually exclusive choices: a real radio group, styled as one segmented track."""
    return html.Div(className="segmented-field", children=[
        html.Span(label, className="field-label", id=f"{control_id}-label"),
        dcc.RadioItems(id=control_id, options=options, value=value, inline=True, className="segmented",
                       labelClassName="segment", **{"aria-labelledby": f"{control_id}-label"}),
    ])
