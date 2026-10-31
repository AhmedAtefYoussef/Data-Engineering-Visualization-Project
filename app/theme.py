"""Design tokens: one source for the page (CSS variables) and the charts (plotly template).

`css_variables()` is written into the page head, and app/assets/site.css uses only those
variables, so a colour, size or radius is defined in exactly one place. The data colours are
the notebook's Okabe-Ito colours (housing blue, crash vermilion), so a chart on the site and a
figure in the notebook read the same way.
"""
from __future__ import annotations

import plotly.graph_objects as go

# Type: Schibsted Grotesk sets titles and figures; Hanken Grotesk sets text, controls and charts. Both are
# variable faces from Google Fonts with tabular figures, loaded with font-display: swap. The "... Fallback"
# families are declared in assets/css/00-base.css as local system fonts with matched metrics.
FONT_STACK = '"Hanken Grotesk", "Hanken Grotesk Fallback", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'
FONT_DISPLAY = '"Schibsted Grotesk", "Schibsted Grotesk Fallback", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'
FONTS_URL = ("https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400..700"
             "&family=Hanken+Grotesk:wght@400..700&display=swap")

COLOR = {
    # surfaces and lines (cool neutrals, light theme only)
    "page": "#f7f8f9",
    "surface": "#ffffff",
    "sunken": "#f0f2f4",
    "active": "#e8ebee",
    "line": "#e4e7eb",
    "line-strong": "#cdd2d9",
    # text
    "ink": "#0e1116",
    "ink-2": "#3c4450",
    "ink-3": "#5c6673",
    # the one primary action is near-black; colour is kept for data
    "primary": "#0e1116",
    "primary-hover": "#2a303a",
    "on-primary": "#ffffff",
    # links and the focus ring
    "accent": "#0b5f96",
    "accent-strong": "#084a75",
    "accent-soft": "#e2edf5",
    "on-accent": "#ffffff",
    # notices
    "caution-ink": "#7a4300",
    "caution-soft": "#fdf1e2",
    # data: identity colours, validated for colour-vision deficiency with the dataviz validator
    "housing": "#0072B2",
    "preservation": "#009E73",
    "crash": "#D55E00",
    "crash-ink": "#b34e00",       # the crash hue at text contrast (4.5:1 on white)
    "comparison": "#6b7783",
    # the not-yet-completed line of the event study: a dark neutral, because a mid grey sits too close
    # to the blue and the green it is drawn with
    "reference": "#3c4450",
    "placebo": "#B26F00",
}
BOROUGH_COLOR = {"Bronx": "#E69F00", "Brooklyn": "#0072B2", "Manhattan": "#D55E00",
                 "Queens": "#009E73", "Staten Island": "#CC79A7"}
# A second, non-colour channel for the five borough lines (two of the colours are close for deuteranopes).
BOROUGH_SYMBOL = {"Bronx": "circle", "Brooklyn": "square", "Manhattan": "diamond",
                  "Queens": "triangle-up", "Staten Island": "x"}
CONSTRUCTION_COLOR = {"New Construction": COLOR["housing"], "Preservation": COLOR["preservation"]}
CONSTRUCTION_SYMBOL = {"New Construction": "circle", "Preservation": "square"}
# Magnitude: one hue (the crash vermilion), light to dark.
SEQUENTIAL = [[0.0, "#fdf0e6"], [0.2, "#fbd3b5"], [0.4, "#f5ac7a"], [0.6, "#e98443"],
              [0.8, "#d55e00"], [1.0, "#7f3500"]]

SPACE = {"1": "0.25rem", "2": "0.5rem", "3": "0.75rem", "4": "1rem", "5": "1.5rem", "6": "2rem",
         "7": "3rem", "8": "4.5rem"}
# One type scale for a working screen: titles near 26px, figures up to 56px.
TYPE = {"xs": "0.75rem", "sm": "0.875rem", "base": "0.9375rem", "md": "1.0625rem", "lg": "1.25rem",
        "xl": "1.625rem", "2xl": "2.125rem", "3xl": "2.75rem", "4xl": "3.5rem"}
RADIUS = "8px"
SHADOW = "0 8px 24px rgb(14 17 22 / 0.12)"      # things that truly float: menus, dialogs
SHADOW_RAISED = SHADOW
# Motion: one easing curve (a fast start that settles) and three durations. Interface feedback stays
# under 250 ms; assets/css/00-base.css switches all of it off under prefers-reduced-motion.
EASE_OUT = "cubic-bezier(0.16, 1, 0.3, 1)"
DURATION = {"fast": "120ms", "base": "200ms", "slow": "360ms"}
CHART_FONT_SIZE = 13
CHART_TICK_SIZE = 12
# Chart toolbar: only "download as image" and "reset" stay on the charts (dragging still zooms and a
# double click resets), so the toolbar is two quiet buttons and cannot sit on top of a legend.
MODEBAR_REMOVE = ["zoom", "pan", "zoomin", "zoomout", "autoscale", "lasso", "select"]


def css_variables() -> str:
    """The tokens as a `:root { --name: value; }` block for the page head."""
    lines = [f"--color-{name}: {value};" for name, value in COLOR.items()]
    lines += [f"--space-{name}: {value};" for name, value in SPACE.items()]
    lines += [f"--text-{name}: {value};" for name, value in TYPE.items()]
    lines += [f"--duration-{name}: {value};" for name, value in DURATION.items()]
    lines += [f"--radius: {RADIUS};", f"--shadow: {SHADOW};", f"--shadow-raised: {SHADOW_RAISED};",
              f"--ease-out: {EASE_OUT};", f"--font: {FONT_STACK};", f"--font-display: {FONT_DISPLAY};",
              "color-scheme: light;"]
    return ":root {" + " ".join(lines) + "}"


def plotly_template() -> go.layout.Template:
    """Chart chrome built from the same tokens: recessive grid, page font, no in-figure title."""
    axis = {"gridcolor": COLOR["sunken"], "gridwidth": 1, "linecolor": COLOR["line-strong"],
            "zerolinecolor": COLOR["line-strong"], "zerolinewidth": 1, "ticks": "outside", "ticklen": 4,
            "tickcolor": COLOR["line-strong"], "automargin": True,
            "title": {"font": {"size": CHART_TICK_SIZE, "color": COLOR["ink-2"]}, "standoff": 12},
            "tickfont": {"size": CHART_TICK_SIZE, "color": COLOR["ink-3"]}}
    return go.layout.Template(layout={
        "font": {"family": FONT_STACK, "size": CHART_FONT_SIZE, "color": COLOR["ink-2"]},
        "paper_bgcolor": COLOR["surface"], "plot_bgcolor": COLOR["surface"],
        "colorway": [COLOR["housing"], COLOR["crash"], COLOR["preservation"]],
        "margin": {"l": 8, "r": 12, "t": 40, "b": 8},
        "xaxis": axis, "yaxis": axis,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                   "font": {"size": CHART_TICK_SIZE, "color": COLOR["ink-2"]}, "bgcolor": "rgba(0,0,0,0)",
                   "itemsizing": "constant"},
        "hoverlabel": {"bgcolor": COLOR["surface"], "bordercolor": COLOR["line-strong"], "align": "left",
                       "font": {"family": FONT_STACK, "size": CHART_FONT_SIZE, "color": COLOR["ink"]}},
        "annotationdefaults": {"font": {"size": CHART_TICK_SIZE, "color": COLOR["ink-3"]}},
        # the toolbar is quiet until it is used, and still passes 3:1 against the chart surface
        "modebar": {"bgcolor": "rgba(255,255,255,0.9)", "color": COLOR["ink-3"], "activecolor": COLOR["accent"],
                    "remove": MODEBAR_REMOVE},
        "separators": ".,",
        "hovermode": "closest",
        "dragmode": "zoom",
    })


# Plotly config shared by the explorer charts and the story figures.
GRAPH_CONFIG = {"displaylogo": False, "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
