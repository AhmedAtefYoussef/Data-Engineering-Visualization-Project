"""Design tokens: one source for the page (CSS variables) and the charts (plotly template).

`css_variables()` is written into the page head, and app/assets/site.css uses only those
variables, so a colour, size or radius is defined in exactly one place. The data colours are
the notebook's Okabe-Ito colours (housing blue, crash vermilion), so a chart on the site and a
figure in the notebook read the same way.
"""
from __future__ import annotations

import plotly.graph_objects as go

FONT_STACK = '"Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'

COLOR = {
    # surfaces and lines (cool neutrals, light theme only)
    "page": "#f3f5f7",
    "surface": "#fcfdfd",
    "sunken": "#e9edf1",
    "line": "#d3dae1",
    "line-strong": "#aeb9c4",
    # text
    "ink": "#111a22",
    "ink-2": "#3d4a57",
    "ink-3": "#586574",
    # the one interface accent (links, focus ring, primary button); same hue as the housing series
    "accent": "#0b5f96",
    "accent-strong": "#084a75",
    "accent-soft": "#e2edf5",
    "on-accent": "#fcfdfd",
    # notices
    "caution-ink": "#7a4300",
    "caution-soft": "#fdf1e2",
    # data: identity colours, validated for colour-vision deficiency with the dataviz validator
    "housing": "#0072B2",
    "preservation": "#009E73",
    "crash": "#D55E00",
    "comparison": "#6b7783",
    # the not-yet-completed line of the event study: a dark neutral, because a mid grey sits too close
    # to the blue and the green it is drawn with (the dataviz validator's separation checks pass for
    # all three pairs; it is darker than the categorical band on purpose, being a reference line)
    "reference": "#3d4a57",
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
TYPE = {"xs": "0.8125rem", "sm": "0.875rem", "base": "1rem", "md": "1.125rem", "lg": "1.375rem",
        "xl": "1.75rem", "2xl": "2.375rem", "3xl": "3rem"}
RADIUS = "8px"
SHADOW = "0 10px 28px rgb(17 26 34 / 0.14)"      # open dropdown menus
SHADOW_RAISED = "0 1px 2px rgb(17 26 34 / 0.2), 0 6px 16px rgb(11 95 150 / 0.24)"   # the primary button only
# Motion: one easing curve (a fast start that settles) and three durations. Interface feedback stays
# under 250 ms; site.css switches all of it off under prefers-reduced-motion.
EASE_OUT = "cubic-bezier(0.16, 1, 0.3, 1)"
DURATION = {"fast": "120ms", "base": "200ms", "slow": "360ms"}
CHART_FONT_SIZE = 13


def css_variables() -> str:
    """The tokens as a `:root { --name: value; }` block for the page head."""
    lines = [f"--color-{name}: {value};" for name, value in COLOR.items()]
    lines += [f"--space-{name}: {value};" for name, value in SPACE.items()]
    lines += [f"--text-{name}: {value};" for name, value in TYPE.items()]
    lines += [f"--duration-{name}: {value};" for name, value in DURATION.items()]
    lines += [f"--radius: {RADIUS};", f"--shadow: {SHADOW};", f"--shadow-raised: {SHADOW_RAISED};",
              f"--ease-out: {EASE_OUT};", f"--font: {FONT_STACK};", "color-scheme: light;"]
    return ":root {" + " ".join(lines) + "}"


def plotly_template() -> go.layout.Template:
    """Chart chrome built from the same tokens: recessive grid, page font, no in-figure title."""
    axis = {"gridcolor": COLOR["sunken"], "gridwidth": 1, "linecolor": COLOR["line-strong"],
            "zerolinecolor": COLOR["line-strong"], "zerolinewidth": 1, "ticks": "outside", "ticklen": 4,
            "tickcolor": COLOR["line-strong"], "automargin": True,
            "title": {"font": {"size": CHART_FONT_SIZE, "color": COLOR["ink-2"]}, "standoff": 10},
            "tickfont": {"size": 12, "color": COLOR["ink-3"]}}
    return go.layout.Template(layout={
        "font": {"family": FONT_STACK, "size": CHART_FONT_SIZE, "color": COLOR["ink-2"]},
        "paper_bgcolor": COLOR["surface"], "plot_bgcolor": COLOR["surface"],
        "colorway": [COLOR["housing"], COLOR["crash"], COLOR["preservation"]],
        "margin": {"l": 8, "r": 12, "t": 34, "b": 8},
        "xaxis": axis, "yaxis": axis,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0,
                   "font": {"size": 12, "color": COLOR["ink-2"]}, "bgcolor": "rgba(0,0,0,0)",
                   "itemsizing": "constant"},
        "hoverlabel": {"bgcolor": COLOR["surface"], "bordercolor": COLOR["line-strong"], "align": "left",
                       "font": {"family": FONT_STACK, "size": 13, "color": COLOR["ink"]}},
        # the zoom toolbar stays quiet until it is used
        "modebar": {"bgcolor": "rgba(0,0,0,0)", "color": COLOR["line-strong"], "activecolor": COLOR["accent"]},
        "separators": ".,",
        "hovermode": "closest",
        "dragmode": "zoom",
    })
