"""Design tokens: one source for the page (CSS variables) and the charts (plotly template).

`css_variables()` is written into the page head, and app/assets/site.css uses only those
variables, so a colour, size or radius is defined in exactly one place. The data colours are
the notebook's Okabe-Ito colours (housing blue, crash vermilion), so a chart on the site and a
figure in the notebook read the same way.
"""
from __future__ import annotations

import plotly.graph_objects as go

# Type: Bodoni Moda, a high-contrast display face with optical sizes, sets the question, headings and figures;
# Hanken Grotesk sets text, controls and charts, with tabular figures. Both come from Google Fonts with
# font-display: swap. The "... Fallback" families in assets/css/00-base.css hold the layout while they load.
FONT_STACK = '"Hanken Grotesk", "Hanken Grotesk Fallback", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'
FONT_DISPLAY = '"Bodoni Moda", "Bodoni Moda Fallback", Georgia, "Times New Roman", serif'
FONTS_URL = ("https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,400..800;1,6..96,400..700"
             "&family=Hanken+Grotesk:wght@400..700&display=swap")

# New York at night: a midnight ground, buildings as warm light, crashes in coral. Gold marks what can be done.
COLOR = {
    "page": "#0a0d1a",
    "surface": "#131829",
    "sunken": "#0f1322",
    "active": "#1f2640",
    "line": "#262d47",
    "line-strong": "#3c4566",
    # text: warm white, stepping down in three contrasts that all pass 4.5:1 on page and surface
    "ink": "#f6f2e9",
    "ink-2": "#c9cad6",
    "ink-3": "#9ca2b8",
    # the one action colour
    "primary": "#f4b942",
    "primary-hover": "#ffd071",
    "on-primary": "#171022",
    # links and the focus ring
    "accent": "#ffd071",
    "accent-strong": "#ffe2a3",
    "accent-soft": "#2e2a2c",
    "on-accent": "#171022",
    # notices
    "caution-ink": "#ffd9a0",
    "caution-soft": "#33261c",
    # data: identity colours chosen for a dark ground
    "housing": "#f4b942",
    "preservation": "#5fd4c4",
    "crash": "#ff6f59",
    "crash-ink": "#ff8a77",
    "comparison": "#8d95b3",
    # the not-yet-completed line of the event study
    "reference": "#c9cad6",
    "placebo": "#c9a2ff",
}
BOROUGH_COLOR = {"Bronx": "#f4b942", "Brooklyn": "#7fb0ff", "Manhattan": "#ff6f59",
                 "Queens": "#5fd4c4", "Staten Island": "#dba2ec"}
# A second, non-colour channel for the five borough lines.
BOROUGH_SYMBOL = {"Bronx": "circle", "Brooklyn": "square", "Manhattan": "diamond",
                  "Queens": "triangle-up", "Staten Island": "x"}
CONSTRUCTION_COLOR = {"New Construction": COLOR["housing"], "Preservation": COLOR["preservation"]}
CONSTRUCTION_SYMBOL = {"New Construction": "circle", "Preservation": "square"}
# Magnitude: dark to hot, so a busy cell glows on the night ground.
SEQUENTIAL = [[0.0, "#241a30"], [0.2, "#4a2238"], [0.4, "#84303d"], [0.6, "#c7463f"],
              [0.8, "#ff6f59"], [1.0, "#ffd0a8"]]

SPACE = {"1": "0.25rem", "2": "0.5rem", "3": "0.75rem", "4": "1rem", "5": "1.5rem", "6": "2rem",
         "7": "3rem", "8": "5rem", "9": "8rem"}
# One type scale: comfortable reading text, and display sizes that carry the page.
TYPE = {"xs": "0.8125rem", "sm": "0.9375rem", "base": "1.0625rem", "md": "1.25rem", "lg": "1.625rem",
        "xl": "2.25rem", "2xl": "3.25rem", "3xl": "4.5rem", "4xl": "6rem"}
RADIUS = "14px"
SHADOW = "0 18px 50px rgb(0 0 0 / 0.5)"      # things that truly float: menus, dialogs
SHADOW_RAISED = SHADOW
# Motion: one easing curve (a fast start that settles) and three durations. Interface feedback stays
# under 250 ms; assets/css/00-base.css switches all of it off under prefers-reduced-motion.
EASE_OUT = "cubic-bezier(0.16, 1, 0.3, 1)"
DURATION = {"fast": "140ms", "base": "260ms", "slow": "520ms", "epic": "1100ms"}
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
              "color-scheme: dark;"]
    return ":root {" + " ".join(lines) + "}"


def plotly_template() -> go.layout.Template:
    """Chart chrome built from the same tokens: recessive grid, page font, no in-figure title."""
    axis = {"gridcolor": COLOR["line"], "gridwidth": 1, "linecolor": COLOR["line-strong"],
            "zerolinecolor": COLOR["line-strong"], "zerolinewidth": 1, "ticks": "outside", "ticklen": 4,
            "tickcolor": COLOR["line-strong"], "automargin": True,
            "title": {"font": {"size": CHART_TICK_SIZE, "color": COLOR["ink-2"]}, "standoff": 12},
            "tickfont": {"size": CHART_TICK_SIZE, "color": COLOR["ink-3"]}}
    return go.layout.Template(layout={
        "font": {"family": FONT_STACK, "size": CHART_FONT_SIZE, "color": COLOR["ink-2"]},
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
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
        "modebar": {"bgcolor": "rgba(19,24,41,0.9)", "color": COLOR["ink-3"], "activecolor": COLOR["accent"],
                    "remove": MODEBAR_REMOVE},
        "separators": ".,",
        "hovermode": "closest",
        "dragmode": "zoom",
    })


# Plotly config shared by the explorer charts and the story figures.
GRAPH_CONFIG = {"displaylogo": False, "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
