"""Entry point. `app.app:server` is the WSGI server Vercel runs; `python -m app.app` runs it locally.

The process loads the exported tables once, builds the layout, registers the callbacks and
computes the default report so the first visitor does not wait for the bootstraps. It reads
only app/data/ and imports nothing from the pipeline, the analysis package or the root config.
"""
from __future__ import annotations

import gzip
import json
import logging
import os

import dash
from flask import Response, request

from app import data
from app.callbacks import register, report_outputs
from app.filters import make_filters
from app.layout import build_layout, missing_data_layout
from app.report import GEOJSON_URL
from app.theme import css_variables

TITLE = "Affordable housing and injury crashes in NYC"
DESCRIPTION = ("An interactive report on affordable-housing buildings and police crash reports in "
               "New York City, 2016 to 2025.")
DEFAULT_PORT = 8050
GEOJSON_MAX_AGE_S = 86_400

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("site")

app = dash.Dash(
    __name__, title=TITLE, update_title=None, suppress_callback_exceptions=False,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "description", "content": DESCRIPTION},
               {"name": "color-scheme", "content": "light"}])
server = app.server

# The design tokens are written into the head as CSS variables; app/assets/site.css uses only them.
app.index_string = """<!DOCTYPE html>
<html lang="en">
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
<style id="design-tokens">""" + css_variables() + """</style>
{%css%}
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


def _serve_grid(payload: bytes):
    """A view that serves the map grid, gzip-compressed, cacheable for a day."""
    compressed = gzip.compress(payload, compresslevel=6)

    def crash_grid() -> Response:
        accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "")
        response = Response(compressed if accepts_gzip else payload, mimetype="application/geo+json")
        if accepts_gzip:
            response.headers["Content-Encoding"] = "gzip"
        response.headers["Cache-Control"] = f"public, max-age={GEOJSON_MAX_AGE_S}"
        response.headers["Vary"] = "Accept-Encoding"
        return response

    return crash_grid


try:
    STORE = data.load()
except FileNotFoundError as error:
    STORE = None
    log.warning("%s", error)
    app.layout = missing_data_layout(str(error))
else:
    app.layout = build_layout(STORE)
    register(app, STORE)
    grid_bytes = json.dumps(data.grid_geojson(STORE), separators=(",", ":")).encode("utf-8")
    server.add_url_rule(GEOJSON_URL, "crash_grid", _serve_grid(grid_bytes))
    report_outputs(STORE, make_filters(), None)        # warm the caches behind the default report
    log.info("data loaded: %s crashes, %s housing rows, built %s",
             f"{STORE.meta['rows']['crashes']:,}", f"{STORE.meta['rows']['housing']:,}", STORE.meta["built_at"])


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("PORT", DEFAULT_PORT)))
