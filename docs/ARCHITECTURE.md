# Architecture

The project has three deliverables that share one code base: a notebook, a website and the documents in `docs/`. The notebook is the source of truth. The website shows the notebook's results and lets a reader filter the underlying counts.

## Data flow

```
 NYC Open Data API (hg8x-zxpr, h9gi-nx95)
          |  downloaded by Section 2 only if the files are missing
          v
 notebooks/housing_hg8x-zxpr.csv       notebooks/crashes_h9gi-nx95.csv.gz
          |                                        |
          +----------------+-----------------------+
                           v
            tools/notebook_src/*.py  (section files, jupytext "percent" format)
                  |                                   |
   tools/assemble_notebook.py                tools/build_app_data.py
   joins the sections, executes them          runs the same sections in one process
                  |                           (except the exploratory Section 3)
                  v                                   v
 notebooks/NYC_Housing_Safety.ipynb              app/data/*.parquet, results.json, meta.json
 (stored outputs, no repository imports)                 |
                                                         v
                                              app/ (Dash, reads only app/data/)
                                                         |
                                              gunicorn on Render
```

## The notebook

The notebook is generated from the files in `tools/notebook_src/`, one file per section, joined in name order.

| File | Section |
|---|---|
| `00_title.py` | Title, abstract, team table, dataset links, map of the assignment's requirements |
| `01_setup.py` | Libraries, constants (window, buffer, bootstrap settings, seed), plot style, helpers |
| `02_data.py` | Loading through the API, overview, why these datasets, feasibility, known issues |
| `03_eda.py` | Exploratory analysis of both tables |
| `04_cleaning.py` | Pre-integration cleaning and the cleaning log |
| `05_integration.py` | The ZIP by year panel, the distance join, sites and clusters, event panel |
| `06_post_integration.py` | New gaps from the joins, ZIP recovery, redundant columns, types, conservation checks, citywide index |
| `07_toolkit.py` | Bootstrap, rank correlation, within-unit slopes, before and after, multiple tests, forecast evaluation, with a self-test |
| `08a_rq_1_5.py`, `08b_rq_6_10.py` | The ten research questions |
| `09_robustness.py` | Robustness checks of the central comparison and of the ZIP-level results |
| `10_story.py` | What was found, limits, impact, prompts, contributions, reproducibility |

`tools/assemble_notebook.py` parses each file, concatenates the cells, runs them with nbclient inside `notebooks/` (where the two data files sit) and writes `notebooks/NYC_Housing_Safety.ipynb`. The notebook imports nothing from the repository. It needs only the two data files and the libraries in `requirements.txt`, and downloads the files if they are missing. Numbers that come from the data are printed by code, so the text follows the data.

## The website data

`tools/build_app_data.py` executes the section sources in one namespace and exports compact tables to `app/data/` (about 3.3 MB in all). Each crash table carries the same seven counts, one per "who was harmed" option, so a filter can change the outcome without a new table. `docs/DATA_DICTIONARY.md` lists every file. After the export the script runs the website's own statistics code on the exported tables and asserts that it reproduces the notebook's RQ3 and RQ7 estimates and intervals.

## The website

The site is a single Dash page served by gunicorn. It loads `app/data/` once at start-up, builds the layout, registers the callbacks and computes the default report so the first visitor does not wait for the bootstraps. It imports nothing from the notebook or the raw data, and it downloads nothing at start-up.

| Module | Responsibility |
|---|---|
| `app.py` | Entry point (`app.app:server` for gunicorn, `python -m app.app` locally); serves the map grid as cached, gzipped GeoJSON |
| `data.py` | Loads the parquet files and filters the pre-aggregated tables |
| `filters.py` | The filter model: borough, from and to year, construction type, building size, contributing factor, time of day, who was harmed, crash location |
| `search.py` | Query parser; a longest-phrase vocabulary lookup plus year patterns, with typo tolerance. It sets filters and reports what it understood and what it ignored |
| `stats.py` | Ports of the notebook's bootstrap and before-and-after statistics |
| `charts.py` | Figure builders (pure functions): map, site selection, borough lines, completions, factors, weekday by hour heatmap, ZIP scatter, event study |
| `report.py` | Turns one filter selection into every panel of the report |
| `story.py` | Findings, limits, impact and data sections, written from `results.json` |
| `layout.py`, `callbacks.py` | Page layout and the callbacks |
| `theme.py`, `assets/` | Design tokens (one source for CSS variables and the plotly template), `site.css`, `site.js` |

Charts and numbers change only when the reader presses Generate report. Typing in the search box or changing a dropdown updates the filters, marks the report as out of date, and nothing is recomputed until the button is pressed. Each chart card states its filters and sample size, and a selection with too little data shows an empty state.

If `app/data/` is missing, the page shows a short "not ready" message and the server log names the build command.

## Deployment

`render.yaml` defines one free Render web service: Python 3.12.8, `pip install -r requirements.txt` as the build, and `gunicorn app.app:server --workers 1 --threads 4 --timeout 120 --preload` as the start command. `Procfile` has the same start command for hosts that read one. `runtime.txt` names the Python version. The site needs no environment variables, disk or database. It has not been deployed yet.

## Tests

`tests/test_site.py` (79 tests, `python -m pytest -q tests`) covers the query parser, the filters, conservation of totals across every table, the chart builders, the callbacks through Flask's test client, and agreement between the site's numbers and the exported results.

## Risks and how the design handles them

| Risk | Handling |
|---|---|
| API limits, timeouts, the default row limit | Section 2 pages on `collision_id`, retries with back-off, appends to a temporary file and resumes after an interruption |
| Memory use with 1.5 million crashes | Typed columns, small integer types and string columns backed by Arrow; the raw crash frame is released after cleaning |
| Distances in degrees | Haversine distances in metres; a ball tree checked against a brute-force computation |
| The same crash counted for several overlapping buffers | One count per site, clusters of linked sites, cluster bootstrap, an overlap measurement in Section 5.4 |
| Reporting change in 2020 | Injury crashes as the outcome; a comparison group over the same years; all-crash counts only as a check |
| Gaps created by joins | Section 6 logs each repair; ZIP recovery is validated on hidden rows and flagged in the data |
| Chart and text drifting from the data | Data-derived numbers are printed by code; the site's story reads `results.json`; a test compares site and notebook |
| Website load time | Pre-aggregated cubes, a gzipped map grid and warmed caches; the site never touches raw data |
| Render free plan, 512 MB | About 3.3 MB of data and one worker; memory was about 280 MB locally and has not been measured on Render |
| A stale site after the notebook changes | `build_app_data.py` regenerates every file from the same section sources and checks RQ3 and RQ7 |
