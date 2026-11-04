# NYC Affordable Housing and Street Safety Analytics

[![Tests](https://github.com/AhmedAtefYoussef/NYC-Affordable-Housing-Street-Safety-Analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/AhmedAtefYoussef/NYC-Affordable-Housing-Street-Safety-Analytics/actions/workflows/tests.yml)

**Live site:** https://data-engineering-visualization-proj-three.vercel.app/

A data engineering and visualization project that tests whether injury crashes change around city-financed affordable housing. It joins two NYC Open Data tables that share no key, runs a documented statistical analysis, and publishes the results as an executed notebook and an interactive Dash website. The website reads only pre-computed, verified data, and a test suite recomputes every figure it shows from the raw files.

## Key findings

- The analysis covers 7,593 located buildings. Of these, 128 new-construction buildings could be compared fairly.
- Injury crashes within 250 m changed by -4.6% after completion, relative to sites not yet completed. The 95% interval runs from -11.9% to +3.3%, so the data cannot distinguish this from no change.
- Before construction, new-construction sites stood on streets with more injury crashes than the typical street in their borough. They sat 19.1 percentile points above the borough median (95% interval 9.4 to 22.3). This is the clearest result in the analysis.
- Three of 15 robustness variants have an interval that excludes zero, two below and one above. The before-and-after estimate therefore depends on how the comparison sites are chosen.

All results are associations. The project does not estimate causal effects.

## The question and the data

The project asks whether the streets around city-financed affordable housing have different crash records from comparable streets, before and after the buildings are completed. The answer combines two NYC Open Data tables:

| Dataset | ID | Page | API (CSV) |
|---|---|---|---|
| Affordable Housing Production by Building | `hg8x-zxpr` | https://data.cityofnewyork.us/d/hg8x-zxpr | https://data.cityofnewyork.us/resource/hg8x-zxpr.csv |
| Motor Vehicle Collisions - Crashes | `h9gi-nx95` | https://data.cityofnewyork.us/d/h9gi-nx95 | https://data.cityofnewyork.us/resource/h9gi-nx95.csv |

The tables are joined by ZIP code and year, and by distance: a crash counts toward a building when it occurs within 250 m of it. The outcome is injury crashes from 2016 to 2025. The copies in `notebooks/` were saved on 10 October 2026. The housing file holds all 9,551 rows. The crash file is a filtered copy with 1,504,722 rows from 2016-01-01 to 2025-12-31 and 19 of its 29 columns.

## Website

The site is one page, read from top to bottom:

| Section | What the reader can do |
|---|---|
| The story | Scroll through five chapters. The located buildings, drawn as points of light, regroup into each chapter's chart: boroughs, pre-construction location, the fair-test funnel, and fifty equally likely answers against zero. |
| Your block | Search by ZIP code, project or borough, or pick a site on a crash-heavy street. The record shows the site's percentile in its borough, its crashes against the borough median, its before-and-after counts and nearby sites. |
| Test it | Change who is compared (sites, borough, size, comparison group, cluster, cohort). The main result is recomputed immediately, and every scenario appears as a mark on the interval axis. |
| Explore | Ask a question in plain words or set filters, then generate a report with four figures, a shareable link and a CSV export. |
| Findings and method | The ten research questions with their intervals, robustness checks, what the data cannot show, the table joins and the full cleaning log. |

Every figure on the page is read from `app/data/`. No number is typed into the code.

**Performance and accessibility.** Charts are built only when the reader approaches them (`app/lazy.py`), so the opening page loads no chart library. Touch devices receive a lighter rendering mode. On an emulated iPhone 13 with the processor slowed four times, first content arrives in about 3.5 seconds (18 seconds before optimization), and the page downloads 2.4 MB (7.6 MB before). axe-core reports no WCAG 2.2 A or AA violations on phone or desktop. Animated text keeps a plain copy for screen readers.

## Architecture

The notebook is the source of truth. Everything else is generated from the same code.

```
tools/notebook_src/*.py ──► assemble_notebook.py ──► notebooks/NYC_Housing_Safety.ipynb   (executed, outputs stored)
        │                        ▲
        │                        └── app/stats.py pasted into Section 7 (one copy of the statistics)
        │
        └──────────────► build_app_data.py ──► app/data/*.parquet, results.json, meta.json
                                 │                  (atomic write, hashes, versions, schema)
                                 └── re-runs the site's statistics and checks they reproduce RQ3 and RQ7
                                                    │
app/ (Dash) reads only app/data and verifies its schema at start-up
        │
git push ──► GitHub Actions (ruff, tests, coverage floor) ──► Vercel preview per pull request ──► main: production
```

Four guards keep the pieces consistent:

- Single source: the notebook prints every data-derived number, and the site reads it from the exported files.
- Contract: the build writes a schema into `meta.json`, and the site refuses to start on data that does not match it.
- Reproducibility: the build records library versions and source hashes. A rebuild from the raw files produces identical tables.
- Change log: every published number that moves is listed in `docs/NUMBER_CHANGES.md` with its reason.

### Statistical methods

- Distance join: buildings and crashes are matched with a haversine ball tree (`scikit-learn`), and the match is checked against a brute-force computation.
- Event study and difference-in-differences, with bootstrap intervals, in a toolkit shared by the notebook and the website (`app/stats.py`).
- Robustness: 15 variants of the before-and-after estimate, reported with the intervals and listed in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

### Tech stack

- Website: Python 3.12+, Dash 3.4 on Flask, Plotly 5.24, pandas, NumPy and fastparquet, with every version pinned.
- Notebook and data build: scikit-learn, SciPy, matplotlib, seaborn, pyarrow and jupytext (see `requirements-dev.txt`).
- Quality: pytest with coverage, ruff and GitHub Actions CI.
- Hosting: Vercel, one Python function with a 60-second limit.

## Repository layout

```
notebooks/
  NYC_Housing_Safety.ipynb     the notebook, with stored outputs (222 cells)
  housing_hg8x-zxpr.csv        housing production data
  crashes_h9gi-nx95.csv.gz     filtered crash data
tools/
  notebook_src/*.py            one file per notebook section (00_title to 10_story), about 5,300 lines
  assemble_notebook.py         joins the sections, executes the notebook and writes it
  build_app_data.py            runs the sections, exports the website data and checks it
app/
  app.py                       Dash entry point (app.app:server) and security headers
  page.py, sections/           the page and its parts: story, block, explorer, findings, method
  data.py                      loads app/data and verifies its schema at start-up
  stats.py                     bootstrap and difference-in-differences code, shared with the notebook
  filters.py, search.py, report.py, charts.py, callbacks.py, scenarios.py, block_data.py,
  story*.py, components.py, layout.py, theme.py
  assets/css/, assets/js/      styles and browser scripts
  data/                        the files the site reads (about 3.3 MB)
tests/                         13 test files: pipeline, data accuracy, statistics, site, tools, hardening
docs/                          architecture, methodology, data dictionary, number changes, AI use record
.github/workflows/tests.yml    CI: ruff, the test suite and the coverage floor on every push
pyproject.toml                 Vercel entry point, pinned dependencies, pytest and ruff settings
vercel.json                    function settings (maxDuration 60 s)
```

## Getting started

You need Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

**Run the notebook.** Open `notebooks/NYC_Housing_Safety.ipynb` and choose Run All. It reads the two data files from its own folder. If either file is missing, Section 2 downloads it from NYC Open Data. The notebook needs no network access when both files are present, and it can be copied alone into any folder that holds them.

**Rebuild the notebook from its sources.** The notebook is generated, so edit `tools/notebook_src/` and then run:

```bash
python tools/assemble_notebook.py --execute
```

A failing cell stops the build and leaves the notebook untouched. For a partial run, pass section prefixes and an output path, for example `--sections 01 02 04 --execute --out /tmp/dev.ipynb`.

**Rebuild the website data.**

```bash
python tools/build_app_data.py
```

The build runs the notebook sections in one process (about 75 seconds, about 3 GB of memory), exports aggregated tables to `app/data/`, records hashes and versions in `meta.json`, writes the folder atomically, and checks that the site's statistics reproduce the notebook's results.

**Run the website locally.**

```bash
python -m app.app
```

Open http://127.0.0.1:8050 (set `PORT` to change the port). The site reads only `app/data/`. Map tiles load from CARTO in the browser, so the map needs an internet connection.

## Testing and quality

```bash
ruff check app tests tools
python -m pytest -q tests
```

The suite has 816 tests and reaches 98% coverage of `app/`. CI runs ruff and the suite on every push, with a coverage floor of 97%. The tests cover:

- Data accuracy: every exported table and every number the site shows, recomputed from the raw files with separate code. Headline figures are checked under 49 filter combinations, 40 of them drawn at random with a fixed seed.
- Pipeline contract: a missing, renamed or retyped column stops the site at start-up and prints the command that rebuilds the data.
- Statistics: the bootstrap and difference-in-differences code, checked against hand-computed cases.
- Notebook and site agreement: the result sentences on the site come from the notebook.
- Site behaviour: callbacks through Flask's test client, the search parser, scenarios, block records and hostile inputs.

The data-accuracy tests skip themselves when the raw files are absent from `notebooks/`.

## Deployment

The site runs on Vercel as one Python function and redeploys on every push to `main`. Each pull request gets its own preview. `pyproject.toml` sets the entry point, and `.vercelignore` keeps the notebook, tools, tests and docs out of the bundle.

The installed packages must stay under Vercel's 500 MB limit for a Python function, and they currently take about 410 MB. Above that limit, Vercel reinstalls packages on each cold start into a small disk, and the page fails. For this reason the site reads its data with fastparquet (about 17 MB) instead of pyarrow (about 170 MB), and every version is pinned. In the deployed configuration, a cold start loads the data in about a second and uses about 300 MB.

An earlier production incident was caused by an unpinned dependency that pushed the function past this limit. It was diagnosed from the runtime logs, reproduced locally with the deployed configuration, and fixed by pinning every package.

## Engineering approach

The code and text were drafted with AI coding agents (Claude models, through Claude Code) under the direction of the project lead. The lead set the constraints, reviewed every result and decided what was merged. `docs/AI_USE.md` records the prompts and the process.

- Ownership boundaries: each agent edited only its own files, against a frozen interface that named the objects passed between sections.
- Separation of writing and judging: reviewers read the code and the executed notebook and reported findings with a location and a fix, but did not edit the work they reviewed.
- Parallel work in isolated git worktrees: later rounds ran several agents at once, and the orchestrator merged a branch only after its checks passed.
- Verification before acceptance: a change counted as done only after it ran on the real data, its figures had been inspected and the tests passed. Performance changes were measured before and after.
- Mechanical checks over judgement: anything a script can verify, such as schema, hashes, number traceability and test results, is checked by code.

## Limitations

- The before-and-after comparison rests on a smaller, less densely built part of the programme. Its estimate depends on how the sites are selected.
- Comparison sites were not assigned at random, so every result is an association.
- The site's trend is a straight line on the transformed scale. It fits 2011 and 2012 well and runs somewhat too steep after that.

## Documentation

| If you want to know | Read |
|---|---|
| The ten research questions and their answers | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| Modules, data flow and how the statistics are shared | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Every table and file, column by column | [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) |
| Every published number that changed, and why | [docs/NUMBER_CHANGES.md](docs/NUMBER_CHANGES.md) |
| How AI agents were used in the project | [docs/AI_USE.md](docs/AI_USE.md) |

## License and credits

The code is released under the MIT licence (see `LICENSE`). The datasets come from NYC Open Data and are used under its [terms of use](https://www.nyc.gov/home/terms-of-use.page). Map tiles are © CARTO, and map data is © OpenStreetMap contributors (ODbL).
