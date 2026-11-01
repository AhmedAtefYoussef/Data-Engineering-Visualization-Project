# Affordable housing and street safety in New York City

ICS 503 Data Engineering and Visualization.

**Live site:** https://data-engineering-visualization-proj-three.vercel.app/

This project asks whether the streets around city-financed affordable housing have different crash records from comparable streets, before and after the buildings are completed. It joins two NYC Open Data tables that share no key, the housing production table and the police crash table, by ZIP code and year and by distance (crashes within 250 m of a building), and it uses injury crashes from 2016 to 2025 as the outcome.

For the 128 new buildings that could be compared fairly, injury crashes within 250 m changed by -4.6% after completion relative to sites not yet completed (95% interval -11.9% to +3.3%), so the data cannot tell this from no change, and the other before-and-after questions are just as inconclusive. The clearest result concerns location before construction: new-construction sites stood on streets with more injury crashes than the typical street point of their borough, 19.1 percentile points above its middle (95% interval 9.4 to 22.3). Every result is an association. The before-and-after comparison rests on a small, less densely built part of the programme, and its estimate depends on how the sites are chosen: three of the 15 robustness variants have an interval that excludes zero, two below it and one above. The answers to the ten research questions are in `docs/METHODOLOGY.md`.

## Contents

- [Datasets](#datasets)
- [What the website does](#what-the-website-does)
- [Repository layout](#repository-layout)
- [Setup and running](#setup-and-running)
- [Tests and checks](#tests-and-checks)
- [Deployment on Vercel](#deployment-on-vercel)
- [How the project was engineered](#how-the-project-was-engineered)
- [Team contributions](#team-contributions)
- [Licence and credits](#licence-and-credits)

## Datasets

| Dataset | ID | Page | API (CSV) |
|---|---|---|---|
| Affordable Housing Production by Building | `hg8x-zxpr` | https://data.cityofnewyork.us/d/hg8x-zxpr | https://data.cityofnewyork.us/resource/hg8x-zxpr.csv |
| Motor Vehicle Collisions - Crashes | `h9gi-nx95` | https://data.cityofnewyork.us/d/h9gi-nx95 | https://data.cityofnewyork.us/resource/h9gi-nx95.csv |

The copies in `notebooks/` were saved on 10 October 2026. The housing file holds all 9,551 rows. The crash file is a filtered copy: crashes dated 2016-01-01 to 2025-12-31 (1,504,722 rows) and 19 of the 29 columns.

## What the website does

One page, read from top to bottom:

| Part | What the reader can do |
|---|---|
| The story | Scroll through five chapters in which the 7,593 located buildings, drawn as points of light, regroup into each chapter's chart: the boroughs, where the sites stood before construction, the fair-test funnel, and fifty equally likely answers against zero. |
| Your block | Search a ZIP code, project or borough, or pick from the sites on the most crash-heavy streets. The record opens beside the list: the site's percentile in its borough, its crashes against the borough median, its before-and-after counts and the sites near it. |
| Test it | Change who is compared (sites, borough, size, comparison group, cluster, cohort) and the main result is recomputed at once, beside the controls, with every scenario tried left as a mark on the interval axis. |
| Explore | Type a question in plain words or set the filters, then Generate a report: four figures and charts over time, place, cause and before-and-after, with a shareable link and a CSV. |
| Findings and Method | The ten research questions with their intervals, the robustness checks, what the data cannot tell, how the tables were joined and the full cleaning log. |

Every figure on the page is read from `app/data/`; none is typed into the code.

**Performance and accessibility.** The charts are built only when the reader comes within about two screens of them (`app/lazy.py`), so the opening page loads no chart library. Phones and touch screens get a light mode: no blur filters, a canvas drawn at one device pixel that draws only while its points move, and viewport units that do not change as the address bar slides. Measured on an emulated iPhone 13 with the processor slowed four times, the first content arrives in about 3.5 seconds (18 before), the main thread is blocked for about 2.5 seconds during load (21.6 before) and the page downloads 2.4 MB (7.6 before). axe-core reports no WCAG 2.2 A or AA violation on phone or desktop, controls are at least 44 px tall on touch screens, and every animated text keeps one plain copy for screen readers.

## Repository layout

```
notebooks/
  NYC_Housing_Safety.ipynb     the notebook, with its outputs stored (222 cells)
  housing_hg8x-zxpr.csv        the housing data
  crashes_h9gi-nx95.csv.gz     the filtered crash data
tools/
  notebook_src/*.py            one file per notebook section (00_title ... 10_story), about 5,300 lines
  assemble_notebook.py         joins the section files into the notebook and executes it
  build_app_data.py            runs the same sections and writes the website data, then checks it
app/
  app.py                       Dash entry point (app.app:server) and security headers
  page.py, sections/           the page and its parts: overview (story), block, lab, explorer, findings, method
  data.py                      loads app/data and verifies its schema at start-up
  stats.py                     the bootstrap and difference-in-differences code, shared with the notebook
  filters.py, search.py, report.py, charts.py, callbacks.py, scenarios.py, block_data.py,
  story.py, story_data.py, story_text.py, components.py, layout.py, theme.py
  assets/css/, assets/js/      styles and browser scripts, loaded in file-name order
  data/                        the files the site reads (about 3.3 MB)
tests/                         13 test files (pipeline, data accuracy, statistics, site, tools, hardening)
docs/
  ARCHITECTURE.md              modules, data flow, how the statistics are shared
  METHODOLOGY.md               methods, the ten questions and their answers
  DATA_DICTIONARY.md           every table and file, column by column
  NUMBER_CHANGES.md            every published number that moved, and why
  AI_USE.md                    the prompts and the full record of how AI agents were used
requirements.txt                site dependencies, pinned (what Vercel installs)
requirements-dev.txt            notebook, data build and tests
pyproject.toml                  Vercel entry point, pinned dependencies, pytest and ruff settings
vercel.json                     function settings for Vercel (maxDuration 60 s)
.vercelignore                   folders left out of the deployment
.github/workflows/tests.yml     CI: ruff, the test suite and a coverage floor on every push
```

## Setup and running

You need Python 3.12 or newer. The notebook stops at its first code cell on an older version.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

**Run the notebook.** Open `notebooks/NYC_Housing_Safety.ipynb` in Jupyter and choose Run All. It reads only the two data files from its own folder. If either is missing, Section 2 downloads it from NYC Open Data (the crash download takes several minutes and resumes if interrupted). With both files present it needs no network, and it can be copied alone into any folder together with them.

**Rebuild the notebook from its sources.** The notebook is generated: edit `tools/notebook_src/`, then run

```bash
python tools/assemble_notebook.py --execute
```

It joins the sections in name order, pastes `app/stats.py` into Section 7 so the notebook imports nothing from the repository, executes every cell inside `notebooks/` and writes the notebook with its outputs. A failing cell is printed and the notebook is left untouched. For a partial run, pass section prefixes and another output: `--sections 01 02 04 --execute --out /tmp/dev.ipynb`.

**Rebuild the website data.**

```bash
python tools/build_app_data.py
```

It executes the notebook sections in one process (about 75 seconds, about 3 GB of memory), exports small aggregated tables to `app/data/`, records hashes, row counts and library versions in `meta.json`, writes the folder atomically, and checks that the website's own statistics reproduce the notebook's RQ3 and RQ7. Two builds from scratch produce identical tables.

**Run the website locally.**

```bash
python -m app.app
```

Open http://127.0.0.1:8050 (set `PORT` for another port). The site reads only `app/data/`. The map's background tiles load from Carto in the browser, so the map needs an internet connection.

## Tests and checks

```bash
ruff check app tests tools
python -m pytest -q tests
```

816 tests pass with 98% coverage of `app/`. They include:

- **Data accuracy:** every exported table and the numbers the site shows, recomputed from the two raw files with separate code. The headline figures and chart totals are checked under 49 filter combinations, 40 of them drawn at random with a fixed seed.
- **Pipeline contract:** a missing, renamed or retyped column stops the site at start-up with the command that rebuilds the data; the export is atomic.
- **Statistics:** the bootstrap and difference-in-differences code against hand-computed cases.
- **Notebook against site:** the result sentences on the site are the notebook's own.
- **Site:** callbacks through Flask's test client, the search parser, scenarios, the block records, hostile inputs and the page structure.

The data-accuracy tests skip themselves when the raw files are not in `notebooks/`. CI runs ruff and the suite on every push, with a coverage floor of 97%.

## Deployment on Vercel

The site runs on Vercel as one Python function, redeployed on every push to `main`; each pull request gets its own preview. `pyproject.toml` names the entry point (`app.app:server`), `vercel.json` gives it a `maxDuration` of 60 seconds, and `.vercelignore` keeps the notebook, tools, tests and docs out of the bundle. The project needs no environment variables.

Keep the installed packages under Vercel's 500 MB limit for a Python function (they take about 410 MB). Above it, Vercel installs the packages on every cold start into a small temporary disk, which fills up, and the page shows "A function needed by this page temporarily failed". That is why the site reads its data with fastparquet (about 17 MB) instead of pyarrow (about 170 MB), and why every version is pinned. Measured in the deployed configuration (the shipped files only, Python 3.12, the pinned packages), a cold start loads the data in about a second and uses about 300 MB.

## How the project was engineered

The code and text were drafted by AI coding agents (Claude models, through Claude Code) under the direction of the project lead, who designed the workflow, set the constraints, reviewed every result and decided what was merged. The full record, with the prompts, is in `docs/AI_USE.md`. This section summarises the method.

### Orchestrating the agents

The work was organised as a small team of agents with one orchestrator, rather than one long conversation:

- **One orchestrator, many bounded workers.** An orchestrating session planned each wave of work, wrote the briefs, routed tasks, integrated the results and verified them. Workers each received one task with a fixed brief: a role, the files to read, the only files they may edit, the quality bar and the shape of their report.
- **Ownership boundaries.** Each worker edited only its own files (one notebook section, one area of the site), against a frozen interface contract that named the objects each section hands to the next. Parallel work could not collide, and every change is traceable to one owner.
- **Separation of writing and judging.** Auditors and reviewers read the code and executed notebook from one point of view each (requirements, statistics, code, prose, charts, design) and returned findings with a location, a severity and a fix, but could not edit what they judged. Writers then applied the findings.
- **Model tiering for cost and quality.** Reasoning-heavy work (method design, the statistics, the interface contract, architecture and final review) went to the strongest model (Opus), and routine code, prose, charts and tests to a cheaper one (Sonnet); the routing policy for the later rounds also reserved repetitive mechanical work for the cheapest (Haiku). Work that a script can check was never left to a model's judgement.
- **Parallel waves in isolated worktrees.** Later rounds ran several agents at once, each in its own git worktree and branch, merged by the orchestrator only after its checks passed. The commit history shows two such waves: `[t1]` statistics corrections, `[t2]` server hardening and CI, `[t3]` the checked data contract; then `[n2]` the Your block and Test it tools, `[n3]` the explorer, `[n4]` Findings and Method.
- **Execution before acceptance.** Nothing counted as done until it had run on the real data, its figures had been looked at, and the tests passed. Interface work was checked in a headless browser at desktop and phone widths, and performance changes were measured before and after, not assumed.

### The automation pipeline

One rule shapes the pipeline: **the notebook is the source of truth, and everything else is generated from the same code.**

```
tools/notebook_src/*.py ──► assemble_notebook.py ──► notebooks/NYC_Housing_Safety.ipynb   (executed, outputs stored)
        │                        ▲
        │                        └── app/stats.py pasted into Section 7 (one copy of the statistics)
        │
        └──────────────► build_app_data.py ──► app/data/*.parquet, results.json, meta.json
                                 │                  (atomic write, hashes, versions, schema)
                                 └── re-runs the site's statistics: must reproduce RQ3 and RQ7
                                                    │
app/ (Dash) reads only app/data, verifies its schema at start-up
        │
git push ──► GitHub Actions: ruff, 816 tests, coverage floor ──► Vercel preview per pull request ──► main: production
```

The guards are mechanical, not promises:

- **Single source:** no number derived from the data is typed into text; the notebook prints it, and the site reads it from the exported files.
- **Contract:** the build writes a schema into `meta.json` and the site refuses to start on data that does not match it.
- **Reproducibility:** the build records library versions and source hashes, and a rebuild from the raw files gives identical tables. The RQ7 reference points were made independent of how a library breaks distance ties, so they give the same result on any machine.
- **Accuracy:** separate test code recomputes what the site shows from the raw files, under fixed and random queries.
- **Change log:** every published number that moves is listed in `docs/NUMBER_CHANGES.md` with the reason.

### Verification and incidents

The final round checked the pipeline, the notebook and the website against each other: a full rebuild from the raw files, notebook outputs against `results.json`, the website against raw recounts under random queries, timing of every interactive path, and a click-through of the rebuilt site. It found and fixed one reproducibility defect (the RQ7 distance ties). An earlier production incident, where the site failed intermittently because an unpinned dependency pushed the function past Vercel's size limit, was diagnosed from the runtime logs, reproduced locally in the exact deployed configuration and fixed by replacing pyarrow and pinning every package.

## Team contributions

Five members, two research questions and two charts each. The table is the same as Section 13 of the notebook. Section numbers refer to the notebook.

| Member | Name | Student ID | Research questions and charts | Data cleaning | Integration | Website | Other work |
|---|---|---|---|---|---|---|---|
| M1 | Ahmed Atef | 16007032 | RQ1 forecast comparison (dot plot of forecast error); RQ2 within-ZIP slopes (coefficient plot with leads and lags) | Housing: blank unit counts, duplicates, geographic identifiers, dates, unit consistency, size outliers (4.1-4.6) | ZIP by year panel (5.2) | Dropdown filters and the Generate report button | Setup and data loading (1, 2), housing exploration (3.1), README run steps |
| M2 | Nour Mostafa Aly | 16001618 | RQ3 event study (lines with interval bands); RQ4 building size (scatter with fitted slope) | Crashes: key, duplicates, dates, time of day, borough and ZIP (4.7, 4.8) | Ball-tree distance join and its brute-force check, choice of the 250 m buffer (5.3) | Event-study cards and the bootstrap callback | Crash exploration (3.2), bootstrap and difference-in-differences toolkit (7) |
| M3 | Yahia Mohamed | 16004934 | RQ5 new construction vs preservation (interval plot); RQ6 pedestrian and cyclist share by borough (interval plot) | Crashes: coordinates, casualty totals, extreme injury counts, factor groups (4.8-4.11) | Sites, clusters and the nearby-completion screen (5.4) | Search box and query parser | Robustness checks (9), data dictionary |
| M4 | Moaz Ahmed | 16003110 | RQ7 site selection (percentile histogram); RQ8 hour of day (interval plot of day parts) | ZIP recovery from coordinates and its holdout validation (6.1) | Tagging crashes with their nearest site (5.5), conservation checks (6.4) | Map and heat-map cards, page layout, accessibility | Deployment configuration, architecture notes |
| M5 | Yassin Fady | 16005869 | RQ9 deep-affordable share across ZIP codes (scatter by borough); RQ10 contributing factors (interval plot) | Post-integration cleaning: borough, redundant columns, data types (6.2, 6.3) | Cumulative stock and citywide index (6.2, 6.5), final schema | Findings and data-and-method sections of the page | Findings, limits, impact, reproducibility (10-14), hosting |

Each member should be able to explain the code and text of their own two questions.

## Licence and credits

The code is released under the MIT licence (`LICENSE`). The two datasets come from NYC Open Data and are used under its terms of use (https://www.nyc.gov/home/terms-of-use.page). The map's background tiles are © CARTO, with map data © OpenStreetMap contributors (ODbL).
