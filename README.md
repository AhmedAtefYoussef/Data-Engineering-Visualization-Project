# Affordable housing and street safety in New York City

ICS 503 Data Engineering and Visualization, Project Milestone 1.

**Live site:** https://data-engineering-visualization-proj-three.vercel.app/

This project asks whether the streets around city-financed affordable housing have different crash records from comparable streets, before and after the buildings are completed. It joins two NYC Open Data tables that share no key, the housing production table and the police crash table, by ZIP code and year and by distance (crashes within 250 m of a building), and it uses injury crashes from 2016 to 2025 as the outcome. For the 128 new buildings that could be compared fairly, injury crashes within 250 m changed by -4.6% after completion relative to sites not yet completed (95% interval -11.9% to +3.3%), so the data cannot tell this from no change, and the other before-and-after questions are just as inconclusive. The clearest result concerns location before construction: new-construction sites stood on streets with more injury crashes than the typical street point of their borough, about 20 percentile points above its middle (95% interval 10.4 to 23.1). Every result is an association, the before-and-after comparison rests on a small, less densely built part of the programme, and its sign changes with how the sites are chosen (one of 11 robustness variants has an interval that excludes zero). The answers to the ten research questions are in `docs/METHODOLOGY.md`.

## Datasets

| Dataset | ID | Page | API (CSV) |
|---|---|---|---|
| Affordable Housing Production by Building | `hg8x-zxpr` | https://data.cityofnewyork.us/d/hg8x-zxpr | https://data.cityofnewyork.us/resource/hg8x-zxpr.csv |
| Motor Vehicle Collisions - Crashes | `h9gi-nx95` | https://data.cityofnewyork.us/d/h9gi-nx95 | https://data.cityofnewyork.us/resource/h9gi-nx95.csv |

The copies in `notebooks/` were saved on 10 October 2026. The housing file holds all 9,551 rows. The crash file is a filtered copy: crashes dated 2016-01-01 to 2025-12-31 (1,504,722 rows) and 19 of the 29 columns.

## Repository layout

```
notebooks/
  NYC_Housing_Safety.ipynb     the notebook, with its outputs stored (220 cells)
  housing_hg8x-zxpr.csv        the housing data
  crashes_h9gi-nx95.csv.gz     the filtered crash data
tools/
  assemble_notebook.py         builds the notebook from the section files and runs it
  notebook_src/*.py            one file per notebook section (00_title ... 10_story), about 5,000 lines
  build_app_data.py            runs the same sections and writes the website data
app/
  app.py                       Dash entry point (app.app:server)
  data.py, filters.py, search.py, stats.py, charts.py, report.py,
  story.py, layout.py, callbacks.py, theme.py
  assets/                      site.css, site.js
  data/                        the files the site reads (about 3.3 MB)
tests/
  test_site.py                 parser, filters, charts, callbacks, totals
  test_data_accuracy.py        site numbers recomputed from the raw files
  conftest.py
docs/
  ARCHITECTURE.md              modules, data flow, how the statistics are shared
  METHODOLOGY.md               methods, the ten questions and their answers
  DATA_DICTIONARY.md           every table and file, column by column
  AI_USE.md                    prompts, build process and the AI-use record
requirements.txt                site dependencies (what Vercel installs)
requirements-dev.txt            notebook, data build and tests
pyproject.toml                  Vercel entry point (and pytest settings)
vercel.json                     function settings for Vercel (maxDuration 60 s)
.vercelignore                   folders left out of the deployment
```

The statistics (cluster bootstrap, difference-in-differences) live once in `app/stats.py`. `tools/assemble_notebook.py` pastes that file into Section 7 of the notebook (the cell `# INLINE: app/stats.py`), so the notebook still imports nothing from the repository and the website uses the same code.

## Setup

You need Python 3.12 or newer. The notebook stops at its first code cell on an older version.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Run the notebook

```bash
jupyter lab notebooks/NYC_Housing_Safety.ipynb
```

Choose Run All. The notebook reads only `housing_hg8x-zxpr.csv` and `crashes_h9gi-nx95.csv.gz` from its own folder. If either file is missing, Section 2 downloads it from NYC Open Data and saves it there. The crash download takes several minutes and resumes from the last saved page if it is interrupted. With both files present the run needs no network. To refresh the data, delete the two files and run again; the written interpretations were checked against the 10 October copy and should be read again after a refresh.

The notebook can be copied alone into any folder together with the two data files.

## Rebuild the notebook from its sources

The notebook is generated. Edit the files in `tools/notebook_src/`, then run:

```bash
python tools/assemble_notebook.py --execute
```

This joins the section files in name order, pastes `app/stats.py` into Section 7, gives every cell a stable id, executes every cell inside `notebooks/`, prints a line budget per section and writes `notebooks/NYC_Housing_Safety.ipynb` with its outputs. If a cell fails, the script prints the failing cell, writes `NYC_Housing_Safety.failed.ipynb` instead of replacing the notebook, and exits with status 1. To work on a few sections, pass their prefixes and write somewhere else (`--src` reads the section files from another folder):

```bash
python tools/assemble_notebook.py --sections 01 02 04 --execute --out /tmp/dev.ipynb --show 04
```

Later sections use objects built by earlier ones, so a partial run must include everything its sections depend on (the research questions in Section 8 need Sections 1, 2 and 4 to 7).

## Rebuild the website data

```bash
python tools/build_app_data.py
```

The script executes the notebook sections (all except the exploratory Section 3) in one process, exports small aggregated tables to `app/data/`, and then checks that the website's own statistics reproduce the notebook's RQ3 and RQ7 estimates and intervals. Pass `--skip-check` to skip that check. It needs the two data files in `notebooks/`, took 47 seconds on the development machine (`app/data/meta.json`), and needs several GB of memory (the script's own note says about 3 GB). The site never reads the raw files.

## Run the website locally

```bash
python -m app.app
```

Open http://127.0.0.1:8050. Set the `PORT` environment variable to use another port. The site reads only `app/data/` and downloads nothing at start-up. The map's background tiles load from Carto in the visitor's browser, so the map needs an internet connection.

## Run the tests

```bash
python -m pytest -q tests
```

206 tests in two files pass. `tests/test_site.py` covers the search parser, the filters, the pre-aggregated tables (totals must agree across all of them), the chart builders, the callbacks through Flask's test client, and agreement between the site's numbers and the exported results. `tests/test_data_accuracy.py` recomputes the numbers the site shows from the two raw files with separate code and compares them with `app/data/`; it skips itself when the raw files are not in `notebooks/`.

## Deploy on Vercel

**Live site: https://data-engineering-visualization-proj-three.vercel.app/**

The site is deployed on Vercel from `main`; every push to `main` redeploys it, and each pull request gets its own preview deployment.

Vercel runs the site as one Python function. `pyproject.toml` names the entry point (`app.app:server`, the Flask server that Dash creates), `vercel.json` gives `app/app.py` a `maxDuration` of 60 seconds, `requirements.txt` lists the five packages the site needs, and `.vercelignore` keeps the notebook, tools, tests and docs out of the deployment.

1. Merge the pull request so that `main` holds the site and `app/data/`.
2. In the Vercel dashboard choose Add New > Project and import the GitHub repository.
3. Leave the framework preset and the build settings as Vercel detects them. The project needs no environment variables.
4. Deploy and wait for the build to finish.
5. Open the `vercel.app` address in a private window. Check that the page is styled, that Generate report redraws every chart, that a search such as "Brooklyn 2022 new construction" fills the filters, and that the page works at phone width.
6. Write the address into the submission email.

Two things to know. The function starts on demand, and nothing is precomputed at import, so the first report after a quiet period takes longer than later ones. The installed packages take about 350 MB, under Vercel's 500 MB limit for Python functions; pyarrow is the largest part that could be dropped if that limit ever becomes a problem.

## Licence and credits

The code is released under the MIT licence (`LICENSE`). The two datasets come from NYC Open Data and are used under its terms of use (https://www.nyc.gov/home/terms-of-use.page). The map's background tiles are © CARTO, with map data © OpenStreetMap contributors (ODbL).

## Submission checklist

Email ICS503Data@gmail.com by 9 November 2026 with:

- [ ] the student IDs of all five members
- [ ] the GitHub repository link
- [ ] the deployment link (https://data-engineering-visualization-proj-three.vercel.app/)
- [ ] the links to the two datasets (the table above)

## Team contributions

Five members, two research questions and two charts each. The table is the same as Section 13 of the notebook. Section numbers refer to the notebook. The answer to each research question is in `docs/METHODOLOGY.md`.

| Member | Name | Student ID | Research questions and charts | Data cleaning | Integration | Website | Other work |
|---|---|---|---|---|---|---|---|
| M1 | Ahmed Atef | 16007032 | RQ1 forecast comparison (dot plot of forecast error); RQ2 within-ZIP slopes (coefficient plot with leads and lags) | Housing: blank unit counts, duplicates, geographic identifiers, dates, unit consistency, size outliers (4.1-4.6) | ZIP by year panel (5.2) | Dropdown filters and the Generate report button | Setup and data loading (1, 2), housing exploration (3.1), README run steps |
| M2 | Nour Mostafa Aly | 16001618 | RQ3 event study (lines with interval bands); RQ4 building size (scatter with fitted slope) | Crashes: key, duplicates, dates, time of day, borough and ZIP (4.7, 4.8) | Ball-tree distance join and its brute-force check, choice of the 250 m buffer (5.3) | Event-study cards and the bootstrap callback | Crash exploration (3.2), bootstrap and difference-in-differences toolkit (7) |
| M3 | Yahia Mohamed | 16004934 | RQ5 new construction vs preservation (interval plot); RQ6 pedestrian and cyclist share by borough (interval plot) | Crashes: coordinates, casualty totals, extreme injury counts, factor groups (4.8-4.11) | Sites, clusters and the nearby-completion screen (5.4) | Search box and query parser | Robustness checks (9), data dictionary |
| M4 | Moaz Ahmed | 16003110 | RQ7 site selection (percentile histogram); RQ8 hour of day (interval plot of day parts) | ZIP recovery from coordinates and its holdout validation (6.1) | Tagging crashes with their nearest site (5.5), conservation checks (6.4) | Map and heat-map cards, page layout, accessibility | Deployment configuration, architecture notes |
| M5 | Yassin Fady | 16005869 | RQ9 deep-affordable share across ZIP codes (scatter by borough); RQ10 contributing factors (interval plot) | Post-integration cleaning: borough, redundant columns, data types (6.2, 6.3) | Cumulative stock and citywide index (6.2, 6.5), final schema | Findings and data-and-method sections of the page | Findings, limits, impact, reproducibility (10-14), hosting |

Each member should be able to explain the code and text of their own two questions.

## How this was built

AI coding assistants drafted the code and text, and the team directs and answers for the result. `docs/AI_USE.md` holds the prompts, the build process (including the lean-pipeline refactor) and the checks each student should do. Each student should be able to explain their own research questions and the cleaning and integration choices behind them.
