# Affordable housing and street safety in New York City

ICS 503 Data Engineering and Visualization, Project Milestone 1.

This project asks whether the streets around city-financed affordable housing have different crash records from comparable streets, before and after the buildings are completed. It joins two NYC Open Data tables that share no key, the housing production table and the police crash table, by ZIP code and year and by distance (crashes within 250 m of a building), and it uses injury crashes from 2016 to 2025 as the outcome. For the 128 new buildings that could be compared fairly, injury crashes within 250 m changed by -4.6% after completion relative to sites not yet completed (95% interval -11.9% to +3.3%), so the data cannot tell this from no change, and the other before-and-after questions are just as inconclusive. The clearest result concerns location before construction: new-construction sites stood on streets with more injury crashes than the typical street point of their borough, about 20 percentile points above its middle (95% interval 16.3 to 22.8). Every result is an association, the before-and-after comparison rests on a small, less densely built part of the programme, and its sign changes with how the sites are chosen.

## Datasets

| Dataset | ID | Page | API (CSV) |
|---|---|---|---|
| Affordable Housing Production by Building | `hg8x-zxpr` | https://data.cityofnewyork.us/d/hg8x-zxpr | https://data.cityofnewyork.us/resource/hg8x-zxpr.csv |
| Motor Vehicle Collisions - Crashes | `h9gi-nx95` | https://data.cityofnewyork.us/d/h9gi-nx95 | https://data.cityofnewyork.us/resource/h9gi-nx95.csv |

The copies in `notebooks/` were saved on 9 October 2026. The housing file holds all 9,551 rows. The crash file is a filtered copy: crashes dated 2016-01-01 to 2025-12-31 (1,504,722 rows) and 19 of the 29 columns.

## Repository layout

```
notebooks/
  NYC_Housing_Safety.ipynb     the notebook, with its outputs stored
  housing_hg8x-zxpr.csv        the housing data
  crashes_h9gi-nx95.csv.gz     the filtered crash data
tools/
  assemble_notebook.py         builds the notebook from the section files and runs it
  notebook_src/*.py            one file per notebook section (00_title ... 10_story)
  build_app_data.py            runs the same sections and writes the website data
app/
  app.py                       Dash entry point (app.app:server)
  data.py, filters.py, search.py, stats.py, charts.py, report.py,
  story.py, layout.py, callbacks.py, theme.py
  assets/                      site.css, site.js
  data/                        the files the site reads (about 3.3 MB)
tests/
  test_site.py, conftest.py
docs/                          architecture, methodology, data dictionary, contributions,
                               prompts, agent workflow
requirements.txt                site dependencies (what Vercel installs)
requirements-dev.txt            notebook, data build and tests
pyproject.toml, .vercelignore   Vercel entry point and the folders left out of the deployment
```

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

Choose Run All. The notebook reads only `housing_hg8x-zxpr.csv` and `crashes_h9gi-nx95.csv.gz` from its own folder. If either file is missing, Section 2 downloads it from NYC Open Data and saves it there. The crash download takes several minutes and resumes from the last saved page if it is interrupted. With both files present the run needs no network. To refresh the data, delete the two files and run again; the written interpretations were checked against the 9 October copy and should be read again after a refresh.

The notebook can be copied alone into any folder together with the two data files.

## Rebuild the notebook from its sources

The notebook is generated. Edit the files in `tools/notebook_src/`, then run:

```bash
python tools/assemble_notebook.py --execute
```

This joins the section files in name order, executes every cell inside `notebooks/`, and writes `notebooks/NYC_Housing_Safety.ipynb` with its outputs. It took between 1 and 2 minutes on the development machine. If a cell fails, the script prints the failing cell and exits with an error, but it still writes the notebook, so check the exit status before committing. To work on a few sections, pass their prefixes and write somewhere else:

```bash
python tools/assemble_notebook.py --sections 01 02 04 --execute --out /tmp/dev.ipynb --show 04
```

Later sections use objects built by earlier ones, so a partial run must include everything its sections depend on (the research questions in Section 8 need Sections 1, 2 and 4 to 7).

## Rebuild the website data

```bash
python tools/build_app_data.py
```

The script executes the notebook sections (all except the exploratory Section 3) in one process, exports small aggregated tables to `app/data/`, and then checks that the website's own statistics reproduce the notebook's RQ3 and RQ7 estimates and intervals. Pass `--skip-check` to skip that check. It needs the two data files in `notebooks/`, takes about a minute on the development machine, and needs several GB of memory (the script's own note says about 3 GB). The site never reads the raw files.

## Run the website locally

```bash
python -m app.app
```

Open http://127.0.0.1:8050. Set the `PORT` environment variable to use another port. The site reads only `app/data/` and downloads nothing at start-up. The map's background tiles load from Carto in the visitor's browser, so the map needs an internet connection.

## Run the tests

```bash
python -m pytest -q tests
```

The 79 tests cover the search parser, the filters, the pre-aggregated tables (totals must agree across all of them), the chart builders, the callbacks through Flask's test client, and agreement between the site's numbers and the notebook's. They passed in about 9 seconds on Python 3.14.3.

## Deploy on Vercel

The site has not been deployed yet. Add the deployment link below once it exists.

Vercel runs the site as one Python function. `pyproject.toml` names the entry point (`app.app:server`, the Flask server that Dash creates), `requirements.txt` lists the five packages the site needs, and `.vercelignore` keeps the notebook, tools, tests and docs out of the deployment.

1. Merge the pull request so that `main` holds the site and `app/data/`.
2. In the Vercel dashboard choose Add New > Project and import the GitHub repository.
3. Leave the framework preset and the build settings as Vercel detects them. The project needs no environment variables.
4. Deploy and wait for the build to finish.
5. Open the `vercel.app` address in a private window. Check that the page is styled, that Generate report redraws every chart, that a search such as "Brooklyn 2022 new construction" fills the filters, and that the page works at phone width.
6. Write the address into the submission email and add it here: Deployment link: TO ADD after the first deploy.

Three things to know. The configuration follows Vercel's documentation but has not been tried on Vercel, so the first build is the test. The function starts on demand, so the first visit after a quiet period waits a few seconds while the data loads (about 3 seconds locally). The installed packages take about 350 MB, under Vercel's 500 MB limit for Python functions; pyarrow is the largest part that could be dropped if that limit ever becomes a problem.

## Submission checklist

Email ICS503Data@gmail.com by 9 November 2026 with:

- [ ] the student IDs of all five members
- [ ] the GitHub repository link
- [ ] the deployment link
- [ ] the links to the two datasets (the table above)

## Team contributions

Five members, two research questions each. The full split of cleaning, integration, website and documentation work is in `docs/CONTRIBUTIONS.md` and in Section 13 of the notebook. `docs/ANSWERS.md` answers each member's two research questions in plain words, with how the answer was measured and what it means for the city.

| Member | Name | Student ID | Research questions |
|---|---|---|---|
| M1 | Ahmed Atef | 16007032 | RQ1 forecast of a ZIP's injury crashes, RQ2 within-ZIP change as the unit stock grows |
| M2 | Nour Mostafa Aly | 16001618 | RQ3 before and after at new buildings, RQ4 building size |
| M3 | Yahia Mohamed | 16004934 | RQ5 new construction against preservation, RQ6 pedestrian and cyclist share |
| M4 | Moaz Ahmed | 16003110 | RQ7 where the sites stand before construction, RQ8 hour of day |
| M5 | Yassin Fady | 16005869 | RQ9 deep-affordable share across ZIPs, RQ10 contributing factors |

## How this was built

AI coding assistants drafted the code and text, and the team directs and answers for the result. `docs/AGENT_WORKFLOW.md` describes the process and `docs/PROMPTS.md` lists the prompts. Each student should be able to explain their own research questions and the cleaning and integration choices behind them.
