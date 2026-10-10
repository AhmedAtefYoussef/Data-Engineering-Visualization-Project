"""Build notebooks/NYC_Housing_Safety.ipynb with nbformat.

Run: python tools/build_notebook.py
The notebook is a narrated walkthrough: markdown justifies every decision, code cells are
short and call the pipeline/analysis modules. It is written WITHOUT outputs on purpose;
findings shown in the notebook are generated at run time from data/processed/*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

OUT = ROOT / "notebooks" / "NYC_Housing_Safety.ipynb"
CELLS: list = []


def md(text: str) -> None:
    """Append a markdown cell."""
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    """Append a code cell."""
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


# Duty split (mirrors docs/CONTRIBUTIONS.md; research questions come from config.TEAM).
DUTY_COLS = ["Data cleaning", "Integration", "EDA", "Notebook", "Website", "Deployment", "Docs"]
DUTIES = {
    "M1": ["Housing rules H1 to H10", "ZIP x year panel", "Housing EDA (2.1)",
           "Sections 1 and 3.1 to 3.5", "Explore tab: filters and Generate report", "Test the deployed site in a private window",
           "README setup and run"],
    "M2": ["Crash rules C1 to C5 (ids, dates, hour, borough, ZIP)", "BallTree exposure flag and brute-force check",
           "Crash EDA (2.2)", "Sections 2.2 and 3.6 to 3.10", "Event-study cards and bootstrap callback",
           "Memory and speed check on Render", "METHODOLOGY: event study"],
    "M3": ["Crash rules C6 to C9 (coordinates, casualties, cap, derived fields)", "Sites (connected components) and contamination flags",
           "Outliers, correlations, factor mapping", "Sections 3.11 to 3.13 and 4", "Search box and query parser",
           "Procfile and alternative host check", "DATA_DICTIONARY"],
    "M4": ["ZIP recovery grid and its validation", "Cube conservation and panel uniqueness checks",
           "Maps and time series", "Section 5", "Map and heatmap cards, layout, accessibility",
           "render.yaml and requirements", "ARCHITECTURE"],
    "M5": ["Streaming extract, caching, post-integration cleaning", "Artifact contract and pipeline CLI",
           "Missingness and integrated-data view (5.4)", "Sections 6 intro and 7 to 11",
           "Story and Data and method tabs", "Render blueprint, environment variables, build log",
           "CONTRIBUTIONS and AGENT_WORKFLOW"],
}
DUTY_TABLE = ("| Member | " + " | ".join(DUTY_COLS) + " |\n|" + "---|" * (len(DUTY_COLS) + 1) + "\n"
              + "\n".join(f"| {m} | " + " | ".join(v) + " |" for m, v in DUTIES.items())
              + "\n\nEach member also writes the section 6 text for their own research questions and should be able to explain every cleaning and integration decision.")



# --------------------------------------------------------------------------- 0 title
def sec0() -> None:
    md("""
# Are streets near new affordable housing different in crash patterns? New York City, 2016 to 2025

ICS 503 data-integration project. Datasets: HPD *Affordable Housing Production by Building* (`hg8x-zxpr`) and NYPD *Motor Vehicle Collisions - Crashes* (`h9gi-nx95`), both from NYC Open Data.

Question. Are streets near newly completed affordable housing different, in crash patterns, from comparable streets? Concretely: how do police-reported crashes within a short walk (250 m) of a building before and after its completion compare with the same change near housing completed only later? Does the pattern differ by borough, building size, construction type (new building or preservation) and crash type?

Why it matters. HPD and DOT planners and street-safety advocates decide where housing goes and where safety money is spent. Planners would want to know early if crash patterns near new housing look different from comparable places. A finding of no clear shift is also useful. The results here are associations with stated uncertainty, not causal claims.

Where to find each required item:

| Part | Content |
|---|---|
| 1 Datasets | overview with row and column counts, links, why these two, aggregation plan, how the 1.5 million crash rows are loaded |
| 2 EDA | statistics and plots for each raw dataset on its own |
| 3 Pre-integration cleaning | one rule at a time, each with a validation plot |
| 4 Integration | why a ZIP x year panel plus a 250 m buffer and not a row join, with a toy example and a check against brute force |
| 5 Post-integration cleaning | missing values created by the join, ZIP recovery and its validation, types |
| 6 Results | ten research questions, each with one chart and a headline generated from `rq_results.json` |
| 7 to 11 | limits and ethics, impact, prompts used, contributions, reproducibility |

No finding is typed into this notebook by hand. Numbers come from cells that compute them, except the row counts quoted in section 1.1, which are from the API query dated there.
""")
    code("""
# --- Setup: works locally and on Google Colab -------------------------------------------
import os, sys, json, subprocess, warnings
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/TODO-team/nyc-housing-safety.git"   # TODO: set to the team repository (placeholder, not a real URL)
if "TODO" in REPO_URL:
    print("REMINDER: REPO_URL is still a placeholder. Set it to the team's GitHub repository before running on Colab, "
          "and before submitting.")
if IN_COLAB and not Path("config.py").exists():                      # fresh Colab runtime
    subprocess.run(["git", "clone", REPO_URL, "/content/proj"], check=True)
    os.chdir("/content/proj")
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

ROOT = Path.cwd()
while not (ROOT / "config.py").exists() and ROOT != ROOT.parent:     # started inside notebooks/ or a subfolder
    ROOT = ROOT.parent
if not (ROOT / "config.py").exists():
    raise FileNotFoundError("config.py not found: open the notebook from inside the repository")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))                # make config, pipeline and analysis importable
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display, Markdown, HTML

if IN_COLAB:
    pio.renderers.default = "colab"

import config
from config import Y_START, Y_END, BUFFER_M, SEED
pd.set_option("display.max_columns", 60, "display.width", 140)
np.random.seed(SEED)
print("project root:", ROOT, "| processed dir:", config.PROCESSED_DIR, "| fixture mode:", config.FIXTURE)
""")
    code("""
# --- Processed artifacts and the fixture banner ---------------------------------------------
# On Colab (or with RUN_PIPELINE=1) a missing artifact set is built by the pipeline, which streams the API.
RUN_PIPELINE_IF_MISSING = IN_COLAB or os.environ.get("RUN_PIPELINE") == "1"
meta_path = config.PROCESSED_DIR / "meta.json"
if not meta_path.exists() and RUN_PIPELINE_IF_MISSING:
    print("No processed artifacts found: running `python -m pipeline.run` (several minutes on real data).")
    subprocess.run([sys.executable, "-m", "pipeline.run"], check=True)
META = json.loads(meta_path.read_text()) if meta_path.exists() else {}
FIXTURE_MODE = bool(config.FIXTURE or META.get("fixture"))

def fixture_banner() -> None:
    \"\"\"Print and display a loud warning that the data in use is synthetic.\"\"\"
    msg = "FIXTURE MODE: every number and chart below is SYNTHETIC test data. It is not a finding about New York City."
    print("!" * 78)
    print("!!  " + msg)
    print("!!  Unset NYC_FIXTURE and run `python -m pipeline.run` to work with real data.")
    print("!" * 78)
    display(HTML('<div style="border:4px solid #A12A2A;background:#FBE9E9;color:#7A1111;padding:14px;'
                 'font:700 18px/1.4 system-ui,sans-serif">' + msg + '</div>'))

if FIXTURE_MODE:
    fixture_banner()
elif not META:
    print("No processed artifacts yet. Sections 1 to 3 work from the API; run `python -m pipeline.run` "
          "before sections 4 to 6.")
else:
    print("Real-data artifacts found, generated at", META.get("generated_at"))
""")
    code('''
# --- Small helpers used by many cells (plot defaults, missingness, numeric view) ----------
def show(fig, height=420):
    """Apply a consistent layout and display a plotly figure."""
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=60, b=50), template="plotly_white")
    fig.show()

def missing_bar(df, title, top=25):
    """Bar chart of the share of missing values per column (largest first)."""
    s = (df.isna().mean() * 100).sort_values(ascending=False).head(top)
    show(px.bar(s[::-1], orientation="h", labels={"value": "% missing", "index": "column"}, title=title),
         height=max(300, 18 * len(s) + 120))

def numeric_view(df, cols):
    """Coerce the listed columns to numbers (the CSV/JSON API returns text); bad values -> NaN."""
    return df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")

def geo_scatter(df, **kw):
    """Scatter map on free carto tiles (no access token); uses the newer API if available."""
    if hasattr(px, "scatter_map"):
        return px.scatter_map(df, lat="latitude", lon="longitude", zoom=9.2, map_style="carto-positron", **kw)
    return px.scatter_mapbox(df, lat="latitude", lon="longitude", zoom=9.2, mapbox_style="carto-positron", **kw)

def before_after(before, after, title, labels=("before", "after")):
    """Side-by-side bar chart of two category-count Series (validation of a cleaning rule)."""
    d = pd.concat({labels[0]: before, labels[1]: after}, axis=1).fillna(0).reset_index()
    d = d.melt(id_vars=d.columns[0], var_name="stage", value_name="rows")
    show(px.bar(d, x=d.columns[0], y="rows", color="stage", barmode="group", title=title))
''')


# --------------------------------------------------------------------------- 1 datasets
def sec1() -> None:
    md(f"""
## 1. Datasets

### 1.1 Overview

| | Housing | Crashes |
|---|---|---|
| Name | Affordable Housing Production by Building | Motor Vehicle Collisions - Crashes |
| Publisher | NYC Department of Housing Preservation and Development (HPD) | NYC Police Department (NYPD) |
| Socrata ID | `{config.HOUSING_ID}` | `{config.CRASH_ID}` |
| Dataset page | {config.DATASET_PAGES['housing']} | {config.DATASET_PAGES['crashes']} |
| API endpoint (CSV) | {config.SODA_URL.format(config.HOUSING_ID)} | {config.SODA_URL.format(config.CRASH_ID)} |
| Size, queried on 9 Oct 2026 | 9,551 rows x 41 columns | 1,541,146 rows since 2016-01-01 x 29 columns |
| Grain | one row per building in a financed project | one row per police-reported collision |
| Geographic attributes | `borough`, `postcode`, `community_board`, `council_district`, `census_tract`, `latitude`, `longitude`, `bbl`, `bin` | `borough`, `zip_code`, `latitude`, `longitude`, `location`, street names |
| Temporal attributes | `project_start_date`, `project_completion_date`, `building_completion_date` | `crash_date`, `crash_time` |
| Other attributes used | construction type, units by income band and bedroom count, total units | persons injured and killed by road-user type, contributing factors, vehicle types |

About 7.6 thousand of the 9,551 housing rows have both a ZIP and coordinates. In the crash table (rows since 2016-01-01) 92% of rows have latitude, 66% have a ZIP and 66% have a borough. The portal changes daily, so a rerun can show different counts. The next cell asks the API for the current counts and prints them next to the figures above.
""")
    code('''
# Listing 0 - live row counts (skipped in fixture mode or when the API is not reachable)
import io, requests
from config import SODA_URL, HOUSING_ID, CRASH_ID

def soda_count(dataset_id, where=None):
    """Row count from the API with a server-side count(*), so no rows are downloaded."""
    params = {"$select": "count(*)", **({"$where": where} if where else {})}
    headers = {"X-App-Token": config.APP_TOKEN} if config.APP_TOKEN else {}
    r = requests.get(SODA_URL.format(dataset_id), params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return int(pd.read_csv(io.StringIO(r.text)).iloc[0, 0])

if config.FIXTURE:
    print("Fixture mode: live counts skipped. Quoted counts (queried 9 Oct 2026): housing 9,551 rows x 41 columns; "
          "crashes 1,541,146 rows since 2016-01-01 x 29 columns.")
else:
    try:
        print(f"housing rows now: {soda_count(HOUSING_ID):,}   (quoted: 9,551)")
        since = "crash_date >= '2016-01-01'"
        print(f"crash rows since 2016-01-01 now: {soda_count(CRASH_ID, since):,}   (quoted: 1,541,146)")
    except Exception as exc:               # offline, blocked or rate limited
        print("Could not reach the API:", type(exc).__name__ + "; continuing with the quoted counts.")
''')
    md(f"""
### 1.2 Why these two datasets

* They answer the question together. Housing says where and when a change happened (completion date at building level). Crashes give a repeated safety outcome at the same places and years. Neither table alone says anything about change near housing.
* They share only weak keys. Both carry a ZIP (a number in housing, text in crashes, and missing in about a third of crashes) and a date, and both carry coordinates (92% present in crashes). No building id exists in the crash table, so a row-level join is impossible. Section 4 explains the two integrations used instead.
* They differ in grain and in quality problems. One is small and administrative, the other very large and operational. That makes a realistic integration task: scale, missing geography, text and number type conflicts, duplicates.
* Both are public, documented and updated, so anyone can rerun the pipeline.
* Alternatives we considered. Data on complaints about existing buildings describes the housing stock, not new production, so it has no completion event to compare before and after. Census or ACS tables by ZIP would add confounders such as income and population, but they are a third source and are left for future work.

### 1.3 Aggregation plan

The crash table is too large to hold row by row, and the questions need only counts. Each row is cleaned, assigned to a place and a year, counted into a small table, and then discarded. Every grouping column of the crash cube is a filter on the website, so no question is lost by aggregating.

| Table | Grain | Built from | Used by |
|---|---|---|---|
| `housing_clean` | one row per building | housing | map (RQ7), event-study inputs, ZIP panel |
| `crash_cube` | ZIP, borough, year, hour band, factor group, severity, exposure, pedestrian/cyclist flag | crashes, streamed | RQ6, RQ10; crash side of RQ1, RQ2, RQ8, RQ9 |
| `crash_grid` | 0.005 degree map cell, year, borough, severity | crashes, streamed | RQ7 |
| `event_building` | building x year relative to completion (-3 to +3) | housing and crashes within 250 m | RQ3, RQ4, RQ5 |
| `panel_zip_year` | ZIP x year, housing side only | housing | RQ1, RQ2, RQ8, RQ9 |
| `panel_zip_year_full` | ZIP x year, housing plus unfiltered crash counts | panel and cube | this notebook |

### 1.4 Known issues and where they are handled

| Issue | Dataset | Consequence | Handling (section) |
|---|---|---|---|
| Zero or null unit counts are omitted from the API record | housing | missing columns after load | re-create and fill 0 (3.1) |
| `postcode` stored as a number | housing | leading zero lost, out-of-city codes | zero-pad, regex-validate (3.2) |
| Completion before start, duplicate buildings | housing | wrong years, double counting | flag, keep latest (3.4) |
| About 34% missing ZIP and borough, about 8% missing coordinates, (0, 0) coordinates | crashes | biased area counts | coordinate check, grid ZIP recovery (3.8, 5.2) |
| Text hour `H:MM`, free-text contributing factors | crashes | no direct analysis | parse, map to groups (3.7, 3.12) |
| 1.5 million rows | crashes | memory failure on a laptop or free Colab runtime | stream pages, aggregate (1.5) |
| Latest months incomplete | crashes | false drop in the last year | full calendar years only (`config.Y_END`) |

### 1.5 Loading strategy for the 1.5 million crash rows

The SODA API returns 1,000 rows per request unless told otherwise, and large reads are rate limited. Loading every column of every crash row into one `DataFrame` would exceed the memory of a free Colab runtime or a small web server. The approach has five parts.

1. Filter on the server. `$where=crash_date>='2016-01-01'` removes rows before they are sent. `$select` lists only the columns the pipeline uses (16 of 29).
2. Page with a key. Each request asks for `collision_id > last_seen`, ordered by `collision_id`, and the next request resumes from the largest id received. Ordering by the internal `:id` was tested and rejected because it does not sort as a string key (SPEC A1). The housing table is small and comes in one request.
3. Stream. Each page is cleaned and counted into the aggregate tables, then dropped. Only the aggregates stay in memory.
4. Cache. Each page is stored under `data/raw/`, so a failed run resumes and a rerun reads from disk. An optional `SOCRATA_APP_TOKEN` raises the rate limit.
5. Explore on a bounded sample. Section 2 uses one full recent calendar year, capped at 300,000 rows. A year is used instead of the first N rows because rows are ordered by id, so the first N would be one narrow slice of time.

Alternatives considered: the full CSV export (same memory problem and no server-side filter); `$group` aggregation on the server (fast, but cleaning and the spatial step need row-level data); one small random sample for everything (rare outcomes such as fatal crashes and small ZIPs would be unreliable).
""")
    code('''
# Listing 1 - assignment-style loading with pd.read_csv on the rows.csv?accessType=download pattern.
# Only the small housing table (about 9.5 thousand rows) is downloaded this way. The crash table
# (1.5 million rows) would exhaust memory and time on a laptop or free Colab runtime, so it is loaded
# with Listing 2 ($select, $where, $order and $limit through the SODA API) and keyset paging on
# collision_id in pipeline.extract, one page at a time (section 1.5).
import pandas as pd
from config import HOUSING_ID, CRASH_ID                      # config.HOUSING_ID = housing_id, config.CRASH_ID = transport_id
housing_id, transport_id = HOUSING_ID, CRASH_ID               # the assignment's housing_id / transport_id placeholders
DOWNLOAD_URL = "https://data.cityofnewyork.us/api/views/{}/rows.csv?accessType=DOWNLOAD"

if config.FIXTURE:
    print("Fixture mode: skipping the download. URL pattern:", DOWNLOAD_URL.format(housing_id))
else:
    try:
        housing_df = pd.read_csv(DOWNLOAD_URL.format(housing_id))        # small table only
        print(f"housing_df: {housing_df.shape[0]:,} rows x {housing_df.shape[1]} columns")
    except Exception as exc:                                              # offline or blocked network
        print("Could not download the housing table:", type(exc).__name__)
# transport_df = pd.read_csv(DOWNLOAD_URL.format(transport_id))           # NOT run: 1.5 million rows, see Listing 2
''')
    code("""
# Listing 1b - the raw SODA request pattern (pipeline.extract.fetch_soda wraps it with paging, retries,
# caching and the optional SOCRATA_APP_TOKEN header)
params = {"$select": "collision_id,crash_date,zip_code", "$where": "crash_date >= '2025-01-01'",
          "$order": "collision_id", "$limit": 5}
print(SODA_URL.format(CRASH_ID) + "?" + "&".join(f"{k}={v}" for k, v in params.items()))
print(SODA_URL.format(HOUSING_ID) + "?$limit=50000&$order=building_id")
""")
    code("""
# Listing 2 - load both datasets through the pipeline's extractor (housing in full, one crash year as a sample).
from pipeline.extract import fetch_soda

CRASH_COLS = ["collision_id", "crash_date", "crash_time", "borough", "zip_code", "latitude", "longitude",
              "on_street_name", "number_of_persons_injured", "number_of_persons_killed",
              "number_of_pedestrians_injured", "number_of_pedestrians_killed",
              "number_of_cyclist_injured", "number_of_cyclist_killed",
              "number_of_motorist_injured", "number_of_motorist_killed", "contributing_factor_vehicle_1"]
EDA_YEAR = Y_END                       # one full recent calendar year for exploration
raw_h = pd.concat(fetch_soda(HOUSING_ID, page_size=50_000), ignore_index=True)
raw_c = pd.concat(fetch_soda(CRASH_ID, select=",".join(CRASH_COLS),
                             where=f"crash_date >= '{EDA_YEAR}-01-01' AND crash_date < '{EDA_YEAR + 1}-01-01'",
                             max_rows=300_000), ignore_index=True)
print(f"housing raw: {raw_h.shape[0]:,} rows x {raw_h.shape[1]} cols")
print(f"crash sample ({EDA_YEAR}): {raw_c.shape[0]:,} rows x {raw_c.shape[1]} cols")
""")
    code("""
# Listing 3 - stream the crash table: look at each page once, keep only a running count, drop the page.
# The demo stops after STREAM_ROWS rows; the pipeline itself streams every row (max_rows=None).
from pipeline.extract import crash_chunks

STREAM_ROWS = 200_000
per_year, n_rows, biggest_page = pd.Series(dtype="float64"), 0, 0
for page in crash_chunks(max_rows=STREAM_ROWS)():
    n_rows += len(page)
    biggest_page = max(biggest_page, len(page))
    per_year = per_year.add(pd.to_datetime(page["crash_date"], errors="coerce").dt.year.value_counts(), fill_value=0)
print(f"streamed {n_rows:,} rows; at most {biggest_page:,} rows were in memory at once")
print("crashes per year in this slice (ordered by collision_id, so mostly the earliest years):")
print(per_year.astype(int).sort_index().to_string())
""")


# --------------------------------------------------------------------------- 2 EDA
def sec2() -> None:
    md("""
## 2. Exploratory data analysis (raw data, before any cleaning)

Goal: understand shape, types, missingness, distributions, time and space *before* deciding what to clean. Everything here is on the raw download so that section 3 can show what changed. Values arrive as text, so numeric summaries use a coercing view (`numeric_view`); this itself is a finding about the source: the type information is not trustworthy.

### 2.1 Housing: shape, types, summary statistics
""")
    code("""
print("shape:", raw_h.shape)
print(raw_h.dtypes.value_counts().to_string(), "\\n")
unit_cols = [c for c in raw_h.columns if c.endswith("_units") or c in ("total_units", "all_counted_units")]
num_h = numeric_view(raw_h, unit_cols + ["latitude", "longitude", "postcode"])
display(num_h.describe().T.round(2))
""")
    md("""
Reading the table. Count below the row total means missing values. Unit columns with a small count are columns the API leaves out when a value is zero, which is why these are *structural* zeros, not unknowns (justified in 3.1). A `postcode` mean in the tens of thousands confirms it is stored as a number.
""")
    code("""
missing_bar(raw_h, "Housing (raw): % missing by column")
""")
    md("""
Missingness. Coordinates and ZIP are the columns that matter for integration. Their missing share bounds how many buildings can be located. The unit columns are mostly "missing" because zero is omitted.
""")
    code("""
yr = pd.to_datetime(raw_h["building_completion_date"], errors="coerce").dt.year
fig = px.histogram(yr.dropna().astype(int), nbins=40, title="Housing (raw): buildings by completion year",
                   labels={"value": "completion year"})
show(fig.update_layout(showlegend=False, yaxis_title="buildings"))
cnt = raw_h["borough"].value_counts()
show(px.bar(cnt, title="Housing (raw): buildings per borough (spelling variants would show as extra bars)",
            labels={"value": "buildings", "index": "borough"}))
""")
    md("""
Time and place. The completion-year histogram shows which years can serve as treated cohorts (we need three full years on each side inside the crash window) and where the 2014-2015 buildings used for cumulative units sit. The borough bar reveals spelling or case variants, if any.
""")
    code("""
tu = numeric_view(raw_h, ["total_units"])["total_units"]
show(px.histogram(tu.dropna(), nbins=60, log_y=True, title="total_units (log count axis)"), 350)
show(px.box(raw_h.assign(total_units=tu), x="reporting_construction_type", y="total_units",
            log_y=True, title="total_units by construction type (log)"), 400)
""")
    md("""
Outliers. Building size is strongly right-skewed. The boxplot uses a log axis; points above the whiskers are the large projects. Whether these are errors or real is decided in 3.5.
""")
    code("""
corr = numeric_view(raw_h, ["total_units", "all_counted_units", "extremely_low_income_units",
                            "very_low_income_units", "low_income_units", "moderate_income_units",
                            "middle_income_units", "other_income_units", "studio_units", "_1_br_units",
                            "_2_br_units", "_3_br_units"]).corr(method="spearman")
show(px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
               title="Housing (raw): Spearman correlation of unit columns"), 560)
pts = raw_h.assign(latitude=pd.to_numeric(raw_h["latitude"], errors="coerce"),
                   longitude=pd.to_numeric(raw_h["longitude"], errors="coerce")).dropna(subset=["latitude", "longitude"])
show(geo_scatter(pts.sample(min(3000, len(pts)), random_state=SEED), color="borough",
                 hover_name="project_name", title="Housing (raw): sample of buildings"), 520)
""")
    md("""
Correlation and map. Income-band units sum to the total, so strong correlations among them are expected and are not evidence of anything; they warn against putting several of them in one regression. The map is a sanity check that coordinates fall on land in the five boroughs (isolated dots far outside would be errors).

### 2.2 Crashes: shape, types, statistics
""")
    code("""
print("shape:", raw_c.shape)
print(raw_c.dtypes.value_counts().to_string(), "\\n")
cas = [c for c in raw_c.columns if c.startswith("number_of_")] + ["latitude", "longitude"]
display(numeric_view(raw_c, cas).describe().T.round(3))
missing_bar(raw_c, f"Crashes (raw, {EDA_YEAR} sample): % missing by column")
""")
    md("""
Missingness is the main data problem here. Borough and ZIP are missing together for a large block of rows, and coordinates are missing for a smaller block. If we kept only rows with ZIP, ZIP-level counts would be undercounts that vary by place. Section 5 tests whether coordinates let us recover ZIP.
""")
    code("""
d = pd.to_datetime(raw_c["crash_date"], errors="coerce")
monthly = d.dt.to_period("M").value_counts().sort_index()
show(px.line(x=monthly.index.astype(str), y=monthly.values, markers=True,
             title=f"Crashes (raw): per month in {EDA_YEAR}", labels={"x": "month", "y": "crashes"}), 350)
hour = pd.to_numeric(raw_c["crash_time"].astype(str).str.split(":").str[0], errors="coerce")
show(px.histogram(hour.dropna(), nbins=24, title="Crashes (raw): hour of day",
                  labels={"value": "hour"}).update_layout(showlegend=False, yaxis_title="crashes"), 350)
show(px.bar(raw_c["borough"].fillna("(missing)").value_counts(), title="Crashes (raw): crashes per borough"), 350)
""")
    md("""
Time. A one-year sample shows seasonality and the daily rhythm; the 10-year trend (including 2020) is shown in section 5 from the aggregated cube. The borough bar includes a "(missing)" bar, the size of the geographic gap.
""")
    code("""
inj = numeric_view(raw_c, ["number_of_persons_injured", "number_of_persons_killed"])
print((inj > 0).mean().rename("share of crashes with >0").round(4).to_string())
show(px.box(inj, y="number_of_persons_injured", points="outliers", log_y=False,
            title="Crashes (raw): persons injured per crash (outliers shown)"), 380)
show(px.imshow(numeric_view(raw_c, [c for c in cas if "number_of" in c]).corr(method="spearman").round(2),
               text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
               title="Crashes (raw): Spearman correlation of casualty columns"), 520)
top = raw_c["contributing_factor_vehicle_1"].fillna("(missing)").value_counts().head(15)
show(px.bar(top[::-1], orientation="h", title="Crashes (raw): top 15 contributing factors (vehicle 1)"), 460)
cpts = raw_c.assign(latitude=pd.to_numeric(raw_c["latitude"], errors="coerce"),
                    longitude=pd.to_numeric(raw_c["longitude"], errors="coerce")).dropna(subset=["latitude", "longitude"])
show(geo_scatter(cpts.sample(min(4000, len(cpts)), random_state=SEED), opacity=0.4, title="Crashes (raw): sample"), 520)
""")
    md("""
What the plots tell us to do. (1) Most crashes have zero injuries, so injury is a rare-event outcome and must be handled with counts and shares, not means of individual crashes. (2) A long upper tail of injuries per crash and the casualty correlations motivate the consistency and cap rules in 3.11. (3) The top factors list has one dominant "unspecified" value and many near-synonyms; grouping them (3.12) is needed for any factor analysis. (4) Any dot plotted at latitude 0 or far outside the city on a full map would be a coordinate error (3.9).
""")


# --------------------------------------------------------------------------- 3 cleaning
def sec3() -> None:
    md("""
## 3. Pre-integration cleaning

Each dataset is cleaned on its own before anything is joined, because a cleaning rule needs the dataset's own context (e.g. a duplicate building makes sense only inside the housing table). Every rule is implemented in `pipeline/clean_housing.py` or `pipeline/clean_crashes.py`, appends a record to a shared `cleaning_log` (rule, rows before, rows after, rationale), and is validated below with a before/after plot.

The principle for drop, impute or flag: (a) *Drop* only when the record cannot be interpreted at all (duplicate key, impossible value that cannot be repaired) and dropping does not bias counts. (b) *Impute* when there is a defensible structural or auxiliary-information value (zero units; ZIP from coordinates). (c) *Flag and keep* when a value is unusual but possibly real (large buildings), so downstream analysis can run with and without it. (d) Never silently: every action is counted in the log.
""")
    code("""
from pipeline.clean_housing import clean_housing
from pipeline.clean_crashes import clean_crash_chunk, build_zip_grid

log = []                                                  # shared cleaning log (list of dicts)
clean_h = clean_housing(raw_h.copy(), log)
clean_c = clean_crash_chunk(raw_c.copy(), log)            # no ZIP grid yet: recovery is a later step (5.2)

def pick(df, *names):
    \"\"\"Return the first column of df that exists (cleaned column names are checked, not guessed).\"\"\"
    for n in names:
        if n in df.columns:
            return df[n]
    raise KeyError(f"none of {names} in {list(df.columns)[:12]}...")

print(f"housing: {len(raw_h):,} -> {len(clean_h):,} rows;  crashes sample: {len(raw_c):,} -> {len(clean_c):,} rows")
""")
    md("""
### Housing rules

#### 3.1 Missing unit columns -> structural zero
*Problem.* The API omits a unit column in a record when the value is zero or null; after loading, some unit columns may not exist at all, and others are `NaN` where the truth is zero.
*Decision.* Re-create every expected unit column and fill with 0.
*Why not drop or use the mean.* A row without "extremely low income units" does not mean the building is unusable; and a mean would invent income units that do not exist. If the true value were unknown rather than zero, 0 would bias the counts downward. The API omits zero and null values (SPEC section 2), and `total_units` is present as a cross-check (3.5).
*Validation.* Missing share per unit column before (left) and after (right).
""")
    code("""
ucols = [c for c in clean_h.columns if c.endswith("_income_units")]
b = raw_h.reindex(columns=ucols).apply(pd.to_numeric, errors="coerce").isna().mean() * 100
a = clean_h[ucols].isna().mean() * 100
before_after(b.round(1), a.round(1), "Unit columns: % missing before vs after (zero-fill)")
absent = [c for c in ucols if c not in raw_h.columns]
print("unit columns absent from the raw download (re-created as 0):", absent or "none in this download")
print("remaining NaN in cleaned unit columns:", int(clean_h[ucols].isna().sum().sum()))
""")
    md("""
#### 3.2 ZIP code: number -> validated 5-character string
*Problem.* `postcode` is stored as a number, and out-of-city codes (for example Nassau County) can appear.
*Decision.* Cast to text, left-pad to 5 characters (a safeguard: NYC codes start with 1), accept only codes matching the NYC pattern (`config.NYC_ZIP_RE`), otherwise missing.
*Alternatives.* Keep invalid ZIPs (would create phantom areas in the panel); try to correct them by guessing (unsafe). For buildings we later have coordinates, so ZIP can be derived if needed.
""")
    code("""
import re
zre = re.compile(config.NYC_ZIP_RE)
def zip_status(s):
    \"\"\"Classify raw postcodes as valid / invalid / missing under the NYC pattern.\"\"\"
    txt = pd.to_numeric(s, errors="coerce").dropna().astype("int64").astype(str).str.zfill(5)
    st = pd.Series("missing", index=s.index)
    st.loc[txt.index] = np.where(txt.str.match(zre), "valid", "invalid")
    return st.value_counts()
before_after(zip_status(raw_h["postcode"]), zip_status(pick(clean_h, "zip", "postcode")), "Housing ZIP status")
""")
    md("""
#### 3.3 Coordinates inside the NYC bounding box
*Problem.* Some coordinates are missing or implausible (zero or outside the city).
*Decision.* Values outside `config.NYC_BBOX` (lat 40.45-40.95, lon -74.30 to -73.65) become missing; the building stays (it still has a ZIP) but cannot be used in the 250 m analysis.
*Recovery.* The table has a second coordinate pair (`latitude_internal`, `longitude_internal`); where the public pair is unusable but the internal one is inside the box, the internal pair is used (logged), so fewer buildings are lost.
*Why a bounding box.* It is simple, transparent and catches gross errors; it will not catch a point that is in the box but in the harbour. A polygon test would need the borough boundary file, a third dataset. The map below is the check.
""")
    code("""
lat = pd.to_numeric(raw_h["latitude"], errors="coerce"); lon = pd.to_numeric(raw_h["longitude"], errors="coerce")
bb = config.NYC_BBOX
ok = lat.between(*bb["lat"]) & lon.between(*bb["lon"])
print(f"raw: {lat.isna().sum()} missing, {(~ok & lat.notna()).sum()} outside bbox, {ok.sum()} usable;  "
      f"cleaned usable (after internal-coordinate fallback): {clean_h['latitude'].notna().sum()}")
pl = pd.DataFrame({"lat": lat, "lon": lon, "status": np.where(ok, "kept", "set to missing")}).dropna(subset=["lat"])
show(px.scatter(pl, x="lon", y="lat", color="status", opacity=0.5,
                title="Housing coordinates: kept vs removed (axes are lon/lat)"), 420)
""")
    md("""
#### 3.4 Dates, completion year and duplicate buildings
*Problem.* Building completion dates are sometimes missing, completion can precede the project start, and a building can occur twice.
*Decisions.* (1) Completion year = building completion date, else the project completion date (coarser but real). (2) If completion < start, flag `date_inconsistent` and keep the completion date: it is the event we observe; the start date is not used in the analysis. (3) Duplicate `building_id`: keep the row with the latest completion (assuming later records are revisions). (4) Rows with no completion date are kept with `completed=False` and excluded only from time-based analysis.
*Alternative.* Drop rows with inconsistent dates (loses real buildings); average the two dates (invents a date).
""")
    code("""
rt = pd.DataFrame({"raw": raw_h["building_id"].duplicated().sum(), "after": clean_h["building_id"].duplicated().sum()},
                  index=["duplicate building_id"])
display(rt)
cy = pd.to_numeric(clean_h["comp_year"], errors="coerce")
show(px.histogram(cy.dropna(), nbins=30, title="Cleaned completion year (building date, else project date)")
     .update_layout(showlegend=False, xaxis_title="comp_year", yaxis_title="buildings"), 350)
print("buildings without usable completion year:", int(cy.isna().sum()), "->", "kept, completed=False")
""")
    md("""
#### 3.5 Consistency (`total_units >= all_counted_units`) and outliers
*Consistency.* A building cannot have fewer total units than the units counted toward program targets, so `total_units` is raised to `all_counted_units` where violated. We prefer this over dropping because the building exists either way.
*Outliers.* We use the Tukey IQR rule: a value above `Q3 + 1.5 x IQR` of `total_units` is flagged `is_large_outlier`. Large buildings are common in affordable-housing finance and are real, so they are kept and flagged, not removed or capped in the data; statistics that are sensitive (e.g. dose-response) use size *bins* (`1-10, 11-50, 51-150, 151+`), which are not moved by extreme values. IQR is chosen over z-scores because the distribution is strongly skewed, where a z-score is distorted by the very outliers it tries to detect.
""")
    code("""
tu0 = pd.to_numeric(raw_h["total_units"], errors="coerce"); ca0 = pd.to_numeric(raw_h["all_counted_units"], errors="coerce")
viol = (tu0 < ca0)
q1, q3 = clean_h["total_units"].quantile([.25, .75]); fence = q3 + 1.5 * (q3 - q1)
print(f"total < counted in raw: {int(viol.sum())} rows; after cleaning: {int((clean_h['total_units'] < clean_h['all_counted_units']).sum())}")
print(f"IQR fence = {fence:.1f} units; flagged large: {int(clean_h['is_large_outlier'].sum())} of {len(clean_h)}")
show(px.box(clean_h, y="total_units", x="is_large_outlier", points="outliers", log_y=True,
            title="total_units by outlier flag (kept in data; log axis)"), 380)
show(px.histogram(clean_h, x="size_bin", category_orders={"size_bin": ["1-10", "11-50", "51-150", "151+"]},
                  title="Derived size bins (fixed edges 10 / 50 / 150)"), 330)
""")
    md("""
### Crash rules (applied page by page; here on the sample)

#### 3.6 Dates and the analysis window
*Decision.* Parse `crash_date`; keep Y_START..Y_END full calendar years only. Rows after the last full year are dropped because the most recent months are incomplete (reporting lag) and would look like a drop in crashes.
""")
    code("""
yr_raw = pd.to_datetime(raw_c["crash_date"], errors="coerce").dt.year.value_counts().sort_index()
yr_cl = pick(clean_c, "year").value_counts().sort_index()
before_after(yr_raw, yr_cl, f"Crashes per calendar year: raw vs kept (window {Y_START}-{Y_END}); duplicates also removed")
""")
    md("""
#### 3.7 Hour of day
*Problem.* `crash_time` is text `H:MM`; some are malformed.
*Decision.* Extract the hour; invalid -> missing hour, band `Unknown` (kept). Bands: Night 0-5, Morning 6-11, Afternoon 12-17, Evening 18-23. Dropping rows with bad time would lose crashes from all time-agnostic analyses for no reason.
""")
    code("""
hb = pick(clean_c, "hour_band")
before_after(pd.Series({"valid hour": int(hour.notna().sum()), "invalid / missing": int(hour.isna().sum())}),
             hb.value_counts().rename(index=str), "Hour: raw validity vs cleaned bands", ("raw", "cleaned bands"))
""")
    md("""
#### 3.8 Borough and ZIP
*Problem.* About a third of rows lack borough and ZIP. Borough case differs from housing (`MANHATTAN` vs `Manhattan`).
*Decision at this stage.* Canonicalise borough to Title case and keep ZIP only if it matches the NYC pattern. Recovery of missing ZIPs is deliberately postponed to section 5.2: it uses coordinates plus the ZIPs of other crashes, which is information from the whole cleaned dataset (a post-cleaning, integration-era step).
""")
    code("""
zc = pick(clean_c, "zip", "zip_code")
st = pd.DataFrame({"raw": [raw_c["zip_code"].isna().mean(), raw_c["borough"].isna().mean()],
                   "cleaned": [zc.isna().mean(), pick(clean_c, "borough").isna().mean()]},
                  index=["ZIP missing/invalid", "borough missing"]) * 100
show(px.bar(st.reset_index().melt("index"), x="index", y="value", color="variable", barmode="group",
            labels={"value": "% of rows", "index": ""}, title="Geography gaps before and after cleaning"), 340)
""")
    md("""
#### 3.9 Coordinates, including the (0, 0) default
*Decision.* A coordinate pair outside the NYC box, including the common `(0, 0)` default, becomes missing; the crash is kept for borough/ZIP counts. A crash without coordinates gets `exposure = -1` in section 4 (cannot be tested for proximity) but is never dropped from ZIP counts.
""")
    code("""
la = pd.to_numeric(raw_c["latitude"], errors="coerce"); lo = pd.to_numeric(raw_c["longitude"], errors="coerce")
inb = la.between(*bb["lat"]) & lo.between(*bb["lon"])
print(f"raw: missing {int(la.isna().sum()):,}; present but outside box (incl. 0,0) {int((la.notna() & ~inb).sum()):,}; "
      f"usable {int(inb.sum()):,};  cleaned usable {int(pick(clean_c,'lat').notna().sum()):,}")
smp = pd.DataFrame({"lat": la, "lon": lo, "status": np.where(inb, "kept", "removed")}).dropna()
show(px.scatter(smp.sample(min(8000, len(smp)), random_state=SEED), x="lon", y="lat", color="status", opacity=.5,
                title="Crash coordinates kept vs removed (sample)"), 400)
""")
    md("""
#### 3.10 Duplicate `collision_id`
*Decision.* Keep one row per `collision_id` (also across pages in the streaming pipeline, using a sorted array of seen ids so memory stays small). Duplicate rows would inflate counts.
""")
    code("""
print("duplicated collision_id: raw", int(raw_c["collision_id"].duplicated().sum()),
      "-> cleaned", int(pick(clean_c, "collision_id").duplicated().sum()))
""")
    md("""
#### 3.11 Casualty columns: type, consistency, cap
*Problem.* Counts arrive as text; `persons_injured` can be smaller than the sum of pedestrian + cyclist + motorist injured; a few rows show implausibly large numbers.
*Decisions.* Cast to non-negative integers. If `persons_injured` < sum of parts, set it to the sum (the parts are more specific). Cap `persons_injured` at 20 per crash as a domain rule (a collision of road vehicles rarely injures more; larger values are likely entry errors) and log the number capped. Capping rather than dropping keeps the crash in the count.
*Alternative.* Winsorise at the 99.9th percentile (data-driven but would cap different amounts every run, harder to explain).
""")
    code("""
inj0 = pd.to_numeric(raw_c["number_of_persons_injured"], errors="coerce")
inj1 = pick(clean_c, "number_of_persons_injured", "injured", "persons_injured")
print(f"raw max {inj0.max():.0f}, cleaned max {inj1.max():.0f}; rows capped: {int((inj0 > 20).sum())}")
show(px.box(pd.concat([inj0.rename("raw"), inj1.rename("cleaned")], axis=1).melt(), x="variable", y="value",
            title="Persons injured per crash: raw vs cleaned"), 360)
""")
    md("""
#### 3.12 Contributing factor -> factor group; severity; pedestrian/cyclist flag
The free-text factors are mapped to seven groups by keyword rules, first match wins: Distraction/Inattention, Speed/Aggressive, Failure to Yield/Signal, Impairment/Fatigue, Vehicle/Road Defect, Other (no rule matched) and Unspecified. The mapping is a judgement call, so it is shown below where it can be criticised. `severity` = Fatal (any killed), Injury (any injured), Property damage only; `ped_cyc` = a pedestrian or cyclist was hurt. *Alternative:* keep raw factors (too sparse for cells) or use only the top 5 (loses the rest).
""")
    code("""
fcol = pick(clean_c, "factor_group")
rawf = (raw_c.assign(cid=pd.to_numeric(raw_c["collision_id"], errors="coerce")).drop_duplicates("cid")
        .set_index("cid")["contributing_factor_vehicle_1"].fillna("(missing)"))
mapping = (pd.DataFrame({"raw factor": rawf.reindex(pick(clean_c, "collision_id")).values, "group": fcol.values})
           .value_counts().rename("rows").reset_index().sort_values("rows", ascending=False))
display(mapping.head(20))
show(px.bar(fcol.value_counts(), title="Factor groups after mapping"), 330)
""")
    md("""
#### 3.13 The cleaning log
The log is the audit trail: each rule, rows before and after, and the rationale string written by the code. The same table is on the website (Data & Method tab).
""")
    code("""
pd.set_option("display.max_colwidth", 110)
display(pd.DataFrame(log)[["step", "dataset", "rule", "rows_before", "rows_after", "n_affected", "rationale"]])
""")


# --------------------------------------------------------------------------- 4 integration
def sec4() -> None:
    md("""
## 4. Integration

### 4.1 The problem
The datasets have no common key: housing rows are buildings, crash rows are collisions. We need a rule that says "this crash is near this building" and a way to combine counts by place and time.

### 4.2 Options considered

| Option | Verdict |
|---|---|
| Row-level join on a key | Impossible: no shared id |
| Join on ZIP (many-to-many) | Rejected as a row join: each crash would be repeated for every building in the ZIP (multiplying rows, no information). Used instead after aggregating both sides to ZIP x year |
| Nearest-building join (1-NN) | Rejected: assigns every crash in the city to some building, however far |
| Point-in-polygon with ZIP/neighbourhood boundary file | Needs a third dataset; ZIP is already in both |
| ZIP x year panel (chosen, administrative) | Both sides have ZIP and a date; about 190 ZIPs x 10 years gives adequate N; ZIPs are what planners know |
| Distance buffer of 250 m (chosen, spatial) | Matches the street-level mechanism (foot and car traffic generated by a building affects nearby blocks); gives building-level before/after |

Why not a row join, in numbers. The cell below counts how many rows a join on ZIP alone would create: every crash would be paired with every building in its ZIP.
""")
    code("""
ART = {}
for name in ["housing_clean", "crash_cube", "crash_grid", "event_building", "panel_zip_year", "panel_zip_year_full"]:
    p = config.PROCESSED_DIR / f"{name}.parquet"
    if p.exists():
        ART[name] = pd.read_parquet(p)
if "crash_cube" in ART:
    n_b = ART["housing_clean"].dropna(subset=["zip"]).groupby("zip").size()
    n_c = (ART["crash_cube"].query("zip != 'UNLOCATED'").assign(zip=lambda d: d["zip"].astype(str))
           .groupby("zip")["crashes"].sum())
    pairs = int((n_b * n_c).dropna().sum())
    print(f"crashes with a ZIP: {int(n_c.sum()):,}   buildings with a ZIP: {int(n_b.sum()):,}")
    print(f"rows a ZIP-only join would produce: {pairs:,}  ({pairs / n_c.sum():.0f} rows per crash on average)")
else:
    print("run the pipeline first")
""")
    md("""
Each crash is repeated once per building in its ZIP, which adds no information about which building is near it. That is the reason the ZIP side is aggregated to a ZIP x year panel before the two datasets meet.

### 4.3 Granularity, and why two
* ZIP x year answers a *neighbourhood-level* question (is more completed housing in a ZIP associated with more crashes?). Strength: simple, all crashes with a ZIP or recoverable ZIP count. Weakness: ZIPs are large and uneven (ecological fallacy).
* 250 m x site x year answers a *street-level before/after* question. Strength: close to the mechanism, supports event-study and a control group. Weakness: needs coordinates; crashes near several buildings are counted for each building (accepted at building level, but inference is at site level).
* Year is the common time grain. Completion dates are known to the day, but occupancy lags completion and crash counts per building per day are mostly zero.
* The two give different answers to different questions, so a difference between them is information, not a bug.

```
 HOUSING (building rows)                           CRASHES (collision rows, streamed)
   zip, lat/lon, comp_year, units                    zip (66%), lat/lon (92%), year, factor, severity
        |                                                   |
        |  (A) ZIP x year                                   |
        +--> units_completed, cum_units ----+     ZIP recovered from coordinates (5.2)
        |                                    +--> ZIP x year panel  (RQ1, RQ2, RQ8, RQ9)
        |                                    |
        |  (B) 250 m buffer                  |
        +--> BallTree(haversine) <-- crash lat/lon
                 |                           
                 +--> exposure per crash: -1 none, 0 never near, 1 near-but-not-yet, 2 near-and-completed
                 |       -> crash_cube  (RQ6, RQ10)
                 +--> sites + control sites, crashes per building x relative year
                         -> event_building  (RQ3, RQ4, RQ5)
```

### 4.4 The exposure flag, explained on a toy example
Exposure is defined per crash with a proximity test (haversine distance, because a degree of longitude is shorter than a degree of latitude at New York's latitude, so a plain Euclidean test on degrees would be wrong). `0` = no building ever within 250 m; `1` = every building within 250 m is completed *later* than the crash year (a placebo: dense places that have not yet changed); `2` = some building within 250 m was completed in or before the crash year. Comparing `2` with `0` alone would compare built-up with empty areas, so `1` vs `0` is shown as a placebo: if "not yet" areas already differ from `0`, the 2-vs-0 gap is partly just location.

The cell below uses five invented points (a toy, not data) to show what the function returns.
""")
    code("""
from pipeline import integrate as integ
toy_b = pd.DataFrame({"building_id": [1, 2], "latitude": [40.7000, 40.7100], "longitude": [-74.0000, -74.0000],
                      "comp_year": [2018, 2022]})
toy_c = pd.DataFrame({"name": ["A 100 m from bldg 1", "B 100 m from bldg 2", "C ~1 km away", "D no coords", "E ~220 m from bldg 1"],
                      "year": [2019, 2019, 2019, 2019, 2017],
                      "latitude": [40.7009, 40.7109, 40.7500, np.nan, 40.7020], "longitude": [-74.0] * 3 + [np.nan, -74.0]})
tree = integ.build_tree(toy_b)                      # haversine BallTree over located buildings
comp = integ.comp_array(integ.tree_rows(toy_b))     # completion year per tree point
has = toy_c["latitude"].notna().to_numpy()          # crashes with coordinates
owner, bldg = integ.neighbour_pairs(toy_c["latitude"][has].to_numpy(), toy_c["longitude"][has].to_numpy(), tree)
toy_c["exposure"] = -1                              # default: no coordinates
toy_c.loc[has, "exposure"] = integ.exposure(owner, bldg, int(has.sum()), toy_c["year"][has].to_numpy(), comp)
display(toy_c)
""")
    md("""
Reading the toy. A is within 250 m of building 1, which was completed (2018) before the crash year (2019): exposure 2. B is near building 2, which is completed only in 2022: exposure 1, the placebo. C has no building within 250 m: 0. D has no coordinates: -1, kept for ZIP counts only. E is near building 1 but a year *before* its completion: 1. The same rule is applied to all crashes in the pipeline (streamed, a chunk at a time).

### 4.5 Validating the spatial join
A spatial index is only useful if it is right. We check the BallTree result against a brute-force haversine distance (every crash against every building) on the cleaned crash sample from section 3; the agreement must be 100%. The same check exists in the unit tests.
""")
    code("""
hb_ = clean_h.dropna(subset=["latitude", "longitude"])
cs = clean_c.dropna(subset=["lat", "lon"]).sample(min(1000, clean_c["lat"].notna().sum()), random_state=SEED)
tree_real = integ.build_tree(clean_h)
fast = integ.flag_near(cs["lat"], cs["lon"], tree_real)                      # BallTree answer
la1, lo1 = np.radians(cs[["lat", "lon"]].to_numpy().T)[:, :, None]
la2, lo2 = np.radians(hb_[["latitude", "longitude"]].to_numpy().T)[:, None, :]
a_ = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
dist_m = 2 * 6_371_000 * np.arcsin(np.sqrt(a_))                              # haversine, metres
brute = (dist_m <= BUFFER_M).any(axis=1)
print(f"{len(cs):,} crashes x {len(hb_):,} buildings: agreement {np.mean(fast == brute):.2%}; "
      f"near-flag share {fast.mean():.1%}")
show(px.histogram(dist_m.min(axis=1), nbins=60, title="Distance from each sampled crash to its nearest building (m)",
                  labels={"value": "metres"}).add_vline(x=BUFFER_M, line_dash="dash").update_layout(showlegend=False), 340)
""")
    md("""
### 4.6 What the integrated tables look like
""")
    code("""
for name, df_ in ART.items():
    print(f"{name:22s} {df_.shape[0]:>10,} rows x {df_.shape[1]:>2} cols   {df_.memory_usage(deep=True).sum() / 1e6:6.1f} MB")
if not ART:
    print("no processed artifacts: run `python -m pipeline.run`")
""")


# --------------------------------------------------------------------------- 5 post-integration
def sec5() -> None:
    md("""
## 5. Post-integration cleaning

Some problems cannot be seen until the datasets meet: left joins create missing values, types differ between sides, and the integration itself needs a repaired ZIP. Rules are in `pipeline/integrate.py` / `pipeline/clean_crashes.py` and logged in the same `cleaning_log`.

### 5.1 Missing values introduced by the join (and what we do about each)

A naive merge of housing-by-ZIP-year onto crash-by-ZIP-year leaves `NaN` in three different meanings. Treating them all the same would be wrong:

| Where | Meaning | Resolution |
|---|---|---|
| Housing metrics for a ZIP-year with no completions | Structural zero (nothing was completed) | fill 0; build rows from the full ZIP x year grid so they exist |
| Crash counts for a ZIP seen in the crash data but with no row for a year | zero crashes recorded | fill 0 |
| Crash counts for a ZIP never seen in the crash data | unknown, not zero | leave out of the panel (ZIPs with fewer than `MIN_ZIP_CRASHES` located crashes are excluded: rates from so few crashes are unstable) |

*Alternative.* Fill every `NaN` with 0 (inflates the sample with ZIPs we know nothing about) or drop every `NaN` row (removes all zero-completion years, which are the comparison group).
""")
    code("""
# Naive merge vs final panel: NaN counts and dtypes
if "panel_zip_year_full" in ART:
    hc, cube = ART["housing_clean"], ART["crash_cube"]
    h_zy = (hc.dropna(subset=["zip", "comp_year"]).groupby(["zip", "comp_year"], observed=True)["total_units"]
              .sum().rename("units").reset_index().rename(columns={"comp_year": "year"}))
    c_zy = (cube[cube["zip"] != "UNLOCATED"].groupby(["zip", "year"], observed=True)["crashes"].sum().reset_index())
    h_zy["year"] = h_zy["year"].astype(int); h_zy["zip"] = h_zy["zip"].astype(str); c_zy["zip"] = c_zy["zip"].astype(str)
    naive = c_zy.merge(h_zy, on=["zip", "year"], how="outer")
    final = ART["panel_zip_year_full"]
    nan_tbl = pd.DataFrame({"naive outer merge": naive.isna().sum(), "final panel": final.isna().sum()}).fillna("n/a")
    display(nan_tbl)
    print("naive rows:", len(naive), "| final panel rows:", len(final))
    print("unique (zip, year) in final panel:", not final.duplicated(["zip", "year"]).any(),
          "| negative counts:", bool((final.select_dtypes("number") < 0).any().any()))
else:
    print("run the pipeline first")
""")
    md("""
### 5.2 Recovering missing ZIPs from coordinates (grid mode imputation)

*Problem.* About a third of crashes have no ZIP; most of those do have coordinates. Dropping them would undercount ZIPs where officers record ZIP less often.
*Method.* Cut the city into 0.002-degree cells (roughly 170 to 220 m on a side). For every crash with both ZIP and coordinates, record the most common ZIP in its cell. A crash without ZIP but with coordinates receives its cell's modal ZIP and is flagged `zip_imputed`. Cells with no known neighbours stay missing; those crashes are counted as `UNLOCATED` in citywide totals only. Borough is then taken from the ZIP's modal borough.
*Why this and not the alternatives.* K-nearest-neighbour imputation needs a spatial query per crash and was not tried; a polygon join needs a ZIP boundary file (a third dataset); dropping is biased; filling with the borough mode leaves ZIP missing.
*Validation.* An imputation can look plausible and still be wrong, so it is scored. Crashes with `collision_id % 20 == 0` (5%) are excluded when building the grid and their known ZIP is then predicted from the grid; accuracy is reported overall and for thinly supported cells. Caveat: crashes with missing ZIP concentrate on highways and bridges, where ZIP is ambiguous, so holdout accuracy on crashes that *have* a ZIP is probably an upper bound for the accuracy on the ones that do not.
""")
    code("""
# Demonstration on the cleaned sample (full-data numbers come from meta.json below)
grid = build_zip_grid(lambda: iter([raw_c]))     # the grid is built from RAW rows (needs zip_code, collision_id)
imputed = clean_crash_chunk(raw_c.copy(), [], zip_grid=grid)
zi = imputed["zip"]
print(f"sample ZIP missing before recovery: {pick(clean_c,'zip','zip_code').isna().mean():.1%}  ->  after: {zi.isna().mean():.1%}")
for k in ["imputation_holdout_acc", "imputation_acc_low_support", "unlocated_share"]:
    print(f"{k:30s}", META.get(k, "n/a"))
sh = META.get("imputed_share_by_year") or META.get("imputed_share")
if isinstance(sh, dict):
    show(px.bar(x=list(sh), y=[100 * v for v in sh.values()], labels={"x": "year", "y": "% of crashes with ZIP imputed"},
                title="Share of crashes whose ZIP was imputed, by year"), 330)
""")
    md("""
Reading the validation. If holdout accuracy is high overall but clearly lower in thinly supported cells, then ZIP-level results for sparse areas are less reliable; the website and notebook show `n` for that reason. If it were low overall, the right response would be to switch to the polygon method, not to proceed.

### 5.3 Types and redundant columns
Types are enforced because the sides disagree: ZIP is a number in housing and text in crashes (joins on mixed types silently match nothing); counts must be integers (not float with NaN); `year` is a small integer. Columns not used downstream (`latitude_internal`, `longitude_internal`, `bbl`, `bin`, community board, council district, census tract, neighbourhood area, street address, bedroom-count unit columns and the raw date columns) are dropped from `housing_clean`; `building_id` and `project_id` remain as identifiers.
""")
    code("""
if "housing_clean" in ART:
    dropped = sorted(set(raw_h.columns) - set(ART["housing_clean"].columns))
    print(f"raw housing columns: {raw_h.shape[1]}  ->  housing_clean columns: {ART['housing_clean'].shape[1]}")
    print("not carried into housing_clean:", ", ".join(dropped))
    display(pd.concat({k: ART[k].dtypes.astype(str).value_counts() for k in ART if k != 'crash_grid'}, axis=1).fillna(0).astype(int))
""")
    md("""
### 5.4 Whole-period view of the integrated crash data
Now that all years are processed (streamed, aggregated), the long time series can be shown. The citywide index (located crashes in year / mean over the window) is the denominator used to remove citywide swings, such as the 2020 drop, from building-level counts.
""")
    code("""
if "crash_cube" in ART:
    cube = ART["crash_cube"]
    yr_tot = cube.groupby("year", observed=True)[["crashes", "injured", "killed"]].sum()
    loc_tot = cube[cube["exposure"] >= 0].groupby("year", observed=True)["crashes"].sum()
    idx = loc_tot / loc_tot.mean()
    fig = go.Figure()
    fig.add_bar(x=yr_tot.index, y=yr_tot["crashes"], name="all crashes (cube)")
    fig.add_scatter(x=idx.index, y=idx.values, name="city index (right axis)", yaxis="y2", mode="lines+markers")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="index"), yaxis_title="crashes",
                      title="Crashes per year and the citywide normalisation index")
    show(fig, 380)
    print("city_index stored in meta.json:", {k: round(v, 3) for k, v in (META.get("city_index") or {}).items()})
""")



# --------------------------------------------------------------------------- 6 results
RQ_TEXT = {
    1: ("ZIP-level association between completed units and growth in crashes",
        "Spearman rank correlation, which tolerates skew and outliers, between units completed from 2016 to the end year and the growth ratio (crashes in the last 3 years + 1) / (crashes in the first 3 years + 1). Only ZIPs with at least 30 crashes in the first window are used, to avoid ratios of tiny counts. It is a ratio and not a rate because there is no population or traffic denominator.",
        "The comparison is between areas. A ZIP-level link says nothing about streets next to the housing, and ZIPs differ in traffic, size and trends that also attract housing."),
    2: ("Within-ZIP: do years with completions coincide with crash changes?",
        "Two-way within transformation (remove each ZIP's average and each year's average) of log(1 + crashes) on units completed, so only change inside a ZIP relative to the city in the same year is used. The interval comes from a bootstrap that resamples whole ZIPs. An any-completion indicator is shown as a second specification.",
        "Removing ZIP and year means does not remove ZIP-specific trends. Completion timing may follow neighbourhood change. A few ZIPs with large completions can drive the slope."),
    3: ("Event study: treated sites against comparison sites",
        "A site is a group of buildings within 250 m of each other completed in the same year. Treated sites are completions in cohort year c with three full years on either side inside the window. Comparison sites are completed four or more years later, so they are not yet treated. Crash counts within 250 m are divided by the citywide index. The estimate is the difference-in-differences of (mean after minus mean before), with comparison sites reweighted to the cohort and borough mix of the treated sites and a bootstrap over sites.",
        "Parallel trends are checked only by eye in the pre-period lines. Sites completed later may differ systematically from earlier ones. Construction work and anticipation are ignored."),
    4: ("By building size: how does the difference-in-differences estimate vary?",
        "The same difference-in-differences computed separately for size bins 1-10, 11-50, 51-150 and 151+ units, shown as bars with bootstrap intervals. Bins are used instead of a continuous slope because size is strongly skewed.",
        "Large buildings are rarer, so their intervals are wider. Size also tracks location: bigger projects tend to sit in different kinds of area."),
    5: ("New construction against preservation",
        "Difference-in-differences by construction type. Preservation renovates an existing building and adds no new structure, so it serves as a rough placebo for new construction.",
        "Preservation and new-build sites differ in location and in the age of their surroundings, so a gap between them cannot be attributed to construction type alone."),
    6: ("Pedestrian and cyclist harm share near completed housing, by borough",
        "Among crashes within 250 m of completed housing (exposure 2) and crashes never near housing (exposure 0), the risk difference in the share that harmed a pedestrian or cyclist, with a 95% interval and Cramér's V as effect size. Exposure 1 (near housing that is completed only later) is shown as a placebo.",
        "Differences between near and far areas can reflect land use and street design. With hundreds of thousands of crashes, tiny differences reach statistical significance, so effect size matters more than the p-value."),
    7: ("Map: housing over crash density",
        "Descriptive map of buildings (bubble size = units, colour = completion year) over a grid of crash density for the selected years. No test.",
        "The picture shows overlap, not a measured relationship. Crash density is dominated by the busiest corridors."),
    8: ("Borough by year heatmap",
        "Colour is units completed; cell text is the year-over-year percentage change in crashes. Descriptive, no test.",
        "Percentage changes in small boroughs and years are noisy, and 2020 dominates the series."),
    9: ("Low-income share against injury severity at ZIP level",
        "Spearman correlation between the share of completed units that are extremely or very low income and injuries per crash, by ZIP, coloured by borough.",
        "The comparison is between areas. Severity depends on speed limits, road types and reporting at least as much as on who lives nearby."),
    10: ("Which contributing factors are over-represented near completed housing?",
        "Log lift: the log of P(factor | exposure 2) / P(factor | exposure 0), with delta-method intervals and the exposure-1 placebo. Factors are the seven groups from 3.12.",
        "A share can rise only because another falls, since shares sum to 1. The Unspecified group is large and uninformative."),
}
OWNER = {i: t["member"] for t in config.TEAM for i in t["rqs"]}


def sec6() -> None:
    n = config.N_RQ
    md(f"""
## 6. Results: {n} research questions

Each question has one figure, built by the function named in `analysis/rq.py`. The sentence under each figure is generated from `rq_results.json`, which the pipeline writes. The wording ("is associated with", "no clear difference") follows the sign of the confidence interval, the sentence states n, and it ends with a reminder that this is an association. Statistics are suppressed and reported as insufficient data when fewer than {config.MIN_N} analysis units exist.

Reading guide: the estimate is the difference being reported (an association). CI is a 95% percentile interval from a bootstrap that resamples sites or ZIPs (seed {config.SEED}), so it measures sampling noise only and says nothing about bias from confounding. With {n} questions and several estimates each, one interval that barely excludes zero is weak evidence (see section 7).
""")
    code("""
from analysis import rq as RQ
from analysis.data import Artifacts

A = Artifacts.load()                                           # lazy handle on the processed directory
SPECS = [s for s in RQ.RQS if int(str(s["id"]).lower().replace("rq", "")) <= config.N_RQ]   # exactly N_RQ questions
rq_path = config.PROCESSED_DIR / "rq_results.json"
RES = json.loads(rq_path.read_text()) if rq_path.exists() else {}
if FIXTURE_MODE:
    fixture_banner()

def _res(i):
    \"\"\"Stored result for RQ i, whatever the key spelling (1, '1', 'RQ1'); computed live if absent.\"\"\"
    for k in (f"RQ{i}", i, str(i)):
        if k in RES:
            return RES[k]
    spec = next((s for s in SPECS if int(s["id"]) == i), None)
    if spec is None:
        return {}
    stat = spec["stat"](A, None, config.BOOT_APP)              # not stored: compute with the app bootstrap size
    return {"headline": RQ.headline(spec, stat), "stat": stat, "ignores": spec["ignores"]}

def show_rq(i):
    \"\"\"Draw RQ i, then print its generated headline, the scalar statistics and the filters it ignores.\"\"\"
    spec = next(s for s in SPECS if int(s["id"]) == i)
    show(spec["fig"](A, {}), 520)                              # figure builder named in spec["chart"]
    r = _res(i)
    stat = {k: v for k, v in (r.get("stat") or {}).items() if np.isscalar(v)}
    txt = f"{spec['title']} (owner {spec.get('member', '')})\\n\\n> {r.get('headline', 'no stored result: run the pipeline')}\\n"
    if r.get("ignores"):
        txt += "\\nFilters this chart cannot honour: " + ", ".join(r["ignores"]) + "\\n"
    if stat:
        txt += "\\n| statistic | value |\\n|---|---|\\n" + "\\n".join(
            f"| {k} | {round(v, 4) if isinstance(v, float) else v} |" for k, v in stat.items())
    display(Markdown(txt))
""")
    for i in range(1, n + 1):
        title, design, limit = RQ_TEXT[i]
        md(f"""
### RQ{i}. {title}
Owner: {OWNER.get(i, 'unassigned')}.

Design and why. {design}

Limits specific to this question. {limit}
""")
        code(f"show_rq({i})")
    md("""
### Summary of generated headlines
""")
    code("""
rows = [{"RQ": s["id"], "headline": _res(int(s["id"])).get("headline", "")} for s in SPECS]
display(pd.DataFrame(rows).style.set_properties(subset=["headline"], **{"text-align": "left", "white-space": "normal"}))
""")


# --------------------------------------------------------------------------- 7-11
def sec7_11() -> None:
    md("""
## 7. Limits and ethics

Statistical limits

* Observational and non-causal. Housing goes where land, policy and demand allow, and those also affect streets. Comparison sites completed later and a citywide index help with common trends, but they cannot remove local confounders such as transit openings, street redesigns, Vision Zero projects and neighbourhood change.
* Ecological fallacy. ZIP-level analyses (RQ1, RQ2, RQ8, RQ9) describe areas, not residents or drivers.
* COVID-19 and reporting changes. 2020 shifted traffic sharply. The index removes the citywide level, not local differences. Police reporting practices and thresholds changed over the period.
* Measurement. Only police-reported crashes are counted. Completion date is not move-in date. The 250 m radius is a judgement. At building level a crash near several buildings is counted once for each, and the bootstrap treats sites as independent although neighbouring sites overlap.
* Imputed ZIPs. The holdout accuracy in 5.2 is probably an upper bound.
* Multiple comparisons. Ten questions with several estimates each, and no correction for that.
* Small samples. Cells with fewer than 8 analysis units are suppressed. Rare categories such as fatal crashes in small ZIPs stay noisy.
* Selection of comparison sites. A site can be a comparison only if it is completed later, so comparison sites are not a random draw of all sites.

Ethical considerations

* Stigma. A reported association near affordable housing could be read as "affordable housing is dangerous". The wording here is non-causal, and "no clear difference" is as likely a result as any other. Any summary should state the limits next to the number.
* Privacy. The data are public and hold no personal identifiers. The website shows aggregates by ZIP and grid cell, not individual crash locations.
* Unequal data quality. Reporting and enforcement differ between neighbourhoods, so crash counts partly measure policing and reporting.
* Use in decisions. The results are exploratory and should not alone decide siting or funding.
* AI assistance is disclosed in sections 9 and 10.

## 8. Impact

* Planners at HPD and DOT get a reproducible way to monitor street safety around new housing, and a view of which building sizes, boroughs and crash types show the clearest or the most uncertain pattern, to decide where to review street design.
* Advocates and the public get a website with filters (borough, years, construction type, factor, severity) and plain-language findings with their uncertainty.
* For the class and other cities, the pipeline pattern applies to any large open dataset joined to a small one by geography: stream, aggregate, validate imputations, ship small artifacts.
* What would add the most: traffic volume or population denominators, street-network distance instead of straight-line distance, a ZIP or neighbourhood boundary file, and a sensitivity analysis over radius and window.
""")


def sec9_11() -> None:
    md("""
## 9. Prompts used

AI coding agents drafted the design, review, code and documentation. No student review is recorded in this table: the last column lists what each student should personally check before submission. The full prompts are in `docs/PROMPTS.md`, and the process (roles, model classes, reasons) is in `docs/AGENT_WORKFLOW.md`. The table abridges them.

| Step | Agent role | Prompt purpose (abridged) | To verify before submission |
|---|---|---|---|
| 1 | Designer | Choose the question and datasets, write the SPEC: integration design, cleaning rules, artifact contract, research questions | a student confirms the question and datasets are what the team wants |
| 2 | Adversarial reviewer | Find flaws in the SPEC before implementation (paging key, ZIP regex, exposure definition, event-study unit) | read SPEC section 11 and confirm you can explain each amendment |
| 3 | Pipeline engineer | Implement extract, cleaning, integration and the CLI to the contract | run the tests and the real-data pipeline; read the cleaning log |
| 4 | Analysis engineer | Implement the research-question figures and statistics with hedged generated headlines | run the analysis on real data; read the generated headlines against the numbers |
| 5 | Web engineer | Dash app with filters, search and Generate report | click through the site and test it deployed |
| 6 | Educator and writer | Build this notebook and the documentation: justify decisions, add validation plots, no hard-coded findings | run this notebook top to bottom on real data and read every cell you present |
| 7 | Independent auditor | Compare code to SPEC, rerun tests, search for hard-coded findings | rerun the audit checks yourself |
""")
    code("""
pp = ROOT / "docs" / "PROMPTS.md"
print(pp.read_text()[:1500] + "\\n..." if pp.exists() else "docs/PROMPTS.md is not in this checkout")
""")
    md("""
## 10. Contributions

Names, IDs and research-question ownership come from `config.TEAM`, so edit them there. The duty split below is the plan in `docs/CONTRIBUTIONS.md`; replace it with what each member actually did before submission. The code was drafted by AI agents, and each member must read, run and be able to explain their own parts.
""")
    code("""
team = pd.DataFrame(config.TEAM)
team["rqs"] = team["rqs"].map(lambda r: ", ".join(f"RQ{x}" for x in r))
display(team.rename(columns={"member": "Member", "name": "Name", "student_id": "Student ID", "rqs": "Research questions"}))
""")
    md(DUTY_TABLE)
    md("""
## 11. Reproducibility

* Pipeline: `python -m pipeline.run` (real data) writes `data/processed/`. Fixture mode (`NYC_FIXTURE=1`) writes `data/processed_fixture/` and sets `fixture: true`.
* Randomness: seed 42 for the bootstrap and for sampling in plots.
* Versions and the data timestamp are printed below. The live datasets change, so exact numbers can differ between runs on different days. `meta.json` records when the artifacts were generated and how many rows each stage had.
* Tests: `python -m pytest -q tests` covers the cleaning rules, the near-flag against brute force, cube conservation, panel uniqueness, and that every research question returns a figure and a statistic.
* Website: `gunicorn app.app:server`; deployment steps are in `README.md`.
* Colab: the first cell clones the repository (set `REPO_URL`) and installs `requirements.txt`; the second runs the pipeline if no processed artifacts exist.
* The committed notebook has no stored outputs. Run all cells to produce them.
""")
    code("""
import platform, importlib
print("python", platform.python_version())
for m in ["pandas", "numpy", "scipy", "sklearn", "plotly", "pyarrow", "dash"]:
    try:
        print(f"{m:8s}", importlib.import_module(m).__version__)
    except Exception:
        print(f"{m:8s} not installed")
print("artifacts generated_at:", META.get("generated_at"), "| fixture:", META.get("fixture"))
print("row counts per stage:", META.get("rows"))
if FIXTURE_MODE:
    fixture_banner()
""")



def main() -> None:
    sec0(); sec1(); sec2(); sec3(); sec4(); sec5(); sec6(); sec7_11(); sec9_11()
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
