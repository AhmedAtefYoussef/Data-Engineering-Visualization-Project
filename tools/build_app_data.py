"""Build the website's data files from the notebook's own code, so both show the same numbers.

The script executes the notebook section sources in tools/notebook_src/ (every file except the
EDA section, in order, in one namespace, inside notebooks/ where the two raw files live) and
then exports small pre-aggregated tables to app/data/. The website reads only those files: it
never imports the notebook, the old pipeline or the raw data, and downloads nothing.

    python tools/build_app_data.py            # about 90 seconds, needs about 3 GB of free memory

Exports (parquet with categorical and small-integer columns, plus two JSON files):

    crash_cube      year x month x borough x factor group x hour band x nearest-site kind
    hour_cube       year x borough x factor group x weekday x hour x nearest-site kind
    zip_cube        ZIP x year x factor group x hour band x nearest-site kind   (panel ZIPs only)
    map_grid        500 m cell x year x borough x factor group x hour band x nearest-site kind
    housing_cube    borough x ZIP x completion year x construction type x size band
    zip_year        the notebook's ZIP x year panel
    sites           located housing sites, with their pre-construction crash percentile (RQ7)
    event_sites     one row per clean site and cohort, yearly counts by relative year (RQ3, RQ5)
    city_year       citywide yearly totals and the 2019 = 100 indices
    robustness      every specification of the notebook's Section 9 (central, checks, comparison cuts, placebos)
    cleaning_log    every cleaning rule with the rows it touched
    results.json    the ten research-question results with their sentences, and the story facts
    meta.json       row counts, data-as-of dates, constants, build time, file sizes

Every crash table carries the same seven counts, one per "who was harmed" option.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"          # must be set before matplotlib is imported: no window opens

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"
OUTPUT_DIR = ROOT / "app" / "data"
sys.path.insert(0, str(ROOT / "tools"))
from assemble_notebook import read_cells, section_files  # noqa: E402
SKIPPED_SECTIONS = ("03_",)               # the EDA section defines nothing that later sections use

# Map grid: square cells of about 500 m, twice the 250 m buffer, so a cell is one "street
# neighbourhood". One degree of latitude is 111.2 km; one degree of longitude at 40.7 N is 84.3 km.
GRID_LAT_STEP = 0.0045                    # degrees, about 500 m
GRID_LON_STEP = 0.0060                    # degrees, about 506 m

NOT_NEAR = "Not near a site"              # label of crashes farther than BUFFER_M from every site
UNKNOWN_BOROUGH = "Unknown"               # crashes with neither a borough nor a recoverable ZIP
EVENT_OUTCOMES = {"injury": "injury_crashes", "all": "crashes", "pedcyc": "ped_cyc_crashes"}
COUNT_COLUMNS = ["n_all", "n_injury", "n_pedcyc", "n_ped", "n_cyc", "n_mot", "n_fatal"]
MAX_TOTAL_MB = 15                         # the files are committed and loaded on a 512 MB host
BYTES_PER_MB = 1_000_000
# Every file the build writes. An output folder holding anything else is refused, never emptied.
OUTPUT_TABLES = ["crash_cube", "hour_cube", "zip_cube", "map_grid", "housing_cube", "zip_year", "sites",
                 "event_sites", "city_year", "robustness", "cleaning_log"]
OUTPUT_FILES = {f"{name}.parquet" for name in OUTPUT_TABLES} | {"results.json", "meta.json"}
# Libraries that shape the numbers, by import name: scikit-learn's ball tree finds every distance and neighbour.
PACKAGES = ["numpy", "pandas", "pyarrow", "fastparquet", "scipy", "sklearn", "statsmodels", "matplotlib", "IPython"]
REBUILD_COMMAND = "python tools/build_app_data.py"


class DataCheckError(RuntimeError):
    """A check on the exported data failed. Nothing has been written to the output folder."""


def require(condition, message: str) -> None:
    """Raise DataCheckError unless `condition` holds. Unlike `assert`, `python -O` cannot remove it."""
    if not bool(condition):
        raise DataCheckError(message)
PARQUET_OPTIONS = {"engine": "pyarrow", "compression": "zstd", "compression_level": 19, "index": False}


# --------------------------------------------------------------------------------------------
# 1. Run the notebook
# --------------------------------------------------------------------------------------------
def run_notebook_sections() -> dict:
    """Execute the notebook's code cells in order and return the resulting namespace."""
    import IPython.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Outside a kernel `display` would print every table; the exports need the variables only.
    IPython.display.display = lambda *args, **kwargs: None
    plt.show = lambda *args, **kwargs: None

    namespace = {"__name__": "__notebook__"}
    started = time.time()
    previous_dir = Path.cwd()
    os.chdir(NOTEBOOK_DIR)                # the notebook reads its two data files from "."
    try:
        for path in section_files(None):
            if path.name.startswith(SKIPPED_SECTIONS):
                continue
            for cell in read_cells([path]):   # same parser as the notebook build, shared code inlined
                if cell.cell_type == "code":
                    exec(compile(cell.source, cell.id, "exec"), namespace)
                    plt.close("all")      # figures are not needed; closing them frees memory
            print(f"  ran {path.name:<26} {time.time() - started:5.0f}s", flush=True)
    finally:
        os.chdir(previous_dir)
    return namespace


# --------------------------------------------------------------------------------------------
# 2. Small helpers
# --------------------------------------------------------------------------------------------
def smallest_unsigned(series: pd.Series) -> pd.Series:
    """Cast a non-negative integer column to the narrowest unsigned type that holds it."""
    require((series >= 0).all(), f"{series.name} has negative values")
    return pd.to_numeric(series, downcast="unsigned")


def as_category(values, categories: list[str]) -> pd.Categorical:
    """A categorical with a fixed category order, failing loudly on a label outside it."""
    result = pd.Categorical(values, categories=categories)
    require(not pd.isna(result).any(), f"values outside {categories}")
    return result


def write_table(frame: pd.DataFrame, name: str, out_dir: Path, schema: dict) -> None:
    """Write one parquet file, record its schema for meta.json and print its shape and size."""
    path = out_dir / f"{name}.parquet"
    frame.to_parquet(path, **PARQUET_OPTIONS)
    schema[name] = {column: str(dtype) for column, dtype in frame.dtypes.items()}
    print(f"  {name:<13} {len(frame):>9,} rows x {frame.shape[1]:>2} columns  {path.stat().st_size / BYTES_PER_MB:6.2f} MB")


def to_plain(value):
    """Convert numpy and pandas scalars and containers to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_plain(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if value is pd.NA or value is None:
        return None
    return value


# --------------------------------------------------------------------------------------------
# 3. Crash tables
# --------------------------------------------------------------------------------------------
def near_kind_labels(construction_types: list[str], size_bands: list[str]) -> list[str]:
    """Labels of the nearest-site kind: 'Not near a site', then every type and size combination."""
    return [NOT_NEAR] + [f"{kind} | {band}" for kind in construction_types for band in size_bands]


def crash_facts(crashes: pd.DataFrame, sites: pd.DataFrame, boroughs: list[str],
                near_labels: list[str], grid: dict) -> pd.DataFrame:
    """One compact row per crash: the filter dimensions and the seven outcome flags.

    `near` is the kind (construction type and size band) of the nearest housing site when that
    site is within the buffer, which is how the notebook tags crashes in Section 5.
    """
    site_kind = (sites["construction_type"].astype(str) + " | " + sites["size_band"].astype(str))
    site_kind.index = sites["site_id"].to_numpy()
    near = crashes["near_site_id"].map(site_kind).fillna(NOT_NEAR)

    facts = pd.DataFrame({
        "year": crashes["year"].astype("uint16"),
        "month": crashes["month"].astype("uint8"),
        "weekday": crashes["crash_date"].dt.dayofweek.astype("uint8"),      # 0 = Monday
        "hour": crashes["hour"].astype("uint8"),
        "borough": as_category(crashes["borough"].astype(object).fillna(UNKNOWN_BOROUGH),
                               boroughs + [UNKNOWN_BOROUGH]),
        "factor": crashes["factor_group"],
        "hour_band": crashes["hour_band"],
        "near": as_category(near, near_labels),
        "zip": crashes["zip"],
        "in_zip_panel": crashes["in_zip_panel"],
        "n_all": np.uint8(1),
        "n_injury": crashes["is_injury_crash"].astype("uint8"),
        "n_pedcyc": crashes["ped_cyc_harm"].astype("uint8"),
        "n_ped": ((crashes["ped_injured"] + crashes["ped_killed"]) > 0).astype("uint8"),
        "n_cyc": ((crashes["cyc_injured"] + crashes["cyc_killed"]) > 0).astype("uint8"),
        "n_mot": ((crashes["mot_injured"] + crashes["mot_killed"]) > 0).astype("uint8"),
        "n_fatal": (crashes["killed"] > 0).astype("uint8"),
    })
    # Every harmed-road-user flag implies an injury crash, so "injury crashes" is the widest
    # of the six harm options and the others are subsets of it.
    for column in ["n_pedcyc", "n_ped", "n_cyc", "n_mot", "n_fatal"]:
        require((facts[column] <= facts["n_injury"]).all(), f"{column} outside injury crashes")
    require((facts["n_pedcyc"] == np.maximum(facts["n_ped"], facts["n_cyc"])).all(),
            "n_pedcyc is not the larger of n_ped and n_cyc")

    located = crashes["is_located"].to_numpy()
    row = np.floor((crashes["lat"].to_numpy() - grid["lat_min"]) / grid["lat_step"])
    column = np.floor((crashes["lon"].to_numpy() - grid["lon_min"]) / grid["lon_step"])
    cell = np.where(located, row * grid["n_columns"] + column, -1).astype("int32")
    require((cell[located] >= 0).all() and (cell.max() < grid["n_rows"] * grid["n_columns"]),
            "a located crash falls outside the map grid")
    facts["cell"] = cell
    return facts


def aggregate(facts: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Sum the seven counts over the given dimensions, keeping only combinations that occur."""
    cube = (facts.groupby(dimensions, observed=True, sort=True)[COUNT_COLUMNS].sum().reset_index())
    for column in COUNT_COLUMNS:
        cube[column] = smallest_unsigned(cube[column])
    require(int(cube["n_all"].sum()) == len(facts), "aggregation must conserve the crash count")
    return cube


# --------------------------------------------------------------------------------------------
# 4. Housing and site tables
# --------------------------------------------------------------------------------------------
def housing_cube(housing: pd.DataFrame, years: list[int], repeat: pd.Series) -> pd.DataFrame:
    """Completed buildings and units by borough, ZIP, completion year, type and size band.

    Confidential rows have no ZIP; they stay in with an empty ZIP so borough totals are whole.
    A building completed again under a later project counts as a building but adds no units
    (`repeat`, the notebook's rule in Section 5), so unit totals match the ZIP x year panel.
    """
    keep = housing["is_completed"] & housing["completion_year"].isin(years)
    done, units = housing[keep], (~repeat[keep]).astype("int64")
    table = pd.DataFrame({
        "borough": done["borough"], "zip": done["zip"].fillna("").astype(str),
        "year": done["completion_year"].astype("uint16"),
        "construction_type": done["construction_type"], "size_band": done["size_band"],
        "is_located": done["is_located"],
        "buildings": 1, "total_units": done["total_units"].astype("int64") * units,
        "affordable_units": done["all_counted_units"].astype("int64") * units,
        "deep_affordable_units": done["deep_affordable_units"].astype("int64") * units,
    })
    measures = ["buildings", "total_units", "affordable_units", "deep_affordable_units"]
    dimensions = [name for name in table.columns if name not in measures]
    cube = table.groupby(dimensions, observed=True, sort=True)[measures].sum().reset_index()
    for column in measures:
        cube[column] = smallest_unsigned(cube[column])
    cube["zip"] = cube["zip"].astype("category")
    require(int(cube["buildings"].sum()) == len(done), "housing_cube must conserve the building count")
    return cube


def site_table(sites: pd.DataFrame, rq7_sites: pd.DataFrame) -> pd.DataFrame:
    """Located housing sites for the map, with the RQ7 baseline percentile where it exists."""
    percentile = rq7_sites.set_index("site_id")[["baseline_crashes", "borough_percentile"]]
    table = sites[["site_id", "cluster_id", "project_name", "borough", "zip", "lat", "lon",
                   "start_year", "completion_year", "construction_type", "size_band",
                   "total_units", "all_counted_units"]].join(percentile, on="site_id")
    table = table.rename(columns={"all_counted_units": "affordable_units"})
    table["lat"] = table["lat"].astype("float32")
    table["lon"] = table["lon"].astype("float32")
    table["zip"] = table["zip"].fillna("").astype(str).astype("category")
    table["project_name"] = table["project_name"].astype(str).str.title().str.slice(0, 60)
    table["completion_year"] = table["completion_year"].astype("float32")       # NaN = not completed
    table["baseline_crashes"] = table["baseline_crashes"].astype("float32")     # NaN = not in RQ7
    # borough_percentile stays float64: the RQ7 mean is recomputed from it and must match the notebook
    require(table["site_id"].is_unique and len(table) == len(sites), "sites table lost or duplicated a site")
    return table


def event_site_table(event_panel: pd.DataFrame, notebook_event_sites: pd.DataFrame,
                     rel_years: list[int]) -> pd.DataFrame:
    """One row per clean site and cohort with yearly counts for three outcomes.

    The rows are in the order of the notebook's `event_sites`, because the bootstrap draws rows
    by position: the same order and seed give the website the notebook's intervals exactly.
    """
    clean = event_panel[~event_panel["contaminated"]]
    keys = ["site_id", "cohort"]
    attributes = clean[clean["rel_year"] == 0].set_index(keys)[
        ["group", "cluster_id", "borough", "construction_type", "size_band", "total_units"]]
    table = attributes
    for short, column in EVENT_OUTCOMES.items():
        counts = clean.pivot(index=keys, columns="rel_year", values=column)[rel_years]
        counts.columns = [f"{short}_{rel_year:+d}" for rel_year in rel_years]
        table = table.join(smallest_unsigned_frame(counts), validate="one_to_one")
    table = table.reset_index()
    table["treated"] = table["group"] == "treated"
    table = table.drop(columns="group")

    # Same rows, same order and same injury counts as the notebook's own table.
    require(len(table) == len(notebook_event_sites), "event_sites has a different row count from the notebook")
    require((table[keys].to_numpy() == notebook_event_sites[keys].to_numpy()).all(),
            "event_sites rows differ in order from the notebook")
    injury = [f"injury_{rel_year:+d}" for rel_year in rel_years]
    notebook_counts = [f"rel_{rel_year:+d}" for rel_year in rel_years]
    require((table[injury].to_numpy() == notebook_event_sites[notebook_counts].to_numpy()).all(),
            "event_sites injury counts differ from the notebook")
    return table


def smallest_unsigned_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply `smallest_unsigned` to every column."""
    return frame.apply(smallest_unsigned)


# --------------------------------------------------------------------------------------------
# 5. Results and metadata
# --------------------------------------------------------------------------------------------
# Short keys for the rows of the notebook's RQ7 table (the notebook's row labels are sentences).
RQ7_ROWS = {"new": "RQ7_NEW", "pooled": "RQ7_POOLED", "within_zip": "RQ7_WITHIN_ZIP", "one_vote": "RQ7_ONE_VOTE"}


def estimate_row(row: pd.Series, estimate: str, low: str, high: str, **extra: str) -> dict:
    """One estimate with its interval, under the same three keys whatever the source columns are called."""
    record = {"estimate": row[estimate], "low": row[low], "high": row[high]}
    record.update({name: row[column] for name, column in extra.items()})
    return record


def unscreened_variants(robustness: pd.DataFrame) -> list[str]:
    """Robustness rows that admit treated sites with other completions nearby."""
    return robustness.loc[~robustness["keeps_screen"].astype(bool), "variant"].astype(str).tolist()


def results_document(ns: dict) -> dict:
    """The ten headline results with their sentences, plus the facts quoted in the story."""
    summary = ns["summary_table"]
    questions = {}
    for key, stored in ns["RESULTS"].items():
        row = summary.loc[key]
        questions[key] = {
            "question": stored["question"], "sentence": stored["sentence"],
            "estimate": stored["estimate"], "low": stored["ci_low"], "high": stored["ci_high"],
            "level": stored["level"], "adjusted_p": stored["adjusted_p"],
            "n": stored["n"], "owner": row["owner"], "label": row["label"], "unit": row["unit"],
            "n_unit": row["n_unit"], "direction": row["direction"],
            "excludes_zero": row["excludes_zero"], "excludes_zero_reported": row["excludes_zero_reported"],
            "kept": row["kept"],
        }
    contrast = ns["rq5_contrast"]
    rq7_sites, rq7_estimates = ns["rq7_sites"], ns["rq7_estimates"]
    is_new = rq7_sites["construction_type"].astype(str) == ns["NEW_CONSTRUCTION"]
    by_type = rq7_sites.groupby("construction_type", observed=True)["borough_percentile"].mean()
    reported_ped, rq9_all = ns["reported_ped"], ns["rq9_result"].loc["ped_cyc_share_pct"]
    detail = {
        # n_control counts comparison rows, one per cohort a site serves; n_comparison_sites counts sites.
        "rq3_did": {**{key: ns["new_build_did"][key] for key in
                       ["percent_change", "percent_low", "percent_high", "p_value", "n_treated", "n_control",
                        "n_clusters"]},
                    **{key: ns["new_build_sizes"][key] for key in ["n_comparison_sites", "n_treated_clusters"]}},
        "rq3_pretrend": {key: ns["pretrend_did"][key] for key in
                         ["percent_change", "percent_low", "percent_high"]},
        "rq5_by_type": {name: {"percent": contrast.loc[name, "estimate_pct"],
                               "low": contrast.loc[name, "low_pct"], "high": contrast.loc[name, "high_pct"],
                               "n": contrast.loc[name, "n_treated"]} for name in contrast.index},
        "rq7": {short: {**estimate_row(rq7_estimates.loc[ns[constant]], "estimate", "low", "high"),
                        # One vote per cluster: n is the clusters, the units that were averaged.
                        "n": int(rq7_estimates.loc[ns[constant], "units_resampled" if short == "one_vote" else "n_sites"]),
                        "units_resampled": int(rq7_estimates.loc[ns[constant], "units_resampled"])}
                for short, constant in RQ7_ROWS.items()},
        "rq7_new_sites": {
            "share_above_borough_median_pct": float((rq7_sites.loc[is_new, "borough_percentile"]
                                                     > ns["NEUTRAL_PERCENTILE"]).mean() * 100),
            "median_baseline_crashes": float(rq7_sites.loc[is_new, "baseline_crashes"].median()),
            "median_baseline_crashes_reference": float(ns["rq7_reference"]["baseline_crashes"].median()),
            "reference_points": len(ns["rq7_reference"]),
            "share_clusters_above_borough_median_pct": ns["rq7_one_vote"]["share_above_pct"],
            "largest_clusters_shown": ns["LARGEST_RQ7_CLUSTERS"],
            "largest_clusters_share_pct": ns["rq7_largest_share"],
        },
        "rq7_by_borough": ns["rq7_by_borough"].reset_index().to_dict("records"),
        "rq7_mean_percentile_by_type": by_type.to_dict(),
        "rq9_adjusted_p": rq9_all["p_adjusted"],
        # Section 9.10: the three ZIP-level results again, on crashes with a reported ZIP code only.
        "reported_zips": {
            "rq1": estimate_row(ns["reported_forecast"], "mae_minus_baseline", "low", "high"),
            "rq2_stock": estimate_row(ns["reported_stock"], "estimate_pct", "low_pct", "high_pct"),
            "rq2_same_year": estimate_row(ns["same_year_reported"], "estimate_pct", "low_pct", "high_pct",
                                          adjusted_p="p_adjusted"),
            "rq2_same_year_all_zips": estimate_row(ns["same_year_all"], "estimate_pct", "low_pct", "high_pct",
                                                   adjusted_p="p_adjusted"),
            "rq9": estimate_row(reported_ped, "rho_within_borough", "low", "high", adjusted_p="p_adjusted"),
        },
        # Section 9.6: the RQ2 slope beside its headline.
        "rq2_checks": {
            "new_construction_stock": ns["rq2_new_stock"], "preservation_stock": ns["rq2_preserved_stock"],
            "leave_one_zip_out": {key: ns["rq2_without_one"][key] for key in ["low", "high", "n_zips"]},
            "most_influential_zip": {**ns["rq2_without_one"]["without"],
                                     **{key: ns["rq2_without_one"][key] for key in
                                        ["zip", "share_of_stock_variation_pct", "units_completed", "preservation_units",
                                         "new_construction_units"]}},
        },
        "false_discovery_rate": ns["FALSE_DISCOVERY_RATE"],
        "unscreened_variants": unscreened_variants(ns["robustness_table"]),
        "comparison_subset_check": ns["CHECK_COMPARISON_SUBSET"],
        "comparison_subset_pretrend": {key: ns["pretrend_not_started"][key] for key in
                                       ["percent_change", "percent_low", "percent_high"]},
        "placebo_overlap": ns["placebo_overlap"],
        "largest_clusters": ns["LARGEST_CLUSTERS_SHOWN"],
    }
    return to_plain({"questions": questions, "detail": detail, "facts": ns["facts"],
                     "limits": ns["limits"],
                     "stages": [{"stage": stage, "table": table, "rows": rows}
                                for (stage, table), rows in ns["stages"]["rows"].items()]})


def file_sha256(path: Path) -> str:
    """SHA-256 of one file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path = ROOT) -> list[Path]:
    """The code that decides the numbers: the notebook sections and the statistics module."""
    folder = root / "tools" / "notebook_src"
    sections = [path for path in folder.rglob("*") if path.is_file() and "__pycache__" not in path.parts
                and path.suffix != ".pyc"]
    return sorted(sections) + [root / "app" / "stats.py"]


def source_hash(root: Path = ROOT) -> str:
    """One SHA-256 over the names and contents of `source_files`. Line endings are ignored so a
    Windows checkout and a Linux checkout of the same commit give the same hash."""
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + bytes([0]))
        digest.update(path.read_bytes().replace(bytes([13, 10]), bytes([10])) + bytes([0]))
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Versions of the libraries that shape the numbers, read from the modules this process imports (installed
    package records can disagree with them when two copies are installed)."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name.lower()] = importlib.import_module(name).__version__
        except ImportError:
            continue
    versions["python"] = platform.python_version()
    return versions


def meta_document(ns: dict, grid: dict, near_labels: list[str], out_dir: Path, seconds: float,
                  schema: dict, previous: dict | None) -> dict:
    """Row counts, dates, constants, the category orders the website relies on, and what it was built from.

    The data date is read from the crash and housing content. `raw_files_saved` is carried over from
    the previous meta.json while the raw file's hash is unchanged (or was never recorded); a file's
    modification date is never read, because a fresh clone resets it.
    """
    crashes, housing = ns["crashes"], ns["housing"]
    latest_crash = crashes["crash_date"].max().strftime("%Y-%m-%d")
    raw_sha256 = {path.name: file_sha256(NOTEBOOK_DIR / path) for path in [ns["HOUSING_FILE"], ns["CRASH_FILE"]]}
    before = previous or {}
    saved_before = before.get("data_as_of", {}).get("raw_files_saved", {})
    hashes_before = before.get("raw_sha256")
    raw_files_saved = {name: saved_before[name] if name in saved_before
                       and (hashes_before is None or hashes_before.get(name) == digest) else latest_crash
                       for name, digest in raw_sha256.items()}
    files = {path.name: path.stat().st_size for path in sorted(out_dir.glob("*")) if path.name != "meta.json"}
    return to_plain({
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "build_seconds": round(seconds),
        "datasets": {
            "housing": {"id": ns["HOUSING_ID"], "name": "Affordable Housing Production by Building",
                        "url": ns["DATASET_PAGE"].format(ns["HOUSING_ID"])},
            "crashes": {"id": ns["CRASH_ID"], "name": "Motor Vehicle Collisions - Crashes",
                        "url": ns["DATASET_PAGE"].format(ns["CRASH_ID"])},
        },
        "data_as_of": {
            "raw_files_saved": raw_files_saved,
            "latest_crash": latest_crash,
            "earliest_crash": crashes["crash_date"].min().strftime("%Y-%m-%d"),
            "latest_housing_completion": housing["completion_date"].max().strftime("%Y-%m-%d"),
        },
        "raw_sha256": raw_sha256,
        "source_sha256": source_hash(),
        "packages": package_versions(),
        "schema": schema,
        "rows": {
            "housing": len(housing), "housing_located": int(housing["is_located"].sum()),
            "housing_confidential": int(housing["is_confidential"].sum()),
            "crashes": len(crashes), "injury_crashes": int(crashes["is_injury_crash"].sum()),
            "crashes_time_unknown": int(crashes["time_unknown"].sum()),
            "crashes_located": int(crashes["is_located"].sum()),
            "crashes_near_a_site": int(crashes["near_site_id"].notna().sum()),
            "sites": len(ns["sites"]), "zip_year": len(ns["zip_year"]),
            "panel_zips": int(ns["zip_year"]["zip"].nunique()),
            "event_site_cohorts": len(ns["event_site_table"]),
        },
        "constants": {
            "year_start": ns["YEAR_START"], "year_end": ns["YEAR_END"], "buffer_m": ns["BUFFER_M"],
            "event_window": ns["EVENT_WINDOW"], "n_boot": ns["N_BOOT"], "n_boot_joint": ns["N_BOOT_JOINT"],
            "ci_level": ns["CI_LEVEL"],
            "min_group_n": ns["MIN_GROUP_N"], "seed": ns["SEED"],
            "log_ratio_offset": ns["LOG_RATIO_OFFSET"], "baseline_years": list(ns["BASELINE_YEARS"]),
        },
        "dimensions": {
            "boroughs": ns["BOROUGHS"], "unknown_borough": UNKNOWN_BOROUGH,
            "construction_types": list(housing["construction_type"].cat.categories),
            "size_bands": ns["SIZE_BAND_LABELS"], "factor_groups": ns["FACTOR_GROUPS"],
            "hour_bands": list(ns["HOUR_BANDS"]),
            "hour_band_ranges": {name: list(hours) for name, hours in ns["HOUR_BANDS"].items()},
            "near_kinds": near_labels, "not_near": NOT_NEAR,
            "event_outcomes": EVENT_OUTCOMES,
        },
        "grid": grid,
        "files": files,
    })


# --------------------------------------------------------------------------------------------
# 6. Export
# --------------------------------------------------------------------------------------------
def check_output_folder(out_dir: Path) -> None:
    """Refuse an output folder that holds anything the build did not write (for example `--out .`)."""
    if out_dir.exists() and not out_dir.is_dir():
        raise DataCheckError(f"{out_dir} is a file, not a folder")
    if out_dir.exists():
        foreign = sorted(path.name for path in out_dir.iterdir() if path.name not in OUTPUT_FILES)
        if foreign:
            raise DataCheckError(f"{out_dir} holds files the build did not write ({', '.join(foreign[:5])}); "
                                 "refusing to replace it. Choose an empty or build-only folder with --out")


def export(ns: dict, out_dir: Path, seconds: float, skip_check: bool = False) -> None:
    """Write every file the website reads, atomically.

    Everything is written to a temporary folder beside `out_dir`, the conservation, budget and
    website checks run there, and only then does the folder replace `out_dir`. A failed check
    leaves the old output untouched and removes the temporary folder.
    """
    out_dir = out_dir.resolve()
    check_output_folder(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if (out_dir / "meta.json").exists():
        previous = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-build-", dir=out_dir.parent))
    try:
        write_all(ns, staging, seconds, previous)
        if not skip_check:
            check_against_notebook(ns, staging)
        swap_in(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def swap_in(staging: Path, out_dir: Path) -> None:
    """Replace `out_dir` with the finished `staging` folder; restore the old folder if that fails."""
    backup = out_dir.with_name(f".{out_dir.name}-old-{os.getpid()}")
    if out_dir.exists():
        out_dir.rename(backup)
    try:
        staging.rename(out_dir)
    except OSError:
        if backup.exists():
            backup.rename(out_dir)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def write_all(ns: dict, out_dir: Path, seconds: float, previous: dict | None) -> None:
    """Write every table, results.json and meta.json into `out_dir`, checking each against the notebook."""
    crashes, housing, sites = ns["crashes"], ns["housing"], ns["sites"]
    bounds = ns["NYC_BOUNDS"]
    schema: dict = {}
    grid = {"lat_min": bounds["lat"][0], "lon_min": bounds["lon"][0],
            "lat_step": GRID_LAT_STEP, "lon_step": GRID_LON_STEP,
            "n_rows": int(np.ceil((bounds["lat"][1] - bounds["lat"][0]) / GRID_LAT_STEP)) + 1,
            "n_columns": int(np.ceil((bounds["lon"][1] - bounds["lon"][0]) / GRID_LON_STEP)) + 1}
    construction_types = list(housing["construction_type"].cat.categories)
    near_labels = near_kind_labels(construction_types, ns["SIZE_BAND_LABELS"])

    facts = crash_facts(crashes, sites, ns["BOROUGHS"], near_labels, grid)
    write_table(aggregate(facts, ["year", "month", "borough", "factor", "hour_band", "near"]), "crash_cube", out_dir, schema)
    write_table(aggregate(facts, ["year", "borough", "factor", "weekday", "hour", "near"]), "hour_cube", out_dir, schema)

    in_panel = facts[facts["in_zip_panel"]].copy()
    in_panel["zip"] = in_panel["zip"].astype(str).astype("category")
    zip_cube = aggregate(in_panel, ["zip", "year", "factor", "hour_band", "near"])
    write_table(zip_cube, "zip_cube", out_dir, schema)

    located = facts[facts["cell"] >= 0].copy()
    located["cell"] = located["cell"].astype("uint16")
    write_table(aggregate(located, ["cell", "year", "borough", "factor", "hour_band", "near"]), "map_grid", out_dir, schema)

    homes = housing_cube(housing, ns["STUDY_YEARS"], ns["is_repeat_building"](housing))
    write_table(homes, "housing_cube", out_dir, schema)

    zip_year = ns["zip_year"].copy()
    zip_year["zip"] = zip_year["zip"].astype(str).astype("category")
    for column in zip_year.columns.difference(["zip", "year", "borough"]):
        zip_year[column] = smallest_unsigned(zip_year[column])
    write_table(zip_year, "zip_year", out_dir, schema)

    write_table(site_table(sites, ns["rq7_sites"]), "sites", out_dir, schema)
    write_table(event_site_table(ns["event_panel"], ns["event_site_table"], ns["REL_YEARS"]), "event_sites", out_dir, schema)
    write_table(ns["city_year"], "city_year", out_dir, schema)
    write_table(ns["robustness_table"], "robustness", out_dir, schema)
    write_table(pd.DataFrame(ns["CLEANING_LOG"]), "cleaning_log", out_dir, schema)
    require(set(schema) == set(OUTPUT_TABLES), "the build wrote a different set of tables from OUTPUT_TABLES")

    # The cubes must reproduce the notebook's ZIP x year panel, cell for cell.
    panel = ns["zip_year"].set_index(["zip", "year"])
    from_cube = zip_cube.groupby(["zip", "year"], observed=True)[["n_all", "n_injury", "n_pedcyc"]].sum()
    from_cube.index = from_cube.index.set_levels(from_cube.index.levels[0].astype(str), level=0)
    aligned = from_cube.reindex(pd.MultiIndex.from_arrays(
        [panel.index.get_level_values(0).astype(str), panel.index.get_level_values(1)]), fill_value=0)
    require((aligned["n_all"].to_numpy() == panel["crashes"].to_numpy()).all(), "zip_cube crashes differ from the panel")
    require((aligned["n_injury"].to_numpy() == panel["injury_crashes"].to_numpy()).all(),
            "zip_cube injury crashes differ from the panel")
    require((aligned["n_pedcyc"].to_numpy() == panel["ped_cyc_crashes"].to_numpy()).all(),
            "zip_cube pedestrian and cyclist crashes differ from the panel")
    housing_by_zip = (homes[homes["zip"].isin(zip_year["zip"].cat.categories)]
                      .groupby(["zip", "year"], observed=True)["affordable_units"].sum())
    require(int(housing_by_zip.sum()) == int(ns["zip_year"]["affordable_units_completed"].sum()),
            "housing_cube affordable units differ from the panel")

    (out_dir / "results.json").write_text(json.dumps(results_document(ns), indent=1), encoding="utf-8")
    meta = meta_document(ns, grid, near_labels, out_dir, seconds, schema, previous)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    total_mb = sum(path.stat().st_size for path in out_dir.glob("*")) / BYTES_PER_MB
    print(f"  total {total_mb:.2f} MB in {out_dir}")
    require(total_mb < MAX_TOTAL_MB, f"exports are {total_mb:.1f} MB, above the {MAX_TOTAL_MB} MB budget")


def check_against_notebook(ns: dict, data_dir: Path) -> None:
    """Recompute three headline numbers with the website's own code, reading `data_dir`, and compare with the notebook."""
    sys.path.insert(0, str(ROOT))
    from app import data as site_data
    from app.filters import make_filters

    site_data.load.cache_clear()
    try:
        store = site_data.load(data_dir)     # also checks every table against the schema the site expects
    finally:
        site_data.load.cache_clear()         # never leave a store that reads the staging folder in the cache
    new_only = make_filters(construction=[ns["NEW_CONSTRUCTION"]])
    study = site_data.event_study(store, new_only)
    notebook = ns["new_build_did"]
    for key in ["percent_change", "percent_low", "percent_high"]:
        require(abs(study["treated"][ns["NEW_CONSTRUCTION"]]["did"][key] - notebook[key]) < 1e-9,
                f"website RQ3 {key} differs from the notebook")
    # RQ7: the headline (new construction) and the pooled row, each with its cluster-bootstrap interval.
    estimates = ns["rq7_estimates"]
    for filters, row_name in [(new_only, ns["RQ7_NEW"]), (make_filters(), ns["RQ7_POOLED"])]:
        selection = site_data.site_selection(store, filters)
        for site_key, notebook_key in [("excess", "estimate"), ("low", "low"), ("high", "high")]:
            require(abs(selection[site_key] - estimates.loc[row_name, notebook_key]) < 1e-6,
                    f"website RQ7 {site_key} differs from the notebook ({row_name})")
        require(selection["n"] == int(estimates.loc[row_name, "n_sites"]), f"website RQ7 site count differs ({row_name})")
    one_vote = site_data.site_selection(store, new_only)["one_vote"]
    for site_key, notebook_key in [("estimate", "estimate"), ("low", "low"), ("high", "high")]:
        require(abs(one_vote[site_key] - estimates.loc[ns["RQ7_ONE_VOTE"], notebook_key]) < 1e-6,
                f"website RQ7 one-vote {site_key} differs from the notebook")
    require(study["n_comparison_sites"] == ns["new_build_sizes"]["n_comparison_sites"],
            "website comparison site count differs from the notebook")
    print("  website code reproduces the notebook's RQ3 and RQ7 estimates, intervals and site counts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    parser.add_argument("--skip-check", action="store_true", help="do not re-run the website's statistics")
    args = parser.parse_args()

    check_output_folder(args.out.resolve())       # fail before the 90 seconds of work, not after
    started = time.time()
    print("Running the notebook sections")
    namespace = run_notebook_sections()
    print("Exporting")
    try:
        export(namespace, args.out, time.time() - started, args.skip_check)
    except DataCheckError as error:
        print(f"BUILD FAILED, the output folder was not changed: {error}", file=sys.stderr)
        return 1
    print(f"Done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
