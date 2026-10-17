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
    robustness      every specification of the notebook's Section 9 (central, checks, placebos)
    cleaning_log    every cleaning rule with the rows it touched
    results.json    the ten research-question results with their sentences, and the story facts
    meta.json       row counts, data-as-of dates, constants, build time, file sizes

Every crash table carries the same seven counts, one per "who was harmed" option.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
    assert (series >= 0).all(), f"{series.name} has negative values"
    return pd.to_numeric(series, downcast="unsigned")


def as_category(values, categories: list[str]) -> pd.Categorical:
    """A categorical with a fixed category order, failing loudly on a label outside it."""
    result = pd.Categorical(values, categories=categories)
    assert not pd.isna(result).any(), f"values outside {categories}"
    return result


def write_table(frame: pd.DataFrame, name: str, out_dir: Path) -> None:
    """Write one parquet file and print its shape and size."""
    path = out_dir / f"{name}.parquet"
    frame.to_parquet(path, **PARQUET_OPTIONS)
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
        assert (facts[column] <= facts["n_injury"]).all(), f"{column} outside injury crashes"
    assert (facts["n_pedcyc"] == np.maximum(facts["n_ped"], facts["n_cyc"])).all()

    located = crashes["is_located"].to_numpy()
    row = np.floor((crashes["lat"].to_numpy() - grid["lat_min"]) / grid["lat_step"])
    column = np.floor((crashes["lon"].to_numpy() - grid["lon_min"]) / grid["lon_step"])
    cell = np.where(located, row * grid["n_columns"] + column, -1).astype("int32")
    assert (cell[located] >= 0).all() and (cell.max() < grid["n_rows"] * grid["n_columns"])
    facts["cell"] = cell
    return facts


def aggregate(facts: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Sum the seven counts over the given dimensions, keeping only combinations that occur."""
    cube = (facts.groupby(dimensions, observed=True, sort=True)[COUNT_COLUMNS].sum().reset_index())
    for column in COUNT_COLUMNS:
        cube[column] = smallest_unsigned(cube[column])
    assert int(cube["n_all"].sum()) == len(facts), "aggregation must conserve the crash count"
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
    assert int(cube["buildings"].sum()) == len(done)
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
    assert table["site_id"].is_unique and len(table) == len(sites)
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
    assert len(table) == len(notebook_event_sites)
    assert (table[keys].to_numpy() == notebook_event_sites[keys].to_numpy()).all()
    injury = [f"injury_{rel_year:+d}" for rel_year in rel_years]
    notebook_counts = [f"rel_{rel_year:+d}" for rel_year in rel_years]
    assert (table[injury].to_numpy() == notebook_event_sites[notebook_counts].to_numpy()).all()
    return table


def smallest_unsigned_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply `smallest_unsigned` to every column."""
    return frame.apply(smallest_unsigned)


# --------------------------------------------------------------------------------------------
# 5. Results and metadata
# --------------------------------------------------------------------------------------------
# Short keys for the rows of the notebook's RQ7 table (the notebook's row labels are sentences).
RQ7_ROWS = {"new": "RQ7_NEW", "pooled": "RQ7_POOLED", "within_zip": "RQ7_WITHIN_ZIP"}


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
            "excludes_zero": row["excludes_zero"], "kept": row["kept"],
        }
    contrast = ns["rq5_contrast"]
    rq7_sites, rq7_estimates = ns["rq7_sites"], ns["rq7_estimates"]
    is_new = rq7_sites["construction_type"].astype(str) == ns["NEW_CONSTRUCTION"]
    by_type = rq7_sites.groupby("construction_type", observed=True)["borough_percentile"].mean()
    reported_ped, rq9_all = ns["reported_ped"], ns["rq9_result"].loc["ped_cyc_share_pct"]
    detail = {
        "rq3_did": {key: ns["new_build_did"][key] for key in
                    ["percent_change", "percent_low", "percent_high", "p_value", "n_treated", "n_control"]},
        "rq3_pretrend": {key: ns["pretrend_did"][key] for key in
                         ["percent_change", "percent_low", "percent_high"]},
        "rq5_by_type": {name: {"percent": contrast.loc[name, "estimate_pct"],
                               "low": contrast.loc[name, "low_pct"], "high": contrast.loc[name, "high_pct"],
                               "n": contrast.loc[name, "n_treated"]} for name in contrast.index},
        "rq7": {short: {**estimate_row(rq7_estimates.loc[ns[constant]], "estimate", "low", "high"),
                        "n": int(rq7_estimates.loc[ns[constant], "n_sites"]),
                        "units_resampled": int(rq7_estimates.loc[ns[constant], "units_resampled"])}
                for short, constant in RQ7_ROWS.items()},
        "rq7_new_sites": {
            "share_above_borough_median_pct": float((rq7_sites.loc[is_new, "borough_percentile"]
                                                     > ns["NEUTRAL_PERCENTILE"]).mean() * 100),
            "median_baseline_crashes": float(rq7_sites.loc[is_new, "baseline_crashes"].median()),
            "median_baseline_crashes_reference": float(ns["rq7_reference"]["baseline_crashes"].median()),
            "reference_points": len(ns["rq7_reference"]),
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
        "false_discovery_rate": ns["FALSE_DISCOVERY_RATE"],
        "unscreened_variants": unscreened_variants(ns["robustness_table"]),
        "largest_clusters": ns["LARGEST_CLUSTERS_SHOWN"],
    }
    return to_plain({"questions": questions, "detail": detail, "facts": ns["facts"],
                     "limits": ns["limits"],
                     "stages": [{"stage": stage, "table": table, "rows": rows}
                                for (stage, table), rows in ns["stages"]["rows"].items()]})


def meta_document(ns: dict, grid: dict, near_labels: list[str], out_dir: Path, seconds: float) -> dict:
    """Row counts, dates, constants and the category orders the website relies on."""
    crashes, housing = ns["crashes"], ns["housing"]
    raw_files = {path.name: datetime.fromtimestamp((NOTEBOOK_DIR / path).stat().st_mtime, timezone.utc)
                 .strftime("%Y-%m-%d") for path in [ns["HOUSING_FILE"], ns["CRASH_FILE"]]}
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
            "raw_files_saved": raw_files,
            "latest_crash": crashes["crash_date"].max().strftime("%Y-%m-%d"),
            "earliest_crash": crashes["crash_date"].min().strftime("%Y-%m-%d"),
            "latest_housing_completion": housing["completion_date"].max().strftime("%Y-%m-%d"),
        },
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
            "event_window": ns["EVENT_WINDOW"], "n_boot": ns["N_BOOT"], "ci_level": ns["CI_LEVEL"],
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
def export(ns: dict, out_dir: Path, seconds: float) -> None:
    """Write every file the website reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*"):
        if stale.suffix in {".parquet", ".json"}:
            stale.unlink()

    crashes, housing, sites = ns["crashes"], ns["housing"], ns["sites"]
    bounds = ns["NYC_BOUNDS"]
    grid = {"lat_min": bounds["lat"][0], "lon_min": bounds["lon"][0],
            "lat_step": GRID_LAT_STEP, "lon_step": GRID_LON_STEP,
            "n_rows": int(np.ceil((bounds["lat"][1] - bounds["lat"][0]) / GRID_LAT_STEP)) + 1,
            "n_columns": int(np.ceil((bounds["lon"][1] - bounds["lon"][0]) / GRID_LON_STEP)) + 1}
    construction_types = list(housing["construction_type"].cat.categories)
    near_labels = near_kind_labels(construction_types, ns["SIZE_BAND_LABELS"])

    facts = crash_facts(crashes, sites, ns["BOROUGHS"], near_labels, grid)
    write_table(aggregate(facts, ["year", "month", "borough", "factor", "hour_band", "near"]), "crash_cube", out_dir)
    write_table(aggregate(facts, ["year", "borough", "factor", "weekday", "hour", "near"]), "hour_cube", out_dir)

    in_panel = facts[facts["in_zip_panel"]].copy()
    in_panel["zip"] = in_panel["zip"].astype(str).astype("category")
    zip_cube = aggregate(in_panel, ["zip", "year", "factor", "hour_band", "near"])
    write_table(zip_cube, "zip_cube", out_dir)

    located = facts[facts["cell"] >= 0].copy()
    located["cell"] = located["cell"].astype("uint16")
    write_table(aggregate(located, ["cell", "year", "borough", "factor", "hour_band", "near"]), "map_grid", out_dir)

    homes = housing_cube(housing, ns["STUDY_YEARS"], ns["is_repeat_building"](housing))
    write_table(homes, "housing_cube", out_dir)

    zip_year = ns["zip_year"].copy()
    zip_year["zip"] = zip_year["zip"].astype(str).astype("category")
    for column in zip_year.columns.difference(["zip", "year", "borough"]):
        zip_year[column] = smallest_unsigned(zip_year[column])
    write_table(zip_year, "zip_year", out_dir)

    write_table(site_table(sites, ns["rq7_sites"]), "sites", out_dir)
    write_table(event_site_table(ns["event_panel"], ns["event_site_table"], ns["REL_YEARS"]), "event_sites", out_dir)
    write_table(ns["city_year"], "city_year", out_dir)
    write_table(ns["robustness_table"], "robustness", out_dir)
    write_table(pd.DataFrame(ns["CLEANING_LOG"]), "cleaning_log", out_dir)

    # The cubes must reproduce the notebook's ZIP x year panel, cell for cell.
    panel = ns["zip_year"].set_index(["zip", "year"])
    from_cube = zip_cube.groupby(["zip", "year"], observed=True)[["n_all", "n_injury", "n_pedcyc"]].sum()
    from_cube.index = from_cube.index.set_levels(from_cube.index.levels[0].astype(str), level=0)
    aligned = from_cube.reindex(pd.MultiIndex.from_arrays(
        [panel.index.get_level_values(0).astype(str), panel.index.get_level_values(1)]), fill_value=0)
    assert (aligned["n_all"].to_numpy() == panel["crashes"].to_numpy()).all()
    assert (aligned["n_injury"].to_numpy() == panel["injury_crashes"].to_numpy()).all()
    assert (aligned["n_pedcyc"].to_numpy() == panel["ped_cyc_crashes"].to_numpy()).all()
    housing_by_zip = (homes[homes["zip"].isin(zip_year["zip"].cat.categories)]
                      .groupby(["zip", "year"], observed=True)["affordable_units"].sum())
    assert int(housing_by_zip.sum()) == int(ns["zip_year"]["affordable_units_completed"].sum())

    (out_dir / "results.json").write_text(json.dumps(results_document(ns), indent=1), encoding="utf-8")
    meta = meta_document(ns, grid, near_labels, out_dir, seconds)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    total_mb = sum(path.stat().st_size for path in out_dir.glob("*")) / BYTES_PER_MB
    print(f"  total {total_mb:.2f} MB in {out_dir}")
    assert total_mb < MAX_TOTAL_MB, f"exports are {total_mb:.1f} MB, above the {MAX_TOTAL_MB} MB budget"


def check_against_notebook(ns: dict) -> None:
    """Recompute three headline numbers with the website's own code and compare with the notebook."""
    sys.path.insert(0, str(ROOT))
    from app import data as site_data
    from app.filters import make_filters

    site_data.load.cache_clear()
    store = site_data.load()
    new_only = make_filters(construction=[ns["NEW_CONSTRUCTION"]])
    study = site_data.event_study(store, new_only)
    notebook = ns["new_build_did"]
    for key in ["percent_change", "percent_low", "percent_high"]:
        assert abs(study["treated"][ns["NEW_CONSTRUCTION"]]["did"][key] - notebook[key]) < 1e-9, key
    # RQ7: the headline (new construction) and the pooled row, each with its cluster-bootstrap interval.
    estimates = ns["rq7_estimates"]
    for filters, row_name in [(new_only, ns["RQ7_NEW"]), (make_filters(), ns["RQ7_POOLED"])]:
        selection = site_data.site_selection(store, filters)
        for site_key, notebook_key in [("excess", "estimate"), ("low", "low"), ("high", "high")]:
            assert abs(selection[site_key] - estimates.loc[row_name, notebook_key]) < 1e-6, (row_name, site_key)
        assert selection["n"] == int(estimates.loc[row_name, "n_sites"])
    print("  website code reproduces the notebook's RQ3 and RQ7 estimates and intervals")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    parser.add_argument("--skip-check", action="store_true", help="do not re-run the website's statistics")
    args = parser.parse_args()

    started = time.time()
    print("Running the notebook sections")
    namespace = run_notebook_sections()
    print("Exporting")
    export(namespace, args.out, time.time() - started)
    if not args.skip_check and args.out == OUTPUT_DIR:
        check_against_notebook(namespace)
    print(f"Done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
