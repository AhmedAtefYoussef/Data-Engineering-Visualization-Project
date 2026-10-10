"""Fabricate small, schema-exact processed artifacts (SPEC s5 + A3/A4/A8/A9) for developing app/analysis.

Run: python tests/make_mock_artifacts.py   (writes to $NYC_PROCESSED_DIR or data/mock_processed). Random, NOT real data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

BOROUGHS = config.BOROUGHS
FACTORS = ["Distraction/Inattention", "Speed/Aggressive", "Failure to Yield/Signal", "Impairment/Fatigue",
           "Vehicle/Road Defect", "Other", "Unspecified"]
HOURS = ["Night", "Morning", "Afternoon", "Evening", "Unknown"]
SEVS = ["Fatal", "Injury", "Property damage only"]
CENTRES = {"Bronx": (40.85, -73.88), "Brooklyn": (40.65, -73.95), "Manhattan": (40.78, -73.97),
           "Queens": (40.72, -73.80), "Staten Island": (40.58, -74.12)}
ZIP_BASE = {"Bronx": 10451, "Brooklyn": 11201, "Manhattan": 10001, "Queens": 11354, "Staten Island": 10301}


def _zips(rng: np.random.Generator, per: int = 12) -> pd.DataFrame:
    """Synthetic ZIP universe: `per` zips per borough with a centre point and crash weight."""
    rows = [(str(ZIP_BASE[b] + i), b, CENTRES[b][0] + rng.normal(0, .06), CENTRES[b][1] + rng.normal(0, .06),
             rng.lognormal(0, .5)) for b in BOROUGHS for i in range(per)]
    return pd.DataFrame(rows, columns=["zip", "borough", "lat", "lon", "w"])


def _housing(rng: np.random.Generator, z: pd.DataFrame, n: int = 700) -> pd.DataFrame:
    """Building-level housing table with ~15% not completed and comp years 2014-2027."""
    zi = rng.integers(0, len(z), n)
    pid = np.sort(rng.integers(0, n // 2, n))
    py = rng.choice(np.arange(2014, 2028), n // 2 + 1, p=np.r_[np.full(12, .075), .05, .05])
    units = np.maximum(1, rng.lognormal(3.3, 1.1, n).astype(int))
    elis = (rng.beta(2, 3, n) * units).astype(int)
    h = pd.DataFrame({
        "building_id": np.arange(100000, 100000 + n), "project_id": pid,
        "project_name": [f"Project {p}" for p in pid], "borough": z.borough.values[zi],
        "zip": z.zip.values[zi], "latitude": z.lat.values[zi] + rng.normal(0, .01, n),
        "longitude": z.lon.values[zi] + rng.normal(0, .01, n), "comp_year": py[pid].astype("float"),
        "construction_type": rng.choice(["New Construction", "Preservation", "Other"], n, p=[.45, .5, .05]),
        "total_units": units, "all_counted_units": units, "extremely_low_income_units": elis // 2,
        "very_low_income_units": elis - elis // 2, "low_income_units": units // 4,
        "moderate_income_units": units // 8, "middle_income_units": units // 10, "other_income_units": 0})
    h["income_low_share"] = elis / units
    h["size_bin"] = pd.cut(units, [0, 10, 50, 150, np.inf], labels=["1-10", "11-50", "51-150", "151+"]).astype(str)
    h["is_large_outlier"] = units > np.percentile(units, 97)
    h["completed"] = rng.random(n) > .1
    h.loc[~h.completed, "comp_year"] = np.nan
    return h


def _crashes(rng: np.random.Generator, z: pd.DataFrame, n: int = 220_000) -> pd.DataFrame:
    """Row-level synthetic crashes (later aggregated into cube and grid)."""
    yr = rng.choice(np.arange(2016, 2026), n, p=np.array([1, 1, 1, 1, .7, .85, 1, 1, 1, 1]) / 9.55)
    zi = rng.choice(len(z), n, p=z.w / z.w.sum())
    sev = rng.choice(SEVS, n, p=[.004, .26, .736])
    ped = rng.random(n) < np.where(sev == "Property damage only", .01, .2)
    df = pd.DataFrame({
        "zip": z.zip.values[zi], "borough": z.borough.values[zi], "year": yr.astype("int16"),
        "hour_band": rng.choice(HOURS, n, p=[.12, .3, .3, .26, .02]), "factor_group": rng.choice(FACTORS, n, p=[.2, .1, .12, .04, .05, .09, .4]),
        "severity": sev, "exposure": rng.choice([-1, 0, 1, 2], n, p=[.08, .3, .3, .32]).astype("int8"),
        "ped_cyc": ped.astype("int8"), "lat": z.lat.values[zi] + rng.normal(0, .02, n), "lon": z.lon.values[zi] + rng.normal(0, .02, n)})
    df["injured"] = np.where(sev == "Property damage only", 0, 1 + rng.poisson(.3, n))
    df["killed"] = (sev == "Fatal").astype(int)
    df["ped_injured"] = np.where(ped & (df.injured > 0), 1, 0)
    df["cyc_injured"] = np.where(~ped & (rng.random(n) < .02) & (df.injured > 0), 1, 0)
    unl = rng.random(n) < .1
    df.loc[unl, ["zip", "exposure"]] = ["UNLOCATED", -1]
    return df


def _cube_grid(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate row-level crashes to the cube and the 0.005-degree grid (A9)."""
    keys = ["zip", "borough", "year", "hour_band", "factor_group", "severity", "exposure", "ped_cyc"]
    cube = c.assign(crashes=1).groupby(keys, observed=True)[["crashes", "injured", "killed", "ped_injured", "cyc_injured"]].sum().reset_index()
    cube[["crashes", "injured", "killed", "ped_injured", "cyc_injured"]] = cube[["crashes", "injured", "killed", "ped_injured", "cyc_injured"]].astype("int32")
    for col in ["zip", "borough", "hour_band", "factor_group", "severity"]:
        cube[col] = cube[col].astype("category")
    g = config.GRID_DEG
    c = c[c.zip != "UNLOCATED"]
    grid = c.assign(cell_lat=(c.lat / g).round() * g, cell_lon=(c.lon / g).round() * g, crashes=1,
                    ped_cyc_crashes=c.ped_cyc).groupby(["cell_lat", "cell_lon", "year", "borough", "severity"], observed=True)[
        ["crashes", "injured", "ped_cyc_crashes"]].sum().reset_index()
    return cube, grid


def _events(rng: np.random.Generator, h: pd.DataFrame, cidx: dict[int, float]) -> pd.DataFrame:
    """Building x rel_year table: treated cohorts 2019-2022, controls = comp_year >= cohort+4 (A4)."""
    h = h[h.completed].copy()
    h["base"] = rng.gamma(2, 3, len(h))
    parts = []
    for c in range(config.Y_START + 3, config.Y_END - 2):
        t = h[h.comp_year == c].assign(group="treated")
        k = h[h.comp_year >= c + 4].assign(group="control")
        for g in (t, k):
            r = np.arange(-3, 4)
            e = g.loc[g.index.repeat(7)].copy()
            e["rel_year"] = np.tile(r, len(g)).astype("int8")
            e["cohort"] = c
            parts.append(e)
    e = pd.concat(parts, ignore_index=True)
    e["year"] = e.cohort + e.rel_year
    e["crashes"] = rng.poisson(e.base * e.year.map(cidx)).astype("int32")
    e["adj"] = e.crashes / e.year.map(cidx)
    e["site_id"] = e.project_id
    e["contaminated"] = (e.project_id % 7 == 0)
    e["ped_cyc_crashes"] = (e.crashes * .2).astype("int32")
    e["injured"] = (e.crashes * .3).astype("int32")
    cols = ["building_id", "site_id", "group", "cohort", "contaminated", "comp_year", "rel_year", "construction_type",
            "borough", "size_bin", "total_units", "crashes", "adj", "ped_cyc_crashes", "injured"]
    return e[cols]


def _panel(h: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """ZIP x year housing panel (A8) with cumulative units from 2014."""
    d = h[h.completed].assign(y=lambda x: x.comp_year.astype(int), eli=lambda x: x.extremely_low_income_units + x.very_low_income_units)
    d["new"] = np.where(d.construction_type == "New Construction", d.total_units, 0)
    d["pres"] = np.where(d.construction_type == "Preservation", d.total_units, 0)
    d["oth"] = np.where(d.construction_type == "Other", d.total_units, 0)
    a = d.groupby(["zip", "y"]).agg(units_completed=("total_units", "sum"), counted_units=("all_counted_units", "sum"), eli_vli_units=("eli", "sum"),
                                    new_con_units=("new", "sum"), preservation_units=("pres", "sum"), other_units=("oth", "sum"),
                                    buildings_completed=("building_id", "count"))
    full = pd.MultiIndex.from_product([z.zip, range(2014, config.Y_END + 1)], names=["zip", "y"])
    a = a.reindex(full, fill_value=0)
    a["cum_units"] = a.groupby(level=0).units_completed.cumsum()
    a = a.reset_index().rename(columns={"y": "year"})
    a = a[a.year >= config.Y_START].merge(z[["zip", "borough"]], on="zip")
    a = a[["zip", "borough", "year"] + [c for c in a.columns if c not in ("zip", "borough", "year")]]
    a["year"] = a.year.astype("int16")
    return a.astype({c: "int32" for c in a.columns[3:]})


def make(out: Path, seed: int = 42) -> Path:
    """Write all artifacts into `out` and return it."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    z = _zips(rng)
    h = _housing(rng, z)
    c = _crashes(rng, z)
    cube, grid = _cube_grid(c)
    loc = c[c.zip != "UNLOCATED"].groupby("year").size()
    cidx = (loc / loc.mean()).to_dict()
    ev = _events(rng, h, cidx)
    h["event_eligible"] = h.comp_year.between(config.Y_START + 3, config.Y_END - 3)
    for col in ["pre3_crashes", "post3_crashes", "pre3_adj", "post3_adj", "crashes_250m_total", "injuries_250m_total"]:
        h[col] = rng.gamma(2, 5, len(h)).round(1)
    h["borough"] = h.borough.astype("category")
    h.to_parquet(out / "housing_clean.parquet", index=False)
    cube.to_parquet(out / "crash_cube.parquet", index=False)
    grid.to_parquet(out / "crash_grid.parquet", index=False)
    ev.to_parquet(out / "event_building.parquet", index=False)
    _panel(h, z).to_parquet(out / "panel_zip_year.parquet", index=False)
    log = [dict(step="mock", dataset="housing", rule="mock rule", rows_before=1000, rows_after=900, rationale="synthetic")]
    meta = dict(fixture=True, mock=True, Y_START=config.Y_START, Y_END=config.Y_END, generated_at=pd.Timestamp.now().isoformat(),
                rows=dict(housing=len(h), crashes=len(c), cube=len(cube)), imputed_share=.04, imputation_holdout_acc=.9,
                imputation_acc_low_support=.7, unlocated_share=float((c.zip == "UNLOCATED").mean()),
                city_index={str(k): float(v) for k, v in cidx.items()}, cleaning_log=log, n_sites_treated=int(ev[ev.group == "treated"].site_id.nunique()),
                n_sites_control=int(ev[ev.group == "control"].site_id.nunique()), eligible_cohorts=[2019, 2020, 2021, 2022],
                sources=config.DATASET_PAGES)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return out


if __name__ == "__main__":
    d = Path(os.environ.get("NYC_PROCESSED_DIR") or ROOT / "data" / "mock_processed")
    print("wrote", make(d))
