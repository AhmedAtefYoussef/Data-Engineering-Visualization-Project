"""CLI: python -m pipeline.run [--fixture] [--max-rows N] -> parquet artifacts, meta.json, rq_results.json."""
from __future__ import annotations

import os
import sys

if "--fixture" in sys.argv:                      # must precede `import config` (read at import time)
    os.environ["NYC_FIXTURE"] = "1"

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from pipeline import integrate as ig
from pipeline.clean_crashes import SeenIds, build_zip_grid, clean_crash_chunk
from pipeline.clean_housing import clean_housing, log_rule
from pipeline.extract import crash_chunks, housing_raw

T0 = time.time()


def say(msg: str) -> None:
    """Concise progress line with elapsed seconds."""
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def stream_crashes(chunks, grid: dict, tree, comp: np.ndarray, nb: int, log: list) -> dict:
    """Pass 2: clean + impute + expose + aggregate every chunk; nothing but aggregates is retained."""
    seen, cube, gagg = SeenIds(), None, None
    counts = np.zeros((3, nb, ig.NY))
    zcats = [*grid["zips"], "UNLOCATED"]
    yr = {k: np.zeros(ig.NY, np.int64) for k in ("total", "imputed", "located", "unlocated")}
    n_raw = n_clean = 0
    for raw in chunks():
        n_raw += len(raw)
        clean = clean_crash_chunk(raw, log, grid, seen)
        if clean.empty:
            continue
        n_clean += len(clean)
        expo, cnt = ig.process_chunk(clean, tree, comp, nb)
        cube = ig.add_part(cube, ig.cube_part(clean, expo, zcats), ig.CUBE_KEYS, ig.CUBE_MEAS)
        gagg = ig.add_part(gagg, ig.grid_part(clean), ig.GRID_KEYS, ig.GRID_MEAS)
        counts += cnt
        year_idx = clean["year"].to_numpy() - config.Y_START
        for name, mask in (("total", None), ("imputed", clean["zip_imputed"]), ("located", clean["lat"].notna()),
                           ("unlocated", clean["zip"].isna())):
            yr[name] += np.bincount(year_idx, weights=None if mask is None else mask.to_numpy(), minlength=ig.NY).astype(np.int64)
        say(f"  chunk done: raw={n_raw:,} clean={n_clean:,}")
    return dict(cube=cube, grid=gagg, counts=counts, yr=yr, n_raw=n_raw, n_clean=n_clean)


def _share(num: np.ndarray, den: np.ndarray) -> dict:
    """Per-year share as {year: float}."""
    return {int(y): float(a / b) if b else None for y, a, b in zip(ig.YEARS, num, den)}


def build_meta(res: dict, grid: dict, ev: dict, cidx: pd.Series, h: pd.DataFrame, n_house_raw: int,
               dates: tuple, log: list, max_rows: int | None) -> dict:
    """Assemble meta.json (SPEC section 5 + A6) from pass-2 aggregates."""
    y, tot = res["yr"], int(res["yr"]["total"].sum())
    return {
        "fixture": bool(config.FIXTURE), "partial": max_rows is not None, "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_rows": max_rows, "Y_START": config.Y_START, "Y_END": config.Y_END, "buffer_m": config.BUFFER_M,
        "event_window": config.EVENT_WINDOW, "seed": config.SEED, "sources": config.DATASET_PAGES,
        "rows": {"housing_raw": n_house_raw, "housing_clean": len(h), "housing_located": int(h["latitude"].notna().sum()),
                 "housing_completed": int(h["completed"].sum()), "crashes_raw": res["n_raw"],
                 "crashes_clean_in_window": res["n_clean"], "crashes_located": int(y["located"].sum())},
        "dates": {"crash_years": [config.Y_START, config.Y_END], "housing_completion_min": dates[0],
                  "housing_completion_max": dates[1]},
        "zip_imputed_share": float(y["imputed"].sum() / tot) if tot else None,
        "imputation_holdout_acc": grid["holdout_acc"], "imputation_acc_low_support": grid["holdout_acc_low_support"],
        "imputation_holdout_n": grid["holdout_n"], "imputation_holdout_coverage": grid["holdout_coverage"],
        "imputation_note": "Missing ZIPs cluster on highways, so holdout accuracy is an upper bound.",
        "unlocated_share": float(y["unlocated"].sum() / tot) if tot else None,
        "imputed_share_by_year": _share(y["imputed"], y["total"]),
        "located_share_by_year": _share(y["located"], y["total"]),
        "city_index": {int(k): (None if pd.isna(v) else float(v)) for k, v in cidx.items()},
        "eligible_cohorts": [int(cohort) for cohort in ev["cohorts"]],
        "n_sites_treated": int(len(np.unique(ev["site"][ev["treated"]]))),
        "n_sites_control": int(len(np.unique(ev["site"][~ev["treated"]]))),
        "n_sites_contaminated": int(len(np.unique(ev["site"][ev["contaminated"]]))),
        "definitions": {
            "exposure": "-1 no coordinates; 0 no housing within 250 m ever; 1 every building within 250 m completes "
                        "after the crash year (placebo); 2 some building within 250 m completed <= crash year",
            "pre3/post3": "pre3_* = sum over years c-3..c-1, post3_* = sum over c+1..c+3 (adj = crashes / city_index)",
            "units_completed": "sum of total_units of buildings completed in the year",
            "panel_located": "ZIP-assigned crashes (reported or grid-imputed); UNLOCATED excluded",
            "event_counts": "crashes with valid coordinates within 250 m (haversine) of each building"},
        "cleaning_log": log}


def write_all(out: Path, arts: dict, meta: dict) -> None:
    """Write parquet artifacts then meta.json; stale parquet files not in `arts` are removed."""
    for old in out.glob("*.parquet"):
        if old.stem not in arts:
            old.unlink()
            say(f"removed stale artifact {old.name}")
    for name, df in arts.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        say(f"wrote {name}: {len(df):,} rows")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


def run_rq(out: Path) -> None:
    """Compute rq_results.json via analysis.rq.run_all when that module is available."""
    try:
        from analysis.rq import run_all
    except ImportError:
        say("analysis.rq not available -> skipping rq_results.json")
        return
    (out / "rq_results.json").write_text(json.dumps(run_all(out), indent=1, default=str))
    say("wrote rq_results.json")


def run(max_rows: int | None = None, out_dir: str | Path | None = None, with_rq: bool = True) -> dict:
    """Run the full ETL and return the meta dict."""
    out = Path(out_dir or config.PROCESSED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    log: list = []
    say(f"fixture={config.FIXTURE} max_rows={max_rows} out={out}")
    raw_h = housing_raw()
    h = clean_housing(raw_h, log)
    dates = tuple(str(h["completion_date"].agg(f).date()) if h["completed"].any() else None for f in ("min", "max"))
    say(f"housing cleaned: {len(raw_h):,} -> {len(h):,}")
    chunks = crash_chunks(max_rows)
    grid = build_zip_grid(chunks)
    say(f"pass 1 done: {len(grid['key']):,} grid cells, holdout acc={grid['holdout_acc']}")
    h = ig.prepare_housing(h, grid["zip_borough"], log)
    rows, tree = ig.tree_rows(h), ig.build_tree(h)
    ev = ig.event_sites(rows, tree)
    res = stream_crashes(chunks, grid, tree, ig.comp_array(rows), len(rows), log)
    cube = res["cube"]
    n_noc = int((cube.loc[cube["exposure"] < 0, "crashes"]).sum())
    log_rule(log, "post-integration", "crashes", "exposure = -1 for crashes without valid coordinates",
             res["n_clean"], res["n_clean"], "Kept in ZIP counts; excluded from exposure analyses (A3).", n_noc)
    n_unl = int(cube.loc[cube["zip"] == "UNLOCATED", "crashes"].sum())
    log_rule(log, "post-integration", "crashes", "no ZIP after imputation -> zip='UNLOCATED'",
             res["n_clean"], res["n_clean"], "Counted citywide only; excluded from the ZIP panel.", n_unl)
    cidx = ig.city_index(cube)
    h = ig.attach_counts(h, rows, ev, res["counts"], cidx, log)
    eb = ig.event_building(rows, ev, res["counts"], cidx)
    panel = ig.build_panel(h, cube, log)
    arts = {"housing_clean": h, "crash_cube": cube, "crash_grid": ig.finish_grid(res["grid"]),
            "event_building": eb, "panel_zip_year": panel, "panel_zip_year_full": ig.panel_full(panel, cube)}
    arts = {k: ig.enforce_schema(k, v, log) for k, v in arts.items()}
    meta = build_meta(res, grid, ev, cidx, h, len(raw_h), dates, log, max_rows)
    write_all(out, arts, meta)
    if with_rq:
        run_rq(out)
    say("done")
    return meta


def main() -> None:
    """Parse CLI arguments and run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", action="store_true", help="use synthetic data in tests/fixtures (writes to processed_fixture/)")
    ap.add_argument("--max-rows", type=int, default=None, help="limit crash rows (dev/test)")
    a = ap.parse_args()
    run(a.max_rows)


if __name__ == "__main__":
    main()
