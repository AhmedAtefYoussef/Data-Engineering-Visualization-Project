"""Housing cleaning (SPEC section 4 + A2/A7): building-level, vectorised, every rule logged."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config
from pipeline.extract import HOUSING_COLS

UNIT_COLS = [c for c in HOUSING_COLS if c.endswith("_units") and c != "total_units"]
SIZE_EDGES = [0, 10, 50, 150, np.inf]
SIZE_LABELS = ["1-10", "11-50", "51-150", "151+"]
CONSTRUCTION = ["New Construction", "Preservation", "Other"]
BORO_MAP = {"manhattan": "Manhattan", "new york": "Manhattan", "bronx": "Bronx", "the bronx": "Bronx",
            "brooklyn": "Brooklyn", "bklyn": "Brooklyn", "queens": "Queens",
            "staten island": "Staten Island"}


def log_rule(log: list, step: str, dataset: str, rule: str, before: int, after: int,
             rationale: str, n_affected: int | None = None) -> None:
    """Append (or merge, for per-chunk calls) one cleaning-log entry."""
    n = before - after if n_affected is None else int(n_affected)
    for e in log:
        if (e["step"], e["dataset"], e["rule"]) == (step, dataset, rule):
            e["rows_before"] += int(before)
            e["rows_after"] += int(after)
            e["n_affected"] += n
            return
    log.append(dict(step=step, dataset=dataset, rule=rule, rows_before=int(before),
                    rows_after=int(after), rationale=rationale, n_affected=n))


def snake(name: str) -> str:
    """snake_case a column name but keep API-style leading underscores (e.g. _1_br_units)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).lower().rstrip("_")


def in_bbox(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """True where both coordinates lie inside the NYC bounding box (rejects NaN and 0,0)."""
    (la0, la1), (lo0, lo1) = config.NYC_BBOX["lat"], config.NYC_BBOX["lon"]
    return lat.between(la0, la1) & lon.between(lo0, lo1)


def _normalise(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Snake-case names, add absent columns, convert numerics, drop rows without building_id."""
    df = df.rename(columns=snake)
    if "building_id" not in df:
        raise KeyError("housing extract has no building_id column (schema drift?)")
    missing = [c for c in HOUSING_COLS if c not in df]
    df = df.reindex(columns=[*df.columns, *missing])
    # Socrata JSON/CSV omit unit keys that are 0/null for every row of a page -> 0 is the true value.
    log_rule(log, "clean", "housing", "unit columns absent/blank -> 0", len(df), len(df),
             "API omits zero/null unit values; a missing unit count means no units of that type.",
             int(df[UNIT_COLS].isna().to_numpy().sum()))
    num = [*UNIT_COLS, "total_units", "latitude", "longitude", "latitude_internal",
           "longitude_internal", "building_id", "project_id"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[UNIT_COLS] = df[UNIT_COLS].fillna(0).astype("int32")
    n = len(df)
    df = df[df["building_id"].notna()].copy()
    log_rule(log, "clean", "housing", "missing building_id dropped", n, len(df),
             "building_id is the key; rows without it cannot be deduplicated or joined.")
    df["building_id"] = df["building_id"].astype("int64")
    return df


def _dates(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Parse dates, null implausible years, derive completion date/year, flags (spec section 4)."""
    for c in ["project_start_date", "project_completion_date", "building_completion_date"]:
        d = pd.to_datetime(df[c], errors="coerce", format="ISO8601")
        df[c] = d.where(d.dt.year.between(2000, 2030))
    # building completion preferred; fall back to project completion; never overwrite the original.
    df["completion_date"] = df["building_completion_date"].fillna(df["project_completion_date"])
    df["completed"] = df["completion_date"].notna()
    df["comp_year"] = df["completion_date"].dt.year.astype("Int16")
    df["date_inconsistent"] = df["completion_date"] < df["project_start_date"]
    log_rule(log, "clean", "housing", "completion date: building -> project fallback", len(df), len(df),
             "Buildings lacking building_completion_date use project_completion_date; rows with neither "
             "stay in housing_clean with completed=False and are excluded from the panel.",
             int((df["building_completion_date"].isna() & df["project_completion_date"].notna()).sum()))
    log_rule(log, "clean", "housing", "completion before project start flagged", len(df), len(df),
             "Kept (date_inconsistent=True): the building date is still the best completion evidence.",
             int(df["date_inconsistent"].sum()))
    return df


def _dedupe(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Keep the latest-completion record per building_id (stable sort, NaT first)."""
    n = len(df)
    df = df.sort_values("completion_date", na_position="first", kind="stable")
    df = df.drop_duplicates("building_id", keep="last").sort_values("building_id").reset_index(drop=True)
    log_rule(log, "clean", "housing", "duplicate building_id -> keep last by completion date", n, len(df),
             "One row per building; later re-reports supersede earlier ones.")
    return df


def _geo(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Valid NYC ZIP (A2), canonical borough, coordinates inside the NYC bbox."""
    z = df["postcode"].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True).str.zfill(5)
    ok = z.str.match(config.NYC_ZIP_RE).fillna(False).astype(bool)
    log_rule(log, "clean", "housing", "postcode -> 5-char str, valid NYC ZIP else NaN", len(df), len(df),
             "postcode arrives as a number (lost leading zeros); non-NYC/garbage ZIPs (e.g. 99999) are nulled.",
             int((z.notna() & ~ok).sum()))
    zv = z.where(ok)
    df["zip"] = zv.astype(object).where(zv.notna(), np.nan)
    b = df["borough"].astype("string").str.strip().str.lower().map(BORO_MAP)
    df["borough"] = pd.Categorical(b, categories=config.BOROUGHS)
    ok1 = in_bbox(df["latitude"], df["longitude"])
    ok2 = in_bbox(df["latitude_internal"], df["longitude_internal"]) & ~ok1
    df["latitude"] = np.where(ok1, df["latitude"], np.where(ok2, df["latitude_internal"], np.nan))
    df["longitude"] = np.where(ok1, df["longitude"], np.where(ok2, df["longitude_internal"], np.nan))
    log_rule(log, "clean", "housing", "coordinates outside NYC bbox -> NaN (fallback to *_internal)", len(df),
             len(df), "Rejects (0,0) junk; the lot-internal point recovers rows whose public point is blank.",
             int((~ok1).sum()))
    return df


def _units(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Repair total_units, flag IQR outliers, derive size_bin, construction type, income share."""
    tu, ac = df["total_units"], df["all_counted_units"]
    bad = tu.isna() | (tu < ac)
    df["total_units"] = tu.where(~bad, ac).astype("int32")
    log_rule(log, "clean", "housing", "total_units < all_counted_units (or missing) -> all_counted_units",
             len(df), len(df), "Total cannot be smaller than the counted subset.", int(bad.sum()))
    q1, q3 = df["total_units"].quantile([0.25, 0.75])
    # Tukey fence: values above Q3 + 1.5*IQR are flagged, never dropped (big projects are real).
    df["is_large_outlier"] = df["total_units"] > q3 + 1.5 * (q3 - q1)
    log_rule(log, "clean", "housing", "total_units IQR(1.5) outlier flag", len(df), len(df),
             "Flag only; any winsorising (99th pct) happens inside the statistics that need it.",
             int(df["is_large_outlier"].sum()))
    df["size_bin"] = pd.cut(df["total_units"], SIZE_EDGES, labels=SIZE_LABELS, include_lowest=True)
    t = df["reporting_construction_type"].astype("string").str.lower()
    ct = np.select([t.str.contains("new", na=False), t.str.contains("preserv", na=False)],
                   CONSTRUCTION[:2], CONSTRUCTION[2])
    df["construction_type"] = pd.Categorical(ct, categories=CONSTRUCTION)
    low = df["extremely_low_income_units"] + df["very_low_income_units"]
    df["income_low_share"] = (low / ac.where(ac > 0)).astype("float64")
    log_rule(log, "clean", "housing", "derived size_bin / construction_type / income_low_share",
             len(df), len(df), "Fixed size bins (A7); construction canon {New, Preservation, Other}.", 0)
    return df


def clean_housing(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Clean raw (all-str) housing rows into one row per building; extra columns kept for integration."""
    df = _normalise(df, log)
    df = _dates(df, log)
    df = _dedupe(df, log)
    df = _geo(df, log)
    return _units(df, log).reset_index(drop=True)
