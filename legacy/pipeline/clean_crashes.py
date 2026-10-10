"""Crash cleaning (SPEC section 4 + A1/A2/A5/A13): per-chunk, memory-lean, every rule logged."""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

import config
from pipeline.clean_housing import in_bbox, log_rule

HOUR_BANDS = ["Night", "Morning", "Afternoon", "Evening", "Unknown"]
FACTOR_GROUPS = ["Distraction/Inattention", "Speed/Aggressive", "Failure to Yield/Signal",
                 "Impairment/Fatigue", "Vehicle/Road Defect", "Other", "Unspecified"]
SEVERITIES = ["Fatal", "Injury", "Property damage only"]
# Keyword rules, first match wins (order matters: "Failure to Yield" before generic "failure").
FACTOR_RULES = [
    ("Impairment/Fatigue", r"alcohol|drug|medication|fatigu|drows|asleep|illnes|conscious|disabilit|impair"),
    ("Distraction/Inattention", r"distract|inattent|cell phone|electronic|texting|headphone|eating|navigation"),
    ("Speed/Aggressive", r"speed|aggress|road rage|following too|passing|lane chang|lane usage|tailgat"),
    ("Failure to Yield/Signal", r"yield|disregard|turning improper|keep right|signal|right-of-way"),
    ("Vehicle/Road Defect", r"defect|failure|inadequate|slippery|glare|obstruct|debris|lane marking|"
                            r"non-working|lighting|tinted|windshield|view"),
]
COUNT_COLS = ["number_of_persons_injured", "number_of_persons_killed", "number_of_pedestrians_injured",
              "number_of_pedestrians_killed", "number_of_cyclist_injured", "number_of_cyclist_killed",
              "number_of_motorist_injured", "number_of_motorist_killed"]
INJ_CAP = 20
HOLDOUT_MOD = 20           # A5: collision_id % 20 == 0 is held out of the grid and used to score it
LOW_SUPPORT = 5
KEY_SHIFT = 1_000_000


class SeenIds:
    """Sorted int64 array of collision_ids already emitted (cross-chunk dedupe, A1)."""

    def __init__(self) -> None:
        self.ids = np.empty(0, dtype=np.int64)

    def fresh(self, ids: np.ndarray) -> np.ndarray:
        """Boolean mask of ids not seen before nor repeated earlier in this chunk; records them."""
        keep = ~pd.Series(ids).duplicated().to_numpy() & ~np.isin(ids, self.ids)
        self.ids = np.union1d(self.ids, ids[keep])
        return keep


def cell_key(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Integer id of the 0.002-degree ZIP-imputation cell containing each coordinate."""
    ci = np.round(np.asarray(lat) / config.ZIP_GRID_DEG).astype(np.int64)
    cj = np.round(np.asarray(lon) / config.ZIP_GRID_DEG).astype(np.int64)
    return ci * KEY_SHIFT + (cj + KEY_SHIFT // 2)


def parse_hour(s: pd.Series) -> pd.Series:
    """Hour 0-23 from text 'H:MM' (or numeric hours); invalid values become NaN."""
    if pd.api.types.is_numeric_dtype(s):
        h = s.astype("float64")
        return h.where(h.between(0, 23))
    p = s.astype("string").str.strip().str.extract(r"^(\d{1,2}):(\d{2})(?::\d{2})?$").astype("float64")
    return p[0].where(p[0].between(0, 23) & p[1].between(0, 59))


def hour_band(s: pd.Series) -> pd.Series:
    """Night 0-5, Morning 6-11, Afternoon 12-17, Evening 18-23, Unknown (A13) as a category."""
    h = parse_hour(s)
    band = pd.cut(h, [-1, 5, 11, 17, 23], labels=HOUR_BANDS[:4]).astype(object).fillna("Unknown")
    return pd.Series(pd.Categorical(band, categories=HOUR_BANDS), index=s.index)


def factor_group(s: pd.Series) -> pd.Series:
    """Map contributing_factor_vehicle_1 strings to 7 groups via keyword rules (unknown -> Other)."""
    u = pd.Series(s.dropna().unique(), dtype="object")
    if u.empty:
        return pd.Series(pd.Categorical(["Unspecified"] * len(s), categories=FACTOR_GROUPS), index=s.index)
    low = u.astype(str).str.lower().str.strip()
    conds = [low.str.contains(rx, regex=True) for _, rx in FACTOR_RULES]
    grp = np.select(conds, [g for g, _ in FACTOR_RULES], "Other")
    grp = np.where(low.isin(["unspecified", ""]), "Unspecified", grp)  # exact label, not a keyword
    out = s.map(dict(zip(u, grp))).fillna("Unspecified")
    return pd.Series(pd.Categorical(out, categories=FACTOR_GROUPS), index=s.index)


def _zip_geo(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Validated ZIP (regex A2) and bbox-checked lat/lon; shared by the grid pass and cleaning."""
    z = df["zip_code"].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    z = z.str.extract(r"^(\d{5})(?:-\d{4})?$")[0]                      # ZIP+4 -> first 5 digits
    z = z.where(z.str.match(config.NYC_ZIP_RE).fillna(False).astype(bool))
    z = z.astype(object).where(z.notna(), np.nan)  # plain object + NaN (not pd.NA) for downstream ops
    lat, lon, _ = _fix_swapped(df)
    ok = in_bbox(lat, lon)
    return z, lat.where(ok), lon.where(ok)


def _fix_swapped(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Numeric lat/lon with swapped pairs (valid only when exchanged) put back; also the swapped mask."""
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    sw = ~in_bbox(lat, lon) & in_bbox(lon, lat)
    return lat.where(~sw, lon), lon.where(~sw, lat), sw


def _modal(counts: pd.Series, by: str) -> pd.DataFrame:
    """Per-group modal label from a (group, label) -> n Series; also the group total."""
    df = counts.rename("n").reset_index()
    tot = df.groupby(by)["n"].transform("sum")
    df["support"] = tot
    return df.sort_values(["n"], ascending=False, kind="stable").drop_duplicates(by).sort_values(by)


def build_zip_grid(chunk_factory: Callable[[], Iterator[pd.DataFrame]], exclude_holdout: bool = True) -> dict:
    """Pass 1: modal ZIP per 0.002-degree cell from known-ZIP rows (holdout excluded), plus accuracy.

    Returns dict(key, zip, support arrays sorted by key; zip_borough; zips; holdout_* stats).
    """
    cz, bz, hold_k, hold_z, zips = None, None, [], [], set()
    for raw in chunk_factory():
        z, lat, lon = _zip_geo(raw)
        zips.update(z.dropna().unique())
        cid = pd.to_numeric(raw["collision_id"], errors="coerce")
        b = raw["borough"].astype("string").str.strip().str.title()
        bpart = pd.DataFrame({"zip": z, "b": b})[z.notna() & b.isin(config.BOROUGHS)]
        bz = _add(bz, bpart.groupby(["zip", "b"]).size())
        k = (z.notna() & lat.notna()).to_numpy()
        key, zz = cell_key(lat[k], lon[k]), z[k].to_numpy()
        hold = exclude_holdout & ((cid[k].fillna(1).astype("int64") % HOLDOUT_MOD) == 0).to_numpy()
        hold_k.append(key[hold]); hold_z.append(zz[hold])
        cz = _add(cz, pd.DataFrame({"key": key[~hold], "zip": zz[~hold]}).groupby(["key", "zip"]).size())
    return _finish_grid(cz, bz, hold_k, hold_z, zips, exclude_holdout)


def _add(acc: pd.Series | None, part: pd.Series) -> pd.Series:
    """Incrementally merge grouped counts across chunks."""
    return part if acc is None else pd.concat([acc, part]).groupby(level=list(range(part.index.nlevels))).sum()


def _finish_grid(cz, bz, hold_k, hold_z, zips, exclude_holdout) -> dict:
    """Collapse counts into sorted arrays and score the grid on the held-out rows."""
    if cz is None:
        cz = pd.Series([], index=pd.MultiIndex.from_arrays([[], []], names=["key", "zip"]), dtype="int64")
    g = _modal(cz, "key")
    grid = dict(key=g["key"].to_numpy(np.int64), zip=g["zip"].to_numpy(object),
                support=g["support"].to_numpy(np.int32), zips=sorted(zips),
                zip_borough={} if bz is None else {r.zip: r.b for r in _modal(bz, "zip").itertuples()},
                holdout_acc=None, holdout_acc_low_support=None, holdout_n=0, holdout_coverage=None)
    if exclude_holdout and hold_k:
        hk, hz = np.concatenate(hold_k), np.concatenate(hold_z)
        pos = np.minimum(np.searchsorted(grid["key"], hk), max(len(grid["key"]) - 1, 0))
        found = (grid["key"][pos] == hk) if len(grid["key"]) else np.zeros(len(hk), bool)
        right = found & (grid["zip"][pos] == hz) if len(grid["key"]) else found
        low = found & (grid["support"][pos] < LOW_SUPPORT) if len(grid["key"]) else found
        grid.update(holdout_n=int(len(hk)), holdout_coverage=float(found.mean()) if len(hk) else None,
                    holdout_acc=float(right[found].mean()) if found.any() else None,
                    holdout_acc_low_support=float(right[low].mean()) if low.any() else None)
    return grid


def _impute(z: pd.Series, lat: pd.Series, lon: pd.Series, grid: dict) -> tuple[pd.Series, pd.Series]:
    """Fill missing ZIPs from the cell-modal ZIP; returns (zip, imputed flag)."""
    need = (z.isna() & lat.notna()).to_numpy()
    out = z.to_numpy(object).copy()
    key = cell_key(lat.to_numpy()[need], lon.to_numpy()[need])
    pos = np.minimum(np.searchsorted(grid["key"], key), max(len(grid["key"]) - 1, 0))
    hit = (grid["key"][pos] == key) if len(grid["key"]) else np.zeros(len(key), bool)
    idx = np.flatnonzero(need)[hit]
    out[idx] = grid["zip"][pos[hit]]
    flag = np.zeros(len(z), bool)
    flag[idx] = True
    return pd.Series(out, index=z.index), pd.Series(flag, index=z.index)


def _counts(df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Injury/kill columns -> non-negative ints; persons >= sum of categories; cap outliers at 20."""
    n = len(df)
    c = {k: pd.to_numeric(df[k], errors="coerce") for k in COUNT_COLS}
    log_rule(log, "clean", "crashes", "missing/negative injury & kill counts -> 0", n, n,
             "A blank count means none recorded; negatives are impossible.",
             sum(int((v.isna() | (v < 0)).sum()) for v in c.values()))
    c = {k: v.fillna(0).clip(lower=0) for k, v in c.items()}
    for kind in ("injured", "killed"):
        sub = sum(c[f"number_of_{g}_{kind}"] for g in ("pedestrians", "cyclist", "motorist"))
        tot = f"number_of_persons_{kind}"
        log_rule(log, "clean", "crashes", f"persons_{kind} < sum of ped/cyc/mot -> sum", n, n,
                 "Total must be at least the sum of its parts.", int((c[tot] < sub).sum()))
        c[tot] = np.maximum(c[tot], sub)
    big = c["number_of_persons_injured"] > INJ_CAP
    log_rule(log, "clean", "crashes", f"persons_injured > {INJ_CAP} capped", n, n,
             "Domain rule: >20 injured in one crash is implausible data entry; capped, row kept.", int(big.sum()))
    c["number_of_persons_injured"] = c["number_of_persons_injured"].clip(upper=INJ_CAP)
    for g in ("pedestrians", "cyclist", "motorist"):             # parts never exceed the capped total
        c[f"number_of_{g}_injured"] = np.minimum(c[f"number_of_{g}_injured"], c["number_of_persons_injured"])
    out = pd.DataFrame({"injured": c["number_of_persons_injured"], "killed": c["number_of_persons_killed"],
                        "ped_injured": c["number_of_pedestrians_injured"], "cyc_injured": c["number_of_cyclist_injured"]})
    out = out.astype("int16")
    pc = (c["number_of_pedestrians_injured"] + c["number_of_pedestrians_killed"]
          + c["number_of_cyclist_injured"] + c["number_of_cyclist_killed"]) > 0
    out["ped_cyc"] = pc.astype("int8")
    out["severity"] = pd.Categorical(np.select([out["killed"] > 0, out["injured"] > 0], SEVERITIES[:2],
                                               SEVERITIES[2]), categories=SEVERITIES)
    return out


def clean_crash_chunk(df: pd.DataFrame, log: list, zip_grid: dict | None = None,
                      seen: SeenIds | None = None) -> pd.DataFrame:
    """Clean one raw (all-str) crash chunk into a lean typed frame; row-level rules are logged."""
    n0 = len(df)
    cid = pd.to_numeric(df["collision_id"], errors="coerce")
    df = df[cid.notna()]
    cid = cid[cid.notna()].astype("int64")
    log_rule(log, "clean", "crashes", "missing collision_id dropped", n0, len(df), "Key required for dedupe.")
    keep = (seen.fresh(cid.to_numpy()) if seen is not None else ~cid.duplicated().to_numpy())
    log_rule(log, "clean", "crashes", "duplicate collision_id dropped (within and across chunks)",
             len(df), int(keep.sum()), "collision_id is unique per crash; repeats are re-exports.")
    df, cid = df[keep], cid[keep]
    d = pd.to_datetime(df["crash_date"], errors="coerce", format="ISO8601")
    inwin = d.dt.year.between(config.Y_START, config.Y_END).to_numpy()
    log_rule(log, "clean", "crashes", f"crash_date unparseable or outside {config.Y_START}-{config.Y_END} dropped",
             len(df), int(inwin.sum()), "Analysis window is full calendar years only (latest months lag).")
    df, cid, d = df[inwin], cid[inwin], d[inwin]
    return _build(df, cid, d, log, zip_grid)


def _build(df: pd.DataFrame, cid: pd.Series, d: pd.Series, log: list, grid: dict | None) -> pd.DataFrame:
    """Derive typed columns for already-filtered rows (zip, coords, borough, bands, groups, counts)."""
    n = len(df)
    z, lat, lon = _zip_geo(df)
    raw_z = df["zip_code"].astype("string").str.strip()
    log_rule(log, "clean", "crashes", "zip_code invalid (A2 regex) -> NaN", n, n, "Excludes Nassau/garbage ZIPs.",
             int((raw_z.notna() & z.isna()).sum()))
    rl = pd.to_numeric(df["latitude"], errors="coerce")
    log_rule(log, "clean", "crashes", "lat/lon swapped but valid when exchanged -> swapped back", n, n,
             "Transposed entry; the pair lies in the NYC bbox only after swapping.", int(_fix_swapped(df)[2].sum()))
    log_rule(log, "clean", "crashes", "lat/lon outside NYC bbox or (0,0) -> NaN", n, n,
             "Junk coordinates would corrupt the spatial join.", int((rl.notna() & lat.isna() & ~_fix_swapped(df)[2]).sum()))
    b = df["borough"].astype("string").str.strip().str.title()
    b = b.where(b.isin(config.BOROUGHS))
    b = b.astype(object).where(b.notna(), np.nan)
    flag = pd.Series(False, index=df.index)
    if grid is not None:
        z, flag = _impute(z, lat, lon, grid)
        bm = z.map(grid["zip_borough"]).astype(object)
        b = b.fillna(bm)
        log_rule(log, "impute", "crashes", "missing ZIP from coordinates via cell-modal ZIP (A5)", n, n,
                 "Grid built from known-ZIP rows (holdout excluded); unsupported cells stay UNLOCATED.",
                 int(flag.sum()))
    out = _counts(df, log)
    out.index = df.index
    out["collision_id"] = cid
    out["year"] = d.dt.year.astype("int16")
    out["borough"] = pd.Categorical(b, categories=config.BOROUGHS)
    out["zip"], out["zip_imputed"] = z, flag
    out["lat"], out["lon"] = lat, lon
    out["hour_band"] = hour_band(df["crash_time"])
    log_rule(log, "clean", "crashes", "crash_time invalid -> hour_band 'Unknown' (A13)", n, n,
             "Rows kept; only the time-of-day dimension is unknown.", int((out["hour_band"] == "Unknown").sum()))
    out["factor_group"] = factor_group(df["contributing_factor_vehicle_1"])
    return out.reset_index(drop=True)
