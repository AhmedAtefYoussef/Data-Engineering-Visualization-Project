"""Integration: BallTree exposure (A3), cube/grid, ZIP panel (A8) and site-level event study (A4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import BallTree

import config
from pipeline.clean_housing import log_rule
from pipeline.clean_crashes import FACTOR_GROUPS, HOUR_BANDS, SEVERITIES

EARTH_M = 6_371_000.0
NO_COMP = 32_000                      # sentinel comp_year for buildings without a completion year
YEARS = np.arange(config.Y_START, config.Y_END + 1)
NY, W = len(YEARS), config.EVENT_WINDOW
BORO_CATS = [*config.BOROUGHS, "Unknown"]
CUBE_KEYS = ["zip", "borough", "year", "hour_band", "factor_group", "severity", "exposure", "ped_cyc"]
CUBE_MEAS = ["crashes", "injured", "killed", "ped_injured", "cyc_injured"]
GRID_KEYS = ["ci", "cj", "year", "borough", "severity"]
GRID_MEAS = ["crashes", "injured", "ped_cyc_crashes"]
REDUNDANT = ["latitude_internal", "longitude_internal", "bbl", "bin", "community_board", "council_district",
             "census_tract", "neighborhood_tabulation_area", "house_number", "street_name",
             "extended_affordability_status", "prevailing_wage_status", "reporting_construction_type",
             "studio_units", "_1_br_units", "_2_br_units", "_3_br_units", "_4_br_units", "_5_br_units",
             "_6_br_units", "unknown_br_units", "counted_rental_units", "counted_homeownership_units",
             "postcode", "project_start_date", "project_completion_date", "building_completion_date",
             "completion_date", "reporting_construction_type"]
INCOME = ["extremely_low_income_units", "very_low_income_units", "low_income_units",
          "moderate_income_units", "middle_income_units", "other_income_units"]

SCHEMA: dict[str, dict[str, str]] = {
    "housing_clean": {
        "building_id": "int64", "project_id": "Int64", "project_name": "object", "borough": "category",
        "zip": "object", "latitude": "float64", "longitude": "float64", "comp_year": "Int16",
        "construction_type": "category", "total_units": "int32", "all_counted_units": "int32",
        **{col: "int32" for col in INCOME}, "income_low_share": "float64", "size_bin": "category",
        "is_large_outlier": "bool", "completed": "bool", "pre3_crashes": "Int32", "post3_crashes": "Int32",
        "pre3_adj": "float64", "post3_adj": "float64", "event_eligible": "bool",
        "crashes_250m_total": "Int32", "injuries_250m_total": "Int32", "date_inconsistent": "bool"},
    "crash_cube": {"zip": "category", "borough": "category", "year": "int16", "hour_band": "category",
                   "factor_group": "category", "severity": "category", "exposure": "int8", "ped_cyc": "int8",
                   **{col: "int32" for col in CUBE_MEAS}},
    "crash_grid": {"cell_lat": "float64", "cell_lon": "float64", "year": "int16", "borough": "category",
                   "severity": "category", **{col: "int32" for col in GRID_MEAS}},
    "event_building": {"building_id": "int64", "site_id": "int32", "group": "category", "cohort": "int16",
                       "contaminated": "bool", "comp_year": "int16", "rel_year": "int8",
                       "construction_type": "category", "borough": "category", "size_bin": "category",
                       "total_units": "int32", "crashes": "int32", "adj": "float64",
                       "ped_cyc_crashes": "int32", "injured": "int32"},
    "panel_zip_year": {"zip": "object", "borough": "category", "year": "int16", "units_completed": "int32",
                       "counted_units": "int32", "eli_vli_units": "int32", "new_con_units": "int32",
                       "preservation_units": "int32", "other_units": "int32", "buildings_completed": "int32",
                       "cum_units": "int32"},
}
SCHEMA["panel_zip_year_full"] = {**SCHEMA["panel_zip_year"], "crashes": "int32", "injured": "int32",
                                 "killed": "int32", "ped_cyc_crashes": "int32"}


# ---------------------------------------------------------------- spatial primitives
def tree_rows(housing: pd.DataFrame) -> pd.DataFrame:
    """Housing rows with coordinates, in the exact order of the BallTree's points."""
    return housing[housing["latitude"].notna() & housing["longitude"].notna()].reset_index(drop=True)


def build_tree(housing: pd.DataFrame) -> BallTree:
    """Haversine BallTree over located buildings (radians); row i == tree_rows(housing).iloc[i]."""
    rows = tree_rows(housing)
    if rows.empty:
        raise ValueError("no housing buildings with valid coordinates")
    return BallTree(np.radians(rows[["latitude", "longitude"]].to_numpy(float)), metric="haversine")


def neighbour_pairs(lat: np.ndarray, lon: np.ndarray, tree: BallTree,
                    r_m: float = config.BUFFER_M) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (owner, building) pairs within r_m metres; owner indexes the query arrays (sorted)."""
    if len(lat) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    ind = tree.query_radius(np.radians(np.column_stack([lat, lon])), r_m / EARTH_M)
    lens = np.fromiter(map(len, ind), dtype=np.int64, count=len(ind))
    bldg = np.concatenate(ind) if lens.sum() else np.empty(0, np.int64)
    return np.repeat(np.arange(len(ind)), lens), bldg.astype(np.int64)


def flag_near(df_lat, df_lon, tree: BallTree, r_m: float = config.BUFFER_M) -> np.ndarray:
    """Boolean: any housing building within r_m metres (False for missing coordinates)."""
    lat, lon = np.asarray(df_lat, float), np.asarray(df_lon, float)
    ok = ~(np.isnan(lat) | np.isnan(lon))
    out = np.zeros(len(lat), bool)
    if ok.any():
        points = np.radians(np.column_stack([lat[ok], lon[ok]]))
        out[ok] = tree.query_radius(points, r_m / EARTH_M, count_only=True) > 0
    return out


def exposure(owner: np.ndarray, bldg: np.ndarray, n: int, crash_year: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """A3 exposure for n located crashes: 0 none, 1 all neighbours complete later, 2 some complete <= year."""
    first = np.full(n, NO_COMP, dtype=np.int32)
    if len(owner):
        # query_radius returns a ragged array (a different number of neighbours per crash). `neighbour_pairs` flattens it into
        # (owner, bldg) pairs sorted by owner, so one np.minimum.reduceat over each owner's run gives the earliest completion
        # year per crash without a Python loop over millions of crashes.
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        first[owner[starts]] = np.minimum.reduceat(comp[bldg], starts)
    return np.where(first <= crash_year, 2, np.where(first < NO_COMP, 1, 0)).astype(np.int8)


def comp_array(rows: pd.DataFrame) -> np.ndarray:
    """comp_year per tree point as int32 with NO_COMP for non-completed buildings."""
    return rows["comp_year"].astype("float64").fillna(NO_COMP).to_numpy().astype(np.int32)


def process_chunk(crashes: pd.DataFrame, tree: BallTree, comp: np.ndarray, nb: int) -> tuple[np.ndarray, np.ndarray]:
    """Exposure per crash (-1 without coordinates) and (3, nb, NY) per-building-year counts for a chunk."""
    has = crashes["lat"].notna().to_numpy()
    yr = crashes["year"].to_numpy()[has]
    owner, bldg = neighbour_pairs(crashes["lat"].to_numpy()[has], crashes["lon"].to_numpy()[has], tree)
    expo = np.full(len(crashes), -1, dtype=np.int8)
    expo[has] = exposure(owner, bldg, int(has.sum()), yr, comp)
    counts = building_year_counts(owner, bldg, yr, crashes["injured"].to_numpy()[has], crashes["ped_cyc"].to_numpy()[has], nb)
    return expo, counts


# ---------------------------------------------------------------- aggregation
def _cat(values, cats: list[str]) -> pd.Categorical:
    """Fixed-category column (fails loudly rather than silently dropping unknown labels)."""
    cat = pd.Categorical(values, categories=cats)
    if (cat.codes < 0).any():
        raise ValueError("label outside fixed categories")
    return cat


def cube_part(crashes: pd.DataFrame, expo: np.ndarray, zip_cats: list[str] | None = None) -> pd.DataFrame:
    """Aggregate one cleaned crash chunk to the cube keys (A3 exposure, UNLOCATED zip)."""
    zip_col = crashes["zip"].astype(object).where(crashes["zip"].notna(), "UNLOCATED")
    frame = pd.DataFrame({
        "zip": pd.Categorical(zip_col, categories=zip_cats) if zip_cats else pd.Categorical(zip_col),
        "borough": _cat(crashes["borough"].astype(object).where(crashes["borough"].notna(), "Unknown"), BORO_CATS),
        "year": crashes["year"].to_numpy(), "hour_band": _cat(crashes["hour_band"], HOUR_BANDS),
        "factor_group": _cat(crashes["factor_group"], FACTOR_GROUPS), "severity": _cat(crashes["severity"], SEVERITIES),
        "exposure": expo, "ped_cyc": crashes["ped_cyc"].to_numpy(), "crashes": 1,
        "injured": crashes["injured"].to_numpy(), "killed": crashes["killed"].to_numpy(),
        "ped_injured": crashes["ped_injured"].to_numpy(), "cyc_injured": crashes["cyc_injured"].to_numpy()})
    if zip_cats and (frame["zip"].cat.codes < 0).any():
        raise ValueError("zip outside the pass-1 ZIP set")
    return frame.groupby(CUBE_KEYS, observed=True, sort=False)[CUBE_MEAS].sum().reset_index()


def grid_part(crashes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate located crashes of one chunk to 0.005-degree map cells (A9), as integer cell ids."""
    crashes = crashes[crashes["lat"].notna()]
    frame = pd.DataFrame({
        "ci": np.floor(crashes["lat"].to_numpy() / config.GRID_DEG).astype(np.int32),
        "cj": np.floor(crashes["lon"].to_numpy() / config.GRID_DEG).astype(np.int32),
        "year": crashes["year"].to_numpy(),
        "borough": _cat(crashes["borough"].astype(object).where(crashes["borough"].notna(), "Unknown"), BORO_CATS),
        "severity": _cat(crashes["severity"], SEVERITIES), "crashes": 1,
        "injured": crashes["injured"].to_numpy(), "ped_cyc_crashes": crashes["ped_cyc"].to_numpy()})
    return frame.groupby(GRID_KEYS, observed=True, sort=False)[GRID_MEAS].sum().reset_index()


def add_part(acc: pd.DataFrame | None, part: pd.DataFrame, keys: list[str], meas: list[str]) -> pd.DataFrame:
    """Incrementally merge a chunk aggregate into the running one (concat + groupby-sum)."""
    if acc is None:
        return part
    return pd.concat([acc, part]).groupby(keys, observed=True, sort=False)[meas].sum().reset_index()


def finish_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """Convert integer cell ids to cell-centre coordinates."""
    out = grid.copy()
    out["cell_lat"] = np.round((out.pop("ci") + 0.5) * config.GRID_DEG, 4)
    out["cell_lon"] = np.round((out.pop("cj") + 0.5) * config.GRID_DEG, 4)
    return out


def city_index(cube: pd.DataFrame) -> pd.Series:
    """A4/A6: located-crash count per year divided by its mean over Y_START..Y_END."""
    per_year = cube.loc[cube["exposure"] >= 0].groupby("year")["crashes"].sum().reindex(YEARS, fill_value=0)
    # Dividing by the mean makes the index average 1, so adj = crashes / index is in "typical-year" crashes: it removes citywide
    # swings (COVID, reporting changes) from site-level counts.
    return (per_year / per_year.mean()).astype("float64").replace(0.0, np.nan)    # 0 -> NaN: no division by zero


def building_year_counts(owner: np.ndarray, bldg: np.ndarray, year: np.ndarray, injured: np.ndarray,
                         ped_cyc: np.ndarray, nb: int) -> np.ndarray:
    """(3, nb, NY) crashes / injured / ped-cyc crashes within 250 m of each building by year."""
    # One flat bin per (building, year) pair, so a single bincount per measure replaces a Python loop over buildings.
    flat = bldg * NY + (year[owner] - config.Y_START)
    tally = lambda weights: np.bincount(flat, weights=weights, minlength=nb * NY)
    ones = np.ones(len(flat))
    return np.stack([tally(ones), tally(injured[owner].astype(float)), tally(ped_cyc[owner].astype(float))]).reshape(3, nb, NY)


# ---------------------------------------------------------------- event study (A4)
def event_sites(rows: pd.DataFrame, tree: BallTree) -> dict:
    """Treated/control buildings grouped into sites (connected comps within 250 m, same comp_year)."""
    cy = rows["comp_year"].astype("float64").to_numpy()
    treated = (cy - W >= config.Y_START) & (cy + W <= config.Y_END)       # NaN -> False
    cohorts = np.unique(cy[treated]).astype(int)
    control = (cy >= cohorts.min() + W + 1) if len(cohorts) else np.zeros(len(cy), bool)
    nodes = np.flatnonzero(treated | control)
    if len(nodes) == 0:
        return dict(nodes=nodes, site=np.empty(0, np.int32), treated=np.empty(0, bool),
                    contaminated=np.empty(0, bool), cohorts=cohorts)
    pos = np.full(len(rows), -1)
    pos[nodes] = np.arange(len(nodes))
    owner, nb = neighbour_pairs(rows["latitude"].to_numpy()[nodes], rows["longitude"].to_numpy()[nodes], tree)
    owner_year = cy[nodes][owner]
    same = (pos[nb] >= 0) & (cy[nb] == owner_year)
    # Sites are connected components of "within 250 m and same completion year" links, so a chain of adjacent buildings is one site.
    adjacency = coo_matrix((np.ones(same.sum()), (owner[same], pos[nb[same]])), shape=(len(nodes),) * 2)
    _, site = connected_components(adjacency, directed=False)
    year_gap = np.abs(cy[nb] - owner_year)
    # contaminated: another completion within 250 m whose year is within +-3 of the site's year.
    bad = np.bincount(owner[(year_gap >= 1) & (year_gap <= W)], minlength=len(nodes)) > 0
    contam = pd.Series(bad).groupby(site).transform("max").to_numpy()
    return dict(nodes=nodes, site=site.astype(np.int32), treated=treated[nodes], contaminated=contam,
                cohorts=cohorts)


def event_building(rows: pd.DataFrame, ev: dict, counts: np.ndarray, cidx: pd.Series) -> pd.DataFrame:
    """One row per building x cohort x rel_year (-3..3); crashes counted within 250 m, adj = / city_index."""
    nodes, cohorts = ev["nodes"], ev["cohorts"]
    cy = rows["comp_year"].astype("float64").to_numpy()[nodes]
    # Cross every node with every cohort, keep valid pairs (treated: completes in the cohort year; control: >= cohort + 4),
    # then expand each pair to its 7 relative years. Controls repeat once per cohort they can serve, on that cohort's calendar window.
    node_pos = np.repeat(np.arange(len(nodes)), len(cohorts))
    cohort_col = np.tile(cohorts, len(nodes))
    ok = np.where(ev["treated"][node_pos], cy[node_pos] == cohort_col, cy[node_pos] >= cohort_col + W + 1)
    node_pos, cohort_col = np.repeat(node_pos[ok], 2 * W + 1), np.repeat(cohort_col[ok], 2 * W + 1)
    rel = np.tile(np.arange(-W, W + 1), ok.sum())
    bldg_idx, year_idx = nodes[node_pos], cohort_col + rel - config.Y_START
    cnt = counts[0][bldg_idx, year_idx]
    take = lambda col: rows[col].iloc[bldg_idx].reset_index(drop=True)
    boro = take("borough")
    out = pd.DataFrame({
        "building_id": rows["building_id"].to_numpy()[bldg_idx], "site_id": ev["site"][node_pos],
        "group": pd.Categorical(np.where(ev["treated"][node_pos], "treated", "control"), categories=["treated", "control"]),
        "cohort": cohort_col, "contaminated": ev["contaminated"][node_pos], "comp_year": cy[node_pos].astype(int), "rel_year": rel,
        "construction_type": take("construction_type"),
        "borough": boro.astype(object).where(boro.notna(), "Unknown"), "size_bin": take("size_bin"),
        "total_units": rows["total_units"].to_numpy()[bldg_idx], "crashes": cnt, "adj": cnt / cidx.to_numpy()[year_idx],
        "ped_cyc_crashes": counts[2][bldg_idx, year_idx], "injured": counts[1][bldg_idx, year_idx]})
    return out


# ---------------------------------------------------------------- housing finalisation
def prepare_housing(h: pd.DataFrame, zip_borough: dict, log: list) -> pd.DataFrame:
    """Post-integration housing step: borough from ZIP mode, drop redundant columns (logged)."""
    miss = h["borough"].isna() & h["zip"].notna()
    # zip -> modal borough learned from crash rows (grid pass) and from housing itself.
    own = h.dropna(subset=["borough", "zip"]).groupby("zip")["borough"].agg(lambda values: values.mode().iat[0])
    fill = h["zip"].map({**zip_borough, **own.to_dict()}).astype(object)
    filled = h["borough"].astype(object).where(h["borough"].notna(), fill)
    h = h.assign(borough=pd.Categorical(filled, categories=config.BOROUGHS))
    log_rule(log, "post-integration", "housing", "missing borough filled from ZIP->modal borough", len(h), len(h),
             "Borough is a key filter; a valid ZIP identifies it.", int(miss.sum() - (miss & h["borough"].isna()).sum()))
    drop = [col for col in dict.fromkeys(REDUNDANT) if col in h]
    log_rule(log, "post-integration", "housing", f"drop {len(drop)} redundant/raw columns", len(h), len(h),
             "bbl, bin, *_internal, community_board, bedroom splits etc. are not used downstream "
             "(latitude/longitude already carry the *_internal fallback).", len(drop))
    return h.drop(columns=drop)


def attach_counts(h: pd.DataFrame, rows: pd.DataFrame, ev: dict, counts: np.ndarray, cidx: pd.Series,
                  log: list) -> pd.DataFrame:
    """Add crashes_250m_total, injuries_250m_total and pre3/post3 stats (eligible treated buildings)."""
    n = len(rows)
    per_building = pd.DataFrame({"building_id": rows["building_id"].to_numpy(),
                                 "crashes_250m_total": counts[0].sum(1), "injuries_250m_total": counts[1].sum(1)})
    elig = np.zeros(n, bool)
    elig[ev["nodes"][ev["treated"]]] = True
    comp_year = rows["comp_year"].astype("float64").fillna(0).to_numpy().astype(int)
    row_idx = np.arange(n)[:, None]
    city_idx = cidx.to_numpy()
    for tag, offs in (("pre3", np.arange(-W, 0)), ("post3", np.arange(1, W + 1))):
        # Fancy-index each building's own 3-year window at once; columns are clipped so ineligible rows index safely, then masked to NaN.
        cols = np.clip(comp_year[:, None] + offs - config.Y_START, 0, NY - 1)
        per_building[f"{tag}_crashes"] = np.where(elig, counts[0][row_idx, cols].sum(1), np.nan)
        per_building[f"{tag}_adj"] = np.where(elig, (counts[0][row_idx, cols] / city_idx[cols]).sum(1), np.nan)
    per_building["event_eligible"] = elig
    for col in ["pre3_crashes", "post3_crashes", "crashes_250m_total", "injuries_250m_total"]:
        per_building[col] = per_building[col].round().astype("Int32")
    out = h.merge(per_building, on="building_id", how="left")
    out["event_eligible"] = out["event_eligible"].fillna(False).astype(bool)
    log_rule(log, "post-integration", "housing", "buildings without coordinates: crash stats stay NaN", len(out),
             len(out), "No location -> cannot be counted within 250 m; NaN (not 0) avoids a false 'no crashes'.",
             int(out["crashes_250m_total"].isna().sum()))
    return out


# ---------------------------------------------------------------- ZIP panel (A8)
def _zip_boro(cube: pd.DataFrame, h: pd.DataFrame) -> pd.Series:
    """Modal borough per ZIP: crash side first, housing borough as fallback, else 'Unknown'."""
    known = cube.loc[(cube["borough"] != "Unknown") & (cube["zip"] != "UNLOCATED")]
    crash_boro = known.groupby(["zip", "borough"], observed=True)["crashes"].sum().reset_index()
    crash_boro = crash_boro.assign(zip=crash_boro["zip"].astype(str)).sort_values("crashes", ascending=False)
    crash_boro = crash_boro.drop_duplicates("zip").set_index("zip")["borough"]
    housing_boro = h.dropna(subset=["zip", "borough"]).groupby("zip")["borough"].agg(lambda values: values.mode().iat[0])
    return pd.concat([crash_boro.astype(object), housing_boro.astype(object)]).groupby(level=0).first()


def _housing_by_zip_year(h: pd.DataFrame) -> pd.DataFrame:
    """Completed buildings aggregated to (zip, comp_year) with the A8 unit measures."""
    h = h[h["completed"] & h["zip"].notna()]
    units = h["total_units"]
    frame = pd.DataFrame({"zip": h["zip"], "year": h["comp_year"].astype(int), "units_completed": units,
                      "counted_units": h["all_counted_units"],
                      "eli_vli_units": h["extremely_low_income_units"] + h["very_low_income_units"],
                      "new_con_units": units.where(h["construction_type"] == "New Construction", 0),
                      "preservation_units": units.where(h["construction_type"] == "Preservation", 0),
                      "other_units": units.where(h["construction_type"] == "Other", 0), "buildings_completed": 1})
    return frame.groupby(["zip", "year"]).sum()


def build_panel(housing: pd.DataFrame, cube: pd.DataFrame, log: list | None = None) -> pd.DataFrame:
    """ZIP x year housing panel for ZIPs with >= MIN_ZIP_CRASHES located crashes (A8)."""
    log = [] if log is None else log
    located = cube.loc[cube["zip"] != "UNLOCATED"].groupby("zip", observed=True)["crashes"].sum()
    zips = [str(z) for z in located.index[located >= config.MIN_ZIP_CRASHES]]
    agg = _housing_by_zip_year(housing)
    union = set(agg.index.get_level_values(0)) | set(located.index.astype(str))
    log_rule(log, "post-integration", "panel", "ZIP universe restricted to >= MIN_ZIP_CRASHES located crashes",
             len(union), len(zips), "Small-N and housing-only ZIPs give unstable rates; UNLOCATED excluded.")
    idx = pd.MultiIndex.from_product([zips, YEARS], names=["zip", "year"])
    full = agg.reindex(idx)
    log_rule(log, "post-integration", "panel", "housing metrics NaN after left join -> 0", len(full), len(full),
             "Structural zero: the ZIP has crash data but no completions that year.",
             int(full["units_completed"].isna().sum()))
    out = full.fillna(0).reset_index()
    out["cum_units"] = _cumulative(agg, zips)
    boro = _zip_boro(cube, housing)
    out["borough"] = out["zip"].map(boro).fillna("Unknown")
    return out


def _cumulative(agg: pd.DataFrame, zips: list[str]) -> np.ndarray:
    """cum_units (zips-major, years-minor): completions with year >= HOUSING_MIN_YEAR (A8, 2014)
    summed up to and incl. the year; earlier completions are excluded."""
    if agg.empty:
        return np.zeros(len(zips) * NY)
    wide = agg["units_completed"].unstack(fill_value=0)
    wide = wide.loc[:, wide.columns >= config.HOUSING_MIN_YEAR]
    cols = range(config.HOUSING_MIN_YEAR, config.Y_END + 1)
    cum = wide.reindex(columns=cols, fill_value=0).cumsum(axis=1)
    return cum.reindex(index=zips, columns=YEARS, fill_value=0).to_numpy().ravel()


def panel_full(panel: pd.DataFrame, cube: pd.DataFrame) -> pd.DataFrame:
    """Panel plus unfiltered located-ZIP crash measures (for the notebook)."""
    located = cube.loc[cube["zip"] != "UNLOCATED"]
    located = located.assign(zip=located["zip"].astype(str), ped_cyc_crashes=located["crashes"] * located["ped_cyc"])
    by_zip_year = located.groupby(["zip", "year"])[["crashes", "injured", "killed", "ped_cyc_crashes"]].sum().reset_index()
    out = panel.merge(by_zip_year, on=["zip", "year"], how="left")
    return out.fillna({"crashes": 0, "injured": 0, "killed": 0, "ped_cyc_crashes": 0})


# ---------------------------------------------------------------- post-integration checks
def enforce_schema(name: str, df: pd.DataFrame, log: list) -> pd.DataFrame:
    """Reorder/cast to the SPEC dtype contract and assert integrity (logged)."""
    sch = SCHEMA[name]
    changed = sum(str(df[col].dtype) != dtype for col, dtype in sch.items() if dtype != "object")
    out = pd.DataFrame({col: df[col].astype(dtype) for col, dtype in sch.items()})
    for col in [col for col, dtype in sch.items() if dtype in ("int32", "Int32")]:     # all count/measure columns
        assert (out[col].fillna(0) >= 0).all(), f"{name}.{col}: negative values"
    if name.startswith("panel"):
        assert not out.duplicated(["zip", "year"]).any(), f"{name}: duplicate (zip, year)"
    log_rule(log, "post-integration", name, "dtype enforcement + assertions (unique keys, no negatives)",
             len(df), len(out), "Contract dtypes (SPEC 5/11); keys unique; counts non-negative.", changed)
    return out
