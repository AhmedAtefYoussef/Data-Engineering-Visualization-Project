"""Pipeline tests: hand-built frames per cleaning rule + an end-to-end fixture run (20k crashes)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
import requests

import config
from pipeline import extract, integrate as ig
from pipeline.clean_crashes import (SeenIds, build_zip_grid, clean_crash_chunk, factor_group, hour_band)
from pipeline.clean_housing import clean_housing, log_rule
from pipeline.run import run
from pipeline.clean_housing import SIZE_LABELS

SIZE_FIRST = SIZE_LABELS[0]

LOG_KEYS = {"step", "dataset", "rule", "rows_before", "rows_after", "rationale"}


# ------------------------------------------------------------------ helpers
def raw_h(rows: list[dict]) -> pd.DataFrame:
    """Raw-style (all str) housing frame from sparse dicts."""
    return pd.DataFrame(rows).astype("string").astype(object)


def raw_c(rows: list[dict]) -> pd.DataFrame:
    """Raw-style crash chunk with every CRASH_COLS column present (missing -> NaN)."""
    return pd.DataFrame(rows).reindex(columns=extract.CRASH_COLS).astype(object)


def crash(i: int, **kw) -> dict:
    """One valid raw crash row; keyword args override fields."""
    base = dict(crash_date="2020-06-01T00:00:00.000", crash_time="8:05", borough="BROOKLYN", zip_code="11201",
                latitude="40.69", longitude="-73.99", collision_id=str(i),
                contributing_factor_vehicle_1="Unspecified")
    return {**base, **kw}


# ------------------------------------------------------------------ regex / small helpers
@pytest.mark.parametrize("z,ok", [("11010", False), ("11550", False), ("99999", False), ("10001", True),
                                  ("11201", True), ("11004", True), ("11693", True), ("11005", True),
                                  ("11001", False), ("00501", False), ("1000", False)])
def test_zip_regex(z, ok):
    assert bool(re.match(config.NYC_ZIP_RE, z)) is ok


def test_hour_band_edges():
    s = pd.Series(["0:00", "5:59", "6:00", "11:59", "12:00", "17:30", "18:00", "23:59", "24:15", "bad", None, "12:7x"])
    assert hour_band(s).astype(str).tolist() == ["Night", "Night", "Morning", "Morning", "Afternoon", "Afternoon",
                                                  "Evening", "Evening", "Unknown", "Unknown", "Unknown", "Unknown"]


@pytest.mark.parametrize("text,grp", [
    ("Driver Inattention/Distraction", "Distraction/Inattention"), ("Cell Phone (hand-held)", "Distraction/Inattention"),
    ("Outside Car Distraction", "Distraction/Inattention"), ("Texting", "Distraction/Inattention"),
    ("Unsafe Speed", "Speed/Aggressive"), ("Following Too Closely", "Speed/Aggressive"),
    ("Aggressive Driving/Road Rage", "Speed/Aggressive"), ("Unsafe Lane Changing", "Speed/Aggressive"),
    ("Passing or Lane Usage Improper", "Speed/Aggressive"),
    ("Failure to Yield Right-of-Way", "Failure to Yield/Signal"), ("Traffic Control Disregarded", "Failure to Yield/Signal"),
    ("Turning Improperly", "Failure to Yield/Signal"), ("Failure to Keep Right", "Failure to Yield/Signal"),
    ("Alcohol Involvement", "Impairment/Fatigue"), ("Fatigued/Drowsy", "Impairment/Fatigue"),
    ("Drugs (illegal)", "Impairment/Fatigue"), ("Lost Consciousness", "Impairment/Fatigue"),
    ("Illnes", "Impairment/Fatigue"), ("Fell Asleep", "Impairment/Fatigue"),
    ("Brakes Defective", "Vehicle/Road Defect"), ("Steering Failure", "Vehicle/Road Defect"),
    ("Pavement Slippery", "Vehicle/Road Defect"), ("Tire Failure/Inadequate", "Vehicle/Road Defect"),
    ("Traffic Control Device Improper/Non-Working", "Vehicle/Road Defect"), ("Glare", "Vehicle/Road Defect"),
    ("Other Vehicular", "Other"), ("Backing Unsafely", "Other"), ("Driver Inexperience", "Other"),
    ("Pedestrian/Bicyclist/Other Pedestrian Error/Confusion", "Other"), ("80", "Other"), ("Something New", "Other"),
    ("Unspecified", "Unspecified")])
def test_factor_group(text, grp):
    assert factor_group(pd.Series([text])).iloc[0] == grp


def test_factor_group_nan_and_empty():
    assert factor_group(pd.Series([np.nan, None])).astype(str).tolist() == ["Unspecified"] * 2
    assert factor_group(pd.Series([], dtype=object)).empty


# ------------------------------------------------------------------ housing rules
def housing_case() -> tuple[pd.DataFrame, list]:
    rows = [
        dict(building_id="1", postcode="10001", borough="BROOKLYN", latitude="40.70", longitude="-73.90",
             building_completion_date="2020-05-01T00:00:00.000", project_start_date="2018-01-01T00:00:00.000",
             total_units="10", all_counted_units="12", extremely_low_income_units="2", very_low_income_units="1",
             reporting_construction_type="New Construction"),
        dict(building_id="2", postcode="99999", borough="Staten Island", latitude="0", longitude="0",
             latitude_internal="40.60", longitude_internal="-74.00", project_completion_date="2021-03-01T00:00:00.000",
             total_units="60", all_counted_units="60", reporting_construction_type="Preservation"),
        dict(building_id="3", postcode="11201.0", borough="bronx", latitude="40.80", longitude="-73.90",
             building_completion_date="2019-01-01T00:00:00.000", total_units="5", all_counted_units="5"),
        dict(building_id="3", postcode="11201", borough="Bronx", latitude="40.80", longitude="-73.90",
             building_completion_date="2022-01-01T00:00:00.000", total_units="5", all_counted_units="5"),
        dict(building_id="4", postcode="1100", borough="Hoboken", latitude="41.5", longitude="-73.9",
             total_units="1000", all_counted_units="1000", reporting_construction_type="Moderate Rehabilitation"),
        dict(building_id="5", postcode="11004", borough="Queens", latitude="40.75", longitude="-73.70",
             building_completion_date="2015-01-01T00:00:00.000", project_start_date="2016-01-01T00:00:00.000",
             total_units="7", all_counted_units="7"),
        dict(building_id="6", borough="Manhattan", total_units="8", all_counted_units="8"),
    ] + [dict(building_id=str(10 + i), total_units="20", all_counted_units="20") for i in range(12)]
    log: list = []
    return clean_housing(raw_h(rows), log), log


def test_housing_rules():
    h, log = housing_case()
    h = h.set_index("building_id")
    assert h.index.is_unique and h.loc[3, "comp_year"] == 2022              # dup keeps latest completion
    assert h.loc[1, "total_units"] == 12                                    # total < counted repaired
    assert h.loc[1, "income_low_share"] == pytest.approx(3 / 12)
    assert h.loc[1, "zip"] == "10001" and pd.isna(h.loc[2, "zip"])         # 99999 rejected
    assert h.loc[3, "zip"] == "11201" and pd.isna(h.loc[4, "zip"])         # '11201.0' normalised, 1100 invalid
    assert str(h.loc[1, "borough"]) == "Brooklyn" and str(h.loc[3, "borough"]) == "Bronx"
    assert pd.isna(h.loc[4, "borough"])
    assert h.loc[2, "latitude"] == 40.60 and h.loc[2, "longitude"] == -74.00  # (0,0) -> *_internal fallback
    assert pd.isna(h.loc[4, "latitude"]) and pd.isna(h.loc[6, "latitude"])    # outside bbox / absent
    assert h.loc[2, "completed"] and h.loc[2, "comp_year"] == 2021            # project-date fallback
    assert not h.loc[4, "completed"] and pd.isna(h.loc[4, "comp_year"])
    assert h.loc[5, "date_inconsistent"] and not h.loc[1, "date_inconsistent"]
    assert h.loc[4, "is_large_outlier"] and not h.loc[1, "is_large_outlier"]
    assert [str(h.loc[i, "construction_type"]) for i in (1, 2, 4, 6)] == ["New Construction", "Preservation", "Other", "Other"]
    assert (h["studio_units"] == 0).all() and h["_3_br_units"].dtype == np.int32  # absent unit cols exist as 0
    assert h["size_bin"].dtype.name == "category"


def test_housing_log_schema():
    _, log = housing_case()
    assert log and all(LOG_KEYS <= set(e) for e in log)
    assert {e["dataset"] for e in log} == {"housing"}
    dup = next(e for e in log if "duplicate" in e["rule"])
    assert dup["rows_before"] - dup["rows_after"] == 1


def test_housing_missing_key_column_raises():
    with pytest.raises(KeyError):
        clean_housing(pd.DataFrame({"borough": ["Bronx"]}), [])


def test_size_bin_edges():
    rows = [dict(building_id=str(i), total_units=str(u), all_counted_units="0") for i, u in
            enumerate([1, 10, 11, 50, 51, 150, 151, 5000])]
    h = clean_housing(raw_h(rows), [])
    assert h["size_bin"].astype(str).tolist() == ["1-10", "1-10", "11-50", "11-50", "51-150", "51-150", "151+", "151+"]


def test_log_rule_merges_chunks():
    log: list = []
    log_rule(log, "clean", "crashes", "r", 10, 8, "why")
    log_rule(log, "clean", "crashes", "r", 5, 5, "why", 2)
    assert len(log) == 1 and (log[0]["rows_before"], log[0]["rows_after"], log[0]["n_affected"]) == (15, 13, 4)


# ------------------------------------------------------------------ crash rules
def test_crash_rules():
    rows = [
        crash(1), crash(1),                                                  # within-chunk dup
        crash(2, crash_date="2015-12-31T00:00:00.000"), crash(3, crash_date="2026-01-01T00:00:00.000"),
        crash(4, crash_date=None), crash(5, zip_code="11550"), crash(6, zip_code="11010"),
        crash(7, latitude="0", longitude="0"), crash(8, borough="staten island", zip_code=None),
        crash(9, crash_time="24:15"), crash(10, number_of_persons_injured="1", number_of_pedestrians_injured="2",
                                          number_of_cyclist_injured="1", number_of_motorist_injured="1"),
        crash(11, number_of_persons_injured="55"), crash(12, number_of_persons_killed="1"),
        crash(13, number_of_persons_injured="1", number_of_motorist_injured="1"),
        crash(14, number_of_pedestrians_killed="1", number_of_persons_killed="1"),
        crash(15, number_of_cyclist_injured="1", number_of_persons_injured="1"),
        crash(16, number_of_persons_injured="-3"),
        crash(17, crash_date="2025-12-31T00:00:00.000"), crash(18, crash_date="2016-01-01T00:00:00.000"),
    ]
    log: list = []
    seen = SeenIds()
    c = clean_crash_chunk(raw_c(rows), log, None, seen).set_index("collision_id")
    assert c.index.is_unique and set(c.index) == {1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
    assert pd.isna(c.loc[5, "zip"]) and pd.isna(c.loc[6, "zip"]) and c.loc[1, "zip"] == "11201"
    assert pd.isna(c.loc[7, "lat"]) and pd.isna(c.loc[7, "lon"]) and c.loc[1, "lat"] == 40.69
    assert str(c.loc[8, "borough"]) == "Staten Island" and str(c.loc[1, "borough"]) == "Brooklyn"
    assert str(c.loc[9, "hour_band"]) == "Unknown" and str(c.loc[1, "hour_band"]) == "Morning"
    assert c.loc[10, "injured"] == 4 and c.loc[10, "ped_injured"] == 2        # persons < sum -> sum
    assert c.loc[11, "injured"] == 20                                         # cap
    assert c.loc[16, "injured"] == 0                                          # negative -> 0
    assert [str(c.loc[i, "severity"]) for i in (12, 13, 1)] == ["Fatal", "Injury", "Property damage only"]
    assert c.loc[14, "ped_cyc"] == 1 and c.loc[15, "ped_cyc"] == 1 and c.loc[13, "ped_cyc"] == 0
    assert c.loc[14, "killed"] == 1
    assert (str(c["ped_cyc"].dtype), str(c["injured"].dtype), str(c["year"].dtype)) == ("int8", "int16", "int16")
    assert all(LOG_KEYS <= set(e) for e in log)
    # cross-chunk dedupe via SeenIds: second chunk with ids 1 and 17 plus a new one
    c2 = clean_crash_chunk(raw_c([crash(1), crash(17), crash(99)]), log, None, seen)
    assert c2["collision_id"].tolist() == [99]
    win = next(e for e in log if "outside" in e["rule"])
    assert win["rows_before"] > win["rows_after"]


# ------------------------------------------------------------------ ZIP grid imputation
def grid_chunks():
    """4000 rows in two ZIPs split at lon -73.95; ids 1..n so 5% (id % 20 == 0) are held out."""
    rng = np.random.default_rng(42)
    n = 4000
    lon = rng.uniform(-74.0, -73.9, n)
    lat = 40.70 + rng.uniform(-0.0005, 0.0005, n)
    df = pd.DataFrame({"zip_code": np.where(lon < -73.95, "11201", "11211"), "borough": "BROOKLYN",
                       "latitude": lat.round(6).astype(str), "longitude": lon.round(6).astype(str),
                       "collision_id": np.arange(1, n + 1).astype(str)})
    return df.reindex(columns=extract.CRASH_COLS), lat, lon


def test_zip_grid_imputation_and_holdout():
    df, lat, lon = grid_chunks()
    grid = build_zip_grid(lambda: iter([df.iloc[:2000], df.iloc[2000:]]))
    assert grid["holdout_n"] == 200 and grid["holdout_acc"] > 0.95
    assert grid["zip_borough"]["11201"] == "Brooklyn"
    assert (np.diff(grid["key"]) > 0).all() and grid["support"].min() >= 1
    # rows with missing ZIP get the cell-modal ZIP; far-away rows (no cell support) stay NaN
    probe = pd.DataFrame([crash(1, zip_code=None, latitude="40.7", longitude="-73.99", borough=None),
                          crash(2, zip_code=None, latitude="40.7", longitude="-73.91", borough=None),
                          crash(3, zip_code=None, latitude="40.9", longitude="-73.70", borough=None)])
    out = clean_crash_chunk(probe.reindex(columns=extract.CRASH_COLS), [], grid).set_index("collision_id")
    assert out.loc[1, "zip"] == "11201" and out.loc[2, "zip"] == "11211" and pd.isna(out.loc[3, "zip"])
    assert out["zip_imputed"].tolist() == [True, True, False]
    assert str(out.loc[1, "borough"]) == "Brooklyn"                          # borough from ZIP mode


def test_zip_grid_excludes_holdout_from_cells():
    df, _, _ = grid_chunks()
    g_all = build_zip_grid(lambda: iter([df]), exclude_holdout=False)
    g_ex = build_zip_grid(lambda: iter([df]), exclude_holdout=True)
    assert g_all["holdout_acc"] is None and g_ex["support"].sum() < g_all["support"].sum()


# ------------------------------------------------------------------ exposure
def tiny_tree():
    """Three buildings: completed 2020, completed 2023, never completed; all within ~100 m."""
    h = pd.DataFrame({"latitude": [40.7000, 40.7003, 40.7005], "longitude": [-73.9] * 3,
                      "comp_year": pd.array([2020, 2023, None], dtype="Int16")})
    return h, ig.build_tree(h), ig.comp_array(ig.tree_rows(h))


def test_exposure_semantics():
    h, tree, comp = tiny_tree()
    lat = np.array([40.7001, 40.7001, 40.7001, 40.7001, 40.80])
    lon = np.full(5, -73.9)
    yr = np.array([2019, 2020, 2022, 2025, 2020])
    owner, bldg = ig.neighbour_pairs(lat, lon, tree)
    e = ig.exposure(owner, bldg, 5, yr, comp)
    assert e.tolist() == [1, 2, 2, 2, 0]       # before any completion=1, from 2020 on=2, far away=0
    assert ig.flag_near(lat, lon, tree).tolist() == [True] * 4 + [False]
    assert ig.flag_near([np.nan], [np.nan], tree).tolist() == [False]
    only_open = pd.DataFrame({"latitude": [40.7], "longitude": [-73.9], "comp_year": pd.array([None], dtype="Int16")})
    t2 = ig.build_tree(only_open)
    o, b = ig.neighbour_pairs(np.array([40.7]), np.array([-73.9]), t2)
    assert ig.exposure(o, b, 1, np.array([2020]), ig.comp_array(ig.tree_rows(only_open))).tolist() == [0]


def haversine_m(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(h))


def test_exposure_matches_bruteforce_on_fixture():
    h = clean_housing(extract.housing_raw(), [])
    first = next(iter(extract.crash_chunks(20_000)()))
    c = clean_crash_chunk(first, [], None)
    c = c[c["lat"].notna()].sample(500, random_state=42).reset_index(drop=True)
    rows, tree = ig.tree_rows(h), ig.build_tree(h)
    owner, bldg = ig.neighbour_pairs(c["lat"].to_numpy(), c["lon"].to_numpy(), tree)
    got = ig.exposure(owner, bldg, len(c), c["year"].to_numpy(), ig.comp_array(rows))
    d = haversine_m(c["lat"].to_numpy()[:, None], c["lon"].to_numpy()[:, None],
                    rows["latitude"].to_numpy()[None, :], rows["longitude"].to_numpy()[None, :])
    comp = rows["comp_year"].astype("float64").to_numpy()[None, :]
    within = (d <= config.BUFFER_M) & ~np.isnan(comp)
    post = (within & (comp <= c["year"].to_numpy()[:, None])).any(1)
    exp = np.where(post, 2, np.where(within.any(1), 1, 0))
    assert (got == exp).all() and set(np.unique(got)) >= {0, 1, 2}


# ------------------------------------------------------------------ event sites
def test_event_sites_and_contamination():
    rows = pd.DataFrame({
        "building_id": [1, 2, 3, 4, 5, 6],
        "latitude": [40.7000, 40.7005, 40.80, 40.60, 40.7003, 40.50],   # 1,2 same site; 5 contaminates site 1-2
        "longitude": [-73.9] * 6, "comp_year": pd.array([2020, 2020, 2021, 2024, 2022, 2030], dtype="Int16"),
        "borough": pd.Categorical(["Bronx"] * 5 + [None], categories=config.BOROUGHS),
        "construction_type": pd.Categorical(["Other"] * 6, categories=["New Construction", "Preservation", "Other"]),
        "size_bin": pd.Categorical(["1-10"] * 6, categories=["1-10", "11-50", "51-150", "151+"]),
        "total_units": 5})
    tree = ig.build_tree(rows)
    ev = ig.event_sites(rows, tree)
    assert ev["cohorts"].tolist() == [2020, 2021, 2022]                      # 2019 has no building
    n = ev["nodes"].tolist()
    assert n == [0, 1, 2, 3, 4, 5] or n == [0, 1, 2, 3, 4]
    site = dict(zip(n, ev["site"]))
    assert site[0] == site[1] and len({site[0], site[2], site[3], site[4]}) == 4
    contam = dict(zip(n, ev["contaminated"]))
    assert contam[0] and contam[1] and contam[4] and not contam[2] and not contam[3]
    treated = dict(zip(n, ev["treated"]))
    assert treated[0] and treated[4] and not treated[3]                       # 2024 >= 2020+4 -> control
    counts = np.zeros((3, len(rows), ig.NY))
    cidx = pd.Series(1.0, index=ig.YEARS)
    eb = ig.event_building(rows, ev, counts, cidx)
    g = eb.groupby(["building_id", "cohort"])["rel_year"].apply(lambda s: sorted(s))
    assert all(v == list(range(-3, 4)) for v in g)
    assert set(eb.loc[eb.building_id == 4, "cohort"]) == {2020}              # control only for cohorts <= 2020
    assert set(eb.loc[eb.building_id == 4, "group"].astype(str)) == {"control"}


# ------------------------------------------------------------------ aggregation merge
def test_incremental_cube_equals_single_pass():
    rng = np.random.default_rng(42)
    rows = [crash(i, zip_code=str(rng.choice(["11201", "10001"])), crash_time=f"{rng.integers(0, 24)}:10",
                  crash_date=f"20{rng.integers(16, 26)}-03-01T00:00:00.000", number_of_persons_injured=str(rng.integers(0, 3)))
            for i in range(1, 301)]
    c = clean_crash_chunk(raw_c(rows), [], None)
    expo = rng.integers(-1, 3, len(c)).astype(np.int8)
    one = ig.cube_part(c, expo)
    acc = None
    for sl in (slice(0, 100), slice(100, 220), slice(220, 300)):
        acc = ig.add_part(acc, ig.cube_part(c.iloc[sl], expo[sl]), ig.CUBE_KEYS, ig.CUBE_MEAS)
    key = lambda d: d.astype({k: str for k in ("zip", "borough", "hour_band", "factor_group", "severity")}) \
        .sort_values(ig.CUBE_KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(key(one), key(acc), check_dtype=False)
    assert acc["crashes"].sum() == len(c)


# ------------------------------------------------------------------ extract (mocked HTTP)
class FakeResp:
    def __init__(self, status: int, text: str = ""):
        self.status_code, self.text = status, text


def test_fetch_soda_keyset_cache_token_and_reindex(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FIXTURE", False)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "APP_TOKEN", "tok123")
    calls, tokens = [], []

    def fake_get(self, url, params=None, timeout=None):
        calls.append(params)
        tokens.append(self.headers.get("X-App-Token"))
        last = int(re.search(r"collision_id>(\d+)", params["$where"]).group(1))
        ids = [i for i in range(1, 6) if i > last][: params["$limit"]]
        return FakeResp(200, "collision_id,zip_code\n" + "".join(f"{i},1120{i}\n" for i in ids))   # latitude omitted

    monkeypatch.setattr(requests.Session, "get", fake_get)
    sel = ["collision_id", "zip_code", "latitude"]
    pages = list(extract.fetch_soda(config.CRASH_ID, sel, "crash_date>='2016-01-01'", page_size=2))
    df = pd.concat(pages)
    assert df["collision_id"].tolist() == ["1", "2", "3", "4", "5"] and list(df.columns) == sel
    assert df["latitude"].isna().all()                                         # omitted column -> NaN
    assert set(tokens) == {"tok123"} and "$order" in calls[0] and calls[0]["$order"] == "collision_id"
    assert "collision_id>0" in calls[0]["$where"] and "collision_id>2" in calls[1]["$where"]
    n = len(calls)
    again = pd.concat(extract.fetch_soda(config.CRASH_ID, sel, "crash_date>='2016-01-01'", page_size=2))
    assert len(again) == 5 and len(calls) - n == 2                              # full pages cached; short tail + empty refetched
    capped = pd.concat(extract.fetch_soda(config.CRASH_ID, sel, "crash_date>='2016-01-01'", page_size=2, max_rows=3))
    assert len(capped) == 3


def test_fetch_soda_retry_and_fail_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FIXTURE", False)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)
    seq = [FakeResp(503), FakeResp(429), FakeResp(200, "building_id\n7\n")]
    monkeypatch.setattr(requests.Session, "get", lambda self, url, params=None, timeout=None: seq.pop(0))
    out = pd.concat(extract.fetch_soda(config.HOUSING_ID, ["building_id", "bbl"]))
    assert out["building_id"].tolist() == ["7"] and not seq
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "b")
    monkeypatch.setattr(requests.Session, "get", lambda self, url, params=None, timeout=None: FakeResp(403, "blocked"))
    with pytest.raises(RuntimeError, match="403"):
        list(extract.fetch_soda(config.HOUSING_ID, ["building_id"]))


def test_fixture_mode_streams_20k_chunks():
    sizes = [len(c) for c in extract.fetch_soda(config.CRASH_ID, extract.CRASH_COLS)]
    assert sizes[0] == 20_000 and sum(sizes) == 150_000
    assert next(extract.crash_chunks(500)()).shape == (500, len(extract.CRASH_COLS))


# ------------------------------------------------------------------ end-to-end (fixture, 20k crashes)
SPEC_DTYPES = {   # SPEC section 5 + amendments A3/A4/A8/A9, written independently of integrate.SCHEMA
    "crash_cube": dict(zip="category", borough="category", year="int16", hour_band="category",
                       factor_group="category", severity="category", exposure="int8", ped_cyc="int8",
                       crashes="int32", injured="int32", killed="int32", ped_injured="int32", cyc_injured="int32"),
    "crash_grid": dict(cell_lat="float64", cell_lon="float64", year="int16", borough="category",
                       severity="category", crashes="int32", injured="int32", ped_cyc_crashes="int32"),
    "event_building": dict(building_id="int64", site_id="int32", group="category", cohort="int16",
                           contaminated="bool", comp_year="int16", rel_year="int8", construction_type="category",
                           borough="category", size_bin="category", total_units="int32", crashes="int32",
                           adj="float64", ped_cyc_crashes="int32", injured="int32"),
    "panel_zip_year": dict(zip="str", borough="category", year="int16", units_completed="int32",
                           counted_units="int32", eli_vli_units="int32", new_con_units="int32",
                           preservation_units="int32", other_units="int32", buildings_completed="int32",
                           cum_units="int32"),
}
SPEC_DTYPES["panel_zip_year_full"] = {**SPEC_DTYPES["panel_zip_year"], "crashes": "int32", "injured": "int32",
                                      "killed": "int32", "ped_cyc_crashes": "int32"}
HOUSING_SPEC_COLS = ["building_id", "project_id", "project_name", "borough", "zip", "latitude", "longitude",
                     "comp_year", "construction_type", "total_units", "all_counted_units",
                     "extremely_low_income_units", "very_low_income_units", "low_income_units",
                     "moderate_income_units", "middle_income_units", "other_income_units", "income_low_share",
                     "size_bin", "is_large_outlier", "completed", "pre3_crashes", "post3_crashes", "pre3_adj",
                     "post3_adj", "event_eligible", "crashes_250m_total", "injuries_250m_total"]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    meta = run(max_rows=20_000, out_dir=out, with_rq=False)
    arts = {p.stem: pd.read_parquet(p) for p in out.glob("*.parquet")}
    return out, meta, arts


def kind_ok(series: pd.Series, want: str) -> bool:
    """dtype check: exact for numerics/bool/category, string-likeness for 'str'."""
    if want == "str":
        return pd.api.types.is_string_dtype(series.dtype) or series.dtype == object
    return str(series.dtype) == want


def test_e2e_files_and_dtype_contract(e2e):
    out, meta, arts = e2e
    assert set(arts) == {"housing_clean", "crash_cube", "crash_grid", "event_building", "panel_zip_year",
                         "panel_zip_year_full"}
    assert not (out / "event_study.parquet").exists() and (out / "meta.json").exists()
    for name, spec in SPEC_DTYPES.items():
        assert list(arts[name].columns) == list(spec), name
        bad = {c: str(arts[name][c].dtype) for c, t in spec.items() if not kind_ok(arts[name][c], t)}
        assert not bad, (name, bad)
    h = arts["housing_clean"]
    assert list(h.columns[: len(HOUSING_SPEC_COLS)]) == HOUSING_SPEC_COLS
    assert h["building_id"].is_unique and h["comp_year"].dtype.name == "Int16"
    assert h["completed"].dtype == bool and h["event_eligible"].dtype == bool
    assert "bbl" not in h and "latitude_internal" not in h


def test_e2e_cube_conservation(e2e):
    _, meta, arts = e2e
    raw = pd.concat(extract.fetch_soda(config.CRASH_ID, ["collision_id", "crash_date"], max_rows=20_000))
    yr = pd.to_datetime(raw["crash_date"], format="ISO8601").dt.year
    expected = int(raw.loc[yr.between(config.Y_START, config.Y_END)].drop_duplicates("collision_id").shape[0])
    cube = arts["crash_cube"]
    assert cube["crashes"].sum() == expected == meta["rows"]["crashes_clean_in_window"]
    assert arts["crash_grid"]["crashes"].sum() == cube.loc[cube["exposure"] >= 0, "crashes"].sum()
    assert not cube.duplicated(ig.CUBE_KEYS).any()
    assert set(cube["exposure"].unique()) <= {-1, 0, 1, 2} and cube["year"].between(2016, 2025).all()
    pf = arts["panel_zip_year_full"]
    in_panel = cube["zip"].astype(str).isin(pf["zip"])
    assert pf["crashes"].sum() == cube.loc[in_panel, "crashes"].sum()


def test_e2e_panel(e2e):
    _, meta, arts = e2e
    p = arts["panel_zip_year"]
    assert not p.duplicated(["zip", "year"]).any()
    assert (p.select_dtypes("number") >= 0).all().all()
    assert p["year"].nunique() == config.Y_END - config.Y_START + 1
    assert p.groupby("zip").size().eq(p["year"].nunique()).all()
    assert (p["units_completed"] == p["new_con_units"] + p["preservation_units"] + p["other_units"]).all()
    assert (p.groupby("zip")["cum_units"].diff().dropna() >= 0).all()          # cumulative is monotone
    cube = arts["crash_cube"]
    located = cube[cube["zip"] != "UNLOCATED"].groupby("zip", observed=True)["crashes"].sum()
    assert set(p["zip"]) == set(located.index[located >= config.MIN_ZIP_CRASHES].astype(str))
    assert "low_income_units" not in p and "UNLOCATED" not in set(p["zip"])


def test_e2e_event_building(e2e):
    _, meta, arts = e2e
    e = arts["event_building"]
    assert len(e) > 0 and set(e["group"].astype(str)) <= {"treated", "control"}
    cover = e.groupby(["building_id", "cohort"])["rel_year"].apply(lambda s: tuple(sorted(s)))
    assert (cover == tuple(range(-3, 4))).all()
    t = e[e["group"] == "treated"]
    assert (t["cohort"] == t["comp_year"]).all()
    assert t["cohort"].between(config.Y_START + 3, config.Y_END - 3).all()
    c = e[e["group"] == "control"]
    assert (c["comp_year"] >= c["cohort"] + 4).all()
    assert e["site_id"].nunique() == meta["n_sites_treated"] + meta["n_sites_control"]
    assert (e["adj"] >= 0).all() and np.isfinite(e["adj"]).all()
    h = arts["housing_clean"]
    el = h[h["event_eligible"]].set_index("building_id")
    tt = t.groupby("building_id")
    pre = t[t["rel_year"].between(-3, -1)].groupby("building_id")["crashes"].sum()
    assert (pre.reindex(el.index) == el["pre3_crashes"]).all()
    assert (el["crashes_250m_total"] >= el["pre3_crashes"] + el["post3_crashes"]).all()
    assert tt.ngroups == len(el)


def test_e2e_meta_and_log(e2e):
    out, meta, _ = e2e
    on_disk = json.loads((out / "meta.json").read_text())
    for k in ["fixture", "Y_START", "Y_END", "generated_at", "sources", "city_index", "imputation_holdout_acc",
              "imputation_acc_low_support", "imputed_share_by_year", "located_share_by_year", "eligible_cohorts",
              "n_sites_treated", "n_sites_control", "unlocated_share", "zip_imputed_share", "rows", "cleaning_log"]:
        assert k in on_disk, k
    assert on_disk["fixture"] is True and meta["max_rows"] == 20_000 and meta["partial"] is True and on_disk["partial"] is True
    assert out != config.ROOT / "data" / "processed"                          # never touches real output dir
    assert np.mean(list(on_disk["city_index"].values())) == pytest.approx(1.0)
    log = on_disk["cleaning_log"]
    assert all(LOG_KEYS <= set(e) for e in log)
    assert {e["step"] for e in log} >= {"clean", "impute", "post-integration"}
    assert {e["dataset"] for e in log} >= {"housing", "crashes", "panel"}
    assert sum(1 for e in log if e["rule"].startswith("dtype enforcement")) == 6
    assert 0.0 <= on_disk["imputation_holdout_acc"] <= 1.0


# ------------------------------------------------------------------ review regressions
def _soda_server(monkeypatch, tmp_path, cap):
    """Fake crash endpoint with ids 1..7 that truncates every page to `cap` rows (like a proxy)."""
    monkeypatch.setattr(config, "FIXTURE", False)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)

    def fake_get(self, url, params=None, timeout=None):
        last = int(re.search(r"collision_id>(\d+)", params["$where"]).group(1))
        ids = [i for i in range(1, 8) if i > last][:cap]
        return FakeResp(200, "collision_id\n" + "".join(f"{i}\n" for i in ids))

    monkeypatch.setattr(requests.Session, "get", fake_get)


def test_keyset_survives_truncated_pages(monkeypatch, tmp_path):
    _soda_server(monkeypatch, tmp_path, cap=2)                    # limit 5 but pages carry 2 rows
    df = pd.concat(extract.fetch_soda(config.CRASH_ID, ["collision_id"], page_size=5))
    assert df["collision_id"].tolist() == [str(i) for i in range(1, 8)]


def test_keyset_fails_when_id_does_not_advance(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FIXTURE", False)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(requests.Session, "get", lambda self, url, params=None, timeout=None: FakeResp(200, "collision_id\n3\n"))
    with pytest.raises(RuntimeError, match="did not advance"):
        list(extract.fetch_soda(config.CRASH_ID, ["collision_id"], page_size=5))


def test_next_key_ignores_non_numeric_ids():
    df = pd.DataFrame({"collision_id": ["4", "oops", None, "9"]})
    assert extract._next_key(df, "collision_id", 0) == 9


def test_short_tail_page_not_cached(monkeypatch, tmp_path):
    _soda_server(monkeypatch, tmp_path, cap=50)
    list(extract.fetch_soda(config.CRASH_ID, ["collision_id"], page_size=5))
    assert len(list(tmp_path.rglob("page_*.csv.gz"))) == 1        # full first page cached; short tail is not


def test_cumulative_units_start_in_housing_min_year():
    agg = pd.DataFrame({"units_completed": [100, 5, 7]},
                       index=pd.MultiIndex.from_tuples([("11201", 2010), ("11201", 2016), ("11201", 2018)], names=["zip", "year"]))
    cum = ig._cumulative(agg, ["11201", "10001"]).reshape(2, -1)
    assert cum[0, 0] == 5 and cum[0, -1] == 12                      # 2010 completions excluded (A8: from 2014)
    assert (cum[1] == 0).all()


def test_zip_plus4_time_seconds_swapped_coords_and_injury_caps():
    rows = [crash(1, zip_code="11201-1234", crash_time="08:05:30"),
            crash(2, latitude="-73.99", longitude="40.69"),                      # swapped pair
            crash(3, latitude="0", longitude="0"),
            crash(4, number_of_persons_injured="35", number_of_pedestrians_injured="30", number_of_cyclist_injured="2")]
    log: list = []
    c = clean_crash_chunk(raw_c(rows), log, None, SeenIds()).set_index("collision_id")
    assert c.loc[1, "zip"] == "11201" and str(c.loc[1, "hour_band"]) == "Morning"
    assert c.loc[2, "lat"] == pytest.approx(40.69) and c.loc[2, "lon"] == pytest.approx(-73.99)
    assert np.isnan(c.loc[3, "lat"])
    assert c.loc[4, "injured"] == 20 and c.loc[4, "ped_injured"] == 20 and c.loc[4, "cyc_injured"] == 2
    assert any("swapped" in r["rule"] and r["rows_affected"] == 1 for r in log if "rows_affected" in r) or any("swapped" in r["rule"] for r in log)


def test_zero_unit_building_gets_size_bin():
    rows = [dict(building_id=str(i), total_units=str(u), all_counted_units="0") for i, u in enumerate([0, 5, 20, 80, 200])]
    h = clean_housing(raw_h(rows), [])
    assert h["size_bin"].notna().all() and str(h.sort_values("total_units")["size_bin"].iloc[0]) == SIZE_FIRST


def test_write_all_removes_stale_parquet(tmp_path):
    from pipeline.run import write_all
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "old.parquet")
    write_all(tmp_path, {"new": pd.DataFrame({"a": [1]})}, {})
    assert [p.name for p in tmp_path.glob("*.parquet")] == ["new.parquet"]
