"""Generate SYNTHETIC NYC housing and crash fixtures (numpy seed 42). NOT real data.

Run: python tests/make_fixtures.py
Writes tests/fixtures/housing.csv and tests/fixtures/crashes.csv using the raw
NYC Open Data column names from SPEC section 2, with API-style messiness: blank
cells for null/zero, bad ZIPs, (0,0) coordinates, duplicates, outliers.
No housing->crash effect is planted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
LAT_LO, LAT_HI = 40.55, 40.90
LON_LO, LON_HI = -74.05, -73.75
GRID_LAT, GRID_LON = 6, 10  # 60 coarse cells, one synthetic ZIP per cell
N_HOUSING = 2940  # + 58 duplicate building_id rows -> ~3000
N_CRASH = 148_500  # + 1_500 duplicate collision_id rows -> 150_000
DATE_FMT = "%Y-%m-%dT00:00:00.000"
OUT_DIR = Path(__file__).resolve().parent / "fixtures"

ZIP_POOL = {
    "Manhattan": (10001, 10282),
    "Staten Island": (10301, 10314),
    "Bronx": (10451, 10475),
    "Brooklyn": (11201, 11256),
    "Queens": (11354, 11436),
}
BORO_CODE = {"Manhattan": 1, "Bronx": 2, "Brooklyn": 3, "Queens": 4, "Staten Island": 5}
NTA_PREFIX = {"Manhattan": "MN", "Bronx": "BX", "Brooklyn": "BK", "Queens": "QN", "Staten Island": "SI"}
STREETS = [
    "BEDFORD AVE", "FLATBUSH AVE", "GRAND CONCOURSE", "JEROME AVE", "ATLANTIC AVE",
    "UTICA AVE", "ROCKAWAY BLVD", "EAST 149 ST", "WEST 125 ST", "ST NICHOLAS AVE",
    "PARK AVE", "HYLAN BLVD", "RICHMOND AVE", "NOSTRAND AVE", "MYRTLE AVE",
    "QUEENS BLVD", "3 AVE", "2 AVE", "LEXINGTON AVE", "CONEY ISLAND AVE",
]
FACTORS = {  # contributing_factor_vehicle_N -> relative weight
    "Unspecified": 32, "Driver Inattention/Distraction": 18,
    "Failure to Yield Right-of-Way": 7, "Unsafe Speed": 6, "Following Too Closely": 5,
    "Passing Too Closely": 2, "Turning Improperly": 3, "Backing Unsafely": 1.5,
    "Traffic Control Disregarded": 2, "Alcohol Involvement": 1.5,
    "Fatigued/Drowsy": 1.5, "Brakes Defective": 0.5, "Pavement Slippery": 1,
    "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion": 2, "Other Vehicular": 4,
}
VEHICLES = {  # vehicle_type_code* -> relative weight
    "Sedan": 40, "PASSENGER VEHICLE": 25, "Station Wagon/Sport Utility Vehicle": 15,
    "Taxi": 3, "Pick-up Truck": 3, "Van": 3, "Box Truck": 1.5, "Bike": 2,
    "Motorcycle": 1.5, "E-Bike": 1, "Bus": 0.5, "Unknown": 2,
}
BAD_ZIPS = ["00000", "N/A", "1000", "ABCDE", "99999", "11"]
INCOME_COLS = [
    "extremely_low_income_units", "very_low_income_units", "low_income_units",
    "moderate_income_units", "middle_income_units", "other_income_units",
]
BED_COLS = ["studio_units"] + [f"_{k}_br_units" for k in range(1, 7)] + ["unknown_br_units"]
HOUSING_COLS = [
    "project_id", "project_name", "project_start_date", "project_completion_date",
    "building_id", "house_number", "street_name", "borough", "postcode", "bbl", "bin",
    "community_board", "council_district", "census_tract", "neighborhood_tabulation_area",
    "latitude", "longitude", "latitude_internal", "longitude_internal",
    "building_completion_date", "reporting_construction_type",
    "extended_affordability_status", "prevailing_wage_status", *INCOME_COLS, *BED_COLS,
    "counted_rental_units", "counted_homeownership_units", "all_counted_units", "total_units",
]
UNIT_COLS = [c for c in HOUSING_COLS if c.endswith("_units") and c != "total_units"]
DATE_COLS = ["project_start_date", "project_completion_date", "building_completion_date"]
CRASH_COLS = [
    "crash_date", "crash_time", "borough", "zip_code", "latitude", "longitude", "location",
    "on_street_name", "off_street_name", "cross_street_name",
    "number_of_persons_injured", "number_of_persons_killed",
    "number_of_pedestrians_injured", "number_of_pedestrians_killed",
    "number_of_cyclist_injured", "number_of_cyclist_killed",
    "number_of_motorist_injured", "number_of_motorist_killed",
    *[f"contributing_factor_vehicle_{k}" for k in range(1, 6)], "collision_id",
    "vehicle_type_code1", "vehicle_type_code2",
    "vehicle_type_code_3", "vehicle_type_code_4", "vehicle_type_code_5",
]


def nan_mask(rng: np.random.Generator, values, p: float) -> pd.Series:
    """Blank a random fraction p of values (integer input becomes nullable Int64)."""
    s = pd.Series(values)
    if pd.api.types.is_integer_dtype(s):
        s = s.astype("Int64")
    return s.mask(rng.random(len(s)) < p)


def weighted_text(rng: np.random.Generator, table: dict, n: int, p_nan: float) -> pd.Series:
    """Draw labels by relative weight, then blank a fraction p_nan."""
    names = np.array(list(table), dtype=object)
    probs = np.array(list(table.values()), dtype=float)
    vals = rng.choice(names, n, p=probs / probs.sum())
    return pd.Series(vals).mask(rng.random(n) < p_nan)


def make_zip_grid(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pick 60 synthetic NYC-valid ZIPs and their borough labels, one per grid cell."""
    pool = [(z, b) for b, (lo, hi) in ZIP_POOL.items() for z in range(lo, hi + 1)]
    picked = [pool[i] for i in rng.choice(len(pool), GRID_LAT * GRID_LON, replace=False)]
    zips = np.array([f"{z:05d}" for z, _ in picked])
    boros = np.array([b for _, b in picked], dtype=object)
    return zips, boros


def cell_index(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Coarse grid cell id (0..59) for coordinates inside the NYC bbox."""
    i = np.clip(((lat - LAT_LO) / (LAT_HI - LAT_LO) * GRID_LAT).astype(int), 0, GRID_LAT - 1)
    j = np.clip(((lon - LON_LO) / (LON_HI - LON_LO) * GRID_LON).astype(int), 0, GRID_LON - 1)
    return i * GRID_LON + j


def sample_in_cells(rng: np.random.Generator, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform lat/lon inside each given grid cell, so ZIP and coordinates agree."""
    n = len(cells)
    ci, cj = cells // GRID_LON, cells % GRID_LON
    lat = LAT_LO + (ci + rng.uniform(0, 1, n)) * (LAT_HI - LAT_LO) / GRID_LAT
    lon = LON_LO + (cj + rng.uniform(0, 1, n)) * (LON_HI - LON_LO) / GRID_LON
    return lat, lon


def housing_dates(rng: np.random.Generator, n: int) -> dict[str, pd.Series]:
    """Start and completion dates in 2014-2025, with missing values and before-start cases."""
    comp = pd.to_datetime("2014-01-01") + pd.to_timedelta(rng.integers(0, 4383, n), unit="D")
    start = comp - pd.to_timedelta(rng.integers(365, 1500, n), unit="D")
    late_start = rng.random(n) < 0.05  # project starts after it completes
    start = start.where(~late_start, comp + pd.to_timedelta(rng.integers(1, 200, n), unit="D"))
    bldg = comp + pd.to_timedelta(rng.integers(-180, 181, n), unit="D")
    early = rng.random(n) < 0.04  # building completes before project start
    bldg = bldg.where(~early, start - pd.to_timedelta(rng.integers(10, 300, n), unit="D"))
    return {
        "project_start_date": start,
        "project_completion_date": comp.where(rng.random(n) >= 0.05),
        "building_completion_date": bldg.where(rng.random(n) >= 0.08),
    }


def housing_units(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Unit counts: heavy-tailed total, income and bedroom splits summing to all_counted."""
    total = np.maximum(1, np.round(rng.lognormal(3.6, 1.0, n))).astype(int)
    total[rng.choice(n, 4, replace=False)] = rng.integers(1000, 2400, 4)  # a few huge projects
    counted = np.where(rng.random(n) < 0.06, total + rng.integers(1, 15, n), total)  # total < counted
    rental = np.round(counted * rng.uniform(0.6, 1.0, n)).astype(int)
    inc = np.floor(rng.dirichlet([0.3, 0.5, 1.0, 2.0, 1.0, 0.2], n) * counted[:, None]).astype(int)
    inc[:, 0] += counted - inc.sum(axis=1)  # remainder keeps the split exact
    bed = np.floor(rng.dirichlet([1, 0.5, 2, 3, 2, 0.8, 0.2, 0.3], n) * counted[:, None]).astype(int)
    bed[:, -1] += counted - bed.sum(axis=1)
    out = {"total_units": total, "all_counted_units": counted,
           "counted_rental_units": rental, "counted_homeownership_units": counted - rental}
    out.update({c: inc[:, k] for k, c in enumerate(INCOME_COLS)})
    out.update({c: bed[:, k] for k, c in enumerate(BED_COLS)})
    return out


def make_housing(rng: np.random.Generator, zips: np.ndarray, boros: np.ndarray) -> pd.DataFrame:
    """Building-level affordable housing rows with raw API names and messy values."""
    n = N_HOUSING
    cells = rng.integers(0, GRID_LAT * GRID_LON, n)
    lat, lon = sample_in_cells(rng, cells)
    zero = rng.random(n) < 0.003  # junk (0, 0) coordinates
    lat, lon = np.where(zero, 0.0, lat), np.where(zero, 0.0, lon)
    boro = boros[cells]
    code = np.array([BORO_CODE[b] for b in boro])
    pid = 1000 + rng.integers(0, 900, n)
    nta_num = rng.integers(1, 99, n)
    cols = {
        "project_id": pid,
        "project_name": [f"Project {p} Apartments" for p in pid],
        "building_id": rng.permutation(np.arange(100_000, 100_000 + n)),
        "house_number": rng.integers(1, 999, n).astype(str),
        "street_name": rng.choice(STREETS, n),
        "borough": boro,
        "postcode": nan_mask(rng, zips[cells].astype(int), 0.10).mask(rng.random(n) < 0.005, 99999),
        "bbl": nan_mask(rng, code * 10**9 + rng.integers(1, 16000, n) * 10**4 + rng.integers(1, 999, n), 0.05),
        "bin": nan_mask(rng, rng.integers(1_000_000, 6_000_000, n), 0.10),
        "community_board": rng.integers(1, 19, n),
        "council_district": rng.integers(1, 52, n),
        "census_tract": nan_mask(rng, rng.integers(1, 9999, n), 0.02),
        "neighborhood_tabulation_area": [f"{NTA_PREFIX[b]}{k:02d}" for b, k in zip(boro, nta_num)],
        "latitude": nan_mask(rng, lat, 0.20),
        "longitude": nan_mask(rng, lon, 0.20),
        "latitude_internal": nan_mask(rng, lat, 0.20),
        "longitude_internal": nan_mask(rng, lon, 0.20),
        "reporting_construction_type": weighted_text(
            rng, {"New Construction": 60, "Preservation": 36, "Moderate Rehabilitation": 1, "TBD": 1}, n, 0.03),
        "extended_affordability_status": weighted_text(
            rng, {"Extended Affordability Only": 1, "Non-Extended": 2}, n, 0.3),
        "prevailing_wage_status": weighted_text(
            rng, {"Prevailing Wage Required": 1, "Prevailing Wage Not Required": 2}, n, 0.4),
    }
    cols.update(housing_dates(rng, n))
    cols.update(housing_units(rng, n))
    df = pd.DataFrame(cols)
    for c in UNIT_COLS:  # API omits zero-valued unit keys -> blank cells
        df[c] = df[c].where(df[c] != 0).astype("Int64")
    dup = df.iloc[rng.choice(n, int(0.02 * n), replace=False)].copy()  # duplicate building_id rows
    dup["building_completion_date"] += pd.Timedelta(days=30)  # later copy wins on dedupe
    df = pd.concat([df, dup], ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    for c in DATE_COLS:
        df[c] = df[c].dt.strftime(DATE_FMT)
    return df[HOUSING_COLS]


def crash_coords(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """70% of crashes cluster around 40 hot-spot centres, the rest are uniform in the bbox."""
    hot_lat, hot_lon = rng.uniform(LAT_LO, LAT_HI, 40), rng.uniform(LON_LO, LON_HI, 40)
    k = rng.integers(0, 40, n)
    near = rng.random(n) < 0.7
    lat = np.where(near, hot_lat[k] + rng.normal(0, 0.006, n), rng.uniform(LAT_LO, LAT_HI, n))
    lon = np.where(near, hot_lon[k] + rng.normal(0, 0.008, n), rng.uniform(LON_LO, LON_HI, n))
    return (np.clip(lat, LAT_LO + 1e-4, LAT_HI - 1e-4),
            np.clip(lon, LON_LO + 1e-4, LON_HI - 1e-4))


def crash_dates(rng: np.random.Generator, n: int) -> pd.Series:
    """Mostly 2016-2025, plus ~2% in 2015 and ~2% in 2026 (outside the analysis window)."""
    year = rng.integers(2016, 2026, n)
    year = np.where(rng.random(n) < 0.02, 2015, year)
    year = np.where(rng.random(n) < 0.02, 2026, year)
    doy = np.where(year == 2026, rng.integers(0, 270, n), rng.integers(0, 365, n))
    base = pd.to_datetime(pd.DataFrame({"year": year, "month": 1, "day": 1}))
    return base + pd.to_timedelta(doy, unit="D")


def crash_time_text(rng: np.random.Generator, n: int) -> pd.Series:
    """Text 'H:MM' times, with ~1.2% invalid strings and ~0.5% blank."""
    h, m = rng.integers(0, 24, n), rng.integers(0, 60, n)
    txt = np.array([f"{a}:{b:02d}" for a, b in zip(h, m)], dtype=object)
    bad = rng.random(n) < 0.012
    txt[bad] = rng.choice(["24:15", "99:00", "bad", "12:7x", "-1:30"], bad.sum())
    return nan_mask(rng, txt, 0.005)


def injury_columns(rng: np.random.Generator, n: int) -> dict[str, pd.Series]:
    """Injury/kill counts; persons_injured sometimes below the sum, rare outliers above 20."""
    ped_i, cyc_i = rng.poisson(0.06, n), rng.poisson(0.04, n)
    mot_i, other = rng.poisson(0.25, n), rng.poisson(0.12, n)
    sub = ped_i + cyc_i + mot_i
    pers_i = sub + other
    short = (rng.random(n) < 0.05) & (sub >= 1)
    pers_i = np.where(short, sub - 1, pers_i)  # inconsistent with subcategories
    outliers = rng.choice(n, 30, replace=False)
    pers_i[outliers] = rng.integers(21, 60, 30)
    ped_k, cyc_k = rng.binomial(1, 0.002, n), rng.binomial(1, 0.0006, n)
    mot_k = rng.binomial(1, 0.0015, n)
    pers_k = ped_k + cyc_k + mot_k + rng.binomial(1, 0.0002, n)
    raw = {
        "number_of_persons_injured": pers_i, "number_of_persons_killed": pers_k,
        "number_of_pedestrians_injured": ped_i, "number_of_pedestrians_killed": ped_k,
        "number_of_cyclist_injured": cyc_i, "number_of_cyclist_killed": cyc_k,
        "number_of_motorist_injured": mot_i, "number_of_motorist_killed": mot_k,
    }
    return {c: nan_mask(rng, v, 0.01) for c, v in raw.items()}


def make_crashes(rng: np.random.Generator, zips: np.ndarray, boros: np.ndarray) -> pd.DataFrame:
    """Crash rows with clustered coordinates; ZIP and borough come from the same grid."""
    n = N_CRASH
    lat, lon = crash_coords(rng, n)
    cells = cell_index(lat, lon)  # ZIP/borough from true position, before blanking
    zero = rng.random(n) < 0.001  # junk (0, 0) coordinates
    lat_r = np.where(zero, 0.0, np.round(lat, 6))
    lon_r = np.where(zero, 0.0, np.round(lon, 6))
    lat_s, lon_s = nan_mask(rng, lat_r, 0.08), nan_mask(rng, lon_r, 0.08)
    location = [f"({a}, {b})" if pd.notna(a) and pd.notna(b) else np.nan for a, b in zip(lat_s, lon_s)]
    zip_arr = zips[cells].astype(object)
    bad_zip = rng.random(n) < 0.01
    zip_arr[bad_zip] = rng.choice(BAD_ZIPS, bad_zip.sum())
    cols = {
        "crash_date": crash_dates(rng, n),
        "crash_time": crash_time_text(rng, n),
        "borough": nan_mask(rng, pd.Series(boros[cells]).str.upper().values, 0.34),
        "zip_code": nan_mask(rng, zip_arr, 0.34),
        "latitude": lat_s,
        "longitude": lon_s,
        "location": location,
        "on_street_name": nan_mask(rng, rng.choice(STREETS, n), 0.15),
        "off_street_name": nan_mask(rng, rng.choice(STREETS, n), 0.90),
        "cross_street_name": nan_mask(rng, rng.choice(STREETS, n), 0.40),
    }
    cols.update(injury_columns(rng, n))
    for k, p in zip(range(1, 6), [0.03, 0.6, 0.9, 0.97, 0.99]):
        cols[f"contributing_factor_vehicle_{k}"] = weighted_text(rng, FACTORS, n, p)
    cols["collision_id"] = rng.permutation(np.arange(3_900_000, 3_900_000 + n))
    for name, p in [("vehicle_type_code1", 0.03), ("vehicle_type_code2", 0.55),
                    ("vehicle_type_code_3", 0.97), ("vehicle_type_code_4", 0.99),
                    ("vehicle_type_code_5", 0.995)]:
        cols[name] = weighted_text(rng, VEHICLES, n, p)
    df = pd.DataFrame(cols)
    dup = df.iloc[rng.choice(n, 1500, replace=False)]  # duplicate collision_id rows
    df = pd.concat([df, dup], ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df["crash_date"] = df["crash_date"].dt.strftime(DATE_FMT)
    return df[CRASH_COLS]


def main() -> None:
    """Generate both fixture CSVs and print row counts and sizes."""
    rng = np.random.default_rng(SEED)
    zips, boros = make_zip_grid(rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    housing = make_housing(rng, zips, boros)
    crashes = make_crashes(rng, zips, boros)
    for name, df in (("housing", housing), ("crashes", crashes)):
        path = OUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}: {len(df):,} rows x {len(df.columns)} cols, "
              f"{path.stat().st_size / 1e6:.2f} MB -> {path}")


if __name__ == "__main__":
    main()
