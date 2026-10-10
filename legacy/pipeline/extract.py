"""Socrata extraction: keyset-paged CSV download with retries, gz cache and fixture streaming."""
from __future__ import annotations

import gzip
import hashlib
import io
import os
import time
from typing import Callable, Iterator, Sequence

import pandas as pd
import requests

import config

CRASH_COLS = [
    "crash_date", "crash_time", "borough", "zip_code", "latitude", "longitude",
    "number_of_persons_injured", "number_of_persons_killed",
    "number_of_pedestrians_injured", "number_of_pedestrians_killed",
    "number_of_cyclist_injured", "number_of_cyclist_killed",
    "number_of_motorist_injured", "number_of_motorist_killed",
    "contributing_factor_vehicle_1", "collision_id",
]
INCOME_COLS = ["extremely_low_income_units", "very_low_income_units", "low_income_units",
               "moderate_income_units", "middle_income_units", "other_income_units"]
HOUSING_COLS = [
    "project_id", "project_name", "project_start_date", "project_completion_date", "building_id",
    "house_number", "street_name", "borough", "postcode", "bbl", "bin", "community_board",
    "council_district", "census_tract", "neighborhood_tabulation_area", "latitude", "longitude",
    "latitude_internal", "longitude_internal", "building_completion_date",
    "reporting_construction_type", "extended_affordability_status", "prevailing_wage_status",
    *INCOME_COLS, "studio_units", *[f"_{k}_br_units" for k in range(1, 7)], "unknown_br_units",
    "counted_rental_units", "counted_homeownership_units", "all_counted_units", "total_units",
]
FIXTURE_FILES = {config.HOUSING_ID: "housing.csv", config.CRASH_ID: "crashes.csv"}
FIXTURE_CHUNK = 20_000
HOUSING_LIMIT = 50_000
RETRIES, TIMEOUT = 5, 180
RETRY_STATUS = {429, 500, 502, 503, 504}


def _cols(select: str | Sequence[str] | None) -> list[str] | None:
    """Normalise a $select spec (comma string or list) to a list of column names."""
    if select is None:
        return None
    return [c.strip() for c in select.split(",")] if isinstance(select, str) else list(select)


def _align(df: pd.DataFrame, cols: list[str] | None) -> pd.DataFrame:
    """Reindex to the requested columns: Socrata omits all-null columns, so add them as NaN."""
    return df if cols is None else df.reindex(columns=cols)


def _read_text(text: str) -> pd.DataFrame:
    """Parse CSV text with every column kept as str (explicit conversion happens in cleaning)."""
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), dtype=str)


def _download(sess: requests.Session, url: str, params: dict) -> str:
    """GET one page with exponential backoff on 429/5xx/timeouts; fail fast on other 4xx."""
    last = "no response"
    for i in range(RETRIES):
        try:
            r = sess.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in RETRY_STATUS:
                break
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = repr(exc)
        time.sleep(2 ** i)
    raise RuntimeError(f"SODA request failed for {url} ({last})")


def _write_cache(path, text: str) -> None:
    """Write a cache page atomically so an interrupted run never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _fixture(dataset_id: str, cols: list[str] | None, max_rows: int | None) -> Iterator[pd.DataFrame]:
    """Stream the synthetic CSV in 20k-row chunks (same str dtype as the API path)."""
    path = config.FIXTURE_DIR / FIXTURE_FILES[dataset_id]
    got = 0
    for chunk in pd.read_csv(path, dtype=str, chunksize=FIXTURE_CHUNK):
        if max_rows is not None:
            chunk = chunk.iloc[: max_rows - got]
        got += len(chunk)
        yield _align(chunk, cols)
        if max_rows is not None and got >= max_rows:
            return


def fetch_soda(dataset_id: str, select: str | Sequence[str] | None = None, where: str | None = None,
               page_size: int = 50_000, max_rows: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield all-str DataFrames; crashes use keyset paging on collision_id, housing one request."""
    cols = _cols(select)
    if config.FIXTURE:
        yield from _fixture(dataset_id, cols, max_rows)
        return
    keyset = "collision_id" if dataset_id == config.CRASH_ID else None
    if keyset and cols and keyset not in cols:
        cols = [*cols, keyset]
    limit = page_size if keyset else HOUSING_LIMIT
    tag = hashlib.md5(f"{cols}|{where}|{limit}".encode()).hexdigest()[:10]
    cache = config.RAW_DIR / f"{dataset_id}_{tag}"
    sess = requests.Session()
    if config.APP_TOKEN:
        sess.headers["X-App-Token"] = config.APP_TOKEN
    last, page, got = 0, 0, 0
    while True:
        f = cache / f"page_{page}.csv.gz"
        if f.exists():
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                text = fh.read()
        else:
            conds = [c for c in (where, f"{keyset}>{last}" if keyset else None) if c]
            params = {"$select": ",".join(cols) if cols else None, "$where": " AND ".join(conds) or None,
                      "$order": keyset or "building_id", "$limit": limit}
            text = _download(sess, config.SODA_URL.format(dataset_id),
                             {k: v for k, v in params.items() if v is not None})
            df = _read_text(text)
            if len(df) >= limit:      # a short page may be the live tail: never cache it (stale on rerun)
                _write_cache(f, text)
        df = _read_text(text)
        n_full = len(df)
        print(f"[extract] {dataset_id} page {page}: {n_full:,} rows", flush=True)
        if n_full == 0:
            return
        if max_rows is not None:
            df = df.iloc[: max_rows - got]
        got += len(df)
        yield _align(df, cols)
        # keyset: stop only on an empty page (proxies may truncate pages) or max_rows
        if keyset is None or (max_rows is not None and got >= max_rows):
            return
        last, page = _next_key(df, keyset, last), page + 1


def _next_key(df: pd.DataFrame, keyset: str, last: int) -> int:
    """Largest id on the page (non-numeric ids dropped); fail loudly if it does not advance."""
    ids = pd.to_numeric(df[keyset], errors="coerce").dropna()
    if ids.empty or int(ids.max()) <= last:
        raise RuntimeError(f"keyset paging on {keyset} did not advance past {last}")
    return int(ids.max())


def crash_chunks(max_rows: int | None = None) -> Callable[[], Iterator[pd.DataFrame]]:
    """Return a re-iterable factory over crash chunks (two passes re-read the cache)."""
    where = f"crash_date>='{config.Y_START}-01-01'"
    return lambda: fetch_soda(config.CRASH_ID, CRASH_COLS, where, max_rows=max_rows)


def housing_raw() -> pd.DataFrame:
    """Load the whole (small) housing dataset as one all-str DataFrame."""
    parts = list(fetch_soda(config.HOUSING_ID, HOUSING_COLS))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=HOUSING_COLS)
