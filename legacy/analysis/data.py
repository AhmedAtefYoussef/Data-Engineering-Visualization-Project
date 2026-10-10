"""Artifact loading (lazy, categorical), filter normalisation and cached aggregations over the crash cube."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

CUBE_CATS = ["zip", "borough", "hour_band", "factor_group", "severity"]
REQUIRED = ["housing_clean", "crash_cube", "crash_grid", "event_building", "panel_zip_year", "meta"]
FACTORS = ["Distraction/Inattention", "Speed/Aggressive", "Failure to Yield/Signal", "Impairment/Fatigue",
           "Vehicle/Road Defect", "Other", "Unspecified"]
SEVERITIES = ["Fatal", "Injury", "Property damage only"]
CONSTRUCTION = ["New Construction", "Preservation", "Other"]
HOUR_BANDS = ["Night", "Morning", "Afternoon", "Evening", "Unknown"]
UNIT_COL = {"New Construction": "new_con_units", "Preservation": "preservation_units", "Other": "other_units"}
FILTER_LABELS = dict(boroughs="Borough", years="Years", construction="Construction", factor="Contributing factor",
                     severity="Severity", hours="Time of day", ped_cyc="Pedestrian and cyclist only")


class FKey(NamedTuple):
    """Hashable, normalised filter set (used as cache key)."""
    boroughs: tuple
    years: tuple
    construction: tuple
    factor: tuple
    severity: tuple
    hours: tuple
    ped_cyc: bool


def default_filters() -> dict:
    """Filters meaning 'everything' (empty list = no restriction)."""
    return dict(boroughs=[], years=(config.Y_START, config.Y_END), construction=[], factor=[], severity=[], hours=[], ped_cyc=False)


def fkey(filters: dict | FKey | None) -> FKey:
    """Normalise a filters dict (missing keys = defaults, None/empty = all) into a hashable FKey."""
    if isinstance(filters, FKey):
        return filters
    f = {**default_filters(), **(filters or {})}
    lo, hi = sorted(int(y) for y in (f["years"] or (config.Y_START, config.Y_END)))
    t = lambda v: tuple(sorted(v or ()))  # noqa: E731
    return FKey(t(f["boroughs"]), (lo, hi), t(f["construction"]), t(f["factor"]), t(f["severity"]), t(f["hours"]), bool(f["ped_cyc"]))


def scope_text(fk: FKey) -> str:
    """Short human description of the active filters, for chart titles."""
    parts = [", ".join(fk.boroughs) or "All boroughs", f"{fk.years[0]} to {fk.years[1]}"]
    for v in (fk.construction, fk.factor, fk.severity, fk.hours):
        if v:
            parts.append(", ".join(v))
    if fk.ped_cyc:
        parts.append("ped/cyclist crashes")
    return "; ".join(parts)


def _read(path: Path, cats: list[str] | None = None) -> pd.DataFrame:
    """Read one parquet file with string columns as categoricals (memory)."""
    return pq.read_table(path).to_pandas(categories=cats or [])


@dataclass(eq=False)
class Artifacts:
    """Lazy handle on the processed parquet/json artifacts in one directory (identity-hashed for caching)."""
    dir: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Artifacts":
        """Open the artifact directory (default config.PROCESSED_DIR)."""
        return cls(Path(path) if path else Path(config.PROCESSED_DIR))

    @property
    def available(self) -> bool:
        """True when every required artifact file exists."""
        return all((self.dir / f"{n}.{'json' if n == 'meta' else 'parquet'}").exists() for n in REQUIRED)

    @cached_property
    def meta(self) -> dict:
        p = self.dir / "meta.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @cached_property
    def rq_results(self) -> dict:
        p = self.dir / "rq_results.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @cached_property
    def cube(self) -> pd.DataFrame:
        c = _read(self.dir / "crash_cube.parquet", CUBE_CATS)
        c["year"] = c["year"].astype("int16")
        for col in ("exposure", "ped_cyc"):
            c[col] = c[col].astype("int8")
        return c

    @cached_property
    def housing(self) -> pd.DataFrame:
        h = _read(self.dir / "housing_clean.parquet", ["borough", "construction_type", "size_bin"])
        return h[h["completed"].astype(bool)] if "completed" in h else h

    @cached_property
    def grid(self) -> pd.DataFrame:
        return _read(self.dir / "crash_grid.parquet", ["borough", "severity"])

    @cached_property
    def event(self) -> pd.DataFrame:
        return _read(self.dir / "event_building.parquet", ["group", "borough", "construction_type", "size_bin"])

    @cached_property
    def panel(self) -> pd.DataFrame:
        return _read(self.dir / "panel_zip_year.parquet", ["borough"])

    @property
    def y_range(self) -> tuple[int, int]:
        return int(self.meta.get("Y_START", config.Y_START)), int(self.meta.get("Y_END", config.Y_END))


def _isin(s: pd.Series, vals: tuple):
    """Membership mask; an empty selection means 'no restriction'."""
    return s.isin(vals).to_numpy() if vals else np.ones(len(s), bool)


def cube_mask(cube: pd.DataFrame, fk: FKey, ignore: tuple = ()) -> np.ndarray:
    """Boolean mask for cube rows under the filters, skipping any listed in `ignore`."""
    m = np.ones(len(cube), bool)
    if "years" not in ignore:
        m &= cube["year"].between(*fk.years).to_numpy()
    for name, col, vals in (("boroughs", "borough", fk.boroughs), ("factor", "factor_group", fk.factor),
                            ("severity", "severity", fk.severity), ("hours", "hour_band", fk.hours)):
        if name not in ignore:
            m &= _isin(cube[col], vals)
    if fk.ped_cyc and "ped_cyc" not in ignore:
        m &= cube["ped_cyc"].to_numpy() == 1
    return m


@lru_cache(maxsize=96)
def cube_agg(art: Artifacts, fk: FKey, by: tuple, ignore: tuple = ()) -> pd.DataFrame:
    """Cube aggregated by `by` under filters (cached on the filter key); ignored filters are not applied."""
    c = art.cube
    meas = ["crashes", "injured", "killed", "ped_injured", "cyc_injured"]
    out = c[cube_mask(c, fk, ignore)].groupby(list(by), observed=True)[meas].sum().reset_index()
    for col in by:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
    return out


def housing_mask(h: pd.DataFrame, fk: FKey) -> np.ndarray:
    """Mask for completed buildings under borough / completion-year / construction filters."""
    return _isin(h["borough"], fk.boroughs) & h["comp_year"].between(*fk.years).to_numpy() & _isin(h["construction_type"], fk.construction)


@lru_cache(maxsize=48)
def zip_year(art: Artifacts, fk: FKey) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Balanced ZIP x year frames (units completed, crashes, injured) under filters; crashes 0-filled."""
    p = art.panel
    p = p[_isin(p["borough"], fk.boroughs) & p["year"].between(*fk.years).to_numpy()]
    u = p["units_completed"] if not fk.construction else sum(p[UNIT_COL[ctype]] for ctype in fk.construction)
    U = p.assign(u=u).pivot_table(index="zip", columns="year", values="u", aggfunc="sum", fill_value=0)
    cy = cube_agg(art, fk, ("zip", "year"))

    def piv(col: str) -> pd.DataFrame:
        return cy.pivot_table(index="zip", columns="year", values=col, aggfunc="sum").reindex(index=U.index, columns=U.columns).fillna(0)
    return U.astype(float), piv("crashes"), piv("injured")


@lru_cache(maxsize=4)
def zip_borough(art: Artifacts) -> pd.Series:
    """ZIP -> borough lookup from the panel."""
    return art.panel.drop_duplicates("zip").set_index("zip")["borough"].astype(str)


def apply_filters(art: Artifacts, filters: dict | FKey | None) -> dict[str, pd.DataFrame]:
    """Filtered views {cube, housing, grid, panel} per the SPEC A10 filter contract (event table: see rq._sites)."""
    fk = fkey(filters)
    c, h, g, p = art.cube, art.housing, art.grid, art.panel
    gm = g["year"].between(*fk.years).to_numpy() & _isin(g["borough"], fk.boroughs) & _isin(g["severity"], fk.severity)
    pm = p["year"].between(*fk.years).to_numpy() & _isin(p["borough"], fk.boroughs)
    return dict(cube=c[cube_mask(c, fk)], housing=h[housing_mask(h, fk)], grid=g[gm], panel=p[pm])
