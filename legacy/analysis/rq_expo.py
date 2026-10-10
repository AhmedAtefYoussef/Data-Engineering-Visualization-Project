"""Exposure-based comparisons on the crash cube (RQ6, RQ10) and the relationship map (RQ7)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sps

import config
from analysis import stats
from analysis.common import fmt_rho, insufficient, make_api
from analysis.data import FACTORS, Artifacts, FKey, _isin, cube_agg
from app import charts as ch

EXPO_NOTE = "exposure 2 = within 250 m of a completed building; 0 = never near housing; placebo 1 = near a building not yet completed"


def _counts(art: Artifacts, fk: FKey, by: str, ignore: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crash counts (all, ped/cyc only) per `by` level x exposure (0,1,2); exposure -1 (no coordinates) dropped."""
    agg = cube_agg(art, fk, (by, "exposure", "ped_cyc"), ignore)
    agg = agg[agg["exposure"] >= 0]
    tot = agg.groupby([by, "exposure"])["crashes"].sum().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    pc = agg[agg["ped_cyc"] == 1].groupby([by, "exposure"])["crashes"].sum().unstack(fill_value=0).reindex(index=tot.index, columns=[0, 1, 2], fill_value=0)
    return tot, pc


# ---------------------------------------------------------------- RQ6
@lru_cache(maxsize=24)
def _core6(art: Artifacts, fk: FKey, B: int) -> dict:
    """Risk difference in ped/cyclist share of crashes, exposure 2 vs 0 (placebo 1 vs 0), by borough + pooled; Cramer's V."""
    tot, pc = _counts(art, fk, "borough", ("ped_cyc",))
    tot.loc["All boroughs"], pc.loc["All boroughs"] = tot.sum(), pc.sum()
    rows = []
    for name in tot.index:
        n_all, n_pc = tot.loc[name], pc.loc[name]      # crashes by exposure level (0, 1, 2): all, and harming pedestrians/cyclists
        if min(n_all[2], n_all[0]) < config.MIN_N:
            continue
        row = dict(borough=name, **stats.risk_diff(n_pc[2], n_all[2], n_pc[0], n_all[0]), share_exposed=float(n_pc[2] / n_all[2]), share_base=float(n_pc[0] / n_all[0]))
        if n_all[1] >= config.MIN_N:
            row["placebo"] = stats.risk_diff(n_pc[1], n_all[1], n_pc[0], n_all[0])["rd"]
        rows.append(row)
    pooled = next((r for r in rows if r["borough"] == "All boroughs"), None)
    if pooled is None:
        return dict(stat=insufficient(int(tot.loc["All boroughs", 2] + tot.loc["All boroughs", 0])))
    n_all, n_pc = tot.loc["All boroughs"], pc.loc["All boroughs"]
    v = stats.cramers_v(np.array([[n_pc[2], n_all[2] - n_pc[2]], [n_pc[0], n_all[0] - n_pc[0]]], float))
    s = dict(n=int(n_all[2] + n_all[0]), rd=pooled["rd"], ci_lo=pooled["ci_lo"], ci_hi=pooled["ci_hi"], cramers_v=v,
             placebo_rd=pooled.get("placebo"), by_borough=rows)
    return dict(stat=s, rows=rows)


def _plot6(d: dict, title: str) -> go.Figure:
    rows = sorted(d["rows"], key=lambda r: (r["borough"] != "All boroughs", r["borough"]), reverse=True)
    boroughs = [r["borough"] for r in rows]
    fig = go.Figure(go.Scatter(
        x=[r["rd"] * 100 for r in rows], y=boroughs, mode="markers", name="Near completed housing vs never near", marker=dict(size=11, color=ch.TREATED),
        error_x=dict(type="data", symmetric=False, array=[(r["ci_hi"] - r["rd"]) * 100 for r in rows], arrayminus=[(r["rd"] - r["ci_lo"]) * 100 for r in rows], color=ch.OI["black"]),
        customdata=[[r["share_exposed"] * 100, r["share_base"] * 100, r["n1"], r["n0"]] for r in rows],
        hovertemplate="%{y}<br>Risk difference %{x:+.2f} pp<br>ped/cyc share: %{customdata[0]:.1f}% vs %{customdata[1]:.1f}%<br>n = %{customdata[2]:,} vs %{customdata[3]:,}<extra></extra>"))
    pl = [r for r in rows if "placebo" in r]
    fig.add_scatter(x=[r["placebo"] * 100 for r in pl], y=[r["borough"] for r in pl], mode="markers", name="Placebo: near not-yet-built housing",
                    marker=dict(size=10, symbol="diamond-open", color=ch.PLACEBO, line=dict(width=2)), hovertemplate="%{y}<br>placebo RD %{x:+.2f} pp<extra></extra>")
    fig.add_vline(x=0, line_color=ch.OI["grey"])
    s = d["stat"]
    note = f"n = {s['n']:,} crashes; pooled Cramér's V = {s['cramers_v']:.3f}; {EXPO_NOTE}"
    return ch.style(fig, title, "Difference in share of crashes harming pedestrians/cyclists (percentage points, 95% CI)", "Borough", note)


# ---------------------------------------------------------------- RQ10
@lru_cache(maxsize=24)
def _core10(art: Artifacts, fk: FKey, B: int) -> dict:
    """Contributing-factor log-lift, exposure 2 vs 0 (placebo 1 vs 0), delta-method CIs."""
    tot, _ = _counts(art, fk, "factor_group", ("factor",))
    tot = tot.reindex([f for f in FACTORS if f in tot.index])
    n = tot.sum()                                      # total crashes per exposure level
    if min(n[2], n[0]) < config.MIN_N:
        return dict(stat=insufficient(int(n[2] + n[0])))
    lift = stats.log_lift(tot[2].to_numpy(float), n[2], tot[0].to_numpy(float), n[0])
    placebo = stats.log_lift(tot[1].to_numpy(float), max(n[1], 1), tot[0].to_numpy(float), n[0])
    df = pd.DataFrame(dict(factor=tot.index, lift=lift["lift"], lo=lift["lo"], hi=lift["hi"], placebo=placebo["lift"], c2=tot[2].values, c0=tot[0].values)).dropna(subset=["lift"])
    sig = df[(df.lo > 1) | (df.hi < 1)]
    # The headline factor comes from the significant set (CI excludes 1). Only when none is significant do we fall back to the
    # largest point estimate, flagged so the text says it cannot be told apart from 1.
    pool = sig if len(sig) else df
    top = pool.loc[np.log(pool.lift).abs().idxmax()]
    s = dict(n=int(n[2] + n[0]), factors=df.to_dict("records"), n_significant=len(sig), top_factor=str(top.factor),
             top_lift=float(top.lift), top_lo=float(top.lo), top_hi=float(top.hi), top_is_significant=bool(len(sig)))
    return dict(stat=s, df=df)


def _plot10(d: dict, title: str) -> go.Figure:
    df = d["df"].sort_values("lift")
    fig = go.Figure(go.Bar(
        y=df.factor, x=df.lift - 1, base=1, orientation="h", name="Near completed housing / never near",
        marker_color=[ch.TREATED if (lo > 1 or hi < 1) else ch.OI["sky"] for lo, hi in zip(df.lo, df.hi)],
        error_x=dict(type="data", symmetric=False, array=df.hi - df.lift, arrayminus=df.lift - df.lo, color=ch.OI["black"]),
        customdata=np.c_[df.lift, df.lo, df.hi, df.c2], hovertemplate="%{y}<br>lift %{customdata[0]:.2f} (95% CI %{customdata[1]:.2f} to %{customdata[2]:.2f})<br>%{customdata[3]:,} crashes near housing<extra></extra>"))
    fig.add_scatter(y=df.factor, x=df.placebo, mode="markers", name="Placebo (not-yet-built)", marker=dict(symbol="diamond-open", size=9, color=ch.PLACEBO, line=dict(width=2)),
                    hovertemplate="%{y}<br>placebo lift %{x:.2f}<extra></extra>")
    fig.add_vline(x=1, line_color=ch.OI["grey"])
    fig.update_xaxes(type="log")
    s = d["stat"]
    note = f"n = {s['n']:,} crashes; dark bars: 95% CI excludes 1; lift = P(factor | near) ÷ P(factor | not near); delta-method CI, crashes treated as independent"
    return ch.style(fig, title, "Lift in share of crashes (1 = same share; log scale)", "Contributing-factor group", note)


# ---------------------------------------------------------------- RQ7
def _centre(x: pd.Series, gd: float) -> pd.Series:
    """Grid-cell centre (floor index + 0.5) * gd: identical to pipeline.integrate.finish_grid so the merge keys match."""
    return ((np.floor(x / gd) + 0.5) * gd).round(4)


@lru_cache(maxsize=24)
def _core7(art: Artifacts, fk: FKey, B: int) -> dict:
    """Cell-level Spearman between crash counts and housing units on the 0.005-degree grid (descriptive map)."""
    grid, housing, cell_deg = art.grid, art.housing, config.GRID_DEG
    grid_mask = grid["year"].between(*fk.years).to_numpy() & _isin(grid["borough"], fk.boroughs) & _isin(grid["severity"], fk.severity)
    measure = "ped_cyc_crashes" if fk.ped_cyc else "crashes"
    cells = grid[grid_mask].groupby(["cell_lat", "cell_lon"], observed=True)[measure].sum().reset_index(name="crashes")
    housing_mask = _isin(housing["borough"], fk.boroughs) & housing["comp_year"].between(*fk.years).to_numpy() & _isin(housing["construction_type"], fk.construction)
    buildings = housing[housing_mask & housing["latitude"].notna().to_numpy()]
    if len(cells) < config.MIN_N or len(buildings) < 1:
        return dict(stat=insufficient(len(cells)))
    units_by_cell = buildings.assign(cell_lat=_centre(buildings.latitude, cell_deg), cell_lon=_centre(buildings.longitude, cell_deg)).groupby(["cell_lat", "cell_lon"])["total_units"].sum().reset_index()
    # Left-merge keeps every crash cell; cells with no housing count as 0 units, which is what the rank correlation should see.
    merged = cells.merge(units_by_cell, how="left", on=["cell_lat", "cell_lon"]).fillna({"total_units": 0})
    rho, p = sps.spearmanr(merged.crashes, merged.total_units) if merged.total_units.std() > 0 else (np.nan, np.nan)
    s = dict(n=len(merged), rho=float(rho), p=float(p), n_buildings=len(buildings), units=int(buildings.total_units.sum()), share_cells_with_housing=float((merged.total_units > 0).mean()))
    return dict(stat=s, cells=cells, b=buildings)


def _plot7(d: dict, title: str) -> go.Figure:
    cells, b, s = d["cells"], d["b"], d["stat"]       # b = buildings plotted
    fig = go.Figure(ch.density_map(lat=cells.cell_lat, lon=cells.cell_lon, z=cells.crashes, radius=14, name="Crash density", showscale=False,
                                   colorscale=[[0, "rgba(0,0,0,0)"], [.4, "rgba(213,94,0,.35)"], [1, "rgba(213,94,0,.85)"]], hoverinfo="skip"))
    fig.add_trace(ch.scatter_map(
        lat=b.latitude, lon=b.longitude, mode="markers", name="Completed housing",
        marker=dict(size=np.clip(np.sqrt(b.total_units) * 1.6 + 4, 5, 26), color=b.comp_year, colorscale=ch.SEQ, showscale=True, opacity=.85,
                    colorbar=dict(title="Completed", thickness=10, len=.6)),
        customdata=np.c_[b.project_name.astype(str) if "project_name" in b else b.building_id, b.total_units, b.comp_year, b.construction_type.astype(str)],
        hovertemplate="%{customdata[0]}<br>%{customdata[1]} units, completed %{customdata[2]:.0f}<br>%{customdata[3]}<extra></extra>"))
    ch.add_mapbox(fig)
    note = f"n = {s['n']:,} grid cells, {s['n_buildings']:,} buildings; orange = crash density, circles = housing (size = units, colour = completion year); cell {fmt_rho(s['rho'])}: both concentrate in dense areas"
    return ch.style(fig, title, note=note, height=430)


rq6_fig, rq6_stat = make_api(_core6, _plot6, "Pedestrian/cyclist share of crashes near housing", "rq6")
rq7_fig, rq7_stat = make_api(_core7, _plot7, "Where housing and crashes overlap", "rq7")
rq10_fig, rq10_stat = make_api(_core10, _plot10, "Contributing factors near new housing", "rq10")
