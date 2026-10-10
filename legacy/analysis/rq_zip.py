"""ZIP / borough level relationships (RQ1, RQ2, RQ8, RQ9): housing completions vs crash outcomes."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sps

import config
from analysis import stats
from analysis.common import SEED, fmt_rho, insufficient, make_api
from analysis.data import UNIT_COL, Artifacts, FKey, cube_agg, zip_borough, zip_year
from app import charts as ch

MIN_FIRST3 = 30   # A11: growth ratio only for ZIPs with >= 30 crashes in the first 3 years


def _scatter_by_borough(fig: go.Figure, df: pd.DataFrame, x: str, y: str, hover: str, size: str = "units") -> None:
    """Add one marker trace per borough (Okabe-Ito colours) with bubble size ~ sqrt(size)."""
    for borough, group in df.groupby("borough"):
        fig.add_scatter(x=group[x], y=group[y], mode="markers", name=borough, customdata=group[["zip", "units", "crashes"]].to_numpy(),
                        marker=dict(size=np.clip(np.sqrt(group[size]) * 0.9 + 5, 5, 22), color=ch.BOROUGH_COLORS.get(borough, ch.OI["grey"]),
                                    opacity=.75, line=dict(width=.5, color="white")), hovertemplate=hover + "<extra>" + borough + "</extra>")


# ---------------------------------------------------------------- RQ1
@lru_cache(maxsize=24)
def _core1(art: Artifacts, fk: FKey, B: int) -> dict:
    """Spearman of units completed vs crash growth ratio ((last3+1)/(first3+1)) across ZIPs."""
    U, C, _ = zip_year(art, fk)
    if C.shape[1] < 6:
        return dict(stat=insufficient(0, "need at least 6 years in range"))
    first, last = C.iloc[:, :3].sum(axis=1), C.iloc[:, -3:].sum(axis=1)
    keep = first >= MIN_FIRST3
    df = pd.DataFrame(dict(zip=U.index, borough=zip_borough(art).reindex(U.index).values, units=U.sum(axis=1).values,
                           crashes=first.values + last.values, growth=((last + 1) / (first + 1)).values))[keep.values]
    if len(df) < config.MIN_N:
        return dict(stat=insufficient(len(df)))
    s = stats.spearman_boot(df.units.to_numpy(), df.growth.to_numpy(), B, SEED)     # Spearman: ranks, so a few huge ZIPs cannot dominate
    s["median_growth"] = float(df.growth.median())
    return dict(stat=s, df=df)


def _plot1(d: dict, title: str) -> go.Figure:
    df, s = d["df"], d["stat"]
    fig = go.Figure()
    _scatter_by_borough(fig, df.assign(x=df.units + 1), "x", "growth",
                        "ZIP %{customdata[0]}<br>Units completed: %{customdata[1]:,.0f}<br>Growth ratio: %{y:.2f}")
    fig.add_hline(y=1, line_dash="dot", line_color=ch.OI["grey"], annotation_text="no change", annotation_position="bottom right")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    note = f"n = {s['n']} ZIPs; Spearman ρ = {s['rho']:.2f} (95% CI {s['ci_lo']:.2f} to {s['ci_hi']:.2f}); ecological: ZIPs, not people"
    return ch.style(fig, title, "Affordable units completed in ZIP (+1, log scale)", "Crash growth ratio (last 3 ÷ first 3 yrs, log)", note)


# ---------------------------------------------------------------- RQ2
@lru_cache(maxsize=24)
def _core2(art: Artifacts, fk: FKey, B: int) -> dict:
    """Two-way fixed-effects slope of log1p(crashes) on units completed, plus an any-completion indicator."""
    U, C, _ = zip_year(art, fk)
    if U.shape[0] < config.MIN_N or U.shape[1] < 2:
        return dict(stat=insufficient(U.shape[0]))
    Y, X = np.log1p(C.to_numpy()), U.to_numpy()
    # log1p stabilises count skew and keeps ZIP-years with zero crashes; the slope is then roughly the proportional change per unit (x100 reported).
    cont = stats.within_slope(Y, X, B, SEED)
    ind = stats.within_slope(Y, (X > 0).astype(float), B, SEED + 1)
    s = dict(n=int(Y.size), n_zips=cont["n_zips"], slope=cont["slope"], ci_lo=cont["ci_lo"], ci_hi=cont["ci_hi"],
             pct_per_100_units=100 * np.expm1(100 * cont["slope"]), ind_slope=ind["slope"], ind_ci_lo=ind["ci_lo"], ind_ci_hi=ind["ci_hi"],
             ind_pct=100 * np.expm1(ind["slope"]))
    return dict(stat=s, x=stats.two_way_demean(X).ravel(), y=stats.two_way_demean(Y).ravel(),
                zips=np.repeat(U.index.to_numpy(), U.shape[1]), years=np.tile(U.columns.to_numpy(), U.shape[0]))


def _plot2(d: dict, title: str) -> go.Figure:
    s, x, y = d["stat"], d["x"], d["y"]
    xs = np.array([x.min(), x.max()])
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", name="ZIP-year", marker=dict(color=ch.OI["blue"], opacity=.35, size=5),
                                 customdata=np.c_[d["zips"], d["years"]],
                                 hovertemplate="ZIP %{customdata[0]}, %{customdata[1]}<br>units (net): %{x:.1f}<br>log crashes (net): %{y:.2f}<extra></extra>"))
    fig.add_scatter(x=xs, y=s["slope"] * xs, mode="lines", name="Within-ZIP OLS fit", line=dict(color=ch.OI["vermillion"], width=3))
    note = (f"n = {s['n']:,} ZIP-years in {s['n_zips']} ZIPs; slope {s['pct_per_100_units']:+.1f}% crashes per +100 units "
            f"(95% CI {100 * np.expm1(100 * s['ci_lo']):+.1f} to {100 * np.expm1(100 * s['ci_hi']):+.1f}); any-completion year: {s['ind_pct']:+.1f}%")
    return ch.style(fig, title, "Units completed, net of ZIP and year averages", "log(1 + crashes), net of ZIP and year averages", note)


# ---------------------------------------------------------------- RQ8
@lru_cache(maxsize=24)
def _core8(art: Artifacts, fk: FKey, B: int) -> dict:
    """Borough x year matrices: units completed and year-over-year % change in crashes (descriptive)."""
    panel = art.panel
    panel = panel[panel["year"].between(*fk.years).to_numpy() & (panel["borough"].isin(fk.boroughs).to_numpy() if fk.boroughs else True)]
    unit_col = "units_completed" if not fk.construction else None
    panel = panel.assign(u=panel[unit_col] if unit_col else sum(panel[UNIT_COL[ctype]] for ctype in fk.construction))
    U = panel.pivot_table(index="borough", columns="year", values="u", aggfunc="sum", observed=True, fill_value=0)
    cy = cube_agg(art, fk, ("borough", "year"))
    C = cy.pivot_table(index="borough", columns="year", values="crashes", aggfunc="sum").reindex(index=U.index.astype(str), columns=U.columns)
    C.index = U.index
    yoy = (C / C.shift(1, axis=1) - 1) * 100
    cells = yoy.notna().to_numpy().sum()
    if cells < config.MIN_N:
        return dict(stat=insufficient(cells))
    ok = yoy.notna().to_numpy()
    rho = sps.spearmanr(U.to_numpy()[ok], yoy.to_numpy()[ok])[0] if U.to_numpy()[ok].std() > 0 else np.nan
    top = U.stack().idxmax()
    s = dict(n=int(cells), rho=float(rho), total_units=int(U.to_numpy().sum()), top_borough=str(top[0]), top_year=int(top[1]),
             top_units=int(U.stack().max()))
    return dict(stat=s, U=U, yoy=yoy)


def _plot8(d: dict, title: str) -> go.Figure:
    U, yoy, s = d["U"], d["yoy"], d["stat"]
    txt = yoy.map(lambda v: "" if pd.isna(v) else f"{v:+.0f}%")
    hover = np.array([[f"{borough}, {year}<br>Units completed: {U.loc[borough, year]:,.0f}<br>Crashes YoY: {txt.loc[borough, year] or 'n/a'}" for year in U.columns] for borough in U.index])
    fig = go.Figure(go.Heatmap(z=U.to_numpy(), x=[str(y) for y in U.columns], y=list(U.index), text=txt.to_numpy(), texttemplate="%{text}",
                               colorscale=ch.SEQ, colorbar=dict(title="Units<br>completed"), hovertext=hover, hoverinfo="text"))
    note = (f"n = {s['n']} borough-years; colour = affordable units completed, text = year-over-year % change in crashes; "
            f"descriptive ({fmt_rho(s['rho'])}), no significance test")
    return ch.style(fig, title, "Year", "Borough", note)


# ---------------------------------------------------------------- RQ9
@lru_cache(maxsize=24)
def _core9(art: Artifacts, fk: FKey, B: int) -> dict:
    """Spearman of ZIP low-income share of completions vs injuries per crash."""
    panel = art.panel
    panel = panel[panel["year"].between(*fk.years).to_numpy() & (panel["borough"].isin(fk.boroughs).to_numpy() if fk.boroughs else True)]
    unit_totals = panel.groupby("zip", observed=True)[["eli_vli_units", "counted_units", "units_completed"]].sum()
    _, C, I = zip_year(art, fk)
    df = unit_totals.join(pd.DataFrame(dict(crashes=C.sum(axis=1), injured=I.sum(axis=1))), how="inner")
    df = df[(df.counted_units > 0) & (df.crashes >= MIN_FIRST3)]
    df = df.assign(share=df.eli_vli_units / df.counted_units, ipc=df.injured / df.crashes, units=df.units_completed)
    df["zip"], df["borough"] = df.index.astype(str), zip_borough(art).reindex(df.index).values
    if len(df) < config.MIN_N:
        return dict(stat=insufficient(len(df)))
    return dict(stat=stats.spearman_boot(df.share.to_numpy(), df.ipc.to_numpy(), B, SEED), df=df)


def _plot9(d: dict, title: str) -> go.Figure:
    df, s = d["df"], d["stat"]
    fig = go.Figure()
    _scatter_by_borough(fig, df.assign(x=df.share * 100), "x", "ipc",
                        "ZIP %{customdata[0]}<br>Extremely/very-low-income share: %{x:.0f}%<br>Injuries per crash: %{y:.3f}")
    note = f"n = {s['n']} ZIPs with ≥{MIN_FIRST3} crashes; Spearman ρ = {s['rho']:.2f} (95% CI {s['ci_lo']:.2f} to {s['ci_hi']:.2f}, p = {s['p']:.3f}); ecological"
    return ch.style(fig, title, "Share of ZIP's completed units for extremely/very-low income (%)", "Injuries per crash (persons)", note)


rq1_fig, rq1_stat = make_api(_core1, _plot1, "ZIP completions vs crash growth", "rq1")
rq2_fig, rq2_stat = make_api(_core2, _plot2, "Within-ZIP co-movement of completions and crashes", "rq2")
rq8_fig, rq8_stat = make_api(_core8, _plot8, "Borough-year completions and crash change", "rq8")
rq9_fig, rq9_stat = make_api(_core9, _plot9, "Low-income share of completions vs injuries per crash", "rq9")
