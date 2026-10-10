"""Site-level event study with a difference-in-differences effect (RQ3, RQ4, RQ5; SPEC A4/A11)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as sps

import config
from analysis import stats
from analysis.common import SEED, insufficient, make_api
from analysis.data import CONSTRUCTION, Artifacts, FKey, _isin
from app import charts as ch

REL = [-3, -2, -1, 1, 2, 3]
RCOLS = [f"r{r}" for r in REL]
OUTC = RCOLS + ["d"]
BINS = ["1-10", "11-50", "51-150", "151+"]


@lru_cache(maxsize=16)
def sites(art: Artifacts, fk: FKey) -> pd.DataFrame:
    """One row per (group, cohort, site) with complete rel-year adj means, pre/post, change d and stratum id.

    Filters (A10): borough applies to treated and control; completion-year range and construction type
    apply to treated buildings only (controls are the not-yet-built comparison pool)."""
    events = art.event
    events = events[events["rel_year"].to_numpy() != 0]
    treated = (events["group"] == "treated").to_numpy()
    keep = _isin(events["borough"], fk.boroughs) & (~treated | (events["cohort"].between(*fk.years).to_numpy() & _isin(events["construction_type"], fk.construction)))
    events = events[keep]
    key = ["group", "cohort", "site_id"]
    # Site-level mean per relative year; dropna keeps only sites observed in all six pre/post years.
    wide = events.groupby(key + ["rel_year"], observed=True)["adj"].mean().unstack("rel_year").reindex(columns=REL).dropna()
    wide.columns = RCOLS
    buildings = events.drop_duplicates(["group", "cohort", "building_id"])
    attrs = buildings.groupby(key, observed=True).agg(borough=("borough", "first"), ctype=("construction_type", "first"),
                                                      units=("total_units", "sum"), contaminated=("contaminated", "max"))
    S = wide.join(attrs).reset_index()
    S["group"] = S["group"].astype(str)
    S["borough"], S["ctype"] = S["borough"].astype(str), S["ctype"].astype(str)
    # post-pre change in adjusted crashes (rel -3..-1 vs +1..+3); rel 0 (completion year) is excluded by design
    S["d"] = S[RCOLS[3:]].mean(axis=1) - S[RCOLS[:3]].mean(axis=1)
    S["size_bin"] = pd.cut(S["units"], [0, 10, 50, 150, np.inf], labels=BINS).astype(str)
    S["st"] = stratum(S["cohort"].to_numpy(), S["borough"])   # stratum = cohort x borough
    return S


def stratum(cohort: np.ndarray, borough: pd.Series) -> np.ndarray:
    """Stratum id = cohort x borough; a borough outside config.BOROUGHS ('Unknown') gets its own slot."""
    nb = len(config.BOROUGHS)
    bidx = borough.map({name: i for i, name in enumerate(config.BOROUGHS)}).fillna(nb).astype(int).to_numpy()
    cohort_year = np.asarray(cohort).astype(int)
    return (cohort_year - (cohort_year.min() if len(cohort_year) else 0)) * (nb + 1) + bidx


def usable_cohorts(S: pd.DataFrame) -> set:
    """Cohorts with at least MIN_N distinct comparison sites. Late cohorts (2021-2022) need buildings completed 2025 or later,
    which can be too few; for those a DiD would rest on a handful of sites, so they are not compared."""
    counts = S[S["group"] == "control"].groupby("cohort")["site_id"].nunique()
    return set(counts[counts >= config.MIN_N].index)


def _did(S: pd.DataFrame, tsel: pd.Series | None, B: int, cols: list[str], seed: int = SEED) -> dict | None:
    """Bootstrapped treated / control / DiD for the treated subset `tsel`, using only cohorts that have enough comparison
    sites; None if either arm then has < MIN_N sites."""
    ok = usable_cohorts(S)
    T = S[(S["group"] == "treated") & (tsel if tsel is not None else True) & S["cohort"].isin(ok)]
    C = S[(S["group"] == "control") & S["cohort"].isin(ok)]
    if len(T) < config.MIN_N or len(C) < config.MIN_N:
        return None
    res = stats.did_boot(T[cols].to_numpy(float), T["st"].to_numpy(), C[cols].to_numpy(float), C["st"].to_numpy(),
                         int(S["st"].max()) + 1, B, seed, T["site_id"].to_numpy(), C["site_id"].to_numpy())
    res.update(n_t=len(T), n_c=len(C))
    return res


def _flat(r: dict, i: int = -1) -> dict:
    """Pull the DiD effect (last column) out of a bootstrap result as plain floats."""
    return dict(did=float(r["diff"]["est"][i]), ci_lo=float(r["diff"]["lo"][i]), ci_hi=float(r["diff"]["hi"][i]),
                treated_change=float(r["treated"]["est"][i]), control_change=float(r["control"]["est"][i]), n_treated=r["n_t"], n_control=r["n_c"])


# ---------------------------------------------------------------- RQ3
@lru_cache(maxsize=24)
def _core3(art: Artifacts, fk: FKey, B: int) -> dict:
    """Treated vs borough-reweighted control event-study lines and the pooled DiD (all sites, plus no-contamination sensitivity)."""
    S = sites(art, fk)
    treated = S[S["group"] == "treated"]
    if len(treated) < config.MIN_N:
        return dict(stat=insufficient(len(treated)))
    missing = sorted(int(cohort) for cohort in set(treated["cohort"]) - usable_cohorts(S))   # treated cohorts without enough comparison sites
    r = _did(S, None, B, OUTC)
    if r is None:
        # No usable comparison group: report treated-only pre/post, with an honest note, instead of a CI that has no basis.
        tb = stats.treated_boot(treated[OUTC].to_numpy(float), treated["site_id"].to_numpy(), B, SEED)
        i_d = OUTC.index("d")
        s = dict(n=len(treated), did_unavailable=True, n_treated=len(treated), n_control=int(S.loc[S["group"] == "control", "site_id"].nunique()),
                 reason=f"fewer than {config.MIN_N} comparison sites were available for each completion-year cohort",
                 treated_pre=float(tb["est"][:3].mean()), treated_post=float(tb["est"][3:6].mean()), treated_change=float(tb["est"][i_d]),
                 t_ci_lo=float(tb["lo"][i_d]), t_ci_hi=float(tb["hi"][i_d]), cohorts_without_control=missing)
        return dict(stat=s, treated_only=tb)
    s = {**_flat(r), "n": r["n_t"], "cohorts_without_control": missing}
    sens = _did(S[~S["contaminated"].astype(bool)], None, max(B // 2, 50), ["d"])
    s["did_excl_contaminated"] = None if sens is None else _flat(sens)
    return dict(stat=s, r=r)


def _band(fig: go.Figure, x: list, est: np.ndarray, lo: np.ndarray, hi: np.ndarray, name: str, color: str) -> None:
    """Line with a translucent 95% CI ribbon."""
    rgb = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.18)"
    fig.add_scatter(x=x + x[::-1], y=list(hi) + list(lo[::-1]), mode="lines", fill="toself", fillcolor=rgb, line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=x, y=est, mode="lines+markers", name=name, line=dict(color=color, width=2.5), marker=dict(size=7),
                    customdata=np.c_[lo, hi], hovertemplate=name + "<br>rel. year %{x}<br>adj. crashes %{y:.2f} (95% CI %{customdata[0]:.2f} to %{customdata[1]:.2f})<extra></extra>")


def _plot3(d: dict, title: str) -> go.Figure:
    s = d["stat"]
    fig = go.Figure()
    if s.get("did_unavailable"):
        tb = d["treated_only"]
        _band(fig, REL, tb["est"][:6], tb["lo"][:6], tb["hi"][:6], "Sites with completion (no comparison)", ch.TREATED)
        fig.add_vline(x=0, line_dash="dot", line_color=ch.OI["grey"], annotation_text="completion year (excluded)", annotation_position="top right")
        fig.add_annotation(text="DiD unavailable: too few comparison sites", xref="paper", yref="paper", x=.99, y=.04,
                           showarrow=False, xanchor="right", font=dict(size=12, color=ch.OI["black"]), bgcolor="rgba(255,255,255,.8)")
        note = (f"n = {s['n_treated']} treated sites, {s['n_control']} comparison sites; DiD unavailable ({s['reason']}); "
                "treated sites only, so the line may reflect citywide trends")
        return ch.style(fig, title, "Years relative to completion", "Adjusted crashes within 250 m (per building-year)", note)
    r = d["r"]
    for key, name, col in (("control", "Comparison sites (not yet built, reweighted)", ch.CONTROL), ("treated", "Sites with completion", ch.TREATED)):
        _band(fig, REL, r[key]["est"][:6], r[key]["lo"][:6], r[key]["hi"][:6], name, col)
    fig.add_vline(x=0, line_dash="dot", line_color=ch.OI["grey"], annotation_text="completion year (excluded)", annotation_position="top right")
    fig.add_annotation(text=f"DiD = {s['did']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]", xref="paper", yref="paper", x=.99, y=.04,
                       showarrow=False, xanchor="right", font=dict(size=12, color=ch.OI["black"]), bgcolor="rgba(255,255,255,.8)")
    left_out = f"; cohorts {', '.join(map(str, s['cohorts_without_control']))} omitted (too few comparison sites)" if s.get("cohorts_without_control") else ""
    note = f"n = {s['n_treated']} treated / {s['n_control']} comparison sites; bands: site-cluster bootstrap 95% CI; association, trend-adjusted by city index{left_out}"
    return ch.style(fig, title, "Years relative to completion", "Adjusted crashes within 250 m (per building-year)", note)


# ---------------------------------------------------------------- RQ4 / RQ5
def _did_by(S: pd.DataFrame, col: str, levels: list[str], B: int) -> list[dict]:
    """DiD per level of a site attribute (controls = full comparison pool); levels with < MIN_N treated sites carry n only."""
    out = []
    for lv in levels:
        sel = S[col] == lv
        r = _did(S, sel, B, ["d"])
        out.append(dict(level=lv, n=int(((S["group"] == "treated") & sel).sum()), **({} if r is None else _flat(r, 0))))
    return out


@lru_cache(maxsize=24)
def _core4(art: Artifacts, fk: FKey, B: int) -> dict:
    """Dose-response: DiD by site size bin, plus Spearman of log(units) vs change among treated sites."""
    S = sites(art, fk)
    rows = _did_by(S, "size_bin", BINS, B)
    n = sum(r["n"] for r in rows)
    if not any("did" in r for r in rows):
        return dict(stat=insufficient(n, f"fewer than {config.MIN_N} treated or comparison sites per cohort for a difference-in-differences"))
    T = S[S["group"] == "treated"]
    rho, p = sps.spearmanr(np.log1p(T["units"]), T["d"])
    return dict(stat=dict(n=n, bins=rows, trend_rho=float(rho), trend_p=float(p)), rows=rows)


def _plot4(d: dict, title: str) -> go.Figure:
    rows = [r for r in d["rows"] if "did" in r]
    est = np.array([r["did"] for r in rows])
    fig = go.Figure(go.Bar(x=[r["level"] for r in rows], y=est, marker_color=ch.OI["blue"],
                           error_y=dict(type="data", symmetric=False, array=[r["ci_hi"] - r["did"] for r in rows],
                                        arrayminus=[r["did"] - r["ci_lo"] for r in rows], color=ch.OI["black"]),
                           customdata=[[r["n"], r["ci_lo"], r["ci_hi"]] for r in rows],
                           hovertemplate="%{x} units<br>DiD %{y:+.2f} (95% CI %{customdata[1]:+.2f} to %{customdata[2]:+.2f})<br>%{customdata[0]} sites<extra></extra>"))
    fig.add_hline(y=0, line_color=ch.OI["grey"])
    s = d["stat"]
    skipped = [f"{r['level']} (n={r['n']})" for r in d["rows"] if "did" not in r]
    note = f"n = {s['n']} treated sites; error bars: site-cluster bootstrap 95% CI; size trend ρ = {s['trend_rho']:.2f}" + (f"; suppressed (<{config.MIN_N}): {', '.join(skipped)}" if skipped else "")
    return ch.style(fig, title, "Site size (total units)", "DiD: change in adjusted crashes vs comparison", note)


@lru_cache(maxsize=24)
def _core5(art: Artifacts, fk: FKey, B: int) -> dict:
    """DiD by construction type; Preservation is a quasi-placebo (no new structure)."""
    S = sites(art, fk)
    rows = _did_by(S, "ctype", CONSTRUCTION, B)
    n = sum(r["n"] for r in rows)
    ok = {r["level"]: r for r in rows if "did" in r}
    if not ok:
        return dict(stat=insufficient(n, f"fewer than {config.MIN_N} treated or comparison sites per cohort for a difference-in-differences"))
    s = dict(n=n, types=rows)
    if "New Construction" in ok and "Preservation" in ok:
        a, b = ok["New Construction"], ok["Preservation"]
        se = lambda r: (r["ci_hi"] - r["ci_lo"]) / (2 * stats.Z95)   # noqa: E731
        diff, sd = a["did"] - b["did"], float(np.hypot(se(a), se(b)))  # independent-arm approximation (shared controls ignored)
        s.update(diff=diff, diff_ci_lo=diff - stats.Z95 * sd, diff_ci_hi=diff + stats.Z95 * sd)
    return dict(stat=s, rows=rows)


def _plot5(d: dict, title: str) -> go.Figure:
    rows = [r for r in d["rows"] if "did" in r]
    fig = go.Figure(go.Scatter(
        y=[r["level"] for r in rows], x=[r["did"] for r in rows], mode="markers", marker=dict(size=12, color=[ch.OI["vermillion"] if r["level"] == "New Construction" else ch.OI["sky"] for r in rows]),
        error_x=dict(type="data", symmetric=False, array=[r["ci_hi"] - r["did"] for r in rows], arrayminus=[r["did"] - r["ci_lo"] for r in rows], color=ch.OI["black"]),
        customdata=[[r["n"], r["ci_lo"], r["ci_hi"]] for r in rows],
        hovertemplate="%{y}<br>DiD %{x:+.2f} (95% CI %{customdata[1]:+.2f} to %{customdata[2]:+.2f})<br>%{customdata[0]} sites<extra></extra>"))
    fig.add_vline(x=0, line_color=ch.OI["grey"])
    s = d["stat"]
    note = f"n = {s['n']} treated sites; Preservation adds no new structure: a quasi-placebo" + (f"; New construction minus preservation = {s['diff']:+.2f} [{s['diff_ci_lo']:+.2f}, {s['diff_ci_hi']:+.2f}]" if "diff" in s else "")
    return ch.style(fig, title, "DiD: change in adjusted crashes vs comparison sites", "Construction type", note, height=400)


rq3_fig, rq3_stat = make_api(_core3, _plot3, "Event study: crashes near new completions vs comparison sites", "rq3")
rq4_fig, rq4_stat = make_api(_core4, _plot4, "Dose-response by building size (DiD)", "rq4")
rq5_fig, rq5_stat = make_api(_core5, _plot5, "New construction vs preservation (DiD)", "rq5")
