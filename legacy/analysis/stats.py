"""Statistical helpers: bootstraps, difference-in-differences, Spearman, proportion and lift CIs."""
from __future__ import annotations

import numpy as np
from scipy import stats

Z95 = 1.959964


def treated_boot(dt: np.ndarray, site_t: np.ndarray, B: int, seed: int) -> dict:
    """Plain (unweighted) column means of treated sites with a site-cluster percentile CI; used when no comparison group exists.

    Resampling whole sites, not rows, keeps the interval honest when one site contributes several rows."""
    groups = _clusters(site_t, len(dt))
    rng = np.random.default_rng(seed)
    reps = np.array([dt[_draw(rng, groups)].mean(axis=0) for _ in range(B)])
    lo, hi = pct_ci(reps)
    return dict(est=dt.mean(axis=0), lo=lo, hi=hi)


def pct_ci(samples: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Percentile 95% interval of bootstrap replicates."""
    return np.nanpercentile(samples, 2.5, axis=axis), np.nanpercentile(samples, 97.5, axis=axis)


def _rank(a: np.ndarray) -> np.ndarray:
    """Row-wise average ranks of a 2D array (so Spearman = Pearson of ranks)."""
    return stats.rankdata(a, axis=-1)


def spearman_boot(x: np.ndarray, y: np.ndarray, B: int, seed: int) -> dict:
    """Spearman rho with asymptotic p and a percentile CI from resampling units with replacement."""
    n = len(x)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    # Spearman is Pearson on ranks. Ranking all B resamples at once (one 2D rankdata call) is far faster than calling
    # spearmanr B times, and the Pearson formula below then runs vectorised across the resamples.
    xr, yr = _rank(x[idx]), _rank(y[idx])
    xr, yr = xr - xr.mean(1, keepdims=True), yr - yr.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xr * yr).sum(1) / np.sqrt((xr ** 2).sum(1) * (yr ** 2).sum(1))
    lo, hi = pct_ci(r)
    return dict(n=int(n), rho=float(rho), p=float(p), ci_lo=float(lo), ci_hi=float(hi))


def two_way_demean(m: np.ndarray) -> np.ndarray:
    """Remove row (ZIP) and column (year) means of a balanced panel: m - row - col + grand.

    This is the two-way fixed-effects "within" transform. Subtracting both sets of means strips out anything constant
    within a ZIP (size, street layout) or common to a year (citywide traffic swings), so a slope on the demeaned
    values only uses variation within a ZIP over time. Adding the grand mean back corrects for subtracting it twice."""
    return m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + m.mean()


def within_slope(y: np.ndarray, x: np.ndarray, B: int, seed: int) -> dict:
    """Two-way fixed-effects slope of y on x (ZIP x year matrices) with a ZIP-cluster bootstrap CI."""
    def slope(yy: np.ndarray, xx: np.ndarray) -> float:
        xt, yt = two_way_demean(xx), two_way_demean(yy)
        denom = (xt ** 2).sum()
        return float((xt * yt).sum() / denom) if denom > 0 else np.nan
    rng = np.random.default_rng(seed)
    # Cluster bootstrap: resample whole ZIPs (rows) so the within-ZIP serial correlation stays intact; re-demeaning inside
    # slope() matters because the resampled panel has different ZIP and year means.
    boot_slopes = np.array([slope(y[i], x[i]) for i in rng.integers(0, len(y), (B, len(y)))])
    lo, hi = pct_ci(boot_slopes)
    return dict(slope=slope(y, x), ci_lo=float(lo), ci_hi=float(hi), n_zips=int(len(y)))


def _strat_means(d: np.ndarray, s: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-stratum sums of d (n x m) and counts, via bincount (much faster than a groupby inside a bootstrap loop)."""
    cnt = np.bincount(s, minlength=k).astype(float)
    sums = np.stack([np.bincount(s, weights=d[:, j], minlength=k) for j in range(d.shape[1])], 1)
    return sums, cnt


def reweighted(dt: np.ndarray, st: np.ndarray, dc: np.ndarray, sc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Treated mean and control mean reweighted to the treated stratum (cohort x borough) mix.

    Control stratum means are weighted by the number of treated sites in the stratum, so the control
    group mimics the treated group's cohort/borough composition; strata missing on either side drop out."""
    t_sum, t_n = _strat_means(dt, st, k)
    c_sum, c_n = _strat_means(dc, sc, k)
    # Reweighting: a stratum with no treated or no control sites cannot be compared, so it is dropped from BOTH arms
    # (otherwise the two means would describe different mixes of cohort and borough).
    ok = (t_n > 0) & (c_n > 0)
    if not ok.any():
        return np.full(dt.shape[1], np.nan), np.full(dt.shape[1], np.nan)
    treated = t_sum[ok].sum(0) / t_n[ok].sum()
    control = (t_n[ok, None] * c_sum[ok] / c_n[ok, None]).sum(0) / t_n[ok].sum()
    return treated, control


def _clusters(ids: np.ndarray | None, n: int) -> list[np.ndarray]:
    """Row-index arrays, one per unique cluster id (each row its own cluster when ids is None)."""
    if ids is None:
        return [np.array([i]) for i in range(n)]
    order = np.argsort(ids, kind="stable")
    cuts = np.flatnonzero(np.r_[True, ids[order][1:] != ids[order][:-1]])
    return np.split(order, cuts[1:])


def _draw(rng: np.random.Generator, groups: list[np.ndarray]) -> np.ndarray:
    """Resample clusters with replacement and concatenate all rows of the picked clusters."""
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def did_boot(dt: np.ndarray, st: np.ndarray, dc: np.ndarray, sc: np.ndarray, k: int, B: int, seed: int,
             site_t: np.ndarray | None = None, site_c: np.ndarray | None = None) -> dict:
    """Site-cluster bootstrap (site_t/site_c = site ids per row; a site seen in several cohorts is one cluster) of treated and reweighted-control means for each column of d.

    Columns are arbitrary outcomes (e.g. adjusted counts per rel year plus post-pre change); returns
    point estimates and 95% intervals for treated, control and treated-minus-control (DiD)."""
    t0, c0 = reweighted(dt, st, dc, sc, k)
    # Percentile bootstrap: each replicate redraws whole sites in each arm and recomputes the reweighted means, so the CI
    # reflects both sampling noise and the changing stratum mix. DiD replicates are paired differences, not a difference of CIs.
    rng = np.random.default_rng(seed)
    gt, gc = _clusters(site_t, len(dt)), _clusters(site_c, len(dc))
    reps = []
    for _ in range(B):
        it, ic = _draw(rng, gt), _draw(rng, gc)
        reps.append(np.stack(reweighted(dt[it], st[it], dc[ic], sc[ic], k)))
    r = np.array(reps)                                     # B x 2 x m
    out = {}
    for name, pt, rr in (("treated", t0, r[:, 0]), ("control", c0, r[:, 1]), ("diff", t0 - c0, r[:, 0] - r[:, 1])):
        lo, hi = pct_ci(rr)
        out[name] = dict(est=pt, lo=lo, hi=hi)
    return out


def risk_diff(x1: float, n1: float, x0: float, n0: float) -> dict:
    """Risk difference p1-p0 with Wald 95% CI (independent binomials)."""
    p1, p0 = x1 / n1, x0 / n0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return dict(rd=float(p1 - p0), ci_lo=float(p1 - p0 - Z95 * se), ci_hi=float(p1 - p0 + Z95 * se), n1=int(n1), n0=int(n0))


def cramers_v(table: np.ndarray) -> float:
    """Cramer's V from a contingency table (chi-square without continuity correction)."""
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def log_lift(c1: np.ndarray, n1: float, c0: np.ndarray, n0: float) -> dict[str, np.ndarray]:
    """Lift P(cat|exposed)/P(cat|baseline) with delta-method CI on the log scale.

    Var(log p) ~ (1-p)/(n p) for a multinomial proportion; the two groups are independent, so variances add."""
    p1, p0 = c1 / n1, c0 / n0
    # Delta method: for a log-ratio of two independent proportions, Var(log p1 - log p0) is the sum of the two
    # (1-p)/(n p) terms, so a normal interval on the log scale is exponentiated back to a multiplicative lift.
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.log(p1 / p0)
        se = np.sqrt((1 - p1) / (n1 * p1) + (1 - p0) / (n0 * p0))
        return dict(lift=np.exp(ll), lo=np.exp(ll - Z95 * se), hi=np.exp(ll + Z95 * se))
