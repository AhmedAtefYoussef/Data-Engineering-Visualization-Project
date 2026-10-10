"""The few statistics the website recomputes for a filtered selection.

These functions are line-for-line ports of the notebook's Section 7 toolkit and of the event
study in Section 8 (same resampling scheme, same seed, same order of random draws). On the
notebook's own sample they return the notebook's estimates and intervals exactly, which
tools/build_app_data.py asserts after every data build. Pure numpy and pandas; no Dash.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

N_BOOT = 1000                 # bootstrap resamples behind every interval (the notebook's value)
CI_LEVEL = 0.95
SEED = 42
LOG_RATIO_OFFSET = 0.5        # added to both counts so a site with zero crashes keeps a finite ratio
MAX_FAILED_SHARE = 0.10       # no interval is reported when more replicates than this fail
PERCENT = 100
PRE_PERIOD_INDEX = 100        # the mean of the pre-completion years is set to 100 in the event chart
NEUTRAL_PERCENTILE = 50       # mean percentile that sites would have if they were typical streets


def percentile_ci(samples, level: float = CI_LEVEL) -> dict:
    """Percentile interval of bootstrap replicates; NaN when too many replicates failed."""
    values = np.asarray(samples, dtype=float)
    n_failed = np.isnan(values).sum(axis=0)
    tail_pct = PERCENT * (1 - level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        low = np.nanpercentile(values, tail_pct, axis=0)
        high = np.nanpercentile(values, PERCENT - tail_pct, axis=0)
    too_many_failed = n_failed > MAX_FAILED_SHARE * values.shape[0]
    low, high = np.where(too_many_failed, np.nan, low), np.where(too_many_failed, np.nan, high)
    if values.ndim == 1:
        return {"low": float(low), "high": float(high)}
    return {"low": low, "high": high}


def bootstrap_p_value(replicates, null_value: float = 0.0) -> float:
    """Two-sided bootstrap p-value: twice the smaller share of replicates on either side of the null."""
    values = np.asarray(replicates, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    at_or_below = int((values <= null_value).sum())
    at_or_above = int((values >= null_value).sum())
    return float(min(1.0, 2 * (min(at_or_below, at_or_above) + 1) / (values.size + 1)))


def cluster_layout(clusters, n_rows: int):
    """(order, starts, sizes): where each cluster's rows sit, so whole clusters can be drawn at once."""
    if clusters is None:
        return np.arange(n_rows), np.arange(n_rows), np.ones(n_rows, dtype=np.int64)
    codes, _ = pd.factorize(np.asarray(clusters))
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    return order, np.cumsum(sizes) - sizes, sizes


def draw_rows(generator, layout) -> np.ndarray:
    """Row numbers of one resample: clusters drawn with replacement, their rows kept together."""
    order, starts, sizes = layout
    picked = generator.integers(0, len(sizes), len(sizes))
    picked_sizes = sizes[picked]
    block_ends = np.cumsum(picked_sizes)
    position_in_block = np.arange(block_ends[-1]) - np.repeat(block_ends - picked_sizes, picked_sizes)
    return order[np.repeat(starts[picked], picked_sizes) + position_in_block]


def bootstrap_ci(statistic, *arrays, clusters=None, n_boot: int = N_BOOT, seed: int = SEED,
                 level: float = CI_LEVEL) -> dict:
    """Percentile bootstrap of `statistic(*arrays)`, resampling rows or whole clusters."""
    arrays = [np.asarray(array) for array in arrays]
    n_rows = len(arrays[0])
    layout = cluster_layout(clusters, n_rows)
    generator = np.random.default_rng(seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        estimate = np.asarray(statistic(*arrays), dtype=float)
        replicates = np.full((n_boot,) + estimate.shape, np.nan)
        for replicate in range(n_boot):
            rows = draw_rows(generator, layout)
            replicates[replicate] = statistic(*[array[rows] for array in arrays])
    interval = percentile_ci(replicates, level)
    if estimate.ndim == 0:
        estimate, p_value = float(estimate), bootstrap_p_value(replicates)
    else:
        p_value = np.array([bootstrap_p_value(column) for column in replicates.T])
    return {"estimate": estimate, "low": interval["low"], "high": interval["high"],
            "p_value": p_value, "n": int(n_rows), "n_clusters": int(len(layout[2]))}


def to_percent(log_points):
    """A difference in logs as a percentage change: 100 * (exp(x) - 1)."""
    return PERCENT * np.expm1(log_points)


def log_ratio(before, after) -> np.ndarray:
    """log((after + 0.5) / (before + 0.5)) for counts, element by element."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    return np.log((after + LOG_RATIO_OFFSET) / (before + LOG_RATIO_OFFSET))


def stratified_difference(change, treated, strata_codes, n_strata: int) -> dict:
    """Treated-minus-comparison mean of `change`, compared inside strata and weighted by treated counts."""
    treated_n = np.bincount(strata_codes[treated], minlength=n_strata)
    control_n = np.bincount(strata_codes[~treated], minlength=n_strata)
    treated_sum = np.bincount(strata_codes[treated], weights=change[treated], minlength=n_strata)
    control_sum = np.bincount(strata_codes[~treated], weights=change[~treated], minlength=n_strata)
    comparable = (treated_n > 0) & (control_n > 0)      # only strata with both arms can be compared
    if not comparable.any():
        return {"estimate": float("nan"), "n_treated": 0, "n_control": 0}
    weights = treated_n[comparable] / treated_n[comparable].sum()
    treated_change = float((weights * treated_sum[comparable] / treated_n[comparable]).sum())
    control_change = float((weights * control_sum[comparable] / control_n[comparable]).sum())
    return {"estimate": treated_change - control_change, "n_treated": int(treated_n[comparable].sum()),
            "n_control": int(control_n[comparable].sum())}


def did_log_ratio(before, after, treated, strata, clusters) -> dict:
    """Difference-in-differences of the log after/before ratio, strata weighted, clusters resampled.

    Returns percent_change, percent_low, percent_high, p_value, n_treated, n_control, n_clusters.
    """
    change = log_ratio(before, after)
    treated = np.asarray(treated, dtype=bool)
    strata_codes, strata_labels = pd.factorize(np.asarray(strata))
    n_strata = len(strata_labels)
    point = stratified_difference(change, treated, strata_codes, n_strata)

    def did_estimate(change_rows, treated_rows, strata_rows):
        return stratified_difference(change_rows, treated_rows, strata_rows, n_strata)["estimate"]

    boot = bootstrap_ci(did_estimate, change, treated, strata_codes, clusters=clusters)
    return {"percent_change": float(to_percent(point["estimate"])),
            "percent_low": float(to_percent(boot["low"])), "percent_high": float(to_percent(boot["high"])),
            "p_value": boot["p_value"], "n_treated": point["n_treated"], "n_control": point["n_control"],
            "n_clusters": boot["n_clusters"]}


def type_contrast(before, after, arm, strata, clusters) -> dict:
    """DiD for two treated arms against one shared comparison arm, and their difference.

    arm: 0 = comparison, 1 and 2 = the two treated types. One joint bootstrap, so the
    difference is paired. Returns three dicts (`first`, `second`, `difference`) in percent.
    """
    change = log_ratio(before, after)
    arm = np.asarray(arm)
    strata_codes, strata_labels = pd.factorize(np.asarray(strata))
    n_strata = len(strata_labels)

    def type_estimates(change_rows, arm_rows, strata_rows):
        by_type = []
        for treated_arm in (1, 2):
            rows = (arm_rows == 0) | (arm_rows == treated_arm)
            by_type.append(stratified_difference(change_rows[rows], arm_rows[rows] == treated_arm,
                                                 strata_rows[rows], n_strata)["estimate"])
        return np.array([by_type[0], by_type[1], by_type[0] - by_type[1]])

    boot = bootstrap_ci(type_estimates, change, arm, strata_codes, clusters=clusters)
    counts = [int((arm == 1).sum()), int((arm == 2).sum())]
    counts.append(sum(counts))
    names = ["first", "second", "difference"]
    return {name: {"percent_change": float(to_percent(boot["estimate"][position])),
                   "percent_low": float(to_percent(boot["low"][position])),
                   "percent_high": float(to_percent(boot["high"][position])),
                   "p_value": float(boot["p_value"][position]), "n_treated": counts[position],
                   "n_control": int((arm == 0).sum()), "n_clusters": boot["n_clusters"]}
            for position, name in enumerate(names)}


def event_profile(counts, treated, cohorts, clusters, n_pre: int) -> dict:
    """Mean crashes per site by relative year, indexed to the pre-completion mean, for both groups.

    counts: (rows, relative years). Both groups are averaged cohort by cohort and the cohorts
    are weighted by the treated counts, so the two lines describe the same mix of calendar
    years. Returns {"treated": {...}, "comparison": {...}}, each with `index`, `low`, `high`
    (arrays over relative years) and `n_sites`.
    """
    counts = np.asarray(counts, dtype=float)
    treated = np.asarray(treated, dtype=bool)
    cohort_codes, cohort_labels = pd.factorize(np.asarray(cohorts))
    clusters = np.asarray(clusters)
    n_cohorts = len(cohort_labels)
    cohort_weights = np.bincount(cohort_codes[treated], minlength=n_cohorts).astype(float)

    def indexed_means(count_rows, codes):
        sizes = np.bincount(codes, minlength=n_cohorts)
        present = (sizes > 0) & (cohort_weights > 0)
        if not present.any():
            return np.full(count_rows.shape[1], np.nan)
        sums = np.column_stack([np.bincount(codes, weights=count_rows[:, column], minlength=n_cohorts)
                                for column in range(count_rows.shape[1])])
        cohort_means = sums[present] / sizes[present, None]
        yearly = (cohort_weights[present] / cohort_weights[present].sum()) @ cohort_means
        return PRE_PERIOD_INDEX * yearly / yearly[:n_pre].mean()

    profile = {}
    for is_treated, group in [(True, "treated"), (False, "comparison")]:
        rows = treated == is_treated
        boot = bootstrap_ci(indexed_means, counts[rows], cohort_codes[rows], clusters=clusters[rows])
        profile[group] = {"index": boot["estimate"], "low": boot["low"], "high": boot["high"],
                          "n_sites": int(rows.sum())}
    return profile


def mean_percentile_excess(percentiles) -> float:
    """Mean percentile minus 50: zero when the places rank like typical streets."""
    return float(percentiles.mean() - NEUTRAL_PERCENTILE)


def spearman_rho(first, second) -> float:
    """Spearman rank correlation (average ranks for ties); NaN when either side is constant."""
    first_ranks = pd.Series(np.asarray(first, dtype=float)).rank().to_numpy()
    second_ranks = pd.Series(np.asarray(second, dtype=float)).rank().to_numpy()
    if first_ranks.std() == 0 or second_ranks.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])
