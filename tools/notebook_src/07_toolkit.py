# %% [markdown]
# ## 7. Analysis toolkit
#
# The research questions in Section 8 reuse the same few statistical steps, so they are written
# once here as small functions that take numpy arrays or pandas objects and return plain
# dictionaries or tables. Nothing in this section touches the housing or crash data. Each
# function is introduced by a short paragraph that says what it measures and when it is the
# right tool, and Section 7.9 checks every one of them on tiny made-up inputs whose answer is
# known in advance.
#
# Three ideas shape the whole section. First, crashes near the same building or in the same ZIP
# are not independent observations, so intervals are built by resampling whole clusters (sites
# or ZIPs, or one-kilometre blocks of sites where the text says so) and never single crashes.
# Treating every crash as an independent draw would make every interval so narrow that any
# difference at all would look real. Second, every function reports
# how many units it actually used, because a result on a handful of sites should not read like a
# result on thousands. Third, the data are observational, so the wording helper at the end only
# ever describes associations.

# %% [markdown]
# ### 7.1 Constants for this section
#
# These thresholds are used only by the toolkit. The resampling settings (`N_BOOT`, `CI_LEVEL`,
# `SEED`, `MIN_GROUP_N`) come from Section 1.

# %%
LOG_RATIO_OFFSET = 0.5        # crashes; added to both counts so a site with zero crashes in one
                              # period still has a finite log ratio. Half a crash is the usual
                              # half-count correction: smaller than any real count, so sites with
                              # ordinary counts are barely moved, but large enough that a change from
                              # 0 to 1 crash is not an infinite ratio
MAX_FAILED_SHARE = 0.10       # share of bootstrap replicates; if more than this fail to produce
                              # a number, no interval is reported at all
FALSE_DISCOVERY_RATE = 0.05   # share of flagged results allowed to be false alarms when several
                              # tests are read together (Benjamini-Hochberg)
PERCENT = 100                 # converts a proportion to a percentage

# %% [markdown]
# ### 7.2 Intervals from resampling
#
# `percentile_ci` turns a set of bootstrap replicates into an interval by reading off the lower
# and upper percentiles that leave `CI_LEVEL` of the replicates in the middle. A replicate can
# fail, for example when a resample happens to contain no control site, and it then comes back
# as NaN. Dropping those quietly would make an unstable estimate look stable, so the function
# counts the failures, returns the count, and gives no interval when more than
# `MAX_FAILED_SHARE` of the replicates failed.
#
# `bootstrap_p_value` answers a narrower question: how often do the replicates fall on the far
# side of a reference value such as zero. It is computed from the same replicates as the
# percentile interval, with one added to the count so that it is never zero. The two are
# different calculations on the same replicates, so for a result close to the threshold the
# interval can touch zero while the p-value is just under the limit, or the other way round. The
# p-value is the one used where several results have to be corrected together in Section 7.6. With `N_BOOT` replicates it cannot go below about two divided by `N_BOOT`, so very small
# p-values are reported as that floor and not as zero.
#
# `bootstrap_ci` is the general tool behind everything else. It takes any statistic written as a
# function of row-aligned arrays, recomputes it on `N_BOOT` resamples of the rows, and returns
# the estimate with its percentile interval. When `clusters` is given, whole clusters are drawn
# with replacement and all their rows travel together, which is the right choice whenever rows
# share a site, a ZIP or any other common source of noise. Each call builds its own random
# generator from `SEED`, so a result does not depend on which cells were run before it.

# %%
def percentile_ci(samples, level=CI_LEVEL):
    """Percentile interval of bootstrap replicates that reports failed (NaN) replicates.

    samples: array of shape (n_replicates,) or (n_replicates, k). Returns a dict with `low`,
    `high`, `n_valid` and `n_failed` (floats and ints for 1-D input, arrays of length k for
    2-D input). `low` and `high` are NaN when more than MAX_FAILED_SHARE of replicates failed.
    """
    values = np.asarray(samples, dtype=float)
    n_replicates = values.shape[0]
    n_failed = np.isnan(values).sum(axis=0)
    tail_pct = PERCENT * (1 - level) / 2
    with warnings.catch_warnings():
        # A column whose replicates all failed makes numpy warn; the NaN it returns is the
        # answer wanted here, and the failure is reported through n_failed.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        low = np.nanpercentile(values, tail_pct, axis=0)
        high = np.nanpercentile(values, PERCENT - tail_pct, axis=0)
    too_many_failed = n_failed > MAX_FAILED_SHARE * n_replicates
    low = np.where(too_many_failed, np.nan, low)
    high = np.where(too_many_failed, np.nan, high)
    if values.ndim == 1:
        return {"low": float(low), "high": float(high),
                "n_valid": int(n_replicates - n_failed), "n_failed": int(n_failed)}
    return {"low": low, "high": high, "n_valid": n_replicates - n_failed, "n_failed": n_failed}


def bootstrap_p_value(replicates, null_value=0.0):
    """Two-sided bootstrap p-value: twice the smaller share of replicates on either side of null_value.

    One is added to the count and to the total so the p-value is never exactly zero. NaN
    replicates are ignored; the result is NaN when no replicate is valid.
    """
    values = np.asarray(replicates, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    at_or_below = int((values <= null_value).sum())
    at_or_above = int((values >= null_value).sum())
    return float(min(1.0, 2 * (min(at_or_below, at_or_above) + 1) / (values.size + 1)))


def cluster_layout(clusters, n_rows):
    """Describe where each cluster's rows sit, so whole clusters can be drawn quickly.

    Returns (order, starts, sizes): `order` lists row numbers sorted by cluster, and cluster j
    owns order[starts[j] : starts[j] + sizes[j]]. With clusters=None every row is its own cluster.
    """
    if clusters is None:
        return np.arange(n_rows), np.arange(n_rows), np.ones(n_rows, dtype=np.int64)
    codes, _ = pd.factorize(np.asarray(clusters))
    assert len(codes) == n_rows, "clusters must have one label per row"
    assert (codes >= 0).all(), "a missing cluster label cannot be resampled"
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.cumsum(sizes) - sizes
    return order, starts, sizes


def draw_rows(generator, layout):
    """Row numbers of one bootstrap resample: clusters drawn with replacement, rows kept together."""
    order, starts, sizes = layout
    picked = generator.integers(0, len(sizes), len(sizes))
    picked_sizes = sizes[picked]
    # Build the row list without a Python loop: each picked cluster contributes the positions
    # start, start + 1, ... start + size - 1 of the sorted order.
    block_ends = np.cumsum(picked_sizes)
    position_in_block = np.arange(block_ends[-1]) - np.repeat(block_ends - picked_sizes, picked_sizes)
    return order[np.repeat(starts[picked], picked_sizes) + position_in_block]


def bootstrap_ci(statistic, *arrays, clusters=None, n_boot=N_BOOT, seed=SEED, level=CI_LEVEL):
    """Percentile bootstrap of `statistic(*arrays)`, resampling rows or whole clusters.

    arrays: one or more arrays aligned on their first axis. statistic returns a float or a 1-D
    array and should return NaN when it cannot be computed. Returns a dict: `estimate`, `low`,
    `high`, `p_value` (against zero), `n` (rows), `n_clusters`, `n_failed`, `replicates`.
    """
    arrays = [np.asarray(array) for array in arrays]
    n_rows = len(arrays[0])
    assert n_rows > 0, "nothing to resample"
    assert all(len(array) == n_rows for array in arrays), "arrays must have the same number of rows"
    layout = cluster_layout(clusters, n_rows)
    generator = np.random.default_rng(seed)   # a fresh generator: the result ignores cell order
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
            "p_value": p_value, "n": int(n_rows), "n_clusters": int(len(layout[2])),
            "n_failed": interval["n_failed"], "replicates": replicates}


# %% [markdown]
# ### 7.3 Rank correlation
#
# Spearman's rho measures whether two quantities rise and fall together in rank order. It is
# used here in place of the ordinary correlation because counts of units and crashes are
# heavily right-skewed, and a rank correlation is not driven by the few largest ZIPs.
# `spearman_ci` returns rho with a bootstrap interval, drops pairs with a missing value, and
# refuses to report anything for fewer than `MIN_GROUP_N` pairs, since a rank correlation on a
# handful of points is mostly noise.
#
# A correlation across ZIPs can come entirely from the boroughs being different from each
# other: Manhattan ZIPs may have more of both quantities without any relationship inside
# Manhattan. `group_adjusted_spearman_ci` subtracts each group's mean from both variables
# before ranking, so only the variation inside groups is left. The group means are recomputed
# inside every resample, which keeps the interval honest about that extra estimation step.
# Groups with a single member carry no within-group information and are left out.

# %%
def pearson_r(first, second):
    """Pearson correlation of two 1-D arrays; NaN when either is constant."""
    first_centred = first - first.mean()
    second_centred = second - second.mean()
    denominator = np.sqrt((first_centred ** 2).sum() * (second_centred ** 2).sum())
    return float((first_centred * second_centred).sum() / denominator) if denominator > 0 else float("nan")


def spearman_rho(x, y):
    """Spearman rank correlation (Pearson correlation of average ranks); NaN for fewer than 3 pairs."""
    if len(x) < 3:
        return float("nan")
    return pearson_r(stats.rankdata(x), stats.rankdata(y))


def demean_by_group(values, group_codes, n_groups):
    """Subtract from each value the mean of its group (group_codes are integers 0..n_groups-1)."""
    group_sums = np.bincount(group_codes, weights=values, minlength=n_groups)
    group_sizes = np.bincount(group_codes, minlength=n_groups)
    # A resample can leave a group empty; dividing by 1 there is harmless because no row uses it.
    return values - (group_sums / np.maximum(group_sizes, 1))[group_codes]


def empty_result(n, **extra):
    """Result dict used when there are too few units to estimate anything."""
    return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"),
            "p_value": float("nan"), "n": int(n), "n_clusters": 0, "n_failed": 0, **extra}


def spearman_ci(x, y, clusters=None, min_n=MIN_GROUP_N, n_boot=N_BOOT, seed=SEED, level=CI_LEVEL):
    """Spearman's rho with a (cluster) bootstrap interval.

    Returns a dict: `estimate` (rho), `low`, `high`, `p_value`, `n` (complete pairs used),
    `n_clusters`, `n_failed`. All four statistics are NaN when n < min_n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    complete = ~(np.isnan(x) | np.isnan(y))
    if complete.sum() < min_n:
        return empty_result(complete.sum())
    kept_clusters = None if clusters is None else np.asarray(clusters)[complete]
    result = bootstrap_ci(spearman_rho, x[complete], y[complete], clusters=kept_clusters,
                          n_boot=n_boot, seed=seed, level=level)
    del result["replicates"]
    return result


def group_adjusted_spearman_ci(x, y, groups, clusters=None, min_n=MIN_GROUP_N,
                               n_boot=N_BOOT, seed=SEED, level=CI_LEVEL):
    """Spearman's rho after removing group means from x and y (within-group association).

    Returns the same dict as spearman_ci plus `n_groups`. Rows with a missing x, y or group and
    groups with a single row are not used; `n` counts the rows that are.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    group_codes, group_labels = pd.factorize(np.asarray(groups))
    usable = ~(np.isnan(x) | np.isnan(y)) & (group_codes >= 0)
    # Group sizes are counted after the incomplete rows are removed: a group of two rows of which
    # one is missing has one usable row, and demeaning that row would give exactly 0.
    usable_sizes = np.bincount(group_codes[usable], minlength=len(group_labels))
    usable &= usable_sizes[np.maximum(group_codes, 0)] >= 2
    n_groups = int(len(np.unique(group_codes[usable])))
    if usable.sum() < min_n:
        return empty_result(usable.sum(), n_groups=n_groups)

    def within_group_rho(x_rows, y_rows, code_rows):
        """Demean inside the resample, then rank: the group means are part of the estimate."""
        x_within = demean_by_group(x_rows, code_rows, len(group_labels))
        y_within = demean_by_group(y_rows, code_rows, len(group_labels))
        return spearman_rho(x_within, y_within)

    kept_clusters = None if clusters is None else np.asarray(clusters)[usable]
    result = bootstrap_ci(within_group_rho, x[usable], y[usable], group_codes[usable],
                          clusters=kept_clusters, n_boot=n_boot, seed=seed, level=level)
    del result["replicates"]
    result["n_groups"] = n_groups
    return result


# %% [markdown]
# ### 7.4 Within-unit slopes in a panel
#
# A balanced panel is a table with one row per unit (a ZIP) and one column per year, with no
# gaps. `two_way_demean` subtracts each row's mean and each column's mean and adds the overall
# mean back once. What remains is how far a ZIP in a given year sits from its own usual level
# and from the citywide level of that year. Anything fixed for a ZIP (its size, its street
# layout) and anything shared by a year (the drop in reported crashes after 2020) is removed.
#
# `within_slope` regresses the demeaned outcome on one or several demeaned regressors, which is
# the two-way fixed-effects estimate. Its interval comes from resampling whole ZIPs, so the
# year-to-year dependence inside a ZIP stays intact, and the demeaning is redone in every
# resample because the resampled panel has different means. Several regressors can be entered
# together, which is how leads and lags are estimated: `lead_lag_panel` builds them from one
# regressor matrix. A lead is the regressor's value in a later year. Future housing cannot
# influence present crashes, so a lead coefficient away from zero is a warning that ZIPs were
# already on different paths, and the lag coefficients should then not be read as a response.
# The slope is still only an association: it assumes that ZIPs with and without new housing
# would otherwise have moved in parallel, which the leads can question but never prove.

# %%
def two_way_demean(matrix):
    """Remove row means and column means from a balanced unit-by-period matrix.

    Returns matrix - row mean - column mean + grand mean (the grand mean is added back because
    the two subtractions each removed it once).
    """
    values = np.asarray(matrix, dtype=float)
    assert values.ndim == 2, "expected a unit-by-period matrix"
    assert not np.isnan(values).any(), "the panel must be balanced: no missing cells"
    return values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()


def within_coefficients(outcome, regressor_stack):
    """Least-squares coefficients of the demeaned outcome on the demeaned regressors.

    outcome: (units, periods). regressor_stack: (units, k, periods). Returns k coefficients, all
    NaN when the regressors are collinear after demeaning (no unique answer exists then).
    """
    n_regressors = regressor_stack.shape[1]
    target = two_way_demean(outcome).ravel()
    design = np.column_stack([two_way_demean(regressor_stack[:, position, :]).ravel()
                              for position in range(n_regressors)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients if rank == n_regressors else np.full(n_regressors, np.nan)


def within_slope(outcome, regressors, n_boot=N_BOOT, seed=SEED, level=CI_LEVEL):
    """Two-way fixed-effects slopes for a balanced panel with a unit-cluster bootstrap.

    outcome: (units, periods) matrix. regressors: one matrix of the same shape, or a dict
    {name: matrix} to estimate several slopes jointly. Returns a DataFrame indexed by regressor
    name with columns `estimate`, `low`, `high`, `p_value`, `n_units`, `n_periods`, `n_failed`.
    """
    named = regressors if isinstance(regressors, dict) else {"regressor": regressors}
    outcome = np.asarray(outcome, dtype=float)
    stack = np.stack([np.asarray(matrix, dtype=float) for matrix in named.values()], axis=1)
    assert stack.shape[0] == outcome.shape[0] and stack.shape[2] == outcome.shape[1], \
        "every regressor must have the shape of the outcome"
    # Rows are units, so resampling rows resamples whole units with all their years.
    result = bootstrap_ci(within_coefficients, outcome, stack, n_boot=n_boot, seed=seed, level=level)
    table = pd.DataFrame({"estimate": result["estimate"], "low": result["low"], "high": result["high"],
                          "p_value": result["p_value"], "n_units": outcome.shape[0],
                          "n_periods": outcome.shape[1], "n_failed": result["n_failed"]},
                         index=pd.Index(list(named), name="regressor"))
    return table


def shift_name(shift):
    """Label of a shifted regressor: negative shifts are leads, positive shifts are lags."""
    if shift < 0:
        return f"lead {-shift}"
    return "same year" if shift == 0 else f"lag {shift}"


def lead_lag_panel(outcome, regressor, shifts):
    """Build lead and lag versions of a regressor and trim the panel to the periods all of them cover.

    shift s > 0 pairs the outcome in period t with the regressor in t - s (a lag); s < 0 pairs
    it with t + |s| (a lead). Returns (outcome_trimmed, {name: matrix}, kept_columns), where
    kept_columns are the positions of the retained periods in the original matrices and the
    dict is ordered from the furthest lead to the furthest lag.
    """
    outcome = np.asarray(outcome, dtype=float)
    regressor = np.asarray(regressor, dtype=float)
    assert outcome.shape == regressor.shape, "outcome and regressor must have the same shape"
    longest_lag = max(max(shifts), 0)
    longest_lead = max(-min(shifts), 0)
    kept_columns = np.arange(longest_lag, outcome.shape[1] - longest_lead)
    assert len(kept_columns) >= 2, "the leads and lags leave fewer than two usable periods"
    shifted = {shift_name(shift): regressor[:, kept_columns - shift] for shift in sorted(shifts)}
    return outcome[:, kept_columns], shifted, kept_columns


# %% [markdown]
# ### 7.5 Before and after at the same places
#
# `log_ratio` expresses a site's change as the logarithm of after over before. A site with no
# crashes in one period would make the logarithm undefined, so `LOG_RATIO_OFFSET` is added to
# both counts; the constant is small enough that sites with ordinary counts are almost
# unaffected. Dropping sites with a zero count was the alternative, and it would remove exactly
# the quiet sites. A change in levels would let a few busy intersections decide the answer,
# while a log ratio treats a site going from 4 to 2 crashes the same as one going from 40 to 20.
# Small values read almost as proportional changes.
#
# `did_log_ratio` is the difference-in-differences. Crashes near a completed building may fall
# simply because reported crashes fell everywhere, so the change at treated sites is compared
# with the change at control sites over the same calendar years, and the estimate is the
# treated mean log ratio minus the control mean log ratio. With `strata` (for example
# completion year by borough) the comparison is made inside each stratum and the strata are
# averaged with the number of treated sites as weights, so the controls are made to resemble
# the treated group in when and where they are. A stratum that lacks either treated or control
# sites cannot be compared and is left out of both sides; the returned counts are the sites
# actually used, with the dropped ones listed separately. Treated and control rows go in one
# table and whole clusters are resampled from it, which stays valid when the same site serves
# as a control for one cohort and is treated in another. The estimate is an association under
# the assumption that the two groups would otherwise have changed alike.
#
# `share_change` is for questions about composition, such as the pedestrian and cyclist share
# of injury crashes. It pools the events of all clusters in each period, takes the difference
# of the two shares, and resamples clusters. Pooling avoids averaging shares from sites that
# have only one or two crashes, where a share can only be zero, a half or one.
#
# `linear_slope` is a plain least-squares slope, provided so that a dose relationship (change
# against building size) can be passed to `bootstrap_ci` with site clusters.

# %%
def log_ratio(before, after):
    """log((after + LOG_RATIO_OFFSET) / (before + LOG_RATIO_OFFSET)) for counts, element by element."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    assert (before >= 0).all() and (after >= 0).all(), "counts cannot be negative"
    return np.log((after + LOG_RATIO_OFFSET) / (before + LOG_RATIO_OFFSET))


def log_points_to_percent(log_points):
    """Percent change that a difference in log points stands for: 100 * (exp(x) - 1).

    Works on a single number or an array, so every section converts in the same way.
    """
    return PERCENT * np.expm1(log_points)


def stratified_difference(change, treated, strata_codes, n_strata):
    """Treated-minus-control mean of `change`, compared inside strata and weighted by treated counts.

    Returns a dict with `estimate`, `treated_change`, `control_change`, `n_treated`, `n_control`
    (units in strata that have both arms) and `n_strata` (such strata). The estimates are NaN
    when no stratum has both arms.
    """
    treated_n = np.bincount(strata_codes[treated], minlength=n_strata)
    control_n = np.bincount(strata_codes[~treated], minlength=n_strata)
    treated_sum = np.bincount(strata_codes[treated], weights=change[treated], minlength=n_strata)
    control_sum = np.bincount(strata_codes[~treated], weights=change[~treated], minlength=n_strata)
    # Only strata with both arms can be compared. Dropping them from both sides keeps the two
    # means describing the same mix of strata.
    comparable = (treated_n > 0) & (control_n > 0)
    if not comparable.any():
        return {"estimate": float("nan"), "treated_change": float("nan"),
                "control_change": float("nan"), "n_treated": 0, "n_control": 0, "n_strata": 0}
    weights = treated_n[comparable] / treated_n[comparable].sum()
    treated_change = float((weights * treated_sum[comparable] / treated_n[comparable]).sum())
    control_change = float((weights * control_sum[comparable] / control_n[comparable]).sum())
    return {"estimate": treated_change - control_change, "treated_change": treated_change,
            "control_change": control_change, "n_treated": int(treated_n[comparable].sum()),
            "n_control": int(control_n[comparable].sum()), "n_strata": int(comparable.sum())}


def did_log_ratio(before, after, treated, strata=None, clusters=None,
                  n_boot=N_BOOT, seed=SEED, level=CI_LEVEL):
    """Difference-in-differences on log((after + offset) / (before + offset)), treated against control.

    before, after: counts per row; offset is LOG_RATIO_OFFSET. treated: boolean per row. strata: optional labels; the
    comparison is made inside strata and weighted by treated counts. clusters: optional labels
    of the units resampled together (rows are resampled singly when omitted).
    Returns a dict: `estimate`, `low`, `high` (log points), `p_value`, `percent_change`,
    `percent_low`, `percent_high` (100 * (exp(.) - 1)), `treated_change`, `control_change`,
    `n_treated`, `n_control`, `n_strata` (all three count what was actually used),
    `n_treated_dropped`, `n_control_dropped`, `n_clusters`, `n_failed`, `replicates`.
    """
    change = log_ratio(before, after)
    treated = np.asarray(treated, dtype=bool)
    if strata is None:
        strata_codes, n_strata = np.zeros(len(change), dtype=np.int64), 1
    else:
        strata_codes, strata_labels = pd.factorize(np.asarray(strata))
        assert (strata_codes >= 0).all(), "every row needs a stratum label"
        n_strata = len(strata_labels)
    point = stratified_difference(change, treated, strata_codes, n_strata)

    def did_estimate(change_rows, treated_rows, strata_rows):
        """The weighted difference recomputed on a resample, including its changed stratum mix."""
        return stratified_difference(change_rows, treated_rows, strata_rows, n_strata)["estimate"]

    boot = bootstrap_ci(did_estimate, change, treated, strata_codes, clusters=clusters,
                        n_boot=n_boot, seed=seed, level=level)
    as_percent = {name: float(log_points_to_percent(boot[key]))
                  for name, key in (("percent_change", "estimate"), ("percent_low", "low"),
                                    ("percent_high", "high"))}
    return {**point, "low": boot["low"], "high": boot["high"], "p_value": boot["p_value"],
            **as_percent,
            "n_treated_dropped": int(treated.sum()) - point["n_treated"],
            "n_control_dropped": int((~treated).sum()) - point["n_control"],
            "n_clusters": boot["n_clusters"], "n_failed": boot["n_failed"],
            "replicates": boot["replicates"]}


def pooled_share_difference(hits_before, total_before, hits_after, total_after):
    """Share after minus share before, each pooled over all rows; NaN if a period has no events."""
    events_before = total_before.sum()
    events_after = total_after.sum()
    if events_before == 0 or events_after == 0:
        return float("nan")
    return float(hits_after.sum() / events_after - hits_before.sum() / events_before)


def share_change(hits_before, total_before, hits_after, total_after, clusters=None,
                 n_boot=N_BOOT, seed=SEED, level=CI_LEVEL):
    """Change in a pooled proportion between two periods for the same clusters.

    One row per cluster (or several rows sharing a label in `clusters`): `hits_*` counts the
    events of interest and `total_*` all events in each period. Returns a dict: `estimate`
    (share after minus share before, as a proportion), `low`, `high`, `p_value`,
    `share_before`, `share_after`, `n_before`, `n_after` (events), `n_clusters`, `n_failed`,
    `replicates`.
    """
    counts = [np.asarray(values, dtype=float)
              for values in (hits_before, total_before, hits_after, total_after)]
    assert all((values >= 0).all() for values in counts), "counts cannot be negative"
    assert (counts[0] <= counts[1]).all() and (counts[2] <= counts[3]).all(), \
        "hits cannot exceed the total"
    boot = bootstrap_ci(pooled_share_difference, *counts, clusters=clusters,
                        n_boot=n_boot, seed=seed, level=level)
    events_before = counts[1].sum()
    events_after = counts[3].sum()
    return {"estimate": boot["estimate"], "low": boot["low"], "high": boot["high"],
            "p_value": boot["p_value"],
            "share_before": float(counts[0].sum() / events_before) if events_before else float("nan"),
            "share_after": float(counts[2].sum() / events_after) if events_after else float("nan"),
            "n_before": int(events_before), "n_after": int(events_after),
            "n_clusters": boot["n_clusters"], "n_failed": boot["n_failed"],
            "replicates": boot["replicates"]}


def linear_slope(x, y):
    """Least-squares slope of y on x; NaN when x does not vary."""
    x_centred = x - x.mean()
    spread = (x_centred ** 2).sum()
    return float((x_centred * (y - y.mean())).sum() / spread) if spread > 0 else float("nan")


# %% [markdown]
# ### 7.6 Several tests at once
#
# When several groups are tested side by side (the contributing-factor groups, the five
# boroughs), some will look unusual by chance alone. `benjamini_hochberg` adjusts a set of
# p-values so that, among the results whose adjusted value is below `FALSE_DISCOVERY_RATE`, at
# most that share is expected to be false alarms. It is less strict than dividing the
# threshold by the number of tests, which suits a handful of related comparisons where missing
# a real pattern is also a cost. The adjusted values are returned in the order they were given.

# %%
def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values, in the input order (NaN stays NaN and is not counted).

    Compare the result with FALSE_DISCOVERY_RATE to decide which tests to flag.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid = ~np.isnan(p_values)
    n_tests = int(valid.sum())
    if n_tests == 0:
        return adjusted
    assert ((p_values[valid] >= 0) & (p_values[valid] <= 1)).all(), "p-values must lie in [0, 1]"
    order = np.argsort(p_values[valid])
    scaled = p_values[valid][order] * n_tests / np.arange(1, n_tests + 1)
    # A larger p-value may not end up with a smaller adjusted value than a smaller one, so
    # each value takes the minimum of itself and everything ranked after it.
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    in_input_order = np.empty(n_tests)
    in_input_order[order] = np.minimum(monotone, 1.0)
    adjusted[valid] = in_input_order
    return adjusted


# %% [markdown]
# ### 7.7 Forecast evaluation
#
# A forecast is judged on years the model never saw. `mean_absolute_error` is the average size
# of the miss in the units of the outcome (crashes per ZIP and year), which is easier to read
# than a squared error and less dominated by the largest ZIPs.
#
# `time_split_evaluation` fits one model per named list of features on all periods before a
# cut-off year and scores each on the periods from the cut-off onwards. Splitting by time and
# not at random matters: a random split would let the model learn from the future of the same
# ZIP. The first feature list is the baseline, and for every other list the function reports
# the difference in mean absolute error against it. That difference gets a bootstrap interval
# from resampling the test units (ZIPs), using the same resample for both models so the
# comparison is paired. The interval describes how much the gain would vary over a different
# set of ZIPs with the fitted models held fixed; it does not include the uncertainty of
# fitting. A negative difference means the extra features are associated with smaller errors.

# %%
def mean_absolute_error(actual, predicted):
    """Mean absolute difference between actual and predicted values."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    assert actual.shape == predicted.shape, "actual and predicted must align"
    return float(np.abs(actual - predicted).mean())


def mean_gap(first, second):
    """Difference of two means (used for the paired difference in absolute errors)."""
    return float(first.mean() - second.mean())


def time_split_evaluation(frame, target, feature_sets, time_column, first_test_period,
                          cluster_column=None, make_model=LinearRegression,
                          n_boot=N_BOOT, seed=SEED, level=CI_LEVEL):
    """Fit on periods before `first_test_period`, score on the rest, and compare feature sets.

    frame: one row per unit and period. feature_sets: dict {name: list of columns}; the first
    entry is the baseline. make_model: a callable returning an unfitted scikit-learn regressor.
    Returns (summary, predictions). summary is a DataFrame indexed by feature-set name with
    columns `mae`, `mae_minus_baseline`, `low`, `high`, `p_value` (the last four refer to the
    paired difference and are 0 / NaN for the baseline), `n_features`, `n_train`, `n_test`,
    `n_test_clusters`. predictions is the test part of `frame` (time, cluster and target
    columns) with one prediction column per feature set.
    """
    used_columns = sorted({column for columns in feature_sets.values() for column in columns})
    assert not frame[used_columns + [target]].isna().any().any(), "features and target must be complete"
    is_test = (frame[time_column] >= first_test_period).to_numpy()
    train, test = frame[~is_test], frame[is_test]
    assert len(train) > 0 and len(test) > 0, "the cut-off leaves an empty training or test set"
    kept = [time_column, target] + ([cluster_column] if cluster_column else [])
    predictions = test[kept].copy()
    for name, columns in feature_sets.items():
        model = make_model().fit(train[columns].to_numpy(dtype=float), train[target].to_numpy(dtype=float))
        predictions[name] = model.predict(test[columns].to_numpy(dtype=float))

    actual = test[target].to_numpy(dtype=float)
    test_clusters = test[cluster_column].to_numpy() if cluster_column else None
    baseline_name = next(iter(feature_sets))
    baseline_errors = np.abs(actual - predictions[baseline_name].to_numpy())
    rows = []
    for name, columns in feature_sets.items():
        errors = np.abs(actual - predictions[name].to_numpy())
        row = {"feature_set": name, "mae": float(errors.mean()), "mae_minus_baseline": 0.0,
               "low": float("nan"), "high": float("nan"), "p_value": float("nan"),
               "n_features": len(columns), "n_train": len(train), "n_test": len(test),
               "n_test_clusters": len(test) if test_clusters is None else len(pd.unique(test_clusters))}
        if name != baseline_name:
            # Both error vectors are indexed by the same resampled rows, so each replicate
            # compares the two models on exactly the same ZIPs.
            gap = bootstrap_ci(mean_gap, errors, baseline_errors, clusters=test_clusters,
                               n_boot=n_boot, seed=seed, level=level)
            row.update(mae_minus_baseline=gap["estimate"], low=gap["low"], high=gap["high"],
                       p_value=gap["p_value"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature_set"), predictions


# %% [markdown]
# ### 7.8 Wording a result
#
# Every research question ends in a sentence that states the estimate, its interval, how many
# units it rests on, and whether the interval includes the reference value. `describe_effect`
# writes that sentence from the numbers, so the text under a chart cannot drift away from the
# result above it. It never uses causal words. When the interval includes the reference value it
# says so and stops; when the interval lies on one side it names the side and what that side
# fits; when the estimate could not be computed it says so and concludes nothing. That every
# result is an association is stated once, in the introduction to Section 8, and not repeated in
# each sentence.

# %%
def describe_effect(estimate, low, high, unit, n, subject="The estimate", n_unit="observations",
                    above=None, below=None, null_value=0.0, digits=2, level=CI_LEVEL):
    """One sentence stating an estimate, its interval and n in associational wording.

    unit: text placed after the estimate ("log points", "percentage points", ""). subject: the
    sentence opening that names what was estimated. above / below: what an interval entirely
    above / below null_value fits, as a clause after "which fits" (for example
    above="more injury crashes near the site"). null_value: the value meaning "no
    difference" (0 for differences and correlations, 1 for ratios). Returns a str.
    """
    sample = f"n = {n:,} {n_unit}"
    if any(np.isnan(value) for value in (estimate, low, high)):
        return f"{subject} could not be estimated reliably ({sample}), so nothing is concluded from it."
    reference = "zero" if null_value == 0 else f"{null_value:g}"
    unit_text = f" {unit}" if unit else ""
    opening = (f"{subject} is {estimate:+.{digits}f}{unit_text} ({PERCENT * level:.0f}% interval "
               f"{low:+.{digits}f} to {high:+.{digits}f}, {sample})")
    if low <= null_value <= high:
        return f"{opening}. The interval includes {reference}."
    is_above = low > null_value
    side = "above" if is_above else "below"
    default_meaning = f"a value {side} {reference}"
    meaning = (above if is_above else below) or default_meaning
    return f"{opening}. The interval lies {side} {reference}, which fits {meaning}."


# %% [markdown]
# ### 7.9 Self-test
#
# Each function is run on a tiny made-up input for which the right answer can be worked out by
# hand, and an `assert` stops the notebook if the answer is wrong. The inputs are built so the
# answer is exact: a panel made of a unit effect, a year effect and known slopes, sites whose
# crashes exactly double, clusters whose rows are copies of each other. A smaller number of
# resamples is used here because only correctness is being checked, not precision.

# %%
SELF_TEST_BOOT = 200      # resamples per check; enough to exercise the code path
SELF_TEST_TOLERANCE = 1e-9
SELF_TEST_MIN_N = 5       # lets the tiny synthetic inputs pass the minimum-n guard


def check(description, condition):
    """Stop on a failed self-test, otherwise print one line."""
    assert condition, f"self-test failed: {description}"
    print(f"ok  {description}")


def close(value, expected):
    """True when value equals expected up to floating-point noise."""
    return bool(np.allclose(value, expected, atol=SELF_TEST_TOLERANCE, equal_nan=True))


def run_self_tests():
    """Run every check on its tiny input; the temporaries stay local so no later cell can pick one up."""
    test_rng = np.random.default_rng(SEED)

    # percentile_ci and bootstrap_p_value
    ladder = np.arange(1001, dtype=float)
    interval = percentile_ci(ladder)
    check("percentile_ci: 0..1000 gives the 2.5th and 97.5th percentiles (25 and 975)",
          close([interval["low"], interval["high"]], [25, 975]) and interval["n_failed"] == 0)
    few_failed = percentile_ci(np.r_[ladder, [np.nan] * 5])
    check("percentile_ci: a few NaN replicates are counted and the interval is kept",
          few_failed["n_failed"] == 5 and few_failed["n_valid"] == 1001 and close(few_failed["low"], 25))
    many_failed = percentile_ci(np.r_[ladder, [np.nan] * 500])
    check("percentile_ci: too many NaN replicates give no interval",
          many_failed["n_failed"] == 500 and np.isnan(many_failed["low"]) and np.isnan(many_failed["high"]))
    check("bootstrap_p_value: 999 positive replicates give 2 / 1000",
          close(bootstrap_p_value(np.ones(999)), 0.002))
    check("bootstrap_p_value: replicates centred on zero give 1",
          close(bootstrap_p_value(np.r_[-np.ones(50), np.ones(50)]), 1.0))

    # bootstrap_ci
    constant = bootstrap_ci(np.mean, np.full(40, 5.0), n_boot=SELF_TEST_BOOT)
    check("bootstrap_ci: the mean of a constant is that constant with a zero-width interval",
          close([constant["estimate"], constant["low"], constant["high"]], [5, 5, 5]) and constant["n"] == 40)
    cluster_values = np.repeat(test_rng.normal(size=20), 50)      # 20 clusters of 50 identical rows
    cluster_labels = np.repeat(np.arange(20), 50)
    by_row = bootstrap_ci(np.mean, cluster_values, n_boot=SELF_TEST_BOOT)
    by_cluster = bootstrap_ci(np.mean, cluster_values, clusters=cluster_labels, n_boot=SELF_TEST_BOOT)
    check("bootstrap_ci: copies inside clusters add no information, so the cluster interval is far wider",
          (by_cluster["high"] - by_cluster["low"]) > 3 * (by_row["high"] - by_row["low"])
          and by_cluster["n_clusters"] == 20 and by_row["n_clusters"] == 1000)
    repeat_run = bootstrap_ci(np.mean, cluster_values, clusters=cluster_labels, n_boot=SELF_TEST_BOOT)
    check("bootstrap_ci: the same seed gives the same interval",
          close([repeat_run["low"], repeat_run["high"]], [by_cluster["low"], by_cluster["high"]]))

    # spearman_ci and group_adjusted_spearman_ci
    steps = np.arange(1.0, 13.0)
    rising = spearman_ci(steps, steps ** 3, min_n=SELF_TEST_MIN_N, n_boot=SELF_TEST_BOOT)
    falling = spearman_ci(steps, -np.exp(steps), min_n=SELF_TEST_MIN_N, n_boot=SELF_TEST_BOOT)
    check("spearman_ci: any increasing relation gives +1 and any decreasing relation gives -1",
          close([rising["estimate"], rising["low"], falling["estimate"], falling["high"]], [1, 1, -1, -1]))
    noisy_x = test_rng.normal(size=60)
    noisy_y = noisy_x + test_rng.normal(size=60)
    check("spearman_ci: the estimate equals scipy.stats.spearmanr",
          close(spearman_ci(noisy_x, noisy_y, n_boot=SELF_TEST_BOOT)["estimate"],
                stats.spearmanr(noisy_x, noisy_y)[0]))
    too_small = spearman_ci(steps, steps, n_boot=SELF_TEST_BOOT)
    check("spearman_ci: fewer pairs than MIN_GROUP_N gives NaN with n reported",
          np.isnan(too_small["estimate"]) and too_small["n"] == 12)
    group_shift = np.repeat([0.0, 100.0], 6)                      # two groups far apart on both axes
    inside = np.tile(np.arange(6.0), 2)
    grouped_x = group_shift + inside
    grouped_y = group_shift - inside                               # falls with x inside each group
    raw = spearman_ci(grouped_x, grouped_y, min_n=SELF_TEST_MIN_N, n_boot=SELF_TEST_BOOT)
    adjusted = group_adjusted_spearman_ci(grouped_x, grouped_y, np.repeat(["a", "b"], 6),
                                          min_n=SELF_TEST_MIN_N, n_boot=SELF_TEST_BOOT)
    check("group_adjusted_spearman_ci: a positive between-group correlation hides a within-group -1",
          raw["estimate"] > 0 and close(adjusted["estimate"], -1) and adjusted["n_groups"] == 2)
    incomplete_x = np.r_[grouped_x, [5.0, np.nan]]                 # a third group of two rows, one incomplete
    incomplete_y = np.r_[grouped_y, [5.0, 7.0]]
    incomplete = group_adjusted_spearman_ci(incomplete_x, incomplete_y, np.r_[np.repeat(["a", "b"], 6), ["c", "c"]],
                                            min_n=SELF_TEST_MIN_N, n_boot=SELF_TEST_BOOT)
    check("group_adjusted_spearman_ci: a group left with one complete row is not used",
          incomplete["n"] == 12 and incomplete["n_groups"] == 2 and close(incomplete["estimate"], -1))

    # two_way_demean, within_slope, lead_lag_panel
    unit_effect = test_rng.normal(size=(15, 1))
    year_effect = test_rng.normal(size=(1, 8))
    check("two_way_demean: a panel made only of unit and year effects becomes all zeros",
          close(two_way_demean(unit_effect + year_effect), 0))
    first_regressor = test_rng.normal(size=(15, 8))
    second_regressor = test_rng.normal(size=(15, 8))
    panel_outcome = unit_effect + year_effect + 2 * first_regressor - 3 * second_regressor
    slopes = within_slope(panel_outcome, {"first": first_regressor, "second": second_regressor},
                          n_boot=SELF_TEST_BOOT)
    check("within_slope: recovers the slopes 2 and -3 through unit and year effects, jointly",
          close(slopes["estimate"], [2, -3]) and close(slopes["low"], [2, -3]) and close(slopes["high"], [2, -3])
          and list(slopes.index) == ["first", "second"] and (slopes["n_units"] == 15).all())
    single = within_slope(unit_effect + year_effect + 0.5 * first_regressor, first_regressor, n_boot=SELF_TEST_BOOT)
    check("within_slope: a single regressor matrix works and gives 0.5", close(single["estimate"], [0.5]))
    lagged_outcome = np.zeros((15, 8))
    lagged_outcome[:, 1:] = 4 * first_regressor[:, :-1]            # outcome follows last year's regressor
    trimmed_outcome, shifted, kept_columns = lead_lag_panel(lagged_outcome, first_regressor, [-1, 0, 1])
    lag_slopes = within_slope(trimmed_outcome, shifted, n_boot=SELF_TEST_BOOT)
    check("lead_lag_panel: an outcome that follows last year's regressor loads on 'lag 1' only",
          list(shifted) == ["lead 1", "same year", "lag 1"] and list(kept_columns) == [1, 2, 3, 4, 5, 6]
          and close(lag_slopes["estimate"], [0, 0, 4]))

    # log_ratio, did_log_ratio, share_change, linear_slope
    check("log_ratio: zero crashes before and after is a change of zero, not an error",
          close(log_ratio([0, 1], [0, 2.5]), [0, np.log(2)]))
    check("log_points_to_percent: log 2 is +100%, minus log 2 is -50%, and arrays work",
          close(log_points_to_percent(np.array([np.log(2), -np.log(2), 0.0])), [100, -50, 0]))
    is_treated = np.r_[np.ones(8, dtype=bool), np.zeros(8, dtype=bool)]
    crashes_before = np.ones(16)
    crashes_after = np.r_[np.full(8, 2.5), np.ones(8)]             # (2.5 + 0.5) / (1 + 0.5) = 2 for treated
    doubled = did_log_ratio(crashes_before, crashes_after, is_treated, n_boot=SELF_TEST_BOOT)
    check("did_log_ratio: treated sites double and controls do not move, so the estimate is log 2 (+100%)",
          close([doubled["estimate"], doubled["low"], doubled["high"]], [np.log(2)] * 3)
          and close(doubled["percent_change"], 100) and doubled["n_treated"] == 8 and doubled["n_control"] == 8)
    shared_trend = did_log_ratio(crashes_before, np.full(16, 2.5), is_treated, n_boot=SELF_TEST_BOOT)
    check("did_log_ratio: when both groups double, the difference-in-differences is zero",
          close(shared_trend["estimate"], 0) and close(shared_trend["treated_change"], np.log(2)))
    # Stratum A: 2 treated (x2) and 3 controls; stratum B: 6 treated (x4) and 2 controls;
    # stratum C: 3 treated and no control, so it cannot be compared and must be left out.
    strata_labels = np.array(["A"] * 5 + ["B"] * 8 + ["C"] * 3)
    strata_treated = np.array([True, True, False, False, False] + [True] * 6 + [False] * 2 + [True] * 3)
    strata_before = np.ones(16)
    strata_after = np.where(strata_treated, np.where(strata_labels == "A", 2.5, 5.5), 1.0)
    weighted = did_log_ratio(strata_before, strata_after, strata_treated, strata=strata_labels,
                             clusters=np.arange(16), n_boot=SELF_TEST_BOOT)
    check("did_log_ratio: strata are weighted by treated counts and a stratum with no control is dropped",
          close(weighted["estimate"], (2 * np.log(2) + 6 * np.log(4)) / 8)
          and (weighted["n_treated"], weighted["n_control"], weighted["n_strata"]) == (8, 5, 2)
          and (weighted["n_treated_dropped"], weighted["n_control_dropped"]) == (3, 0))
    no_control = did_log_ratio(np.ones(4), np.ones(4), np.ones(4, dtype=bool), n_boot=SELF_TEST_BOOT)
    check("did_log_ratio: with no control units the estimate is NaN and n used is zero",
          np.isnan(no_control["estimate"]) and no_control["n_treated"] == 0
          and no_control["n_failed"] == SELF_TEST_BOOT and np.isnan(no_control["low"]))
    shares = share_change(np.full(10, 10), np.full(10, 100), np.full(10, 30), np.full(10, 100),
                          n_boot=SELF_TEST_BOOT)
    check("share_change: 10 of 100 before and 30 of 100 after in every cluster is +0.20 exactly",
          close([shares["estimate"], shares["low"], shares["high"]], [0.2, 0.2, 0.2])
          and close([shares["share_before"], shares["share_after"]], [0.1, 0.3])
          and (shares["n_before"], shares["n_after"], shares["n_clusters"]) == (1000, 1000, 10))
    check("linear_slope: y = 3x + 1 has slope 3", close(linear_slope(steps, 3 * steps + 1), 3))

    # benjamini_hochberg
    check("benjamini_hochberg: [0.005, 0.5, 0.02] becomes [0.015, 0.5, 0.03]",
          close(benjamini_hochberg([0.005, 0.5, 0.02]), [0.015, 0.5, 0.03]))
    check("benjamini_hochberg: evenly spaced p-values 0.01..0.05 all become 0.05, and NaN is passed through",
          close(benjamini_hochberg([0.01, 0.02, np.nan, 0.03, 0.04, 0.05]), [0.05, 0.05, np.nan, 0.05, 0.05, 0.05]))

    # mean_absolute_error and time_split_evaluation
    check("mean_absolute_error: misses of 1, 0 and 2 average to 1",
          close(mean_absolute_error([1, 2, 3], [2, 2, 5]), 1.0))
    forecast_frame = pd.DataFrame({
        "unit": np.repeat(np.arange(12), 6), "year": np.tile(np.arange(2000, 2006), 12),
        "history": test_rng.normal(size=72), "housing": test_rng.normal(size=72)})
    forecast_frame["target"] = 2 * forecast_frame["history"] + 3 * forecast_frame["housing"]
    summary, predictions = time_split_evaluation(
        forecast_frame, "target", {"history only": ["history"], "history + housing": ["history", "housing"]},
        time_column="year", first_test_period=2004, cluster_column="unit", n_boot=SELF_TEST_BOOT)
    check("time_split_evaluation: the feature that truly drives the target removes the error",
          close(summary.loc["history + housing", "mae"], 0) and summary.loc["history only", "mae"] > 1
          and summary.loc["history + housing", "high"] < 0)
    check("time_split_evaluation: training uses only earlier years (48 rows) and testing the later ones (24 rows)",
          (summary["n_train"] == 48).all() and (summary["n_test"] == 24).all()
          and (summary["n_test_clusters"] == 12).all() and predictions["year"].min() == 2004)

    # describe_effect
    includes_zero = describe_effect(-0.03, -0.08, 0.02, "log points", 412, subject="The change", n_unit="sites")
    excludes_zero = describe_effect(0.12, 0.05, 0.2, "log points", 150, subject="The change at treated sites",
                                    n_unit="sites", above="more injury crashes", below="fewer injury crashes")
    not_estimated = describe_effect(float("nan"), float("nan"), float("nan"), "log points", 4)
    causal_words = ("cause", "causes", "caused", "leads to", "effect of", "because of", "due to")
    check("describe_effect: an interval across zero says so and draws no further conclusion",
          "includes zero" in includes_zero and "which fits" not in includes_zero
          and "n = 412 sites" in includes_zero)
    check("describe_effect: an interval away from zero names its side and the reading it fits",
          "which fits more injury crashes" in excludes_zero and "above zero" in excludes_zero
          and not any(word in excludes_zero for word in causal_words))
    check("describe_effect: a NaN estimate concludes nothing",
          "could not be estimated" in not_estimated and "n = 4 observations" in not_estimated)
    check("describe_effect: no causal phrasing in the neutral sentences",
          not any(word in includes_zero or word in not_estimated for word in causal_words))
    print(includes_zero)
    print(excludes_zero)
    print(not_estimated)


run_self_tests()
