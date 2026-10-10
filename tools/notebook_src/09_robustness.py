# %% [markdown]
# ## 9. Robustness checks
#
# The before and after comparison around new-construction sites rests on several choices made in
# Sections 1 and 5: the 250 m buffer, the three-year window, the injury crash as the outcome, the
# screen that removes sites with another completion nearby, and the way the 2020 reporting break
# is handled. None of these values is given by the data. This section re-estimates the same
# comparison with each choice changed, one at a time, and puts every estimate with its interval
# in one table and one chart. A conclusion is only kept at the end if it does not depend on a
# choice that could reasonably have been made differently.
#
# The section uses only the tables of Sections 5 and 6 and the functions of Section 7, so the
# central estimate is rebuilt here from the event table before anything is varied.

# %% [markdown]
# ### 9.1 Constants and the central comparison
#
# The central comparison is defined as follows. The treated sites are new-construction sites of
# the completion cohorts whose full window lies inside the study years. The comparison sites are
# located sites of any construction type that are still not completed at the end of the cohort's
# window. Each group passes its own screen from Section 5.4: a treated site has no neighbour
# completed in a different year of its window, and a comparison site has no completed neighbour
# at all. For every site the injury crashes within
# the buffer are summed over the `EVENT_WINDOW` years before the completion year and the
# `EVENT_WINDOW` years after it; the completion year itself is left out because the building is
# finished part of the way through it. The estimate is the mean log ratio of after to before at
# treated sites minus the same mean at comparison sites, taken inside each cohort so that both
# groups cover the same calendar years, and weighted by the number of treated sites per cohort.
# Clusters of sites are resampled for the interval.

# %%
def assert_state(*names):
    """Stop with a clear message when an object that an earlier section should have left is missing."""
    missing = [name for name in names if name not in globals()]
    assert not missing, f"objects of earlier sections are missing: {', '.join(missing)}"


# Tables and results of Sections 5, 6 and 8 that this section reads.
assert_state("sites", "site_year", "site_pairs", "event_panel", "crashes", "zip_year", "crash_tree",
             "crash_tree_rows", "event_units", "rq6_before", "rq6_after", "rq6_table", "rq8_bands", "rq8_top_band",
             "rq10_factors", "rq10_top_name", "rq10_unrecorded", "rq1", "stock_slope", "timing_slopes",
             "rq9_result")
# Functions of Sections 5, 7 and 8 that it calls.
assert_state("build_site_year", "build_event_panel", "site_crash_pairs", "site_neighbour_pairs",
             "connected_clusters", "aggregate_crashes_to_zip_year", "did_log_ratio", "stratified_difference",
             "log_ratio", "log_points_to_percent", "cohort_unit_table", "period_category_counts",
             "composition_shift", "hour_mix_shift", "factor_mix_shift", "stock_regression", "zip_equity_table",
             "deep_share_associations", "build_forecast_frame", "forecast_comparison")

# --- Constants used only in this section -------------------------------------------------------
# Alternative buffer radii in metres, one clearly smaller and one clearly larger than BUFFER_M.
# Both are among the radii profiled in Section 5.3.
ALTERNATIVE_RADII_M = [150, 400]
NEAR_RADIUS_M = min(ALTERNATIVE_RADII_M)   # the smaller radius, also used for the share questions (RQ6, RQ8, RQ10)
SHORT_WINDOW = 2                 # years; the alternative to EVENT_WINDOW on each side of completion
PLACEBO_SHIFT = EVENT_WINDOW     # years by which the placebo moves the completion date earlier
CENTRAL_TYPE = NEW_CONSTRUCTION
CENTRAL_OUTCOME = "injury_crashes"
ALTERNATIVE_OUTCOME = "crashes"
CLUSTER_UNIT = "cluster_id"      # resampling unit of the central estimate
BLOCK_UNIT = "block_id"          # one-kilometre squares, the alternative resampling unit

# One row per (site, cohort) carries these columns from the event table, next to its keys.
COHORT_UNIT_COLUMNS = ["cluster_id", "block_id", "contaminated", "near_completion_in_window",
                       "near_earlier_completion", "construction_type", "borough"]
COHORT_STRATA = ("cohort",)                 # the central comparison is made inside completion cohorts
COHORT_BOROUGH_STRATA = ("cohort", "borough")   # 9.9: inside cohorts and boroughs

# Forest plot: intervals reaching beyond this many percent on either side are cut at the edge of
# the axis and marked with an arrow, so one very wide interval does not flatten all the others.
# The full interval is always printed next to the row and in the table.
FOREST_AXIS_LIMIT = 30
FOREST_PLOT_SHARE = 0.56         # share of the axes width used by the plot; the rest holds the text columns
LARGEST_CLUSTERS_SHOWN = 3       # clusters whose joint weight is reported when the screen is switched off

# Kinds of row in the table of checks, and the names of the checks that other cells look up.
KIND_CENTRAL, KIND_CHECK, KIND_PLACEBO = "central", "check", "placebo"
CHECK_OUTCOME = "Outcome"                  # the row that counts all crashes (9.4); Section 10 reads it
CHECK_SCREEN = "Nearby completions"        # the rows that change the screen of Section 5.4 (9.5)

# Marker colour of each kind of row in the forest plot.
ROW_KINDS = {KIND_CENTRAL: ("Central estimate", PALETTE["treated"]),
             KIND_CHECK: ("One choice changed", PALETTE["neutral"]),
             KIND_PLACEBO: ("Placebo (no completion in the window)", PALETTE["placebo"])}

assert all(radius in RADIUS_CANDIDATES_M for radius in ALTERNATIVE_RADII_M)
assert SHORT_WINDOW < EVENT_WINDOW <= PLACEBO_SHIFT


# %%
def yearly_counts(site_year_frame, outcome, n_sites):
    """Sites-by-years matrix of one count column; row i is site i, column j is YEAR_START + j."""
    expected_sites = np.repeat(np.arange(n_sites), N_STUDY_YEARS)
    assert (site_year_frame["site_id"].to_numpy() == expected_sites).all(), "site_year must be sorted"
    return site_year_frame[outcome].to_numpy().reshape(n_sites, N_STUDY_YEARS)


def before_after_counts(units, counts, window, drop_years=()):
    """Add `before` and `after`: the outcome summed over `window` years on each side of the cohort year.

    The cohort year itself is never counted. Calendar years in `drop_years` are left out of both
    sums. Treated and comparison sites of one cohort lose the same years, so their log ratios are
    shortened in the same way and the difference between them stays a like-for-like comparison.
    """
    relative_years = np.concatenate([np.arange(-window, 0), np.arange(1, window + 1)])
    years = units["cohort"].to_numpy()[:, None] + relative_years[None, :]
    assert years.min() >= YEAR_START and years.max() <= YEAR_END, "a window leaves the study years"
    values = counts[units["site_id"].to_numpy()[:, None], years - YEAR_START]
    kept = ~np.isin(years, list(drop_years))
    assert kept[:, :window].any(axis=1).all() and kept[:, window:].any(axis=1).all(), "a period lost all its years"
    values = np.where(kept, values, 0)
    return units.assign(before=values[:, :window].sum(axis=1), after=values[:, window:].sum(axis=1))


def central_sample(units, keep_contaminated=False, treated_without_earlier_neighbour=False):
    """Rows of the central comparison: new-construction treated sites and all comparison sites.

    By default only sites that pass the screen of Section 5.4 are kept. `keep_contaminated`
    switches the screen off. `treated_without_earlier_neighbour` tightens it for treated sites,
    which must then have no neighbour completed before their window either.
    """
    is_treated = (units["group"] == "treated").to_numpy()
    keep = ~is_treated | (units["construction_type"] == CENTRAL_TYPE).to_numpy()
    if not keep_contaminated:
        keep &= ~units["contaminated"].to_numpy()
    if treated_without_earlier_neighbour:
        keep &= ~(is_treated & units["near_earlier_completion"].to_numpy())
    return units[keep].reset_index(drop=True)


def estimate_check(units, check, variant, kind=KIND_CHECK, cluster_column=CLUSTER_UNIT,
                   strata_columns=COHORT_STRATA, keeps_screen=True):
    """Difference-in-differences on the log ratio for one specification, as one table row.

    `keeps_screen` records whether both groups pass the screen of Section 5.4 in this
    specification; the summary of 9.11 reads it.

    Treated and comparison sites are compared inside the strata named by `strata_columns`. The
    default is the cohort, so both arms cover the same calendar years. A stratum with sites on
    one side only cannot be compared; its sites are left out and counted in the `dropped`
    columns. Whole clusters are resampled because sites in one cluster share crashes, and a site
    that serves several cohorts as a comparison is resampled with all its copies. No estimate is
    reported when either arm has fewer than MIN_GROUP_N sites.
    """
    strata = units[list(strata_columns)].astype(str).agg("|".join, axis=1).to_numpy()
    result = did_log_ratio(units["before"].to_numpy(), units["after"].to_numpy(),
                           (units["group"] == "treated").to_numpy(), strata=strata,
                           clusters=units[cluster_column].to_numpy())
    row = {"check": check, "variant": variant, "kind": kind, "keeps_screen": keeps_screen,
           "n_treated": result["n_treated"], "n_comparison": result["n_control"],
           "n_treated_dropped": result["n_treated_dropped"], "n_comparison_dropped": result["n_control_dropped"],
           "n_resampled_units": result["n_clusters"], "n_strata": result["n_strata"],
           "treated_change": result["treated_change"], "comparison_change": result["control_change"]}
    large_enough = min(result["n_treated"], result["n_control"]) >= MIN_GROUP_N
    for column in ["estimate", "low", "high", "p_value", "percent_change", "percent_low", "percent_high"]:
        row[column] = result[column] if large_enough else float("nan")
    return row


def relinked_sites(site_frame, radius_m):
    """The site table with `cluster_id` recomputed for buffers of `radius_m` metres.

    Buffers of a different size overlap differently, so the groups of sites that share crashes,
    and are resampled together, have to be rebuilt at the same radius.
    """
    first, second = site_neighbour_pairs(site_frame, radius_m)
    return site_frame.assign(cluster_id=connected_clusters(first, second, len(site_frame)))


def leave_one_cluster_out(units, strata_columns=COHORT_STRATA):
    """Point estimate in log points with each cluster that holds a treated site left out in turn.

    Returns a Series indexed by cluster id. No interval is computed, because the purpose is only
    to see how far one group of neighbouring sites can move the estimate.
    """
    change = log_ratio(units["before"].to_numpy(), units["after"].to_numpy())
    treated = (units["group"] == "treated").to_numpy()
    strata_codes, strata_labels = pd.factorize(units[list(strata_columns)].astype(str).agg("|".join, axis=1))
    clusters = units[CLUSTER_UNIT].to_numpy()
    estimates = {}
    for cluster in np.unique(clusters[treated]):
        kept = clusters != cluster
        estimates[cluster] = stratified_difference(change[kept], treated[kept], strata_codes[kept],
                                                   len(strata_labels))["estimate"]
    return pd.Series(estimates, name="estimate_without_cluster")


def interval_text(estimate, low, high, digits=2):
    """Estimate and interval in one table cell."""
    return f"{estimate:+.{digits}f} ({low:+.{digits}f} to {high:+.{digits}f})"


def reading(row):
    """Estimate, interval and sample of one table row as a phrase for the commentary."""
    if np.isnan(row["estimate"]):
        return (f"not reported, because an arm has fewer than {MIN_GROUP_N} sites ({row['n_treated']:,} treated, "
                f"{row['n_comparison']:,} comparison)")
    return (f"{row['percent_change']:+.1f}% ({CI_LEVEL:.0%} interval {row['percent_low']:+.1f}% to "
            f"{row['percent_high']:+.1f}%; {row['n_treated']:,} treated and {row['n_comparison']:,} comparison "
            f"sites)")


# One list of rows per check, keyed by the name of the check. A cell that is run again replaces
# its own entry, so the table of 9.11 never holds a specification twice.
robustness_checks = {}

# %%
n_sites = len(sites)
base_units = cohort_unit_table(event_panel, COHORT_UNIT_COLUMNS, keep_contaminated=True)
injury_counts = yearly_counts(site_year, CENTRAL_OUTCOME, n_sites)
central_units = central_sample(before_after_counts(base_units, injury_counts, EVENT_WINDOW))

assert (central_units.loc[central_units["group"] == "treated", "construction_type"] == CENTRAL_TYPE).all()
assert not central_units["contaminated"].any()
assert (central_units[["before", "after"]] >= 0).all().all()

central = estimate_check(central_units, "Central estimate",
                         f"{BUFFER_M} m, {EVENT_WINDOW} years, injury crashes", kind=KIND_CENTRAL)
robustness_checks["Central estimate"] = [central]

central_cohort_sizes = (central_units.groupby(["cohort", "group"], observed=True).size().unstack("group"))
display(central_cohort_sizes.rename_axis("completion cohort").rename_axis("sites", axis=1))
say(f"Central estimate: the before-to-after change in injury crashes within {BUFFER_M} m is "
    f"{log_points_to_percent(central['treated_change']):+.1f}% at treated sites and "
    f"{log_points_to_percent(central['comparison_change']):+.1f}% at comparison sites (means of the log ratio, expressed as "
    f"percent). The difference between the two is {reading(central)}, with {central['n_resampled_units']} "
    f"clusters resampled and a bootstrap p-value of {central['p_value']:.2f}.")

# %% [markdown]
# Injury crashes fall at both groups of sites between the two periods, and they fall a little
# more at the treated sites. The difference is small and its interval includes zero, so in the
# central specification the data are compatible with no difference between the streets around a
# newly completed building and the streets around a building that is not finished yet. The
# treated group is small because few new-construction sites pass the screen. Every check below
# is compared with this row.

# %% [markdown]
# ### 9.2 Buffer radius
#
# The 250 m radius was chosen in Section 5.3 as a compromise, and a different radius could give a
# different answer: a smaller buffer keeps only the crashes closest to the building but leaves
# small, noisy counts, and a larger one adds streets that have little to do with the building.
# The comparison is repeated at 150 m and 400 m. A change of radius does two things at once,
# because the screen for nearby completions uses the same distance, so a smaller buffer lets
# more sites pass and a larger one fewer. Each radius is therefore run twice: once with the
# screen and the clusters rebuilt at the new radius, and once with the sites of the central
# estimate kept and only the counting distance changed.

# %%
def radius_units(site_frame, pairs, crash_frame, radius_m, screen_units=None):
    """Central sample with crashes counted within `radius_m` (pairs must hold that radius).

    Without `screen_units` the screen is rebuilt at the same radius: a neighbour contaminates a
    site if it lies within the new buffer. With `screen_units` the sites of that table are kept
    and only the counting radius changes.
    """
    site_year_at_radius = build_site_year(pairs, crash_frame, len(site_frame))
    counts = yearly_counts(site_year_at_radius, CENTRAL_OUTCOME, len(site_frame))
    if screen_units is not None:
        return before_after_counts(screen_units.drop(columns=["before", "after"]), counts, EVENT_WINDOW)
    sites_at_radius = relinked_sites(site_frame, radius_m)
    panel_at_radius = build_event_panel(sites_at_radius, site_year_at_radius, radius_m=radius_m)
    units_at_radius = cohort_unit_table(panel_at_radius, COHORT_UNIT_COLUMNS, keep_contaminated=True)
    return central_sample(before_after_counts(units_at_radius, counts, EVENT_WINDOW))


radius_results = {}
for radius in ALTERNATIVE_RADII_M:
    if radius < BUFFER_M:
        # A smaller buffer is a subset of the pairs already found, so no new tree query is needed.
        pairs_at_radius = site_pairs[site_pairs["distance_m"] <= radius]
    else:
        pairs_at_radius = site_crash_pairs(sites, crash_tree, crash_tree_rows, radius)
    assert not pairs_at_radius.duplicated(["site_id", "crash_row"]).any()
    rescreened = estimate_check(radius_units(sites, pairs_at_radius, crashes, radius), "Buffer radius",
                                f"{radius} m, screen rebuilt at {radius} m")
    same_sites = estimate_check(radius_units(sites, pairs_at_radius, crashes, radius, screen_units=central_units),
                                "Buffer radius", f"{radius} m, same sites as the central estimate")
    radius_results[radius] = (rescreened, same_sites)
release("pairs_at_radius")
robustness_checks["Buffer radius"] = [row for pair in radius_results.values() for row in pair]

for radius, (rescreened, same_sites) in radius_results.items():
    say(f"At {radius} m with the screen rebuilt the difference is {reading(rescreened)}. Counting within "
        f"{radius} m around the sites of the central estimate gives {reading(same_sites)}.")

# %% [markdown]
# When the sites are held fixed, the counting distance hardly moves the estimate; the interval
# is wider at 150 m, where the counts per site are smaller. When the screen is rebuilt, the
# sample changes a great deal and the point estimate changes sign at 150 m, although every one
# of the four intervals includes zero. The radius therefore matters through which sites it
# admits to the comparison and not through how far from the building crashes are counted.

# %% [markdown]
# The questions about shares (RQ6, RQ8 and RQ10) count injury crashes within the same buffer, so
# they are repeated at the smaller radius as well. The sites stay those of Section 8 and only
# the counting distance changes. For RQ8 and RQ10 the row is the part of the day and the factor
# group that had the largest gap at the full radius.

# %%
near_pairs = site_pairs[site_pairs["distance_m"] <= NEAR_RADIUS_M]
near_before, near_after = period_category_counts(
    event_units, near_pairs, crashes, np.where(crashes["ped_cyc_harm"].to_numpy(), 0, 1),
    len(PEDESTRIAN_CYCLIST_LABELS))
rq6_near = composition_shift(event_units, near_before, near_after,
                             PEDESTRIAN_CYCLIST_LABELS)[0].loc[PEDESTRIAN_CYCLIST_LABELS[0]]
_, rq8_near_bands, _, _ = hour_mix_shift(event_units, near_pairs, crashes)
rq10_near_factors, rq10_near_unrecorded, rq10_near_sample = factor_mix_shift(event_units, near_pairs, crashes)
release("near_pairs")

share_rows = [
    ("RQ6: pedestrian and cyclist share", rq6_table.loc[ALL_BOROUGHS_LABEL], rq6_near),
    (f"RQ8: share in the {rq8_top_band.lower()}", rq8_bands.loc[rq8_top_band], rq8_near_bands.loc[rq8_top_band]),
    (f"RQ10: share of \"{rq10_top_name}\"", rq10_factors.loc[rq10_top_name], rq10_near_factors.loc[rq10_top_name]),
    ("RQ10 check: share with no factor recorded", rq10_unrecorded, rq10_near_unrecorded),
]
shares_at_radius = pd.DataFrame(
    [(name, interval_text(full["gap"], full["low"], full["high"]), interval_text(near["gap"], near["low"], near["high"]))
     for name, full, near in share_rows],
    columns=["gap, completed minus comparison sites (percentage points)", f"within {BUFFER_M} m",
             f"within {NEAR_RADIUS_M} m"]).set_index("gap, completed minus comparison sites (percentage points)")
with pd.option_context("display.max_colwidth", None):
    display(shares_at_radius)
say(f"Each cell gives the gap and its {CI_LEVEL:.0%} interval, not widened for several tests. Within "
    f"{NEAR_RADIUS_M} m the shares rest on {rq10_near_sample['treated_crashes']:,} site-level injury crash counts "
    f"with a recorded factor around completed sites. After adjustment, {int(rq8_near_bands['flagged'].sum())} of "
    f"the {len(rq8_near_bands)} parts of the day and {int(rq10_near_factors['flagged'].sum())} of the "
    f"{len(rq10_near_factors)} factor groups keep an interval away from zero at that radius.")

# %% [markdown]
# The pedestrian and cyclist gap and the afternoon gap are almost the same at the smaller radius,
# with wider intervals because fewer crashes are counted. The gap for the factor group is larger
# at the smaller radius, and taken alone its interval excludes zero there. It is one of several
# groups tested together, and after the adjustment no group is kept at either radius, so it is
# reported and not counted. The check on unrecorded factors keeps its sign and an interval that
# excludes zero, so the shares among recorded factors carry the caveat of RQ10 at both radii.

# %% [markdown]
# ### 9.3 Length of the window
#
# Three years on each side is long enough to average out single bad years, but a long window
# reaches further from the completion date, where other changes to the street have had more time
# to happen, and it limits the analysis to the few cohorts with three observed years on both
# sides. A window of 2 years is tried in two ways. The first keeps the sites of the central
# estimate and drops the outermost year on each side. The second rebuilds the event table for a
# 2-year window, which admits two more completion cohorts and applies the screen over the
# shorter span.

# %%
short_same_sites = estimate_check(
    central_sample(before_after_counts(base_units, injury_counts, SHORT_WINDOW)),
    "Window", f"{SHORT_WINDOW} years, same sites as the central estimate")

# With a shorter window more completion years have a full window inside the study years, and
# the screen looks for neighbouring completions in a shorter span, so the table is rebuilt.
short_panel = build_event_panel(sites, site_year, window=SHORT_WINDOW)
short_units = central_sample(before_after_counts(
    cohort_unit_table(short_panel, COHORT_UNIT_COLUMNS, keep_contaminated=True), injury_counts, SHORT_WINDOW))
short_all_cohorts = estimate_check(short_units, "Window", f"{SHORT_WINDOW} years, every cohort with a full window")
robustness_checks["Window"] = [short_same_sites, short_all_cohorts]

short_cohorts = sorted(short_units["cohort"].unique())
say(f"With {SHORT_WINDOW} years on each side and the same sites, the difference is {reading(short_same_sites)}. "
    f"Rebuilding the table for a {SHORT_WINDOW}-year window admits the cohorts {short_cohorts[0]} to "
    f"{short_cohorts[-1]} and gives {reading(short_all_cohorts)}.")
release("short_panel")

# %% [markdown]
# Both versions give a difference close to the central one, slightly further below zero. The
# version with more cohorts has more than twice as many treated sites and a narrower interval,
# whose upper end still lies above zero.

# %% [markdown]
# ### 9.4 Outcome: all crashes against injury crashes
#
# From 2020 the number of recorded crashes without injury drops sharply and stays low
# (Section 3), which is a change in what gets reported and not in what happens on the street.
# Every cohort's window contains years on both sides of that change, so a count of all crashes
# compares a period recorded under the old practice with one recorded under the new practice.
# This check replaces the injury crash by the count of all crashes and changes nothing else. It
# shows how large the reporting break is in a before and after count, and whether the comparison
# sites, which went through the same break in the same calendar years, remove it.

# %%
all_crash_counts = yearly_counts(site_year, ALTERNATIVE_OUTCOME, n_sites)
all_crash_units = central_sample(before_after_counts(base_units, all_crash_counts, EVENT_WINDOW))
all_crashes = estimate_check(all_crash_units, CHECK_OUTCOME, "All crashes, including property damage only")
robustness_checks[CHECK_OUTCOME] = [all_crashes]

outcome_changes = pd.DataFrame(
    {"treated sites (%)": [log_points_to_percent(central["treated_change"]), log_points_to_percent(all_crashes["treated_change"])],
     "comparison sites (%)": [log_points_to_percent(central["comparison_change"]), log_points_to_percent(all_crashes["comparison_change"])],
     "difference (%)": [central["percent_change"], all_crashes["percent_change"]],
     "interval low (%)": [central["percent_low"], all_crashes["percent_low"]],
     "interval high (%)": [central["percent_high"], all_crashes["percent_high"]]},
    index=pd.Index(["Injury crashes", "All crashes"], name="before-to-after change"))
display(outcome_changes.round(1))
say(f"Before any comparison, all crashes near treated sites change by "
    f"{log_points_to_percent(all_crashes['treated_change']):+.1f}% from the {EVENT_WINDOW} years before completion to the "
    f"{EVENT_WINDOW} years after, and injury crashes by {log_points_to_percent(central['treated_change']):+.1f}%. The "
    f"comparison sites change by {log_points_to_percent(all_crashes['comparison_change']):+.1f}% and "
    f"{log_points_to_percent(central['comparison_change']):+.1f}%. After differencing, the estimate for all crashes is "
    f"{reading(all_crashes)}.")

# %% [markdown]
# The break is plain in the undifferenced numbers. Counted as all crashes, the streets around a
# new building appear to lose a large share of their crashes after completion, far more than
# the fall in injury crashes at the same sites (both figures are printed above). A before and
# after count without a comparison group would report this as a large improvement near new
# housing. The comparison sites show almost the same fall, so the difference between the groups
# is again small with an interval that includes zero, and it agrees with the injury-crash
# estimate. Two things follow. The all-crash
# count cannot be used in any comparison over time that lacks a comparison group observed in the
# same years, and the central conclusion does not depend on which of the two outcomes is used
# once such a group is present. The injury crash stays the primary outcome because it does not
# need the comparison group to cancel a reporting change of this size exactly.

# %% [markdown]
# ### 9.5 The screen for nearby completions
#
# The central estimate drops every site with another completion nearby in its window, and every
# comparison site with any completed neighbour. That screen removes most of the candidates, and
# the sites that remain lie in the less dense parts of the programme, so the result may describe
# an unusual subset. The first variant switches the screen off and uses every candidate site,
# which brings back the problem the screen was built for: comparison streets that already have
# a completed building. To see which side of the comparison then moves the estimate, the screen
# is also applied to one group at a time. Without the screen, neighbouring buildings of one
# development all enter as separate rows that count the same streets, so the estimate is
# recomputed with each cluster of treated sites left out in turn, and once more without the
# cluster that moves it most. The last variant makes the screen stricter for treated sites,
# which must have no neighbour completed before their window either.

# %%
injury_units = before_after_counts(base_units, injury_counts, EVENT_WINDOW)
unscreened_units = central_sample(injury_units, keep_contaminated=True)
is_treated_unit = (unscreened_units["group"] == "treated").to_numpy()
is_clean_unit = ~unscreened_units["contaminated"].to_numpy()
unscreened = estimate_check(unscreened_units, CHECK_SCREEN, "Screen switched off: every candidate site",
                            keeps_screen=False)
only_comparison_screened = estimate_check(unscreened_units[is_treated_unit | is_clean_unit], CHECK_SCREEN,
                                          "Only comparison sites screened", keeps_screen=False)
only_treated_screened = estimate_check(unscreened_units[~is_treated_unit | is_clean_unit], CHECK_SCREEN,
                                       "Only treated sites screened", keeps_screen=False)

# How far one cluster can move the estimate, without the screen and in the central sample.
unscreened_without_one = leave_one_cluster_out(unscreened_units)
central_without_one = leave_one_cluster_out(central_units)
dominant_cluster = (unscreened_without_one - unscreened["estimate"]).abs().idxmax()
in_dominant_cluster = (unscreened_units[CLUSTER_UNIT] == dominant_cluster).to_numpy()
without_dominant = estimate_check(unscreened_units[~in_dominant_cluster], CHECK_SCREEN,
                                  "Screen off, without the most influential cluster", keeps_screen=False)
strict = estimate_check(central_sample(injury_units, treated_without_earlier_neighbour=True),
                        CHECK_SCREEN, "Stricter: treated sites with no completed neighbour at all")
robustness_checks[CHECK_SCREEN] = [unscreened, only_comparison_screened, only_treated_screened, without_dominant,
                                   strict]

contaminated_comparison_share = pct((~is_treated_unit & ~is_clean_unit).sum(), (~is_treated_unit).sum())
largest_clusters_share = pct(unscreened_units.groupby(CLUSTER_UNIT).size().nlargest(LARGEST_CLUSTERS_SHOWN).sum(),
                             len(unscreened_units))
say(f"With the screen switched off the difference is {reading(unscreened)}. In that sample "
    f"{contaminated_comparison_share:.0f}% of the comparison rows have a completed building within {BUFFER_M} m, "
    f"and the {LARGEST_CLUSTERS_SHOWN} largest of its {unscreened['n_resampled_units']} clusters hold "
    f"{largest_clusters_share:.0f}% of the rows. With every treated candidate and only the comparison sites "
    f"screened it is {reading(only_comparison_screened)}. With only the treated sites screened and every "
    f"comparison candidate it is {reading(only_treated_screened)}. Requiring in addition to the full screen that "
    f"a treated site has no neighbour completed before its window gives {reading(strict)}.")

# The treated rows of the most influential cluster: which buildings they are and what their streets counted.
assert (sites["site_id"].to_numpy() == np.arange(len(sites))).all(), "site_id must be the row position"
dominant_rows = unscreened_units[in_dominant_cluster & is_treated_unit]
dominant_buildings = sites.iloc[dominant_rows["site_id"].to_numpy()]
dominant_projects = dominant_buildings["project_name"].astype(str).value_counts()
dominant_zips = dominant_buildings["zip"].astype(str).value_counts()
dominant_cohorts = dominant_rows.groupby("cohort").size()
say(f"Leaving out one cluster with treated sites at a time moves the estimate without the screen between "
    f"{log_points_to_percent(unscreened_without_one.min()):+.1f}% and "
    f"{log_points_to_percent(unscreened_without_one.max()):+.1f}% ({len(unscreened_without_one)} clusters). The "
    f"cluster whose removal moves it most holds {len(dominant_rows):,} of the {int(is_treated_unit.sum()):,} "
    f"treated rows ({' and '.join(f'{count} in the {cohort} cohort' for cohort, count in dominant_cohorts.items())}). "
    f"These rows are buildings of {len(dominant_projects)} projects, the largest being "
    f"\"{dominant_projects.index[0]}\" ({dominant_projects.iloc[0]} buildings); "
    f"{pct(dominant_zips.iloc[0], len(dominant_rows)):.0f}% of them stand in ZIP code {dominant_zips.index[0]}, and the median number of units per building "
    f"is {dominant_buildings['total_units'].median():.0f}. Injury crashes within {BUFFER_M} m per treated row of "
    f"the cluster (median): {dominant_rows['before'].median():.0f} in the {EVENT_WINDOW} years before completion "
    f"and {dominant_rows['after'].median():.0f} in the {EVENT_WINDOW} years after. Without that cluster the "
    f"difference is {reading(without_dominant)}.")
say(f"The same exercise on the central sample moves the central estimate between "
    f"{log_points_to_percent(central_without_one.min()):+.1f}% and "
    f"{log_points_to_percent(central_without_one.max()):+.1f}% ({len(central_without_one)} clusters), against "
    f"{central['percent_change']:+.1f}% with all of them.")

# %% [markdown]
# The stricter screen leaves the estimate where it was. Switching the screen off turns the
# estimate positive, with an interval far wider than the central one and very lopsided. The
# comparison group is not the reason. With clean treated sites and every comparison candidate
# the estimate stays negative, and it is positive when every treated candidate enters, whichever
# comparison sites are used. One development is the reason. It is a group of small houses built
# on streets that had almost no injury crashes before. Every house counts those streets again,
# and a rise from near zero is a very large log ratio. Without that cluster the estimate is
# small and its interval includes zero. The rows are printed above. Leaving out any one cluster
# of the central sample hardly moves the central estimate, so the central result is not carried
# by one place in the same way.
#
# The estimate without the screen is therefore not read as a sign of more crashes after
# completion in the dense parts of the programme. Taking out the cluster that raises an estimate
# most has to lower it, so the row without it does not show the opposite either. It shows that
# the positive sign rests on one place. The rows without the screen also remain weaker than the
# screened ones, because most of their comparison streets have a completed building of their
# own, which can hide part of a difference.

# %% [markdown]
# ### 9.6 The year 2020
#
# In 2020 traffic itself changed for several months, on top of the change in reporting, and
# injury crashes dipped in that year. If the dip was different around treated and comparison
# sites, it could produce or hide a difference. Keeping only the cohorts whose window avoids
# 2020 is not possible, as the table below shows: with three years on each side, every usable
# cohort has 2020 inside its window. Two checks are possible. One leaves
# the calendar year 2020 out of every count, so that the cohorts completed before it lose one of
# their after years and those completed later lose one of their before years, in both groups
# alike. The other drops the cohort completed in 2020, whose before years all precede the break
# and whose after years all follow it.

# %%
window_years = pd.DataFrame(
    {"first year": [cohort - EVENT_WINDOW for cohort in sorted(base_units["cohort"].unique())],
     "last year": [cohort + EVENT_WINDOW for cohort in sorted(base_units["cohort"].unique())]},
    index=pd.Index(sorted(base_units["cohort"].unique()), name="completion cohort"))
window_years[f"{BREAK_YEAR} falls in"] = [
    "the completion year (not counted)" if cohort == BREAK_YEAR
    else ("the years after" if cohort < BREAK_YEAR else "the years before") for cohort in window_years.index]
display(window_years)
assert ((window_years["first year"] <= BREAK_YEAR) & (window_years["last year"] >= BREAK_YEAR)).all()

without_break_year = estimate_check(
    central_sample(before_after_counts(base_units, injury_counts, EVENT_WINDOW, drop_years=[BREAK_YEAR])),
    str(BREAK_YEAR), f"Calendar year {BREAK_YEAR} left out of the counts")
without_break_cohort = estimate_check(central_units[central_units["cohort"] != BREAK_YEAR],
                                      str(BREAK_YEAR), f"Without the {BREAK_YEAR} completion cohort")
robustness_checks[str(BREAK_YEAR)] = [without_break_year, without_break_cohort]
say(f"Leaving the calendar year {BREAK_YEAR} out of every count gives {reading(without_break_year)}. Dropping "
    f"the {BREAK_YEAR} completion cohort gives {reading(without_break_cohort)}.")

# %% [markdown]
# Neither change moves the estimate out of the central interval. Without the year 2020 it is
# almost identical, and without the 2020 cohort it is closer to zero with a slightly wider
# interval, as expected from a smaller sample.

# %% [markdown]
# ### 9.7 The unit that is resampled
#
# The interval comes from resampling clusters of sites linked within the buffer. Section 5.4
# showed that a few clusters are very large, and an interval that rests on a handful of dominant
# units can be unstable. The one-kilometre blocks of Section 5.4 cut the city into squares of
# equal size, so no unit can grow along a chain of neighbours. This check keeps the sample and
# the point estimate and only resamples blocks in place of clusters.

# %%
by_block = estimate_check(central_units, "Interval", "One-kilometre blocks resampled instead of clusters",
                          cluster_column=BLOCK_UNIT)
robustness_checks["Interval"] = [by_block]
largest_cluster_share = pct(central_units.groupby(CLUSTER_UNIT).size().max(), len(central_units))
largest_block_share = pct(central_units.groupby(BLOCK_UNIT).size().max(), len(central_units))
say(f"The point estimate is the same by construction. Resampling {by_block['n_resampled_units']} blocks "
    f"instead of {central['n_resampled_units']} clusters gives an interval of {by_block['percent_low']:+.1f}% to "
    f"{by_block['percent_high']:+.1f}%, against {central['percent_low']:+.1f}% to {central['percent_high']:+.1f}% "
    f"for clusters. The largest cluster holds {largest_cluster_share:.0f}% of the rows of the central sample "
    f"and the largest block {largest_block_share:.0f}%.")

# %% [markdown]
# The two intervals are close to each other and both include zero. After the screen, the very
# large clusters are no longer dominant in the sample, since sites in dense chains rarely pass
# it, so the choice of resampling unit makes little difference for the central estimate. That
# does not carry over to analyses that use unscreened sites, as 9.5 shows.

# %% [markdown]
# ### 9.8 Placebo: a completion date that is too early
#
# A difference-in-differences assumes that the two groups would have changed alike without the
# completion. If streets that are about to get a new building were already on a different path,
# the central estimate would pick that up and it would look like a change at completion. The
# placebo tests this where the data allow. Each new-construction site is entered `PLACEBO_SHIFT`
# years before its real completion year, so both the false before years and the false after
# years lie before the building is finished, and it is compared with the clean comparison sites
# of that earlier cohort. Only sites completed late enough to have six observed years before
# completion can be used, which is a different and later set of sites than in the central
# estimate. With the 3-year window the last false after year is the real completion year, in
# which the building opens part of the way through, so the placebo is also run with a 2-year
# window that stops one year short of it. A placebo estimate should be near zero.

# %%
def placebo_units(site_frame, comparison_units, shift, radius_m=BUFFER_M, window=EVENT_WINDOW):
    """Units of a placebo in which treated sites are dated `shift` years before their real completion.

    A new-construction site really completed in year c is entered in cohort c - shift. With
    shift >= window its whole placebo window ends no later than c, so nothing is completed at
    the site during the placebo "after" years. The comparison sites are the clean comparison
    sites of the same cohort, which are still not completed at the end of that window. A placebo
    site is dropped when another site within `radius_m` is completed inside its placebo window
    in a year other than its own, which is the screen of Section 5.4 applied to the false date.
    """
    completion = site_frame["completion_year"].astype("float64").to_numpy()
    cohorts = sorted(comparison_units["cohort"].unique())
    is_new_construction = (site_frame["construction_type"] == CENTRAL_TYPE).to_numpy()
    first, second = site_neighbour_pairs(site_frame, radius_m)
    neighbour_completion = completion[second]
    placebo_parts = []
    for cohort in cohorts:
        in_window = ((first != second) & (neighbour_completion >= cohort - window)
                     & (neighbour_completion <= cohort + window) & (neighbour_completion != completion[first]))
        is_contaminated = np.bincount(first[in_window], minlength=len(site_frame)) > 0
        chosen = site_frame[(completion == cohort + shift) & is_new_construction & ~is_contaminated]
        placebo_parts.append(pd.DataFrame({
            "site_id": chosen["site_id"].to_numpy(), "cohort": np.int16(cohort), "group": "treated",
            "cluster_id": chosen["cluster_id"].to_numpy(), "block_id": chosen["block_id"].to_numpy()}))
    keep_columns = ["site_id", "cohort", "group", "cluster_id", "block_id"]
    clean_comparison = comparison_units.loc[(comparison_units["group"] == "comparison")
                                            & ~comparison_units["contaminated"], keep_columns]
    units = pd.concat([*placebo_parts, clean_comparison.astype({"group": "object"})], ignore_index=True)
    assert not units.duplicated(["site_id", "cohort"]).any(), "a site is on both sides of one cohort"
    return units


placebo_base = placebo_units(sites, base_units, PLACEBO_SHIFT)
placebo_treated_sites = placebo_base.loc[placebo_base["group"] == "treated", "site_id"].to_numpy()
real_completion = sites["completion_year"].astype("float64").to_numpy()
placebo_cohort = placebo_base.loc[placebo_base["group"] == "treated", "cohort"].to_numpy()
# No placebo site is completed before the end of its placebo window.
assert (real_completion[placebo_treated_sites] == placebo_cohort + PLACEBO_SHIFT).all()

placebo_full = estimate_check(before_after_counts(placebo_base, injury_counts, EVENT_WINDOW), "Placebo",
                              f"Completion dated {PLACEBO_SHIFT} years early, {EVENT_WINDOW}-year window",
                              kind=KIND_PLACEBO)
placebo_short = estimate_check(before_after_counts(placebo_base, injury_counts, SHORT_WINDOW), "Placebo",
                               f"Completion dated {PLACEBO_SHIFT} years early, {SHORT_WINDOW}-year window",
                               kind=KIND_PLACEBO)
robustness_checks["Placebo"] = [placebo_full, placebo_short]
real_years = np.unique(real_completion[placebo_treated_sites]).astype(int)
say(f"The placebo can be built for new-construction sites really completed from {real_years.min()} to "
    f"{real_years.max()}. With the {EVENT_WINDOW}-year window, whose last year is the real completion year, the "
    f"difference is {reading(placebo_full)}. With the {SHORT_WINDOW}-year window, which ends the year before "
    f"the real completion, it is {reading(placebo_short)}.")

# %% [markdown]
# Both placebo estimates are close to zero and their intervals include it, so the years before
# completion give no sign that sites about to be completed were already moving away from the
# comparison sites. The placebo intervals are about as wide as the central one. The test can
# therefore rule out a large difference in earlier trends, but not one of roughly the size of
# the central estimate itself.

# %% [markdown]
# ### 9.9 Comparison inside boroughs
#
# The central comparison is made inside completion cohorts only. Treated and comparison sites
# are spread differently over the boroughs, as the table below shows, and neither injury crashes
# nor their reporting moved alike in every borough after the break year. A difference between
# boroughs could then pass for a difference at completion.
# This check compares treated and comparison sites inside each combination of cohort and
# borough. A combination with sites on one side only cannot be compared, so its sites are left
# out and counted. The same stratification is applied to the sample without the screen, with
# and without the cluster singled out in 9.5, and to the citywide gap in the pedestrian and
# cyclist share of RQ6.

# %%
borough_mix = central_units.groupby(["borough", "group"], observed=True).size().unstack("group", fill_value=0)
borough_mix_pct = (PERCENT * borough_mix / borough_mix.sum()).round(1).add_suffix(" (% of the group)")
display(pd.concat([borough_mix.add_suffix(" (sites)"), borough_mix_pct], axis=1))

inside_borough = estimate_check(central_units, "Comparison group", "Compared inside cohort and borough",
                                strata_columns=COHORT_BOROUGH_STRATA)
unscreened_inside_borough = estimate_check(unscreened_units, "Comparison group",
                                           "Inside cohort and borough, screen switched off",
                                           strata_columns=COHORT_BOROUGH_STRATA, keeps_screen=False)
without_dominant_inside_borough = estimate_check(unscreened_units[~in_dominant_cluster], "Comparison group",
                                                 "Inside cohort and borough, screen off, without that cluster",
                                                 strata_columns=COHORT_BOROUGH_STRATA, keeps_screen=False)
robustness_checks["Comparison group"] = [inside_borough, unscreened_inside_borough, without_dominant_inside_borough]

# RQ6 again, with each cohort split by borough before the two arms are compared.
rq6_borough_units = event_units.assign(
    cohort=event_units["cohort"].astype(str) + "|" + event_units["borough"].astype(str))
rq6_inside_table, rq6_inside_sample = composition_shift(rq6_borough_units, rq6_before, rq6_after,
                                                        PEDESTRIAN_CYCLIST_LABELS)
rq6_inside = rq6_inside_table.loc[PEDESTRIAN_CYCLIST_LABELS[0]]
rq6_central = rq6_table.loc[ALL_BOROUGHS_LABEL]

say(f"Inside cohort and borough the difference is {reading(inside_borough)}, in {inside_borough['n_strata']} "
    f"combinations of cohort and borough; {inside_borough['n_treated_dropped']} treated and "
    f"{inside_borough['n_comparison_dropped']} comparison rows are in combinations without both groups and are "
    f"left out. The central estimate, inside cohorts only, is {reading(central)}.")
say(f"Without the screen, the difference inside cohort and borough is {reading(unscreened_inside_borough)} "
    f"({unscreened_inside_borough['n_treated_dropped']} treated and "
    f"{unscreened_inside_borough['n_comparison_dropped']} comparison rows left out), against "
    f"{reading(unscreened)} inside cohorts only. Without the most influential cluster of 9.5 it is "
    f"{reading(without_dominant_inside_borough)}.")
say(f"The gap in the pedestrian and cyclist share of RQ6 is {rq6_inside['gap']:+.2f} percentage points "
    f"({CI_LEVEL:.0%} interval {rq6_inside['low']:+.2f} to {rq6_inside['high']:+.2f}) when it is computed inside "
    f"cohort and borough, against {rq6_central['gap']:+.2f} ({rq6_central['low']:+.2f} to "
    f"{rq6_central['high']:+.2f}) inside cohorts only.")

# %% [markdown]
# The two groups do sit in different boroughs: Brooklyn holds a much larger share of the treated
# sites than of the comparison sites, and Queens the reverse. Comparing inside boroughs does not
# change the reading. The estimate moves slightly further below zero, its interval still
# includes zero, no treated site is lost and only a few comparison rows are. In the sample
# without the screen the estimate stays positive and becomes a little smaller. Without the
# cluster singled out in 9.5 it is close to zero, with a narrow interval that includes zero.
# The RQ6 gap shrinks a little and keeps an interval that includes zero. The different borough mix is therefore not what produces these null results. A borough
# is still a coarse unit, and inside one of them treated and comparison sites can stand in
# different kinds of neighbourhood.

# %% [markdown]
# ### 9.10 ZIP-level results on reported ZIP codes only
#
# The three ZIP-level questions (RQ1, RQ2 and RQ9) count crashes by ZIP code, and part of those
# codes was recovered from coordinates in Section 6.1. A recovered code can be wrong near a
# boundary, and the share of crashes that needed recovery is not the same in every year, so a
# small result at this level could come from the recovery. This check rebuilds the crash columns
# of the panel from the crashes whose ZIP code was reported by the police. The ZIP codes and the
# housing columns stay the same. The forecast comparison of RQ1, the two slopes of RQ2 and the
# two correlations of RQ9 are then estimated again.

# %%
def reported_zip_panel(panel, crash_frame):
    """The ZIP by year panel with its crash columns recounted from crashes whose ZIP code was reported."""
    is_reported = (~crash_frame["zip_imputed"] & crash_frame["in_zip_panel"]).to_numpy()
    counts = aggregate_crashes_to_zip_year(crash_frame[is_reported])
    counts = counts.assign(zip=counts["zip"].astype(str), year=counts["year"].astype(int)).set_index(["zip", "year"])
    keys = pd.MultiIndex.from_arrays([panel["zip"].astype(str), panel["year"].astype(int)])
    aligned = counts[CRASH_PANEL_MEASURES].reindex(keys, fill_value=0)
    reported = panel.copy()
    for column in CRASH_PANEL_MEASURES:
        reported[column] = aligned[column].to_numpy().astype(panel[column].dtype)
    # The panel already holds how many of its crashes rest on a recovered code (Section 6.1).
    assert (reported["crashes"] == panel["crashes"] - panel["crashes_zip_imputed"]).all()
    assert (reported["injury_crashes"] == panel["injury_crashes"] - panel["injury_crashes_zip_imputed"]).all()
    return reported


reported_panel = reported_zip_panel(zip_year, crashes)
reported_forecast = forecast_comparison(build_forecast_frame(reported_panel))[0].loc[FORECAST_FULL]
reported_stock, reported_timing = stock_regression(reported_panel)
reported_equity = deep_share_associations(zip_equity_table(reported_panel))
same_year_all = timing_slopes[timing_slopes["shift"] == 0].iloc[0]
same_year_reported = reported_timing[reported_timing["shift"] == 0].iloc[0]

zip_level_checks = pd.DataFrame(
    [("RQ1: change in forecast error when housing is added (injury crashes per ZIP per year)",
      interval_text(rq1["mae_minus_baseline"], rq1["low"], rq1["high"]),
      interval_text(reported_forecast["mae_minus_baseline"], reported_forecast["low"], reported_forecast["high"])),
     (f"RQ2: slope on the cumulative stock (% per {UNITS_PER_STEP} units)",
      interval_text(stock_slope["estimate_pct"], stock_slope["low_pct"], stock_slope["high_pct"]),
      interval_text(reported_stock["estimate_pct"], reported_stock["low_pct"], reported_stock["high_pct"])),
     (f"RQ2: slope on completions in the same year (% per {UNITS_PER_STEP} units)",
      interval_text(same_year_all["estimate_pct"], same_year_all["low_pct"], same_year_all["high_pct"]),
      interval_text(same_year_reported["estimate_pct"], same_year_reported["low_pct"],
                    same_year_reported["high_pct"])),
     *[(f"RQ9: within-borough rank correlation, {RQ9_OUTCOMES[column][0].split(':')[1].strip()}",
        interval_text(rq9_result.loc[column, "rho_within_borough"], rq9_result.loc[column, "low"],
                      rq9_result.loc[column, "high"], digits=3),
        interval_text(reported_equity.loc[column, "rho_within_borough"], reported_equity.loc[column, "low"],
                      reported_equity.loc[column, "high"], digits=3)) for column in RQ9_OUTCOMES]],
    columns=["estimate", "all ZIP codes of the panel", "reported ZIP codes only"]).set_index("estimate")
with pd.option_context("display.max_colwidth", None):
    display(zip_level_checks)
reported_ped = reported_equity.loc["ped_cyc_share_pct"]
say(f"{pct(zip_year['injury_crashes_zip_imputed'].sum(), zip_year['injury_crashes'].sum()):.1f}% of the injury "
    f"crashes in the panel have a recovered ZIP code and are left out of the right-hand column. Each cell gives "
    f"the estimate and its {CI_LEVEL:.0%} interval. On reported codes the completion-year slope of RQ2 has an "
    f"adjusted p-value of {same_year_reported['p_adjusted']:.3f} (all codes: {same_year_all['p_adjusted']:.3f}), "
    f"and the pedestrian and cyclist correlation of RQ9 an adjusted p-value of {reported_ped['p_adjusted']:.3f} "
    f"(all codes: {rq9_result.loc['ped_cyc_share_pct', 'p_adjusted']:.3f}), on {int(reported_ped['n'])} ZIP codes.")
release("reported_panel")

# %% [markdown]
# The loss of forecast accuracy in RQ1 depends on the recovered codes. It disappears here: on
# reported codes the models with housing features are no worse than crash history alone, and
# the interval includes zero. RQ1 should therefore not be read as
# housing making a forecast worse. The statement that holds in both columns is that housing did
# not improve it. The completion-year slope of RQ2 does not fade on reported codes. It is
# negative in both columns with an interval that excludes zero, and only its adjusted p-value
# moves across the threshold. The reason it is not counted is given in RQ2: it belongs to
# preserved units, which bring no new residents. The RQ9 correlation with the pedestrian and
# cyclist share is almost unchanged and its interval now includes zero, which fits its reading
# as a hint. The
# slope on the cumulative stock and the correlation with deaths include zero in both columns.
#
# Leaving out the recovered codes is not a neutral choice, because it removes a share of the
# crashes that is not the same in every year, so neither column is the true one. A ZIP-level
# result is treated as solid in this notebook only if it appears in both.

# %% [markdown]
# ### 9.11 All checks together
#
# The table lists every specification with its sample, its estimate in log points and in
# percent, the interval and the bootstrap p-value. The chart after it shows the same rows. The
# shaded band is the interval of the central estimate, so a row can be compared with it at a
# glance, and the solid vertical line marks no difference.

# %%
robustness_rows = [row for rows in robustness_checks.values() for row in rows]   # the order of the forest plot
robustness_table = pd.DataFrame(robustness_rows)
robustness_table["interval_excludes_zero"] = (robustness_table["low"] > 0) | (robustness_table["high"] < 0)
assert not robustness_table.duplicated(["check", "variant"]).any()
assert (robustness_table["kind"] == KIND_CENTRAL).sum() == 1
has_estimate = robustness_table["estimate"].notna()
assert (robustness_table.loc[has_estimate, "low"] <= robustness_table.loc[has_estimate, "high"]).all()
assert (robustness_table[["n_treated", "n_comparison"]] > 0).all().all()


def signed_text(values, digits):
    """Format a numeric column with an explicit sign; a missing estimate reads 'not reported'."""
    return values.map(lambda value: "not reported" if pd.isna(value) else f"{value:+.{digits}f}")


robustness_display = pd.DataFrame({
    "treated sites": robustness_table["n_treated"],
    "comparison sites": robustness_table["n_comparison"],
    "units resampled": robustness_table["n_resampled_units"],
    "strata": robustness_table["n_strata"],
    "difference (log points)": signed_text(robustness_table["estimate"], 3),
    "low": signed_text(robustness_table["low"], 3),
    "high": signed_text(robustness_table["high"], 3),
    "difference (%)": signed_text(robustness_table["percent_change"], 1),
    "low (%)": signed_text(robustness_table["percent_low"], 1),
    "high (%)": signed_text(robustness_table["percent_high"], 1),
    "p-value": robustness_table["p_value"].map(lambda value: "not reported" if pd.isna(value) else f"{value:.3f}"),
    "interval excludes zero": robustness_table["interval_excludes_zero"].map({True: "yes", False: "no"}),
})
robustness_display.index = pd.MultiIndex.from_frame(robustness_table[["check", "variant"]])
display(robustness_display)

# %% [markdown]
# A forest plot is the usual form for this kind of result: one row per specification, a point
# for the estimate and a horizontal line for its interval, all on one scale, so that agreement
# and disagreement between rows can be seen without reading numbers. The estimates are shown as
# percent changes. Blue marks the central estimate, dark grey the rows in which one choice was
# changed, and orange the placebo rows, where no difference is expected.

# %%
def forest_layout(table, row_step=1.0, heading_step=0.85, group_gap=0.75):
    """Vertical position of every row and of every group heading, top to bottom."""
    row_positions, headings, cursor = [], [], 0.0
    for check, block in table.groupby("check", sort=False):
        headings.append((cursor, check))
        cursor += heading_step
        row_positions.extend(cursor + row_step * np.arange(len(block)))
        cursor += row_step * len(block) + group_gap
    return np.asarray(row_positions), headings, cursor - group_gap


def plot_percent(value):
    """A percent value for the text column: whole numbers, one decimal for values that round to zero."""
    text = f"{value:+.1f}" if abs(value) < 0.5 else f"{round(value):+d}"
    return text.replace("-", "−")


def forest_plot(table):
    """Every specification as a point with its interval, grouped by check, on a percent scale."""
    row_positions, headings, bottom = forest_layout(table)
    table = table.assign(position=row_positions)
    central_row = table[table["kind"] == KIND_CENTRAL].iloc[0]

    # Plot range: the widest interval, but never beyond the axis limit on either side.
    padding = 2
    left = max(-FOREST_AXIS_LIMIT, np.nanmin(table["percent_low"]) - padding)
    right = min(FOREST_AXIS_LIMIT, np.nanmax(table["percent_high"]) + padding)
    axis_right = left + (right - left) / FOREST_PLOT_SHARE     # room for the two text columns
    value_column, sample_column = FOREST_PLOT_SHARE + 0.03, FOREST_PLOT_SHARE + 0.27

    fig, ax = plt.subplots(figsize=(12.5, 0.34 * bottom + 2.0))
    row_axis = ax.get_yaxis_transform()     # x in axes fractions, y in row positions
    # The interval of the central estimate as a band, so every other row can be read against it.
    band = ax.axvspan(central_row["percent_low"], central_row["percent_high"], color=PALETTE["treated"],
                      alpha=0.10, linewidth=0, label="Interval of the central estimate")
    ax.plot([0, 0], [-0.6, bottom], color=PALETTE["neutral"], linewidth=1.3)

    handles = []
    for kind, (label, color) in ROW_KINDS.items():
        rows = table[(table["kind"] == kind) & table["estimate"].notna()]
        low = rows["percent_low"].clip(lower=left)
        high = rows["percent_high"].clip(upper=right)
        ax.hlines(rows["position"], low, high, color=color, linewidth=2)
        handles.append(ax.scatter(rows["percent_change"].clip(left, right), rows["position"], s=70, color=color,
                                  edgecolor="white", linewidth=1.5, zorder=3, label=label))
        # An arrow head where an interval continues beyond the edge of the axis.
        cut_high = rows[rows["percent_high"] > right]
        ax.scatter(np.full(len(cut_high), right), cut_high["position"], marker=">", s=60, color=color, zorder=3)
        cut_low = rows[rows["percent_low"] < left]
        ax.scatter(np.full(len(cut_low), left), cut_low["position"], marker="<", s=60, color=color, zorder=3)

    for _, row in table.iterrows():
        if np.isnan(row["estimate"]):
            value_text = f"not reported (fewer than {MIN_GROUP_N} sites)"
        else:
            value_text = (f"{plot_percent(row['percent_change'])}%  ({plot_percent(row['percent_low'])} to "
                          f"{plot_percent(row['percent_high'])})")
        weight = "bold" if row["kind"] == KIND_CENTRAL else "normal"
        ax.text(value_column, row["position"], value_text, transform=row_axis, va="center", fontsize=9.5,
                fontweight=weight)
        ax.text(sample_column, row["position"], f"{row['n_treated']:,} / {row['n_comparison']:,}",
                transform=row_axis, va="center", fontsize=9.5, fontweight=weight)
    header_position = -0.75
    ax.text(value_column, header_position, f"Difference ({CI_LEVEL:.0%} interval)", transform=row_axis,
            va="center", fontsize=9.5, fontweight="bold")
    ax.text(sample_column, header_position, "Sites: treated / comparison", transform=row_axis, va="center",
            fontsize=9.5, fontweight="bold")

    ax.set_yticks(table["position"])
    ax.set_yticklabels(table["variant"], fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    for heading_position, check in headings:
        ax.text(-0.012, heading_position, check, transform=row_axis, ha="right", va="center", fontsize=10,
                fontweight="bold")
    ax.set_ylim(bottom, header_position - 0.7)
    ax.set_xlim(left, axis_right)
    tick_step = 10
    ax.set_xticks(np.arange(math.ceil(left / tick_step) * tick_step, right + 1, tick_step))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda value, _: "0%" if value == 0 else f"{value:+.0f}%".replace("-", "−")))
    ax.spines["bottom"].set_bounds(left, right)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.legend(handles=[*handles, band], loc="upper left", bbox_to_anchor=(-0.3, -0.105), ncol=4, columnspacing=1.6,
              handletextpad=0.4)
    finish(ax, "Injury crashes near new-construction sites after completion, under each change of method",
           "Difference between treated and comparison sites in the before-to-after change (percent)", "")
    ax.xaxis.set_label_coords(FOREST_PLOT_SHARE / 2, -0.045)
    ax.annotate(f"Below zero: crashes near treated sites fell more, or rose less, than near comparison sites. The "
                f"Outcome row counts all crashes; every other row counts injury crashes.\nAn arrow marks an interval that continues "
                f"beyond {FOREST_AXIS_LIMIT}%; its full range is given in the text column.",
                xy=(-0.3, -0.16), xycoords="axes fraction", fontsize=8.5, color=PALETTE["neutral"], va="top")
    return fig, ax


forest_plot(robustness_table)
plt.show()

# %%
changed_rows = robustness_table[(robustness_table["kind"] == KIND_CHECK) & has_estimate]
screened_rows = changed_rows[changed_rows["keeps_screen"]]
unscreened_rows = changed_rows[~changed_rows["keeps_screen"]]
placebo_rows = robustness_table[(robustness_table["kind"] == KIND_PLACEBO) & has_estimate]
excluding = changed_rows[changed_rows["interval_excludes_zero"]]
positive = changed_rows[changed_rows["estimate"] > 0]
placebo_estimates = " and ".join(f"{value:+.1f}%" for value in placebo_rows["percent_change"])
say(f"Of the {len(changed_rows)} specifications that change one choice, "
    f"{len(changed_rows) - len(excluding)} have an interval that includes zero. Their point estimates run from "
    f"{changed_rows['percent_change'].min():+.1f}% to {changed_rows['percent_change'].max():+.1f}%, against "
    f"{central['percent_change']:+.1f}% for the central estimate, and {len(positive)} of them are above zero "
    f"({'; '.join(positive['variant'])}). The interval excludes zero in {len(excluding)} of them "
    f"({'; '.join(excluding['variant']) if len(excluding) else 'none'}). Across the {len(screened_rows)} "
    f"specifications that keep the screen on both groups, the intervals run from "
    f"{screened_rows['percent_low'].min():+.1f}% to {screened_rows['percent_high'].max():+.1f}%, so a change "
    f"beyond that range is not supported, while a fall or a rise inside it cannot be excluded; "
    f"{int(screened_rows['interval_excludes_zero'].sum())} of these intervals exclude zero. Of the "
    f"{len(unscreened_rows)} specifications that drop the screen for at least one group, "
    f"{int(unscreened_rows['interval_excludes_zero'].sum())} have an interval that excludes zero. The "
    f"{len(placebo_rows)} placebo estimates are {placebo_estimates}, and "
    f"{int(placebo_rows['interval_excludes_zero'].sum())} of their intervals exclude zero.")

# %% [markdown]
# ### 9.12 What holds and what depends on a choice
#
# One statement holds in every specification that keeps the screen: the change in injury crashes
# around newly completed new-construction sites cannot be told apart from the change around
# comparison sites over the same years. No interval in those rows excludes zero. The intervals
# are not narrow, though. Their range is printed above, and a moderate fall or rise lies inside
# it, so these rows rule out only changes larger than that. The statement does not depend on the
# counting radius, on the length of the window, on how the year 2020 is treated, on the unit
# that is resampled, on whether injury crashes or all crashes are counted, or on whether the
# comparison is made inside boroughs. The placebo gives no sign of different trends before
# completion, within the limits of its own interval. The three questions about shares give the
# same reading at the smaller radius, apart from one factor group whose interval excludes zero
# there when taken alone and which is not kept after adjustment (9.2).
#
# The direction of the point estimate does depend on which sites enter the comparison. It is
# slightly negative in the central specification and in most checks, and positive when the
# screen is rebuilt at 150 m or switched off, while changes that keep the sites fixed leave the
# estimate almost where it was. A reading such as "injury crashes fall a little after
# completion" would therefore be a statement about the selected sites and is not made in this
# notebook. The positive estimate without the screen is carried by one development of small
# houses on streets that had almost no injury crashes before (9.5). Without that cluster the
# estimate for all candidate sites is small and its interval includes zero, inside cohorts and
# inside boroughs (9.9). The rows without the screen are therefore not read as a rise in the
# dense parts of the programme. They are weaker evidence than the screened rows, because most of
# their comparison streets have a completed building of their own.
#
# The reporting break of 2020 is the one place where a choice changes a raw number by a wide
# margin: counted as all crashes, the undifferenced fall at treated sites is far larger than
# the fall in injury crashes. It disappears only because the comparison sites share it. Any result
# in this notebook that compares periods without such a group has to use injury crashes.
#
# The ZIP-level questions were checked separately in 9.10. On reported ZIP codes alone the
# forecast loss of RQ1 is gone and the RQ9 correlation has an interval that includes zero. The
# completion-year slope of RQ2 is negative in both versions and belongs to preserved units. The null results for
# the cumulative stock in RQ2 and for deaths in RQ9 are the same in both versions.
#
# The screened sample is small on the treated side and lies in the less dense parts of the
# programme, so the central result describes those places. Where buildings stand close together
# the only estimates are the ones without the screen. All of these are associations in
# observational data: the comparison sites are similar to the treated ones in being part of the same
# programme, but they were not assigned at random.
