# %% [markdown]
# ## 9. Robustness checks
#
# The before/after comparison around new-construction sites (RQ3) rests on choices that the data do
# not dictate: the 250 m buffer, the injury-crash outcome, the screen against nearby completions,
# the comparison strata and the cohorts used. This section re-estimates it with one choice changed
# at a time and collects every estimate in one table and one chart. A conclusion is kept only if it
# survives choices that could reasonably have been made differently.
#
# **Central comparison.** Treated units are new-construction sites of the cohorts with a full window;
# comparison units are sites still not completed at the end of that window. Both pass the screen of
# Section 5.4. Injury crashes within the buffer are summed over the `EVENT_WINDOW` years before and
# after the completion year (the year itself is skipped). The estimate is the mean log ratio
# after/before at treated minus comparison sites, inside each cohort, weighted by treated sites per
# cohort, with whole clusters resampled for the interval.

# %%
ALTERNATIVE_RADII_M = [150, 400]   # metres; one clearly smaller and one clearly larger than BUFFER_M (Section 5.3)
SHORT_WINDOW = 2                   # years; the placebo is also run with this shorter window
PLACEBO_SHIFT = EVENT_WINDOW       # years by which the placebo moves the completion date earlier
CENTRAL_OUTCOME, ALTERNATIVE_OUTCOME = "injury_crashes", "crashes"
EVENT_UNIT_COLUMNS = ["cluster_id", "contaminated", "construction_type", "borough"]
EARLY_COHORT = YEAR_START + EVENT_WINDOW   # first full-window cohort; its before years start in YEAR_START
LARGEST_CLUSTERS_SHOWN = 3         # clusters whose joint weight Section 10 reports
KIND_CENTRAL, KIND_CHECK, KIND_PLACEBO = "central", "check", "placebo"
CHECK_OUTCOME = "Outcome"          # the all-crashes row; Section 10 reads it
CHECK_SCREEN = "Nearby completions"
assert all(radius in RADIUS_CANDIDATES_M for radius in ALTERNATIVE_RADII_M)


def yearly_counts(site_year_frame, outcome):
    """Sites x study-years matrix of one count column (site_year is sorted by site, then year)."""
    return site_year_frame[outcome].to_numpy().reshape(-1, N_STUDY_YEARS)


def before_after_counts(units, counts, window=EVENT_WINDOW):
    """Add `before` and `after`: the outcome summed over `window` years on each side of the cohort year."""
    relative_years = np.concatenate([np.arange(-window, 0), np.arange(1, window + 1)])
    years = units["cohort"].to_numpy()[:, None] + relative_years[None, :]
    assert years.min() >= YEAR_START and years.max() <= YEAR_END, "a window leaves the study years"
    values = counts[units["site_id"].to_numpy()[:, None], years - YEAR_START]
    return units.assign(before=values[:, :window].sum(axis=1), after=values[:, window:].sum(axis=1))


def comparison_units(panel, counts, keep_contaminated=False):
    """The one event-units builder: new-construction treated and all comparison units, with before/after sums."""
    units = cohort_unit_table(panel, EVENT_UNIT_COLUMNS, keep_contaminated=True)
    keep = ~units["treated"] | (units["construction_type"] == NEW_CONSTRUCTION)
    if not keep_contaminated:
        keep &= ~units["contaminated"]
    return before_after_counts(units[keep].reset_index(drop=True), counts)


def estimate_check(units, check, variant, kind=KIND_CHECK, strata=("cohort",), keeps_screen=True):
    """Difference-in-differences on the log ratio for one specification, as one table row.

    Strata with sites on one side only are left out and counted in the `dropped` columns. No
    estimate is reported when either arm has fewer than MIN_GROUP_N sites.
    """
    result = did_log_ratio(units["before"].to_numpy(), units["after"].to_numpy(), units["treated"].to_numpy(),
                           strata=units[list(strata)].astype(str).agg("|".join, axis=1).to_numpy(),
                           clusters=units["cluster_id"].to_numpy())
    row = {"check": check, "variant": variant, "kind": kind, "keeps_screen": keeps_screen,
           "n_treated": result["n_treated"], "n_comparison": result["n_control"],
           "n_treated_dropped": result["n_treated_dropped"], "n_comparison_dropped": result["n_control_dropped"],
           "n_resampled_units": result["n_clusters"], "n_strata": result["n_strata"],
           "treated_change": result["treated_change"], "comparison_change": result["control_change"]}
    large_enough = min(result["n_treated"], result["n_control"]) >= MIN_GROUP_N
    for column in ["estimate", "low", "high", "p_value", "percent_change", "percent_low", "percent_high"]:
        row[column] = result[column] if large_enough else float("nan")
    return row


def reading(row):
    """Estimate, interval and sample of one table row as a phrase."""
    if np.isnan(row["estimate"]):
        return f"not reported (fewer than {MIN_GROUP_N} sites in an arm)"
    return (f"{row['percent_change']:+.1f}% ({CI_LEVEL:.0%} interval {row['percent_low']:+.1f}% to "
            f"{row['percent_high']:+.1f}%; {row['n_treated']:,} treated and {row['n_comparison']:,} comparison sites)")


injury_counts = yearly_counts(site_year, CENTRAL_OUTCOME)
assert injury_counts.shape == (len(sites), N_STUDY_YEARS)
central_units = comparison_units(event_panel, injury_counts)
central = estimate_check(central_units, "Central estimate", f"{BUFFER_M} m, {EVENT_WINDOW} years, injury crashes",
                         kind=KIND_CENTRAL)
robustness_rows = [central]
show_table(central_units.groupby(["cohort", "group"], observed=True).size().unstack("group"))
say(f"Central estimate: injury crashes change by {log_points_to_percent(central['treated_change']):+.1f}% at "
    f"treated and {log_points_to_percent(central['comparison_change']):+.1f}% at comparison sites; the "
    f"difference is {reading(central)}, bootstrap p = {central['p_value']:.2f}.")

# %% [markdown]
# ### 9.1 Outcome: all crashes
#
# Reports of property-damage-only crashes collapsed in 2020 (Section 3) and every window straddles
# that year. Replacing injury crashes with all crashes shows how large the break is in a raw
# before/after count, and whether comparison sites, which went through the same break, cancel it.

# %%
all_crashes = estimate_check(comparison_units(event_panel, yearly_counts(site_year, ALTERNATIVE_OUTCOME)),
                             CHECK_OUTCOME, "All crashes, including property damage only")
robustness_rows.append(all_crashes)
say(f"Undifferenced, all crashes near treated sites change by {log_points_to_percent(all_crashes['treated_change']):+.1f}% "
    f"(comparison sites {log_points_to_percent(all_crashes['comparison_change']):+.1f}%), against "
    f"{log_points_to_percent(central['treated_change']):+.1f}% for injury crashes. After differencing: "
    f"{reading(all_crashes)}.")

# %% [markdown]
# The raw all-crash fall is far larger than the injury-crash fall, and a before/after count without
# comparison sites would report it as a safety gain near new housing. Comparison sites show almost
# the same fall, so the difference agrees with the central estimate. Injury crashes stay the primary
# outcome because they do not need the comparison group to cancel the break exactly.
#
# ### 9.2 The screen for nearby completions
#
# The screen removes most candidate sites and keeps the less dense part of the programme. Switching
# it off admits every candidate, at the cost of comparison streets that already have a completed
# building. Neighbouring buildings of one development then enter as separate rows that count the
# same streets, so each cluster of treated sites is left out in turn and the estimate is repeated
# without the one that moves it most.

# %%
def leave_one_cluster_out(units):
    """Point estimate (log points) with each cluster that holds a treated site left out in turn."""
    change = log_ratio(units["before"].to_numpy(), units["after"].to_numpy())
    treated = units["treated"].to_numpy()
    strata_codes, strata_labels = pd.factorize(units["cohort"])
    clusters = units["cluster_id"].to_numpy()
    return pd.Series({cluster: stratified_difference(change[clusters != cluster], treated[clusters != cluster],
                                                     strata_codes[clusters != cluster], len(strata_labels))["estimate"]
                      for cluster in np.unique(clusters[treated])})


unscreened_units = comparison_units(event_panel, injury_counts, keep_contaminated=True)
unscreened = estimate_check(unscreened_units, CHECK_SCREEN, "Screen switched off: every candidate site",
                            keeps_screen=False)
without_one = leave_one_cluster_out(unscreened_units)
dominant_cluster = (without_one - unscreened["estimate"]).abs().idxmax()
in_dominant = (unscreened_units["cluster_id"] == dominant_cluster).to_numpy()
without_dominant = estimate_check(unscreened_units[~in_dominant], CHECK_SCREEN,
                                  "Screen off, without the most influential cluster", keeps_screen=False)
robustness_rows += [unscreened, without_dominant]

dominant_rows = unscreened_units[in_dominant & unscreened_units["treated"].to_numpy()]
dominant_projects = sites["project_name"].astype(str).to_numpy()[dominant_rows["site_id"]]
central_without_one = leave_one_cluster_out(central_units)
say(f"Screen off: {reading(unscreened)}. Leaving out one treated cluster at a time moves it between "
    f"{log_points_to_percent(without_one.min()):+.1f}% and {log_points_to_percent(without_one.max()):+.1f}%. The most "
    f"influential cluster holds {len(dominant_rows):,} treated rows, mostly from \"{pd.Series(dominant_projects).mode()[0]}\" "
    f"(median injury crashes per row: {dominant_rows['before'].median():.0f} before, "
    f"{dominant_rows['after'].median():.0f} after). Without it: {reading(without_dominant)}. In the central sample "
    f"the same exercise moves the estimate only between {log_points_to_percent(central_without_one.min()):+.1f}% and "
    f"{log_points_to_percent(central_without_one.max()):+.1f}%.")

# %% [markdown]
# The positive estimate without the screen rests on one development of small houses on streets that
# had almost no injury crashes before: each house counts those streets again, and a rise from near
# zero is a large log ratio. Without it the estimate is small with an interval that includes zero.
# Removing the most influential cluster must lower the estimate, so that row is not evidence of the
# opposite either; it shows the positive sign rests on one place.
#
# ### 9.3 Placebo: a completion date that is too early
#
# If streets about to get a building were already on a different path, the central estimate would
# pick that up. Each new-construction site is entered `PLACEBO_SHIFT` years before its real
# completion, so both false periods precede the building, and compared with the clean comparison
# sites of that earlier cohort. With the 3-year window the last false "after" year is the real
# completion year, so the placebo is repeated with a `SHORT_WINDOW`-year window. Both should be near zero.

# %%
def placebo_units(shift, window=EVENT_WINDOW):
    """Units in which new-construction sites are dated `shift` years before their real completion."""
    completion = sites["completion_year"].astype("float64").to_numpy()
    first, second = site_neighbour_pairs(sites, BUFFER_M)
    base = cohort_unit_table(event_panel, EVENT_UNIT_COLUMNS, keep_contaminated=True)
    parts = [base.loc[~base["treated"] & ~base["contaminated"], ["site_id", "cohort", "group", "treated", "cluster_id"]]]
    for cohort in sorted(base["cohort"].unique()):
        # The screen of Section 5.4 applied to the false date: no other completion in the window.
        neighbour_done = completion[second]
        in_window = ((first != second) & (neighbour_done >= cohort - window) & (neighbour_done <= cohort + window)
                     & (neighbour_done != completion[first]))
        contaminated = np.bincount(first[in_window], minlength=len(sites)) > 0
        chosen = sites[(completion == cohort + shift) & (sites["construction_type"] == NEW_CONSTRUCTION).to_numpy()
                       & ~contaminated]
        parts.append(pd.DataFrame({"site_id": chosen["site_id"].to_numpy(), "cohort": cohort, "group": "treated",
                                   "treated": True, "cluster_id": chosen["cluster_id"].to_numpy()}))
    units = pd.concat(parts, ignore_index=True)
    assert not units.duplicated(["site_id", "cohort"]).any(), "a site is on both sides of one cohort"
    return units


placebo_base = placebo_units(PLACEBO_SHIFT)
for window in (EVENT_WINDOW, SHORT_WINDOW):
    robustness_rows.append(estimate_check(before_after_counts(placebo_base, injury_counts, window), "Placebo",
                                          f"Completion dated {PLACEBO_SHIFT} years early, {window}-year window",
                                          kind=KIND_PLACEBO))
say("; ".join(f"{row['variant']}: {reading(row)}" for row in robustness_rows[-2:]) + ".")

# %% [markdown]
# Both placebo estimates are near zero with intervals that include it, so the pre-completion years
# give no sign that treated streets were already diverging. Their intervals are about as wide as the
# central one, so they rule out a large difference in earlier trends, not one the size of the
# central estimate.
#
# ### 9.4 Strata, cohorts and duplicated projects
#
# Three changes to who is compared with whom. **Inside cohort and borough:** treated and comparison
# sites are spread differently over the boroughs, and reporting did not move alike everywhere after
# 2020. **Without the earliest cohort:** its before years start in `YEAR_START`, when a much larger
# share of crashes lacked usable coordinates (printed below), so fewer crashes could be placed near
# any site. **One building per project:** a few projects hold many treated buildings that count the
# same streets; keeping one building per project (the first site) in each cohort and arm stops
# them from weighing as many independent sites.

# %%
inside_borough = estimate_check(central_units, "Comparison group", "Compared inside cohort and borough",
                                strata=("cohort", "borough"))
without_early = estimate_check(central_units[central_units["cohort"] != EARLY_COHORT], "Cohorts",
                               f"Without the {EARLY_COHORT} cohort")
project_of_unit = sites["project_id"].to_numpy()[central_units["site_id"]]
one_per_project = estimate_check(
    central_units[~central_units.assign(project=project_of_unit).duplicated(["cohort", "group", "project"])],
    "Projects", "One building per project")
robustness_rows += [inside_borough, without_early, one_per_project]

unlocated_by_year = PERCENT * (1 - crashes.groupby("year")["is_located"].mean())
treated_projects = pd.Series(project_of_unit[central_units["treated"].to_numpy()]).value_counts()
say(f"Inside cohort and borough: {reading(inside_borough)} ({inside_borough['n_comparison_dropped']} comparison "
    f"rows in one-sided strata left out). Crashes without usable coordinates: {unlocated_by_year[YEAR_START]:.1f}% in "
    f"{YEAR_START}, {unlocated_by_year.drop(YEAR_START).min():.1f}% to {unlocated_by_year.drop(YEAR_START).max():.1f}% "
    f"later; without the {EARLY_COHORT} cohort: {reading(without_early)}. The {treated_projects.sum():,} treated "
    f"units belong to {len(treated_projects):,} projects, the largest with {treated_projects.max()} buildings; one "
    f"building per project: {reading(one_per_project)}.")

# %% [markdown]
# ### 9.5 Buffer radius
#
# A smaller buffer keeps only the closest crashes but leaves small, noisy counts; a larger one adds
# streets with little to do with the building. The screen and the clusters use the same distance,
# so both are rebuilt at 150 m and 400 m, which changes which sites pass.

# %%
for radius in ALTERNATIVE_RADII_M:
    # A smaller buffer is a subset of the pairs already found; a larger one needs a new tree query.
    pairs = (site_pairs[site_pairs["distance_m"] <= radius] if radius < BUFFER_M
             else site_crash_pairs(sites, crash_tree, crash_tree_rows, radius))
    first, second = site_neighbour_pairs(sites, radius)
    sites_at_radius = sites.assign(cluster_id=connected_clusters(first, second, len(sites)))
    site_year_at_radius = build_site_year(pairs, crashes, len(sites))
    panel = build_event_panel(sites_at_radius, site_year_at_radius, radius_m=radius, neighbour_pairs=(first, second))
    row = estimate_check(comparison_units(panel, yearly_counts(site_year_at_radius, CENTRAL_OUTCOME)),
                         "Buffer radius", f"{radius} m, screen and clusters rebuilt at {radius} m")
    robustness_rows.append(row)
    say(f"{radius} m: {reading(row)}.")
release("pairs", "panel", "site_year_at_radius", "sites_at_radius")

# %% [markdown]
# The radius changes which sites pass the screen, so the sample moves a lot; the point estimate can
# change sign, but every interval includes zero.
#
# ### 9.6 ZIP-level results on reported ZIP codes only
#
# RQ1, RQ2 and RQ9 count crashes by ZIP code, and part of the codes were recovered from coordinates
# (Section 6.1). A recovered code can be wrong near a boundary, and the recovered share varies by
# year. The panel's crash columns are rebuilt from crashes with a police-reported ZIP code and the
# three results are estimated again.

# %%
def reported_zip_panel(panel, crash_frame):
    """The ZIP x year panel with its crash columns recounted from crashes whose ZIP code was reported."""
    is_reported = (~crash_frame["zip_imputed"] & crash_frame["in_zip_panel"]).to_numpy()
    counts = aggregate_crashes_to_zip_year(crash_frame[is_reported])
    counts = counts.assign(zip=counts["zip"].astype(str), year=counts["year"].astype(int)).set_index(["zip", "year"])
    keys = pd.MultiIndex.from_arrays([panel["zip"].astype(str), panel["year"].astype(int)])
    aligned = counts[CRASH_PANEL_MEASURES].reindex(keys, fill_value=0)
    reported = panel.copy()
    for column in CRASH_PANEL_MEASURES:
        reported[column] = aligned[column].to_numpy().astype(panel[column].dtype)
    # Row conservation: the panel already records how many of its crashes rest on a recovered code.
    assert (reported["injury_crashes"] == panel["injury_crashes"] - panel["injury_crashes_zip_imputed"]).all()
    return reported


def interval_text(estimate, low, high, digits=2):
    """Estimate and interval in one table cell."""
    return f"{estimate:+.{digits}f} ({low:+.{digits}f} to {high:+.{digits}f})"


reported_panel = reported_zip_panel(zip_year, crashes)
reported_forecast = forecast_comparison(build_forecast_frame(reported_panel))[0].loc[FORECAST_FULL]
reported_stock, reported_timing = stock_regression(reported_panel)
reported_equity = deep_share_associations(zip_equity_table(reported_panel))
reported_ped = reported_equity.loc["ped_cyc_share_pct"]
same_year_all = timing_slopes[timing_slopes["shift"] == 0].iloc[0]
same_year_reported = reported_timing[reported_timing["shift"] == 0].iloc[0]
rq9_ped_all = rq9_result.loc["ped_cyc_share_pct"]
zip_level_checks = pd.DataFrame(
    [("RQ1: change in forecast error with housing (injury crashes per ZIP-year)",
      interval_text(rq1["mae_minus_baseline"], rq1["low"], rq1["high"]),
      interval_text(reported_forecast["mae_minus_baseline"], reported_forecast["low"], reported_forecast["high"])),
     (f"RQ2: slope on the cumulative stock (% per {UNITS_PER_STEP} units)",
      interval_text(stock_slope["estimate_pct"], stock_slope["low_pct"], stock_slope["high_pct"]),
      interval_text(reported_stock["estimate_pct"], reported_stock["low_pct"], reported_stock["high_pct"])),
     (f"RQ2: slope on same-year completions (% per {UNITS_PER_STEP} units)",
      interval_text(same_year_all["estimate_pct"], same_year_all["low_pct"], same_year_all["high_pct"]),
      interval_text(same_year_reported["estimate_pct"], same_year_reported["low_pct"], same_year_reported["high_pct"])),
     ("RQ9: within-borough rank correlation, pedestrian and cyclist share",
      interval_text(rq9_ped_all["rho_within_borough"], rq9_ped_all["low"], rq9_ped_all["high"], 3),
      interval_text(reported_ped["rho_within_borough"], reported_ped["low"], reported_ped["high"], 3))],
    columns=["estimate", "all ZIP codes", "reported ZIP codes only"]).set_index("estimate")
show_table(zip_level_checks)
say(f"{pct(zip_year['injury_crashes_zip_imputed'].sum(), zip_year['injury_crashes'].sum()):.1f}% of panel injury "
    f"crashes have a recovered ZIP code. Adjusted p on reported codes: RQ2 same-year slope "
    f"{same_year_reported['p_adjusted']:.3f} (all codes {same_year_all['p_adjusted']:.3f}); RQ9 "
    f"{reported_ped['p_adjusted']:.3f} (all codes {rq9_ped_all['p_adjusted']:.3f}).")
release("reported_panel")

# %% [markdown]
# The small forecast loss of RQ1 disappears on reported codes, so RQ1 reads "housing did not improve
# the forecast", not "made it worse". The RQ9 correlation barely moves and stays a hint. The RQ2
# same-year slope keeps its sign in both columns; it belongs to preserved units (see 11). Neither
# column is the true one, because leaving out recovered codes removes a share that varies by year;
# a ZIP-level result counts as solid only if it appears in both.
#
# ### 9.7 All checks together
#
# Every specification of the RQ3 comparison, as a percent difference with its interval. The shaded
# band is the central interval; the dashed line is no difference.

# %%
robustness_table = pd.DataFrame(robustness_rows)
robustness_table["interval_excludes_zero"] = (robustness_table["low"] > 0) | (robustness_table["high"] < 0)
assert not robustness_table.duplicated(["check", "variant"]).any()
show_table(robustness_table.set_index(["check", "variant"])[
    ["n_treated", "n_comparison", "n_resampled_units", "percent_change", "percent_low", "percent_high",
     "p_value", "interval_excludes_zero"]].round(2))

# %%
plotted = robustness_table[robustness_table["estimate"].notna()]
fig, ax = plt.subplots(figsize=(11, 0.42 * len(plotted) + 1.6))
ax.axvspan(central["percent_low"], central["percent_high"], color=PALETTE["treated"], alpha=0.12)
interval_rows(ax, [f"{row.variant} ({row.n_treated:,} / {row.n_comparison:,})" for row in plotted.itertuples()],
              plotted["percent_change"], plotted["percent_low"], plotted["percent_high"],
              title="RQ3 under each change of method: injury crashes near new-construction sites\n"
                    "(treated / comparison sites per row; shaded: central interval)",
              xlabel=f"Difference in the before-to-after change, treated minus comparison (%, {CI_LEVEL:.0%} interval)")
show(fig)

# %%
changed_rows = robustness_table[(robustness_table["kind"] == KIND_CHECK) & robustness_table["estimate"].notna()]
screened_rows = changed_rows[changed_rows["keeps_screen"]]
placebo_rows = robustness_table[robustness_table["kind"] == KIND_PLACEBO]
say(f"Of {len(changed_rows)} specifications that change one choice, {int((~changed_rows['interval_excludes_zero']).sum())} "
    f"have an interval that includes zero; point estimates run from {changed_rows['percent_change'].min():+.1f}% to "
    f"{changed_rows['percent_change'].max():+.1f}% (central {central['percent_change']:+.1f}%). Across the "
    f"{len(screened_rows)} that keep the screen, the intervals span {screened_rows['percent_low'].min():+.1f}% to "
    f"{screened_rows['percent_high'].max():+.1f}%, and {int(screened_rows['interval_excludes_zero'].sum())} exclude "
    f"zero. Placebo intervals excluding zero: {int(placebo_rows['interval_excludes_zero'].sum())} of {len(placebo_rows)}.")

# %% [markdown]
# **What holds.** In every specification that keeps the screen, the change in injury crashes around
# newly completed new-construction sites cannot be told apart from the change around comparison
# sites; the printed span bounds the changes the data can rule out. This does not depend on the
# radius, the outcome, borough strata, the earliest cohort or duplicated projects, and the placebo
# shows no earlier divergence. **What depends on a choice.** The sign of the point estimate moves
# with which sites are admitted, so "crashes fall a little after completion" is not claimed. The
# positive estimate without the screen rests on one development (9.2). The screened sample is the
# less dense part of the programme, and all of this is association, not cause (see 11).
