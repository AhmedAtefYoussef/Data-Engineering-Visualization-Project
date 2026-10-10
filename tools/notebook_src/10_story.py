# %% [markdown]
# ## 10. What we found
#
# Section 8 answered the ten questions one at a time. The table below collects their headline
# numbers from `RESULTS`, as each question stored them, and adds whether the interval includes
# zero. The estimates come in different units, so every row names its own. RQ8 and RQ10 report
# the largest of several gaps, so their interval is the wider one that allows for that choice,
# and RQ9 carries the p-value adjusted for its two outcomes.

# %%
# Objects of Sections 4 to 9 that Sections 10 to 15 read.
assert_state("RESULTS", "rq9_result", "robustness_table", "CHECK_OUTCOME", "KIND_CENTRAL",
             "LARGEST_CLUSTERS_SHOWN", "city_year", "sites", "event_panel", "zip_year", "housing", "crashes")

# --- Constants used only in Sections 10 to 15 -------------------------------------------------
NO_DIFFERENCE = 0.0          # value every headline estimate is compared with
OVERVIEW_COLUMNS = 2         # panels per row in the overview chart of the ten results
COMPLETION_REL_YEAR = 0      # rel_year of the completion year: one row per site and cohort
RQ9_OUTCOME = "ped_cyc_share_pct"   # row of rq9_result behind the RQ9 headline
SURPLUS_FROM_YEAR = BREAK_YEAR + 1  # first year in which injured totals above their parts appear regularly (4.9)
STAGE_ORDER = ["raw", "cleaned", "integrated"]

CORRELATION_DIGITS = 3       # decimals shown for a correlation; the other estimates use two
DEFAULT_DIGITS = 2

# For each question: its owner, a short label for the chart, what its n counts and what a value
# to the right of zero means. These are labels only; every number and unit comes from RESULTS.
RESULT_LAYOUT = {
    "RQ1": ("M1", "Forecast error when housing is added", "ZIP codes",
            "right of zero = forecast less accurate with housing"),
    "RQ2": ("M1", "Injury crashes as the unit stock grows", "ZIP codes",
            "right of zero = more injury crashes with a larger stock"),
    "RQ3": ("M2", "After against before, new buildings", "new-construction sites",
            "right of zero = more injury crashes after completion"),
    "RQ4": ("M2", "Change by units completed nearby", "new-construction sites",
            "right of zero = larger rise where more units were completed"),
    "RQ5": ("M3", "New construction against preservation", "completed sites",
            "right of zero = change more positive at new construction"),
    "RQ6": ("M3", "Pedestrian and cyclist share", "completed sites",
            "right of zero = share rose more at completed sites"),
    "RQ7": ("M4", "Where new-construction sites stand before the project", "new-construction sites",
            "right of zero = more injury crashes than the typical street point"),
    "RQ8": ("M4", "Hour of day (largest band gap)", "completed sites",
            "right of zero = share rose more at completed sites"),
    "RQ9": ("M5", "Deep-affordable share and who is hurt", "ZIP codes",
            "right of zero = higher pedestrian and cyclist share where the stock is deeper"),
    "RQ10": ("M5", "Contributing factor (largest group gap)", "completed sites",
             "right of zero = share rose more at completed sites"),
}


def interval_excludes(low, high, reference=NO_DIFFERENCE):
    """True when the whole interval lies on one side of the reference value."""
    return bool(low > reference or high < reference)


def results_table(results, layout):
    """One row per research question: owner, question, estimate, interval, unit, n and reading.

    `kept` is True when the interval excludes zero and, where the question tested several things
    together, the adjusted p-value is below FALSE_DISCOVERY_RATE as well. A percentile bootstrap
    interval can leave the point estimate outside when the resampled values are skewed, so that
    case is flagged in `estimate_outside_interval` instead of stopping the notebook. The numeric
    columns stay numeric so the chart can use them; `formatted_results` makes the version shown.
    """
    assert set(results) == set(layout), "every research question must have stored exactly one result"
    rows = []
    for key, (owner, label, n_unit, direction) in layout.items():
        stored = results[key]
        excludes_zero = interval_excludes(stored["ci_low"], stored["ci_high"])
        adjusted_p = stored["adjusted_p"]
        rows.append({
            "rq": key, "owner": owner, "label": label, "question": stored["question"],
            "estimate": stored["estimate"], "low": stored["ci_low"], "high": stored["ci_high"],
            "level": stored["level"], "unit": stored["unit"], "n": stored["n"], "n_unit": n_unit,
            "direction": direction, "adjusted_p": float("nan") if adjusted_p is None else adjusted_p,
            "excludes_zero": excludes_zero,
            "kept": excludes_zero and (adjusted_p is None or adjusted_p < FALSE_DISCOVERY_RATE),
            "estimate_outside_interval": not stored["ci_low"] <= stored["estimate"] <= stored["ci_high"],
        })
    return pd.DataFrame(rows).set_index("rq")


def reading_text(row):
    """The reading of one headline in words."""
    if not row["excludes_zero"]:
        return "interval includes zero"
    if row["kept"]:
        return "interval excludes zero"
    margin = min(abs(row["low"]), abs(row["high"]))
    return (f"interval ends {margin:.3f} from zero; not kept after adjustment "
            f"(adjusted p {row['adjusted_p']:.2f})")


def formatted_results(table):
    """The results table as shown: signed numbers, the interval in one cell, the reading in words."""
    digits = [CORRELATION_DIGITS if "correlation" in unit else DEFAULT_DIGITS for unit in table["unit"]]
    shown = pd.DataFrame({
        "owner": table["owner"],
        "question": table["question"],
        "estimate": [f"{value:+.{places}f}" for value, places in zip(table["estimate"], digits)],
        "interval": [f"{low:+.{places}f} to {high:+.{places}f} ({level:.0%})"
                     for low, high, level, places in zip(table["low"], table["high"], table["level"], digits)],
        "unit of the estimate": table["unit"],
        "n": [f"{n:,} {n_unit}" for n, n_unit in zip(table["n"], table["n_unit"])],
        "reading": [reading_text(row) for _, row in table.iterrows()],
    })
    shown.index.name = None
    return shown


summary_table = results_table(RESULTS, RESULT_LAYOUT)
with pd.option_context("display.max_colwidth", None):
    display(formatted_results(summary_table))
outside = summary_table.index[summary_table["estimate_outside_interval"]].tolist()
if outside:
    say(f"Warning: the point estimate lies outside its percentile interval for {', '.join(outside)}, which "
        f"happens when the resampled values are skewed. Those intervals should be read with care.")

# %% [markdown]
# The chart shows the same ten rows. A percentage, a correlation and a number of crashes cannot
# share one scale, so every panel has its own horizontal axis, labelled with its unit and with
# what a value to the right of zero means. The widths of the intervals can therefore not be
# compared between panels. The vertical line is "no difference" in all of them. An interval that
# crosses the line means the data cannot tell that estimate from zero.

# %%
def plot_results_overview(table):
    """Small multiples: one estimate with its interval per research question, each on its own scale."""
    n_rows = math.ceil(len(table) / OVERVIEW_COLUMNS)
    fig, axes = plt.subplots(n_rows, OVERVIEW_COLUMNS, figsize=(12.5, 2.3 * n_rows))
    colors = {True: PALETTE["treated"], False: PALETTE["control"]}
    for ax, (key, row) in zip(axes.ravel(), table.iterrows()):
        color = colors[row["kept"]]
        # The zero line is a short segment inside the panel, so it cannot stick out above the axes.
        ax.plot([NO_DIFFERENCE, NO_DIFFERENCE], [-0.55, 0.55], color=PALETTE["neutral"], linewidth=1.2)
        ax.hlines(0, row["low"], row["high"], color=color, linewidth=3)
        ax.plot(row["estimate"], 0, "o", color=color, markersize=9, markeredgecolor="white")
        # Pad the axis so that zero and both interval ends are always inside the panel.
        reach = max(abs(row["low"]), abs(row["high"]))
        ax.set_xlim(min(row["low"], NO_DIFFERENCE) - 0.25 * reach, max(row["high"], NO_DIFFERENCE) + 0.25 * reach)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        finish(ax, f"{key}  {row['label']}\nn = {row['n']:,} {row['n_unit']}",
               f"{row['unit']}\n{row['direction']}", "")
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(8.5)
    handles = [mpl.lines.Line2D([], [], color=colors[flag], marker="o", linewidth=3, label=label)
               for flag, label in [(True, "Interval excludes zero"),
                                   (False, "Interval includes zero, or is not kept after adjustment")]]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"The ten headline estimates with their bootstrap intervals ({CI_LEVEL:.0%}; wider for RQ8 and "
                 f"RQ10), each in its own unit", fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.035, f"The {len(table)} intervals are shown without a correction for looking at "
                           f"{len(table)} results at once, so one of them can exclude zero by chance.",
             fontsize=8.5, color=PALETTE["neutral"])
    fig.tight_layout(rect=(0, 0.035, 1, 0.97), w_pad=3)
    plt.show()


plot_results_overview(summary_table)

# %%
away_from_zero = summary_table.index[summary_table["excludes_zero"]].tolist()
not_kept = summary_table.index[summary_table["excludes_zero"] & ~summary_table["kept"]].tolist()
rq9_adjusted_p = float(rq9_result.loc[RQ9_OUTCOME, "p_adjusted"])
say(f"Of the {len(summary_table)} headline intervals, {len(away_from_zero)} exclude zero "
    f"({', '.join(away_from_zero) if away_from_zero else 'none'}) and "
    f"{len(summary_table) - len(away_from_zero)} include it. Of those that exclude zero, {len(not_kept)} "
    f"({', '.join(not_kept) if not_kept else 'none'}) are not kept after the adjustment for the tests made "
    f"together with them. For RQ9 the p-value adjusted for its two outcomes is {rq9_adjusted_p:.2f}, against a "
    f"threshold of {FALSE_DISCOVERY_RATE}, and the end of its interval nearest to zero is "
    f"{summary_table.loc['RQ9', ['low', 'high']].abs().min():.3f} away from it.")
say(RESULTS["RQ7"]["sentence"])
say(RESULTS["RQ3"]["sentence"])

# %% [markdown]
# The intervals that exclude zero are different kinds of result. RQ7 is far from zero and is the
# largest result in the notebook; what it does and does not show is taken up in 10.1. RQ1 is on
# the unhelpful side: the forecast is a little less accurate with housing information than
# without it. Its interval is conditional on the fitted models, and the loss is absent when only
# crashes with a reported ZIP code are counted (9.10), so we read it as "housing did not improve
# the forecast" and not as a finding about housing. RQ9 ends a hair from zero, and its p-value
# is above the threshold once its two outcomes are adjusted for together, so it is drawn in grey
# and counted as a hint. All the other headline intervals include zero.
#
# Inside the questions, some side results also have intervals that exclude zero. Two of them
# need a comment: the completion-year slope of RQ2 and the gap in crash reports without a
# contributing factor in RQ10. Both are found for preservation projects, which bring no new
# residents, and 10.1 comes back to them.
#
# ### 10.1 The answer to the main question
#
# We started from a worry that comes up whenever an affordable building is proposed: more
# residents, more traffic, more people hurt on the street. For the buildings we could isolate,
# the crash records do not show it. Around those new buildings, injury crashes fell after
# completion by about as much as they fell around sites of the same programme that were still
# unfinished, and the difference between the two has an interval that includes zero (RQ3). No
# relation with the number of units completed can be detected (RQ4). New construction, which
# brings residents, cannot be told apart from preservation, which brings none (RQ5). Around
# completed sites of either kind, the pedestrian and cyclist share, the hour of the day and the
# mix of recorded contributing factors show no detectable shift (RQ6, RQ8, RQ10). More than half
# of those sites are preservation projects. With the treated group limited to new construction
# the three gaps stay small and their intervals include zero. At the scale of a ZIP code, a
# growing stock of affordable units goes with no detectable movement in injury crashes (RQ2).
# Section 9 changed the radius, the window, the outcome, the handling of 2020 and the strata of
# the comparison, and the reading stayed the same in every version that kept the screen on both
# groups.
#
# The two side results named above both belong to preservation. One of the five timing slopes of RQ2, the one for the completion year, is negative in every version we
# ran. It appears for preserved units and is absent for new ones. In RQ10 the share of crash
# reports that name no contributing factor fell around completed sites relative to comparison
# sites, and that gap is found around preservation sites and is absent around new construction.
# A preservation completion brings no new residents, which is why RQ5 uses it as a placebo, so
# neither result says anything about what new housing does to a street. We do not know what
# produces them. The second one is also a caveat for RQ10: its factor shares are taken among
# reports with a recorded factor, so a change in how completely reports are filled in can hide
# or imitate a shift in the mix.
#
# The interval of RQ3 is wide enough to hold a moderate fall as well as a small rise, so we
# cannot say that the streets stay as they were. Around the isolated sites the data give no
# support to the claim that the streets get worse; the upper end of the interval is printed
# above. For the dense clusters, where most buildings stand, the comparison is weaker, because
# the comparison streets often have a completed building of their own, which can hide part of a
# difference. The estimate without the screen is positive only because of one development of
# small houses on streets that had almost no injury crashes before. Without it the difference
# is small and its interval includes zero (9.5).
#
# The records do show something about where the buildings are. New-construction sites whose
# project started after the baseline years stood, during those years, on streets with more
# injury crashes than most street points of their borough, and they still rank above the middle
# inside their own ZIP code (RQ7). The benchmark is a grid of street points and not other
# housing, and apartment buildings of any kind stand on dense blocks, so this shows where the
# buildings are. It does not show that the programme picks more dangerous streets than other
# housing would. The one comparison between two kinds of housing site that the tables allow
# shows that new-construction sites stand in parts of a borough with more injury crashes than
# preservation sites do. Ranked inside their own ZIP codes, the two kinds cannot be told apart.
# The people who move in arrive on a street that already had this crash record.
#
# ### 10.2 What surprised us
#
# The cell below computes the numbers behind this subsection, so the text can stay free of them.

# %%
def surprise_facts(city, checks, site_frame, panel, crash_frame):
    """Numbers behind Section 10.2, each taken from a table built in Sections 4 to 9."""
    last_year = city.loc[city["year"] == YEAR_END].iloc[0]
    central_row = checks.loc[checks["kind"] == KIND_CENTRAL].iloc[0]
    all_crash_row = checks.loc[checks["check"] == CHECK_OUTCOME].iloc[0]
    cluster_sizes = site_frame["cluster_id"].value_counts()
    # One row per (site, cohort): the completion-year row of every treated candidate.
    treated = panel[(panel["group"] == "treated") & (panel["rel_year"] == COMPLETION_REL_YEAR)]
    surplus = crash_frame[crash_frame["other_injured"] > 0]
    return {
        "crash_index": float(last_year["crash_index"]),
        "injury_index": float(last_year["injury_crash_index"]),
        "all_crash_change": log_points_to_percent(all_crash_row["treated_change"]),
        "injury_change": log_points_to_percent(central_row["treated_change"]),
        "all_crash_difference": float(all_crash_row["percent_change"]),
        "sites": len(site_frame),
        "share_in_largest": pct(cluster_sizes.head(LARGEST_CLUSTERS_SHOWN).sum(), len(site_frame)),
        "median_neighbours": float(site_frame["n_sites_within_buffer"].median()),
        "treated_candidates": len(treated),
        "treated_clean": int((~treated["contaminated"]).sum()),
        "surplus_crashes": len(surplus),
        "surplus_share_late": pct((surplus["year"] >= SURPLUS_FROM_YEAR).sum(), len(surplus)),
    }


facts = surprise_facts(city_year, robustness_table, sites, event_panel, crashes)
assert 0 <= facts["treated_clean"] <= facts["treated_candidates"]
say(f"With {INDEX_BASE_YEAR} set to 100, all reported crashes stand at {facts['crash_index']:.0f} in {YEAR_END} "
    f"and injury crashes at {facts['injury_index']:.0f}. Counted as all crashes, the streets within {BUFFER_M} m "
    f"of the new buildings of RQ3 change by {facts['all_crash_change']:+.0f}% from the {EVENT_WINDOW} years before "
    f"completion to the {EVENT_WINDOW} years after; counted as injury crashes they change by "
    f"{facts['injury_change']:+.0f}%. Against comparison sites the all-crash difference is "
    f"{facts['all_crash_difference']:+.1f}%.")
say(f"The {LARGEST_CLUSTERS_SHOWN} largest clusters hold {facts['share_in_largest']:.0f}% of the {facts['sites']:,} "
    f"located sites, and the median site has {facts['median_neighbours']:.0f} other sites within {BUFFER_M} m. "
    f"Of {facts['treated_candidates']:,} completed sites in the cohorts with a full window, "
    f"{facts['treated_clean']:,} ({pct(facts['treated_clean'], facts['treated_candidates']):.0f}%) had no other "
    f"completion nearby in a different year of their window.")
say(f"{facts['surplus_crashes']:,} crashes report more injured persons than pedestrians, cyclists and motorists "
    f"together, and {facts['surplus_share_late']:.0f}% of them are dated {SURPLUS_FROM_YEAR} or later.")
say(RESULTS["RQ1"]["sentence"])

# %% [markdown]
# The biggest surprise came before any research question. Counted as all crashes, the streets
# around a new building seem to lose a large part of their crashes after completion. The same
# fall appears around unfinished sites and in the city as a whole, and it comes from reports of
# crashes without injuries, which dropped in 2020 and never came back. With the total count and
# no comparison sites we would have reported a large safety gain near new housing, and it would
# have been a product of record keeping.
#
# The second surprise was how tightly the programme is packed. We expected most buildings to
# have a few blocks to themselves. Most of them have other financed buildings within the
# buffer, the chains of neighbours run through whole districts, and only a small part of the
# completed sites had no other completion nearby during their window. The before and after
# comparison rests on far fewer buildings than the housing table has rows, and it describes the
# less dense edges of the programme.
#
# We also expected housing information to help a crash forecast at least a little. It did not.
# In this split the forecast was slightly less accurate with it (RQ1), and a model that knew
# only the ZIP's own injury crashes could not be told from repeating last year's count.
#
# Two smaller things were unexpected. Inside a borough, ZIP codes with a deeper affordable mix
# lean towards a lower pedestrian and cyclist share of injury crashes, where we had expected a
# higher one. This is a weak hint that is not kept after the adjustment for the two outcomes of
# RQ9; its adjusted p-value is printed under the overview chart. And from the year after the
# reporting break the persons-injured total stopped matching pedestrians plus cyclists plus
# motorists. Section 4.9 shows that an e-scooter or e-bike is the first vehicle in a much larger
# share of these crashes than of crashes in general, which fits riders of those devices being
# counted as injured without belonging to the three published categories, although the table
# does not say so.
#
# ### 10.3 What the ten answers add up to
#
# Affordable housing stands on streets with many injury crashes, and that is true before the
# building exists (RQ7). After completion we could not detect a change on any of the measures we
# tried at the sites we could isolate, and the intervals allow a moderate fall or a small rise
# (RQ2 to RQ6, RQ8, RQ10). A map that overlays housing and crashes shows the first of these and
# can be misread as showing a change.
#
# The crash record of a block is known years before a financed building opens on it, so street
# changes for the future tenants can be planned from that record. The results for the years
# after completion are too imprecise to say whether the building itself will change it.

# %% [markdown]
# ## 11. What the data cannot tell us
#
# Every result in this notebook is an association. The reasons are specific to these two tables,
# and the cell below puts a size on the ones that can be measured.

# %%
def limit_facts(housing_frame, crash_frame, results, checks):
    """Sizes of the gaps that Section 11 discusses, from the cleaned and integrated tables."""
    confidential = housing_frame["is_confidential"]
    panel_crashes = crash_frame[crash_frame["in_zip_panel"]]
    first_year = crash_frame[crash_frame["year"] == YEAR_START]
    return {
        "confidential_rows": int(confidential.sum()),
        "confidential_row_share": pct(confidential.sum(), len(housing_frame)),
        "confidential_unit_share": pct(housing_frame.loc[confidential, "total_units"].sum(),
                                       housing_frame["total_units"].sum()),
        "same_day_share": pct(housing_frame["same_day_completion"].sum(), housing_frame["is_completed"].sum()),
        "unlocated_share": pct((~crash_frame["is_located"]).sum(), len(crash_frame)),
        "unlocated_share_first_year": pct((~first_year["is_located"]).sum(), len(first_year)),
        "outside_panel_share": pct((~crash_frame["in_zip_panel"]).sum(), len(crash_frame)),
        "imputed_zip_share": pct(panel_crashes["zip_imputed"].sum(), len(panel_crashes)),
        "no_factor_share": pct((crash_frame["factor_group"] == UNRECORDED_FACTOR).sum(), len(crash_frame)),
        "headline_intervals": len(results),
        "robustness_intervals": len(checks),
    }


limits = limit_facts(housing, crashes, RESULTS, robustness_table)
say(f"Housing: {limits['confidential_rows']:,} rows ({limits['confidential_row_share']:.1f}%) are confidential and "
    f"cannot be located; they hold {limits['confidential_unit_share']:.2f}% of the units. "
    f"{limits['same_day_share']:.0f}% of completed rows carry a completion date equal to their start date.")
say(f"Crashes: {limits['unlocated_share']:.1f}% have no usable coordinates ({limits['unlocated_share_first_year']:.1f}% "
    f"in {YEAR_START}) and cannot be counted near any building. {limits['outside_panel_share']:.1f}% are outside the "
    f"ZIP by year panel, and {limits['imputed_zip_share']:.1f}% of the crashes inside it have a ZIP code recovered "
    f"from coordinates. {limits['no_factor_share']:.1f}% name no contributing factor.")
say(f"Tests: Section 8 reports {limits['headline_intervals']} headline intervals at the {CI_LEVEL:.0%} level, next to "
    f"the borough, hour, factor and timing estimates inside the questions, and Section 9 adds "
    f"{limits['robustness_intervals']} specifications of one comparison.")

# %% [markdown]
# Sites are not chosen at random. RQ7 shows that they stand on streets with more injury crashes
# than the average street point, so a comparison of streets with and without affordable housing
# would mostly measure where apartment buildings are. We compared completed sites with
# unfinished sites of the same programme, which share that pattern, but the two groups still
# differ. They are spread differently over the boroughs (9.9 repeats the comparison inside
# boroughs), and both had to pass a screen that only isolated sites survive. Section 9 shows
# that the direction of the RQ3 estimate changes with which sites are admitted, and that the
# positive estimate for all candidate sites rests on one development (9.5).
#
# Neighbourhoods change for other reasons. A street that gets a new building may get a bike
# lane, a rezoning, a school or a rebuilt junction in the same years, and neither table records
# any of that. For a new building the years before completion are also years of construction,
# with fences and closed pavements, so "before" is not an undisturbed street. Many completion
# dates are administrative, as the share of same-day completions above shows.
#
# Reporting changed in 2020. We use injury crashes because their reporting was far less
# affected, but the same year also changed traffic itself for months, and it falls inside every three-year
# window we could build. No cohort lets us look at a completion without the pandemic in it.
#
# The crash table holds police-reported collisions only. A cyclist who falls without a car
# involved, or a minor collision settled between two drivers, is not in it. Nothing says how
# many people or vehicles used a street, so a count of injury crashes is not a risk per trip,
# and a busy street can have many crashes and still be safe for each person on it. The
# pedestrian and cyclist columns leave out riders of e-scooters and e-bikes.
#
# Location has gaps on both sides. Crashes without usable coordinates cannot be near anything,
# and their share was highest in the first study year, which is a "before" year for the
# earliest cohort. A recovered ZIP code is the most common code of a small grid cell, which is
# sometimes wrong at a boundary, and the forecast loss of RQ1 disappears when the recovered
# codes are left out (9.10). The confidential housing rows have no address at all.
# They are mostly single homes, so the spatial results describe the part of the programme with
# a published address, which is mostly apartment buildings.
#
# ZIP-level results do not describe individual streets. RQ1, RQ2 and RQ9 compare areas where
# tens of thousands of people live. A correlation across such areas says nothing about the
# tenants of the financed buildings or about the corner in front of one of them, and a building
# is a small change for a whole ZIP code.
#
# How completely a report is filled in is not constant. A large share of the reports name no
# contributing factor (the share is printed above), and RQ10 found that this share fell around
# completed preservation sites while it hardly moved around comparison sites or around new
# construction. The factor shares of RQ10 are computed among reports that name a factor, so a
# difference in recording can hide or imitate a shift in the mix. The gap also shows that
# reporting did not move alike everywhere, and every share computed from the reports depends on
# that.
#
# Several tests were run. With this many intervals, one or two can exclude zero by chance. We
# adjusted for multiple tests inside each question but not across the notebook, which is why
# the narrow RQ9 result, the completion-year dip in RQ2, the single hours in RQ8 and the gap in
# unrecorded factors in RQ10 are reported and not presented as findings. The dip in RQ2 and the
# gap in RQ10 have a second reason. Both are found for preservation, the placebo of RQ5, and
# are absent for new construction.
#
# Affordable housing is a subset of all construction. Market-rate buildings without city
# financing are in neither table, so a comparison street may have received a private tower
# during its window without our knowing. For the same reason the results are about this
# programme and should not be read as results about new housing in general, and RQ7 cannot
# compare the programme's sites with the sites of other apartment buildings.

# %% [markdown]
# ## 12. Who can use this and what they could do differently
#
# The Department of Transportation, including its Vision Zero road-safety programme, can use
# RQ7 as a planning list. The housing table lists financed buildings before they are completed,
# and the new-construction addresses among them lean towards streets with more injury crashes
# than is typical for the borough. Street redesign could be scheduled along that pipeline, so
# that it is in place when tenants arrive. RQ7 counts crashes and does not divide them by
# traffic or footfall, so it identifies streets where many people are hurt and not streets where
# each trip is more dangerous. Housing completions did not improve a ZIP-level forecast (RQ1),
# so a ZIP code's own crash history remains the guide for allocating money across
# neighbourhoods. Any before and after evaluation of a street project that spans 2020 should
# also count injury crashes and use comparison streets, because the all-crash count fell
# steeply for reporting reasons alone (9.4).
#
# The Department of Housing Preservation and Development and the developers it finances can
# take two things from this. Around the sites we could isolate, the completion of an affordable
# building is not associated with a detectable change in nearby injury crashes (RQ3 to RQ5),
# which is relevant when a project is opposed on traffic-safety grounds. The interval leaves
# room for a moderate change in either direction, and the sites we could isolate are the less
# dense part of the programme. The second point is RQ7 again: the street outside the future
# front door has a crash record at the time of site selection, and it could be looked at then,
# together with the transport department. The table itself could also be more useful. A date on
# which tenants could actually move in, separate from the financing date, would make every
# before and after comparison sharper.
#
# Community boards review these projects one at a time. When the question of street safety
# comes up, RQ2 and RQ3 say that neither the ZIP code nor the blocks around isolated buildings
# showed a detectable rise in injury crashes after past completions, and RQ7 says the street
# probably had many injury crashes already. A board could ask for the street improvement on the
# strength of the existing record and not make it depend on the building. Our ZIP-level results
# say nothing about one particular corner, and a board knows its corners better than a ten-year
# average does. The before and after result comes from buildings with no other completion
# within the buffer. Inside dense clusters, where many boards review projects, the comparison
# is weaker, because the comparison streets have completed buildings too. There the estimate is
# also small with an interval that includes zero once one development on streets with almost no
# earlier injury crashes is set aside (9.5).
#
# Residents and tenants can read RQ6, RQ8 and RQ10 as a description of the risk outside the
# door. Around completed sites of either kind we could not detect a change in who is hurt, at
# what time of day or with which recorded cause, and the intervals are several points wide, so
# a moderate shift in either direction would not have been seen. The share of crashes with no
# factor named fell near completed preservation buildings, which concerns how reports are filled
# in. The records cover police-reported crashes only and leave e-scooter and
# e-bike riders out of the pedestrian and cyclist share, so they understate harm to people
# outside cars.

# %% [markdown]
# ## 13. Prompts and AI use
#
# AI coding assistants (Claude models from Anthropic, used through Claude Code) drafted code and
# text for this project, including this notebook. The notebook was executed from top to bottom
# on the real data, the outputs shown here are the outputs of that run, and the team is
# responsible for everything in it. The assignment asks for the prompts used in any step.
# Prompts whose text the team has are quoted below in full, and the others are described. The
# notebook itself makes no calls to any model.
#
# ### 13.1 The first requests
#
# The first prompt of the project came with the project description and the data catalogue. It
# asked the assistant to analyse both, to compare the available Claude models, to divide the
# work among several assistants with one responsibility each, and to audit the result against
# the requirements of the project description. Its text is not available word for word, so it
# is described here and not quoted.
#
# The follow-up prompt in the same conversation was:
#
# > apply these skills as necessarily as possible: 21st-ui, design-motion-principles, design-taste-frontend, humanizer, impeccable, ui-ux-pro-max
#
# The team of five members and the light editorial civic look for the website were decided in
# the same conversation.
#
# ### 13.2 The briefs given to the assistants
#
# The project was split into steps, each handled by one assistant with its own brief. Every
# brief had the same parts: a role, the files to read, the files the assistant was allowed to
# edit, the task, the quality bar and the form of the report. The table summarises them.
#
# | Step | Model | Role in the brief | Task | Output asked for |
# |---|---|---|---|---|
# | Spec review | Opus 5.5 | Adversarial architect | Find defects in the written specification: event-study design, meaning of the exposure flag, imputation leakage, paging, memory, gaps in the contract | Numbered findings with severity and exact fix text, under 600 words |
# | Analysis and app | Sonnet 5.5 | Dash engineer and designer | Ten research-question functions, the Dash website, the search parser, the design brief | Files, tests, screenshot findings |
# | Notebook and docs | Sonnet 5.5 | Technical writer | Notebook builder, README, methodology, data dictionary | Notebook execution result |
# | Final audit | Opus 5.5 | Independent auditor | Check every requirement in the project description against the repository | Requirement-by-requirement table |
#
# The prompt patterns were: sections marked with XML-style tags (role, context, ownership, task,
# output format), one responsibility and one set of editable files per assistant, a written
# specification that all of them had to follow, reviewers that could read but not edit what
# they judged, and a cap on the length of each report.
#
# ### 13.3 The request for this notebook
#
# This notebook was requested with the prompt below. The project description and the data
# catalogue were attached to it.
#
# > Reverse engineer the entirety of the codebase in terms of architecture, technical code implementation, documentation and the finalized notebook deliverable. Apply a structural dynamic workflow incorporating multiple sub agents with their respective tasks concurrently, allowing the each subagent to effectively and productively focus on one specific task at a time. For example, a subagent is responsible for code literacy and syntax, another is responsible for logical reasoning and developing the best possible strategies to counteract potential problems in all aspects, etc. The structural workflow should allow each subagent to complete their respective tasks concurrently with maximum level of precision, accuracy and efficiency with the least possible token consumption. Finally, apply all of the necessary plugins, skills, etc. You may research through the Claude chat regarding this project to maximize your knowledge, intel, etc. Finally, the finalized version of the notebook should be perfect in all the aspects such as modularization, code organization, documentation, architecture in addition to coordinating a sub agent that checks whether the finalized notebook satisfies all of the requirements from the project description. Also, use the best datasets from the data catalog folder and I need the notebook to run independently without needing external folders or files, just both of the datasets used. Apply the best skills I have here such as humanizer, impeccable, design-motion-principle, etc. Go all out. Make no mistake.
#
# The same conversation contained three further prompts, and the two-word prompt "Try again" was
# sent twice.
#
# > Build high quality top notch interactive web application according to the project requirements and specification by incorporating these skills as well; /humanizer
#
# > and these skills as well /anthropic-skills:impeccable /anthropic-skills:design-motion-principles /anthropic-skills:design-taste-frontend /anthropic-skills:ui-ux-pro-max
#
# > While you are continuing building this project, I need you to cut down the token usage by incorporating cheaper models for simpler tasks. For example, allow Claude opus 5.5 to execute and perform the complex tasks while sonnet 5.5 is responsible for code syntax and literacy. Basically apply the structural workflow, maintaining the quality and productivity of the work productivity and effectiveness overall while cutting down token usage in a strategic approach.
#
# The notebook is divided into numbered sections. Each section was drafted by a separate
# assistant from a shared brief that fixed the datasets, the injury crash as the outcome, the
# two integration levels and the writing rules, and each draft was executed on the real data
# before it was accepted. Further assistants read the executed notebook, each from one point of
# view: the requirements of the project description, the code, the statistics, the prose and
# the charts. Each was asked for a list of findings with a location, a severity, the problem
# and a proposed fix. The briefs of these reviewers are not available word for word.
#
# ### 13.4 The shared brief for the section authors
#
# The passages of the shared brief that set the content of the notebook are quoted below.
# Passages that only name working folders and build commands are left out.
#
# > Datasets: H2 Affordable Housing Production by Building (hg8x-zxpr) and T4 Motor Vehicle Collisions - Crashes (h9gi-nx95). Crashes are a filtered copy: 2016-01-01 to 2025-12-31 and 19 of 29 columns.
#
# > Crash reporting broke in 2020: total crashes halve (211k in 2019, 113k in 2020, 86k in 2025) while injuries stay near 50-60k a year, because property-damage-only reports collapsed. So the PRIMARY OUTCOME everywhere is the **injury crash** (a crash with at least one person injured or killed). All-crash counts appear only in EDA and as a robustness check. Say this plainly wherever a time comparison is made.
#
# > Two integration levels, both justified in the text: (a) ZIP x year panel; (b) spatial proximity, a crash is "near" a building if within `BUFFER_M` = 250 m (haversine BallTree). No row-level key join exists or is faked.
#
# > Charts: matplotlib/seaborn only, static, no plotly, no map tiles, no network at view time. Use `PALETTE`, `BOROUGH_COLORS`, `finish()`. Every chart has a title that states what it shows, axis labels with units, a legend when there is more than one series, and a labelled log axis when a log axis is used. One figure per code cell, followed by a markdown or `say()` reading of it. Pick the chart type for the question; no pie charts, no 3D, no dual axes.
#
# > Small named functions with docstrings; descriptive names (`housing_clean`, not `h2` or `df1`); no one-letter frames; no semicolon-joined statements; no magic numbers (name them); no dead code; no try/except that hides errors.
#
# > Comment the non-obvious logic (outlier rule, spatial index, imputation, bootstrap, difference-in-differences), explaining why, not restating the line.
#
# > Add `assert` checks for every invariant you claim (row conservation, uniqueness, no negative counts).
#
# > A markdown cell before every code block says what is about to be done and why. After the output, the reading of it. For every cleaning or design decision: the problem as observed in THIS data, the decision (drop, impute, flag, standardise), why, and the alternative considered and why it was rejected.
#
# > Never type a data-derived number into markdown. Numbers that come from the data are printed by code, using `say(f"...")` for sentences. Markdown carries the reasoning; it may state design constants and qualitative conclusions that you have verified in the executed output.
#
# > No claim of causation anywhere. This is observational data: say "is associated with", "coincides with", "differs from". State n and uncertainty with each result.
#
# The brief also listed the ten research questions with their owners, in the wording of the
# team table at the top of this notebook.
#
# ### 13.5 Prompts inside the product
#
# None. Neither the notebook nor the website sends anything to a language model.

# %% [markdown]
# ## 14. Contributions
#
# The table is the team's split of the work. Each member owns two research questions with their
# charts and a share of the cleaning, integration and website work. Code and text were drafted
# with AI assistance as described in Section 13, so owning a part means having directed, checked
# and corrected it and being able to explain it. Section numbers refer to this notebook.
#
# | Member | Name | Research questions and charts | Data cleaning | Integration | Website components | Other work |
# |---|---|---|---|---|---|---|
# | M1 | Ahmed Atef | RQ1 forecast comparison (dot plot of forecast error with intervals); RQ2 within-ZIP slopes (coefficient plot with leads and lags) | Housing: blank unit counts, duplicates on the true key, geographic identifiers, dates, unit consistency, size outliers (4.1 to 4.6) | ZIP by year panel (5.2) | Explore tab: dropdown filters and the Generate report button | Setup and data loading (Sections 1 and 2), housing exploration (3.1), README setup and run steps |
# | M2 | Nour Mostafa Aly | RQ3 event study (lines with interval bands); RQ4 building size (scatter with fitted slope) | Crashes: key, duplicates, dates, time of day, borough and ZIP code (4.7, 4.8) | Distance join with the ball tree and its brute-force check, choice of the 250 m buffer (5.3) | Event-study cards and the bootstrap callback | Crash exploration (3.2), bootstrap and difference-in-differences functions (Section 7) |
# | M3 | Yahia Mohamed | RQ5 new construction against preservation (interval plot); RQ6 pedestrian and cyclist share by borough (interval plot) | Crashes: coordinates, casualty totals against their parts, extreme injury counts, contributing-factor groups (4.8 to 4.11) | Sites, clusters and the screen for nearby completions (5.4) | Search box and the query parser | Robustness checks (Section 9), data dictionary |
# | M4 | Moaz Ahmed | RQ7 site selection (map and percentile distribution); RQ8 hour of day (heat map) | ZIP code recovery from coordinates and its holdout validation (6.1) | Tagging crashes with their nearest site (5.5), conservation and uniqueness checks (6.4) | Map and heat-map cards, page layout, accessibility | Deployment configuration, architecture notes |
# | M5 | Yassin Fady | RQ9 deep-affordable share across ZIP codes (scatter by borough); RQ10 contributing factors (interval plot with adjusted tests) | Post-integration cleaning: borough from two sources, redundant columns, data types (6.2, 6.3) | Cumulative unit stock and the citywide index (6.2, 6.5), final table schema | Story tab and the data and method tab | Findings, limits, impact, prompts and reproducibility (Sections 10 to 15), deployment on the hosting platform |
#
# Each member reviewed the code and text of their own two questions in Section 8 and can explain
# them. Names and student IDs are in the team table at the top.


# %% [markdown]
# ## 15. Reproducibility
#
# The notebook needs nothing but itself. Put it in any folder and run all cells. If the two data
# files, `housing_hg8x-zxpr.csv` and `crashes_h9gi-nx95.csv.gz`, are not in that folder, Section
# 2 downloads them from NYC Open Data and saves them there, which takes several minutes for
# the crash table. Later runs read the saved files and finish in a few minutes without a
# network connection. To refresh the data, delete the two files and run again. The numbers in
# the text are printed by code, so they follow the data, but the written interpretations were
# checked against the copy stamped below and should be read again after a refresh.
#
# It needs Python 3.12 or newer, which the first code cell checks, with pandas 2.0 or newer,
# matplotlib 3.7 or newer, numpy, scipy, scikit-learn, seaborn, requests and pyarrow. The
# matplotlib minimum comes from the RQ7 map, which reads the handles of a legend under the name
# they have had since that version. Every random step uses the seed from Section 1, so a second run on the same files
# gives the same numbers. The cell below records the environment, the size of the tables at
# each stage and how recent the data are.
#
# The code of the website described in Section 14 is in the project repository:
# https://github.com/AhmedAtefYoussef/Data-Engineering-Visualization-Project

# %%
def environment_table():
    """Python and library versions of the run that produced the stored outputs."""
    versions = {
        "Python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": mpl.__version__,
        "seaborn": sns.__version__, "requests": requests.__version__, "pyarrow": pyarrow.__version__,
    }
    return pd.DataFrame({"version": versions})


def raw_rows_from_log(log, dataset):
    """Rows of a table before its first cleaning rule, read from the cleaning log.

    The raw crash frame is released after Section 4 to save memory, so its length is taken from
    the audit trail instead of from the frame.
    """
    first_rule = next(entry for entry in log if entry["dataset"] == dataset)
    return first_rule["rows_before"]


def stage_table(log):
    """Row counts of every table at the raw, cleaned and integrated stage."""
    rows = [
        ("raw", "housing, as loaded", len(raw_housing)),
        ("raw", "crashes, as loaded", raw_rows_from_log(log, "crashes")),
        ("cleaned", "housing: one row per project and building", len(housing)),
        ("cleaned", "crashes: one row per collision", len(crashes)),
        ("integrated", "zip_year: ZIP code by year", len(zip_year)),
        ("integrated", "sites: located buildings", len(sites)),
        ("integrated", "site_year: site by year", len(site_year)),
        ("integrated", "event_panel: site by cohort by relative year", len(event_panel)),
        ("integrated", "crashes tagged with ZIP code and nearest site", len(crashes)),
    ]
    table = pd.DataFrame(rows, columns=["stage", "table", "rows"])
    table["stage"] = pd.Categorical(table["stage"], categories=STAGE_ORDER, ordered=True)
    return table.set_index(["stage", "table"])


def file_stamp(path):
    """Date on which a local data file was last written, which is the day it was downloaded."""
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d")


stages = stage_table(CLEANING_LOG)
assert raw_rows_from_log(CLEANING_LOG, "housing") == len(raw_housing), "the log and the raw housing frame disagree"
assert stages.loc[("raw", "crashes, as loaded"), "rows"] == len(crashes), "cleaning is expected to keep every crash"
assert crashes["collision_id"].is_unique and not zip_year.duplicated(["zip", "year"]).any()

display(environment_table())
display(stages)
say(f"Seed: {SEED}. Bootstrap resamples per interval: {N_BOOT:,}. Study window: {YEAR_START} to {YEAR_END}.")
say(f"Data as of: the latest crash in the copy is dated {crashes['crash_date'].max():%Y-%m-%d} and the latest "
    f"housing completion {housing['completion_date'].max():%Y-%m-%d}. The housing file was saved on "
    f"{file_stamp(HOUSING_FILE)} and the crash file on {file_stamp(CRASH_FILE)}.")
