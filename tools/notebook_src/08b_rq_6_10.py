# %% [markdown]
# ### 8.6 RQ6. Does the pedestrian and cyclist share of injury crashes near affordable-housing sites change after completion, by borough?
#
# Owner: M3
#
# A finished building puts more people on the pavement around it. This question looks at
# whether the people hurt on those streets are, more often than before, the ones outside a
# vehicle. We report the change in the share of injury crashes that hurt a pedestrian or
# cyclist, in percentage points, with an interval, for the city and for each borough that has
# enough sites.
#
# The share is compared at the same places, in the `EVENT_WINDOW` years before and after the
# completion year, using the clean sites of the event table from Section 5. A plain before and
# after comparison is not enough here, because the mix of victims moved across the whole city in
# these years (the pandemic emptied the roads of cars, and cycling and e-bikes grew). The same
# change is therefore measured at comparison sites that were not yet completed, over the same
# calendar years, and the estimate is the gap between the two changes. Each cohort is compared
# with its own comparison sites and the cohorts are averaged with the treated crash counts as
# weights, so that the comparison group has the same calendar mix as the treated group. Shares
# are pooled over sites, since a share computed at one site with two or three crashes can only be
# zero, a half or one. Sites of the same cluster share streets and crashes, so whole clusters are
# resampled in the bootstrap.
#
# Completed sites of both kinds are pooled to keep the sample usable, although only new
# construction brings new residents and preservation is the placebo of RQ5. The pooled gap is
# therefore a statement about completed sites of either kind. It is also estimated with the
# treated group limited to new construction, and printed next to the pooled one together with
# the number of sites of each kind.
#
# The three composition questions of this section (who is hurt here, the hour of day in RQ8 and
# the contributing factor in RQ10) use the same design, so the functions are written once for
# any set of categories.

# %%
# --- Constants used only in Sections 8.6 to 8.10 --------------------------------------------
PEDESTRIAN_CYCLIST_LABELS = ["Pedestrian or cyclist hurt", "Only vehicle occupants hurt"]
ALL_BOROUGHS_LABEL = "All boroughs"
COMPOSITION_UNIT_COLUMNS = ["cluster_id", "borough", "construction_type"]   # carried by every (site, cohort) unit
REFERENCE_SPACING_M = BUFFER_M     # metres between reference points; one buffer radius, so the
                                   # buffers of neighbouring points overlap and no street is skipped
STREET_SNAP_M = 100                # metres; a reference point counts as "on a street" if any crash
                                   # was ever reported this close (about one short block)
NEUTRAL_PERCENTILE = 50            # mean percentile that sites would have if they were typical streets
PERCENTILE_BIN_WIDTH = 10          # percentile points per bar in the RQ7 distribution
MAP_REFERENCE_COLOR = "#D9D9D9"    # light grey for the street grid behind the sites on the map
MAP_MARKER_EDGE = "#333333"        # thin dark outline, so the light end of the colour scale stays visible
HOURS_PER_DAY = 24
UNRECORDED_FACTOR = "Unspecified"  # RQ10: factor group meaning that the report names no factor
UNRECORDED_LABELS = [UNRECORDED_FACTOR, "Recorded factor"]   # the two categories of the RQ10 check
UNRECORDED_ROW = "No factor recorded (check)"   # label of that group in the RQ10 figure
MIN_ZIP_AFFORDABLE_UNITS = 50      # RQ9: units; below this a ZIP's deep-affordable share rests on
                                   # one or two small buildings and jumps between 0 and 100 percent
BOROUGH_MARKERS = dict(zip(BOROUGHS, ["o", "s", "^", "D", "v"]))   # RQ9: a shape per borough next to its colour


# %% [markdown]
# The clean sites, one row per site and cohort, come from `cohort_unit_table` (Section 8). The
# function below counts the injury crashes near each of them by period and category, from the
# site and crash pairs of Section 5. A comparison site can serve several cohorts, so it
# contributes one row per cohort and all its rows are resampled together.

# %%
def period_category_counts(units, pairs, crash_frame, category_codes, n_categories, window=EVENT_WINDOW):
    """Injury crashes near each unit before and after its cohort year, split by category.

    category_codes: one integer per row of `crash_frame`; a negative code leaves the crash out.
    Returns (before, after), two integer arrays of shape (number of units, n_categories). The
    cohort year itself is in neither period, because the building opens part-way through it.
    """
    is_counted = crash_frame["is_injury_crash"].to_numpy() & (category_codes >= 0)
    near = pairs.loc[is_counted[pairs["crash_row"].to_numpy()], ["site_id", "crash_row"]]
    unit_keys = units[["site_id", "cohort"]].assign(unit=np.arange(len(units)))
    # A site in several cohorts matches several unit rows, so this merge is many to many on purpose.
    near = near.merge(unit_keys, on="site_id", how="inner")
    crash_rows = near["crash_row"].to_numpy()
    relative_year = crash_frame["year"].to_numpy()[crash_rows].astype(np.int64) - near["cohort"].to_numpy()
    cell = near["unit"].to_numpy() * n_categories + category_codes[crash_rows]
    shape = (len(units), n_categories)
    is_before = (relative_year >= -window) & (relative_year <= -1)
    is_after = (relative_year >= 1) & (relative_year <= window)
    before = np.bincount(cell[is_before], minlength=shape[0] * shape[1]).reshape(shape)
    after = np.bincount(cell[is_after], minlength=shape[0] * shape[1]).reshape(shape)
    return before, after


# %% [markdown]
# The statistic is a difference-in-differences on shares. For every cohort it computes the share
# of each category before and after, at treated and at comparison sites, and takes the treated
# change minus the comparison change. `composition_shift` wraps it in the cluster bootstrap of
# Section 7 and adjusts the p-values for the number of categories tested together.

# %%
def cohort_weighted_shifts(before, after, treated, cohort_codes, n_cohorts):
    """Change in category shares at treated sites, at comparison sites, and the gap between them.

    before, after: (units, categories) counts. Returns an array of shape (3, categories) holding
    the treated change, the comparison change and their difference, as proportions. Each is an
    average over cohorts weighted by the treated crash count of the cohort. A cohort in which
    either arm has no crashes in either period cannot be compared and is left out of all three.
    """
    n_categories = before.shape[1]
    # Every count belongs to one (arm, cohort, category) cell; one bincount sums all cells at once.
    cell = ((treated.astype(np.int64) * n_cohorts + cohort_codes)[:, None] * n_categories
            + np.arange(n_categories)[None, :]).ravel()
    n_cells = 2 * n_cohorts * n_categories
    before_sums = np.bincount(cell, weights=before.ravel(), minlength=n_cells).reshape(2, n_cohorts, n_categories)
    after_sums = np.bincount(cell, weights=after.ravel(), minlength=n_cells).reshape(2, n_cohorts, n_categories)
    before_totals = before_sums.sum(axis=2)
    after_totals = after_sums.sum(axis=2)
    comparable = ((before_totals > 0) & (after_totals > 0)).all(axis=0)
    if not comparable.any():
        return np.full((3, n_categories), np.nan)
    before_sums, after_sums = before_sums[:, comparable], after_sums[:, comparable]
    before_totals, after_totals = before_totals[:, comparable], after_totals[:, comparable]
    change = after_sums / after_totals[:, :, None] - before_sums / before_totals[:, :, None]
    treated_events = before_totals[1] + after_totals[1]          # arm 1 is the treated arm
    weights = treated_events / treated_events.sum()
    treated_change = (weights[:, None] * change[1]).sum(axis=0)
    comparison_change = (weights[:, None] * change[0]).sum(axis=0)
    return np.vstack([treated_change, comparison_change, treated_change - comparison_change])


def pooled_shares(counts, is_member):
    """Category shares in percent, pooled over the units selected by `is_member`."""
    totals = counts[is_member].sum(axis=0)
    return PERCENT * totals / totals.sum() if totals.sum() > 0 else np.full(counts.shape[1], np.nan)


def composition_shift(units, before, after, labels, n_boot=N_BOOT, seed=SEED):
    """Difference-in-differences of category shares with a cluster bootstrap, in percentage points.

    Returns (table, summary). `table` has one row per category: pooled shares before and after in
    each arm, the cohort-weighted change in each arm, their `gap`, its interval `low` to `high`,
    `p_value`, the Benjamini-Hochberg `p_adjusted` across the categories and `flagged`. The
    columns `low_joint` and `high_joint` hold a wider interval at the level `joint_level` of the
    summary, which is CI_LEVEL with its error share divided by the number of categories
    (Bonferroni). It is the interval to quote for a category that was picked after looking at
    all of them. `summary` also holds the sample sizes: sites, clusters and injury crashes.
    """
    treated = units["treated"].to_numpy()
    cohort_codes, cohort_labels = pd.factorize(units["cohort"].to_numpy(), sort=True)
    n_cohorts, n_categories = len(cohort_labels), len(labels)
    assert before.shape == after.shape == (len(units), n_categories)

    def stacked_shifts(before_rows, after_rows, treated_rows, cohort_rows):
        """The (3, categories) result flattened, because the bootstrap expects a 1-D statistic."""
        return cohort_weighted_shifts(before_rows, after_rows, treated_rows, cohort_rows, n_cohorts).ravel()

    boot = bootstrap_ci(stacked_shifts, before, after, treated, cohort_codes,
                        clusters=units["cluster_id"].to_numpy(), n_boot=n_boot, seed=seed)
    joint_level = 1 - (1 - CI_LEVEL) / n_categories
    joint = percentile_ci(boot["replicates"], level=joint_level)
    in_points = {key: PERCENT * np.asarray(values).reshape(3, n_categories)
                 for key, values in (("estimate", boot["estimate"]), ("low", boot["low"]), ("high", boot["high"]),
                                     ("low_joint", joint["low"]), ("high_joint", joint["high"]))}
    gap_p_values = np.asarray(boot["p_value"]).reshape(3, n_categories)[2]
    table = pd.DataFrame({
        "treated_before": pooled_shares(before, treated), "treated_after": pooled_shares(after, treated),
        "comparison_before": pooled_shares(before, ~treated), "comparison_after": pooled_shares(after, ~treated),
        "treated_change": in_points["estimate"][0], "comparison_change": in_points["estimate"][1],
        "gap": in_points["estimate"][2], "low": in_points["low"][2], "high": in_points["high"][2],
        "low_joint": in_points["low_joint"][2], "high_joint": in_points["high_joint"][2],
        "p_value": gap_p_values, "p_adjusted": benjamini_hochberg(gap_p_values),
    }, index=pd.Index(labels, name="category"))
    table["flagged"] = table["p_adjusted"] < FALSE_DISCOVERY_RATE
    summary = {
        "n_treated_sites": int(units.loc[treated, "site_id"].nunique()),
        "n_comparison_sites": int(units.loc[~treated, "site_id"].nunique()),
        "n_clusters": boot["n_clusters"],
        "treated_crashes": int(before[treated].sum() + after[treated].sum()),
        "comparison_crashes": int(before[~treated].sum() + after[~treated].sum()),
        "joint_level": joint_level,
    }
    return table, summary


def units_of_kind(units, construction_type):
    """Mask of the units left when the treated group is limited to one kind of project.

    Comparison units of any construction type stay, because nothing was completed there.
    """
    return (~units["treated"] | (units["construction_type"] == construction_type)).to_numpy()


# %% [markdown]
# For RQ6 the two categories are an injury crash that hurt at least one pedestrian or cyclist and
# an injury crash that hurt vehicle occupants only. The counts are rebuilt from the pairs and
# checked against the event table, which holds the same totals from an independent aggregation.
# A borough is estimated only if it has at least `MIN_GROUP_N` clean completed sites and as many
# comparison sites. The borough p-values are adjusted together, because looking at several
# boroughs gives chance several tries.

# %%
def pedestrian_cyclist_counts(units, pairs, crash_frame, panel):
    """Injury crashes near each unit before and after, split by PEDESTRIAN_CYCLIST_LABELS.

    The counts are rebuilt from the pairs and must equal the event table, which holds the same
    totals from an independent aggregation by site and year.
    """
    codes = np.where(crash_frame["ped_cyc_harm"].to_numpy(), 0, 1)
    before, after = period_category_counts(units, pairs, crash_frame, codes, len(PEDESTRIAN_CYCLIST_LABELS))
    in_event_window = ~panel["contaminated"] & (panel["rel_year"] != 0)
    expected = (panel[in_event_window].assign(after=lambda rows: rows["rel_year"] > 0)
                .groupby(["site_id", "cohort", "after"])[["injury_crashes", "ped_cyc_crashes"]].sum())
    for is_after, counts in ((False, before), (True, after)):
        from_panel = expected.xs(is_after, level="after").reindex(
            pd.MultiIndex.from_frame(units[["site_id", "cohort"]]))
        assert (from_panel["injury_crashes"].to_numpy() == counts.sum(axis=1)).all()
        assert (from_panel["ped_cyc_crashes"].to_numpy() == counts[:, 0]).all()
    return before, after


def pedestrian_cyclist_share_shift(units, before, after):
    """RQ6 statistic: gap in the pedestrian and cyclist share, citywide and for each borough.

    Returns (table, summaries): one row per area with the shares, the gap in percentage points
    and its interval, and a dict of the sample sizes per area.
    """
    rows, summaries = {}, {}
    for area in [ALL_BOROUGHS_LABEL, *BOROUGHS]:
        in_area = np.ones(len(units), dtype=bool) if area == ALL_BOROUGHS_LABEL else (units["borough"] == area).to_numpy()
        area_units = units[in_area]
        n_treated = int(area_units["treated"].sum())
        n_comparison = int(area_units.loc[~area_units["treated"], "site_id"].nunique())
        if min(n_treated, n_comparison) < MIN_GROUP_N:
            summaries[area] = {"n_treated_sites": n_treated, "n_comparison_sites": n_comparison, "estimated": False}
            continue
        table, summary = composition_shift(area_units, before[in_area], after[in_area], PEDESTRIAN_CYCLIST_LABELS)
        # With two complementary categories there is one test per area, so the joint columns are not needed.
        rows[area] = table.loc[PEDESTRIAN_CYCLIST_LABELS[0]].drop(["p_adjusted", "flagged", "low_joint", "high_joint"])
        summaries[area] = {**summary, "estimated": True}
    result = pd.DataFrame(rows).T.astype(float)
    result.index.name = "area"
    is_borough = result.index != ALL_BOROUGHS_LABEL
    result.loc[is_borough, "p_adjusted"] = benjamini_hochberg(result.loc[is_borough, "p_value"].to_numpy())
    result["flagged"] = result["p_adjusted"] < FALSE_DISCOVERY_RATE
    return result, summaries


event_units = cohort_unit_table(event_panel, COMPOSITION_UNIT_COLUMNS)
is_new_construction_unit = units_of_kind(event_units, NEW_CONSTRUCTION)
is_preservation_unit = units_of_kind(event_units, PRESERVATION)
rq6_before, rq6_after = pedestrian_cyclist_counts(event_units, site_pairs, crashes, event_panel)
rq6_table, rq6_samples = pedestrian_cyclist_share_shift(event_units, rq6_before, rq6_after)
rq6_new_table, rq6_new_sample = composition_shift(
    event_units[is_new_construction_unit], rq6_before[is_new_construction_unit],
    rq6_after[is_new_construction_unit], PEDESTRIAN_CYCLIST_LABELS)
rq6_new = rq6_new_table.loc[PEDESTRIAN_CYCLIST_LABELS[0]]
display(rq6_table.drop(columns=["flagged"]).round(2))

# %% [markdown]
# The figure shows the gap for each area as a dot with its interval. A dot to the right of zero
# means that the pedestrian and cyclist share rose more around completed buildings than around
# sites still waiting to be completed.

# %%
def draw_gap_rows(ax, table, value_labels, label_header, is_check=None, check_label=""):
    """Draw one dot and interval per row of `table` (columns gap, low, high, flagged), top to bottom.

    A filled blue dot marks a gap that is kept after the adjustment for several tests, a hollow
    grey one a gap that is not. Rows selected by `is_check` are drawn as orange diamonds and
    named `check_label` in the legend: they are shown for reference and are not part of the
    adjusted set. `value_labels` are written to the right of the plot under `label_header`.
    Returns True when more than one kind of marker was drawn, so the caller knows whether a
    legend has anything to explain.
    """
    positions = np.arange(len(table))
    is_check = np.zeros(len(table), dtype=bool) if is_check is None else np.asarray(is_check, dtype=bool)
    flagged = table["flagged"].to_numpy(dtype=bool)
    kinds = [(~flagged & ~is_check, PALETTE["control"], "white", "o", "Not kept after adjustment for several tests"),
             (flagged & ~is_check, PALETTE["treated"], PALETTE["treated"], "o", "Kept after adjustment for several tests"),
             (is_check, PALETTE["placebo"], PALETTE["placebo"], "D", check_label)]
    n_kinds_drawn = 0
    for chosen, color, face, marker, label in kinds:
        if not chosen.any():
            continue
        n_kinds_drawn += 1
        ax.errorbar(table["gap"].to_numpy()[chosen], positions[chosen],
                    xerr=[(table["gap"] - table["low"]).to_numpy()[chosen],
                          (table["high"] - table["gap"]).to_numpy()[chosen]],
                    fmt=marker, color=color, markerfacecolor=face, markersize=7, markeredgewidth=1.6,
                    elinewidth=2, capsize=0, label=label)
    ax.axvline(0, color=PALETTE["neutral"], linewidth=1)
    reach = float(np.abs(table[["low", "high"]].to_numpy()).max())
    ax.set_xlim(-1.1 * reach, 1.1 * reach)
    # The labels sit outside the plot area: x in axes fractions, y in row positions. The light
    # row lines lead the eye from a dot to its label.
    for position, text in zip(positions, value_labels):
        ax.text(1.02, position, text, transform=ax.get_yaxis_transform(), va="center", fontsize=9,
                color=PALETTE["neutral"])
    ax.text(1.02, -0.03, label_header, transform=ax.transAxes, va="top", fontsize=8.5, color=PALETTE["neutral"])
    ax.set_yticks(positions)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    ax.grid(axis="x", visible=False)
    return n_kinds_drawn > 1


def plot_pedestrian_cyclist_gap(table, samples):
    """RQ6 figure: gap in the pedestrian and cyclist share by area, with intervals at CI_LEVEL."""
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    value_labels = [f"{row.gap:+.1f} points  ({samples[area]['n_treated_sites']} / "
                    f"{samples[area]['n_comparison_sites']})" for area, row in table.iterrows()]
    shown = table.assign(flagged=table["flagged"].fillna(False).astype(bool))
    has_two_kinds = draw_gap_rows(ax, shown, value_labels, "gap  (completed / comparison sites)")
    ax.axhline(0.5, color=PALETTE["neutral"], linewidth=0.8)   # separates the citywide row from the boroughs
    if has_two_kinds:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    n_boroughs = int((table.index != ALL_BOROUGHS_LABEL).sum())
    finish(ax, f"RQ6: pedestrian and cyclist share of injury crashes within {BUFFER_M} m, change at completed\n"
               f"sites minus change at comparison sites ({int(shown['flagged'].sum())} of {n_boroughs} borough "
               f"gaps kept after adjustment)",
           f"Gap in percentage points, {EVENT_WINDOW} years after against {EVENT_WINDOW} years before "
           f"(dot = estimate, line = {CI_LEVEL:.0%} interval)", "")
    plt.tight_layout()
    plt.show()


plot_pedestrian_cyclist_gap(rq6_table, rq6_samples)

# %%
rq6_city = rq6_table.loc[ALL_BOROUGHS_LABEL]
rq6_sample = rq6_samples[ALL_BOROUGHS_LABEL]
rq6_sentence = describe_effect(
    rq6_city["gap"], rq6_city["low"], rq6_city["high"], "percentage points",
    rq6_sample["n_treated_sites"], n_unit=f"completed sites, against {rq6_sample['n_comparison_sites']:,} "
    f"comparison sites, in {rq6_sample['n_clusters']:,} clusters",
    subject="The gap in the pedestrian and cyclist share of injury crashes (completed minus comparison sites)",
    above="a larger rise in the pedestrian and cyclist share around completed buildings",
    below="a smaller rise in the pedestrian and cyclist share around completed buildings")
say(rq6_sentence)
say(f"Weighted over cohorts, the share changed by {rq6_city['treated_change']:+.2f} points around completed "
    f"sites and by {rq6_city['comparison_change']:+.2f} points around comparison sites over the same calendar "
    f"years; the gap is the difference between the two. Before completion the share was about "
    f"{rq6_city['treated_before']:.0f}% around completed sites and {rq6_city['comparison_before']:.0f}% around "
    f"comparison sites (pooled over {rq6_sample['treated_crashes']:,} and {rq6_sample['comparison_crashes']:,} "
    f"site-level injury crash counts).")
n_completed_sites = rq6_sample["n_treated_sites"]
n_new_construction_sites = rq6_new_sample["n_treated_sites"]
say(f"Of the {n_completed_sites:,} completed sites, {n_new_construction_sites:,} are new construction and "
    f"{n_completed_sites - n_new_construction_sites:,} are preservation. With the treated group limited to new "
    f"construction the gap is {rq6_new['gap']:+.2f} points ({CI_LEVEL:.0%} interval {rq6_new['low']:+.2f} to "
    f"{rq6_new['high']:+.2f}).")
rq6_boroughs = rq6_table.drop(index=ALL_BOROUGHS_LABEL)
rq6_skipped = [f"{area} ({sample['n_treated_sites']} completed sites)"
               for area, sample in rq6_samples.items() if not sample["estimated"]]
say(f"By borough the gaps run from {rq6_boroughs['gap'].min():+.1f} ({rq6_boroughs['gap'].idxmin()}) to "
    f"{rq6_boroughs['gap'].max():+.1f} points ({rq6_boroughs['gap'].idxmax()}). After adjustment for "
    f"{len(rq6_boroughs)} boroughs, {int(rq6_boroughs['flagged'].sum())} of them keep an interval away from zero "
    f"(smallest adjusted p = {rq6_boroughs['p_adjusted'].min():.2f}). Not estimated, because fewer than "
    f"{MIN_GROUP_N} clean sites: {', '.join(rq6_skipped) if rq6_skipped else 'none'}.")
store_result("RQ6", "Does the pedestrian and cyclist share of injury crashes near sites change after completion?",
             rq6_city["gap"], rq6_city["low"], rq6_city["high"], rq6_sample["n_treated_sites"], rq6_sentence,
             "percentage points, completed minus comparison sites")

# %% [markdown]
# The pedestrian and cyclist share changed little around completed sites of either kind and fell
# somewhat more around comparison sites over the same calendar years. The gap between the two is
# positive, but its interval reaches well below zero, so these data do not show a change in who
# is hurt after a completion. The boroughs do not point one way either: the estimates for the
# Bronx and Brooklyn are positive, those for Manhattan and Queens negative, every interval is
# several points wide, and none is kept after the adjustment. Staten Island has too few clean
# sites to be estimated at all.
#
# The two groups start from different levels: before completion the share is higher around
# comparison sites than around completed ones. The design allows a difference in level and
# assumes only that the two groups would have moved alike.
#
# The interval is several percentage points wide, so a real shift smaller than that, in either
# direction, would not be visible. More than half of the completed sites are preservation
# projects, where nobody new moved in. With the treated group limited to new construction, the
# only kind of project that brings new residents, the gap is almost the same and its interval a
# little wider, so pooling the two kinds does not hide a shift at new buildings that these data
# could detect. The clean sites are, by construction, those without another completion next to
# them, which are the less densely built parts of the programme. A share can also stay flat
# while crashes of both kinds rise or fall together, which is the subject of RQ3. The pedestrian
# and cyclist flag follows the police categories, so riders of e-scooters and similar devices,
# who are recorded outside those categories, are not in it.

# %% [markdown]
# ### 8.7 RQ7. Before the project starts, are affordable-housing sites already on streets with more injury crashes than typical street locations of the same borough?
#
# Owner: M4
#
# The questions so far ask what happens after a building opens. This one looks at where the
# buildings stand in the first place. If the sites are on blocks whose streets already injure
# more people than usual, the families who move in inherit that exposure whatever the building
# itself changes, and the city would know where street redesign should accompany new housing.
# Each site is placed in the ranking of all street locations of its borough, measured before
# the project starts.
#
# The method needs a definition of a typical street location. There is no street map in the two
# datasets, so the city is covered with a regular grid of reference points `REFERENCE_SPACING_M`
# metres apart, and a point is kept if any crash of the ten years was reported within
# `STREET_SNAP_M` metres of it. That keeps points on the street network and discards water,
# parks and airports. Each reference point and each site then gets the same measure: the number
# of injury crashes within `BUFFER_M` metres during the baseline years `BASELINE_YEARS`. Only
# sites whose project started after the baseline are used. For new construction the count
# cannot be affected by the building. Preservation buildings already stood in the baseline
# years, with their tenants, so they are reported separately as the places where the programme
# renovates, and the headline is the new-construction figure. Each site is expressed as a
# percentile among the reference points of its own borough. If sites stood on typical street
# locations, the percentiles would be spread evenly and their mean would be `NEUTRAL_PERCENTILE`.
#
# The benchmark is a grid of street points and contains no other residential buildings. It gives
# a highway, an industrial waterfront and a block of row houses the same weight, while apartment
# buildings of any kind stand on dense blocks. A mean percentile above the neutral value
# therefore shows where these buildings are. It does not show that the programme picks more
# dangerous streets than other housing would. The two tables allow one comparison between kinds
# of residential sites: new construction against preservation, which is existing multi-family
# housing. That gap is the part of this question that is about the programme. The two kinds are
# spread differently over the boroughs, so the gap is estimated three times: across the city,
# inside boroughs (the gap of each borough, weighted by its new-construction sites), and with
# every site ranked among the street points of its own ZIP code. The last version separates a
# difference between neighbourhoods from a difference between streets of one neighbourhood.
#
# A comparison of two medians would say less than the percentile, because it hides the shape of
# the distribution. A correlation between housing units and crashes over grid cells, which is
# the usual way to draw this map, mostly measures that both are absent from parks. The
# percentile keeps the borough fixed and uses only places where a crash could be reported.
#
# Sites of one project stand next to each other and share almost the same buffer, so they are
# not independent. The interval resamples whole one-kilometre blocks (`block_id` from Section
# 5). The linked clusters of Section 5.4 are the more cautious unit, but a few of them chain
# through entire neighbourhoods and hold most of the sites, which leaves few units to resample.
# The interval from resampling clusters is given next to the block interval in every row.

# %%
def reference_street_points(crash_frame, tree, tree_rows, spacing_m=REFERENCE_SPACING_M, snap_m=STREET_SNAP_M):
    """Regular grid of points that lie on the street network, with borough and ZIP of the nearest crash.

    A grid point is kept when a located crash of any year lies within `snap_m` metres. The grid
    step in longitude is widened by 1 / cos(latitude) so the spacing is the same in metres both ways.
    """
    lat_low, lat_high = NYC_BOUNDS["lat"]
    lon_low, lon_high = NYC_BOUNDS["lon"]
    lat_step = spacing_m / METRES_PER_DEGREE
    lon_step = lat_step / math.cos(math.radians(MID_LATITUDE_DEG))
    grid_lat, grid_lon = np.meshgrid(np.arange(lat_low, lat_high, lat_step),
                                     np.arange(lon_low, lon_high, lon_step), indexing="ij")
    grid_lat, grid_lon = grid_lat.ravel(), grid_lon.ravel()
    distance, nearest = tree.query(to_radians(grid_lat, grid_lon), k=1)
    on_street = distance[:, 0] * EARTH_RADIUS_M <= snap_m
    nearest_crash = crash_frame.iloc[tree_rows[nearest[on_street, 0]]]
    points = pd.DataFrame({"lat": grid_lat[on_street], "lon": grid_lon[on_street],
                           "borough": nearest_crash["borough"].to_numpy(),
                           "zip": nearest_crash["zip"].astype(object).to_numpy()})
    return points.dropna(subset=["borough"]).reset_index(drop=True)


def baseline_crash_tree(crash_frame, years=BASELINE_YEARS):
    """Ball tree over the located injury crashes of the baseline years."""
    in_baseline = (crash_frame["is_injury_crash"] & crash_frame["is_located"]
                   & crash_frame["year"].between(*years)).to_numpy()
    return build_haversine_tree(crash_frame["lat"].to_numpy()[in_baseline],
                                crash_frame["lon"].to_numpy()[in_baseline])


def crashes_within_radius(tree, lat, lon, radius_m=BUFFER_M):
    """Number of the tree's crashes within `radius_m` metres of each given point."""
    return tree.query_radius(to_radians(lat, lon), radius_m / EARTH_RADIUS_M, count_only=True)


def midrank_percentile(values, reference_values):
    """Percentile of each value within `reference_values`, counting ties as half below.

    With ties counted as half, a value drawn from the reference itself has a mean percentile of
    exactly 50, which is the benchmark the site percentiles are compared with.
    """
    ordered = np.sort(np.asarray(reference_values))
    below = np.searchsorted(ordered, values, side="left")
    below_or_equal = np.searchsorted(ordered, values, side="right")
    return PERCENT * (below + below_or_equal) / (2 * len(ordered))


def percentile_within_groups(values, groups, reference_values, reference_groups, min_reference=MIN_GROUP_N):
    """Midrank percentile of each value among the reference values of its own group.

    Returns NaN where the group has fewer than `min_reference` reference values.
    """
    percentiles = np.full(len(values), np.nan)
    reference = pd.Series(np.asarray(reference_values)).groupby(np.asarray(reference_groups, dtype=object))
    for group, group_reference in reference:
        in_group = np.asarray(groups, dtype=object) == group
        if len(group_reference) >= min_reference and in_group.any():
            percentiles[in_group] = midrank_percentile(np.asarray(values)[in_group], group_reference.to_numpy())
    return percentiles


def mean_percentile_excess(percentiles):
    """Mean percentile minus NEUTRAL_PERCENTILE: zero when the places rank like typical street locations."""
    return float(percentiles.mean() - NEUTRAL_PERCENTILE)


def mean_percentile_gap(percentiles, in_first_group):
    """Mean percentile of the first group minus that of the rest; NaN when a resample lacks one of them."""
    if in_first_group.all() or not in_first_group.any():
        return float("nan")
    return float(percentiles[in_first_group].mean() - percentiles[~in_first_group].mean())


def site_selection_percentiles(site_frame, site_year_frame, crash_frame, tree, tree_rows):
    """RQ7 statistic inputs: baseline counts and percentiles of sites not yet started.

    Returns (future_sites, reference): sites that start after the baseline with `baseline_crashes`,
    `borough_percentile` and `zip_percentile`, and the reference points with `baseline_crashes`.
    """
    baseline_tree = baseline_crash_tree(crash_frame)     # built once, used for the grid and for the sites
    reference = reference_street_points(crash_frame, tree, tree_rows)
    reference["baseline_crashes"] = crashes_within_radius(baseline_tree, reference["lat"].to_numpy(),
                                                          reference["lon"].to_numpy())
    future_sites = site_frame[site_frame["start_year"] > BASELINE_YEARS[1]].copy()
    in_baseline = site_year_frame["year"].between(*BASELINE_YEARS)
    counted = site_year_frame[in_baseline].groupby("site_id")["injury_crashes"].sum()
    future_sites["baseline_crashes"] = counted.reindex(future_sites["site_id"]).to_numpy()
    # The reference points are counted with a new tree; it must reproduce the site table of Section 5.
    recount = crashes_within_radius(baseline_tree, future_sites["lat"].to_numpy(), future_sites["lon"].to_numpy())
    assert (recount == future_sites["baseline_crashes"].to_numpy()).all(), "the two counting routes disagree"
    future_sites["borough_percentile"] = percentile_within_groups(
        future_sites["baseline_crashes"], future_sites["borough"], reference["baseline_crashes"], reference["borough"])
    future_sites["zip_percentile"] = percentile_within_groups(
        future_sites["baseline_crashes"], future_sites["zip"], reference["baseline_crashes"], reference["zip"])
    assert future_sites["borough_percentile"].notna().all()
    assert set(future_sites["construction_type"].astype(str)) <= {NEW_CONSTRUCTION, PRESERVATION}
    return future_sites, reference


# Row names of the RQ7 estimates table.
RQ7_NEW = "New construction, ranked in the borough"
RQ7_PRESERVED = "Preservation, ranked in the borough"
RQ7_POOLED = "Both kinds pooled, ranked in the borough"
RQ7_GAP = "New construction minus preservation, ranked in the borough"
RQ7_GAP_IN_BOROUGH = "New construction minus preservation, compared inside boroughs"
RQ7_WITHIN_ZIP = "New construction, ranked in its own ZIP code"
RQ7_PRESERVED_WITHIN_ZIP = "Preservation, ranked in its own ZIP code"
RQ7_GAP_WITHIN_ZIP = "New construction minus preservation, ranked in their own ZIP code"


def site_selection_estimates(future_sites):
    """RQ7 estimates: mean percentile excess by kind of project and the gap between the kinds.

    Returns a DataFrame with `estimate`, `low`, `high` (percentile points, one-kilometre blocks
    resampled), `n_sites`, `units_resampled`, and `cluster_low`, `cluster_high`,
    `clusters_resampled` for the same estimate with the linked clusters resampled.
    """
    is_new = (future_sites["construction_type"].astype(str) == NEW_CONSTRUCTION).to_numpy()
    borough_percentiles = future_sites["borough_percentile"].to_numpy()
    zip_percentiles = future_sites["zip_percentile"].to_numpy()
    has_zip_rank = ~np.isnan(zip_percentiles)
    everywhere = np.ones(len(future_sites), dtype=bool)
    borough_codes, borough_labels = pd.factorize(future_sites["borough"].astype(str))

    def gap_inside_boroughs(percentiles, in_first_group, codes):
        """Gap computed borough by borough and averaged with the new-construction counts as weights."""
        return stratified_difference(percentiles, in_first_group, codes, len(borough_labels))["estimate"]

    # Row name -> (statistic, arrays it takes, rows it uses).
    specifications = {
        RQ7_NEW: (mean_percentile_excess, [borough_percentiles], is_new),
        RQ7_PRESERVED: (mean_percentile_excess, [borough_percentiles], ~is_new),
        RQ7_POOLED: (mean_percentile_excess, [borough_percentiles], everywhere),
        RQ7_GAP: (mean_percentile_gap, [borough_percentiles, is_new], everywhere),
        RQ7_GAP_IN_BOROUGH: (gap_inside_boroughs, [borough_percentiles, is_new, borough_codes], everywhere),
        RQ7_WITHIN_ZIP: (mean_percentile_excess, [zip_percentiles], is_new & has_zip_rank),
        RQ7_PRESERVED_WITHIN_ZIP: (mean_percentile_excess, [zip_percentiles], ~is_new & has_zip_rank),
        RQ7_GAP_WITHIN_ZIP: (mean_percentile_gap, [zip_percentiles, is_new], has_zip_rank),
    }
    rows = {}
    for name, (statistic, arrays, used) in specifications.items():
        used_arrays = [array[used] for array in arrays]
        by_block = bootstrap_ci(statistic, *used_arrays, clusters=future_sites["block_id"].to_numpy()[used])
        by_cluster = bootstrap_ci(statistic, *used_arrays, clusters=future_sites["cluster_id"].to_numpy()[used])
        rows[name] = {"estimate": by_block["estimate"], "low": by_block["low"], "high": by_block["high"],
                      "n_sites": by_block["n"], "units_resampled": by_block["n_clusters"],
                      "cluster_low": by_cluster["low"], "cluster_high": by_cluster["high"],
                      "clusters_resampled": by_cluster["n_clusters"]}
    return pd.DataFrame(rows).T


rq7_sites, rq7_reference = site_selection_percentiles(sites, site_year, crashes, crash_tree, crash_tree_rows)
rq7_estimates = site_selection_estimates(rq7_sites)
with pd.option_context("display.max_colwidth", None):
    display(rq7_estimates.astype({"n_sites": int, "units_resampled": int, "clusters_resampled": int}).round(2))

rq7_by_borough = pd.DataFrame({
    "reference_points": rq7_reference.groupby("borough", observed=True).size(),
    "median_crashes_reference": rq7_reference.groupby("borough", observed=True)["baseline_crashes"].median(),
    "sites": rq7_sites.groupby("borough", observed=True).size(),
    "median_crashes_sites": rq7_sites.groupby("borough", observed=True)["baseline_crashes"].median(),
    "mean_percentile_sites": rq7_sites.groupby("borough", observed=True)["borough_percentile"].mean(),
    "share_above_borough_median_pct": rq7_sites.groupby("borough", observed=True)["borough_percentile"]
                                               .apply(lambda values: PERCENT * (values > NEUTRAL_PERCENTILE).mean()),
})
display(rq7_by_borough.round(1))
# Where the sites of each kind are, and how each kind ranks inside each borough.
rq7_kind = rq7_sites["construction_type"].astype(str)
rq7_borough = rq7_sites["borough"].astype(str)
rq7_kind_by_borough = pd.concat(
    [(PERCENT * pd.crosstab(rq7_borough, rq7_kind, normalize="columns")).add_suffix(": share of its sites (%)"),
     rq7_sites.groupby([rq7_borough, rq7_kind])["borough_percentile"].mean().unstack()
     .add_suffix(": mean percentile")], axis=1)
display(rq7_kind_by_borough.round(1))

# %% [markdown]
# The map is drawn on plain longitude and latitude axes, without background tiles. One degree of
# longitude is shorter than one degree of latitude at this latitude, so `map_axes` from Section
# 2 sets the aspect ratio to 1 / cos(latitude) to keep distances true in both directions. Grey points are the reference
# street points, and each site is coloured by its percentile within its borough, with the
# highest percentiles drawn last. The panel on the right shows the same percentiles as a
# distribution, split by kind of project, next to the flat line that typical street locations
# would give.

# %%
def plot_site_selection(future_sites, reference):
    """RQ7 figure: map of future sites by borough percentile, and the distribution of the percentiles."""
    fig, (ax_map, ax_bars) = plt.subplots(1, 2, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [1.2, 1]})

    ax_map.scatter(reference["lon"], reference["lat"], s=1.2, color=MAP_REFERENCE_COLOR, linewidths=0,
                   label="Reference street point", rasterized=True)
    # Sites are drawn from the lowest percentile to the highest, so the busy end stays on top.
    drawn = future_sites.sort_values("borough_percentile")
    points = ax_map.scatter(drawn["lon"], drawn["lat"], c=drawn["borough_percentile"], cmap=SEQUENTIAL_CMAP,
                            vmin=0, vmax=PERCENT, s=10, edgecolor=MAP_MARKER_EDGE, linewidths=0.2)
    map_axes(ax_map)     # ground proportions and no grid (Section 2)
    for borough, centre in reference.groupby("borough", observed=True)[["lon", "lat"]].mean().iterrows():
        label = ax_map.text(centre["lon"], centre["lat"], borough, fontsize=9.5, ha="center", color="black",
                            fontweight="bold")
        label.set_path_effects([patheffects.withStroke(linewidth=3.5, foreground="white")])
    colorbar = fig.colorbar(points, ax=ax_map, orientation="horizontal", fraction=0.045, pad=0.1)
    colorbar.set_label("Site: percentile among street points of its borough (100 = most injury crashes)")
    legend = ax_map.legend(loc="upper left", markerscale=3)
    legend.legend_handles[0].set_sizes([20])
    finish(ax_map, f"RQ7: where sites started after {BASELINE_YEARS[1]} stand", "Longitude (degrees east)",
           "Latitude (degrees north)")

    edges = np.arange(0, PERCENT + PERCENTILE_BIN_WIDTH, PERCENTILE_BIN_WIDTH)
    centres = edges[:-1] + PERCENTILE_BIN_WIDTH / 2
    kinds = [(NEW_CONSTRUCTION, PALETTE["housing"], "o"), (PRESERVATION, PALETTE["placebo"], "s")]
    for kind, color, marker in kinds:
        values = future_sites.loc[future_sites["construction_type"].astype(str) == kind, "borough_percentile"]
        shares = PERCENT * np.histogram(values, bins=edges)[0] / len(values)
        ax_bars.plot(centres, shares, color=color, marker=marker, linewidth=2, markersize=6,
                     markeredgecolor="white", label=f"{kind} ({len(values):,} sites)")
    ax_bars.axhline(PERCENTILE_BIN_WIDTH, color=PALETTE["neutral"], linewidth=1.5, linestyle="--",
                    label="Typical street locations (even spread)")
    ax_bars.set_xticks(edges)
    ax_bars.set_xlim(0, PERCENT)
    ax_bars.set_ylim(0, None)
    ax_bars.grid(axis="x", visible=False)
    ax_bars.legend(loc="upper left")
    finish(ax_bars, "How the sites rank among street points of their borough",
           f"Percentile of injury crashes within {BUFFER_M} m, {BASELINE_YEARS[0]} to {BASELINE_YEARS[1]} "
           f"(bins of {PERCENTILE_BIN_WIDTH})", "Share of the sites of that kind (%)")
    plt.tight_layout()
    plt.show()


plot_site_selection(rq7_sites, rq7_reference)

# %%
def rq7_text(row_name):
    """Estimate, interval and sample of one row of the RQ7 table as a phrase."""
    row = rq7_estimates.loc[row_name]
    return (f"{row['estimate']:+.1f} percentile points ({CI_LEVEL:.0%} interval {row['low']:+.1f} to "
            f"{row['high']:+.1f} with {int(row['units_resampled']):,} blocks resampled, {row['cluster_low']:+.1f} to "
            f"{row['cluster_high']:+.1f} with {int(row['clusters_resampled']):,} clusters; "
            f"{int(row['n_sites']):,} sites)")


rq7_new = rq7_estimates.loc[RQ7_NEW]
rq7_new_sites = rq7_sites[rq7_sites["construction_type"].astype(str) == NEW_CONSTRUCTION]
rq7_sentence = describe_effect(
    rq7_new["estimate"], rq7_new["low"], rq7_new["high"], "percentile points", int(rq7_new["n_sites"]),
    n_unit=f"new-construction sites in {int(rq7_new['units_resampled']):,} one-kilometre blocks",
    subject=f"The mean borough percentile of new-construction sites, before their project started, minus "
            f"{NEUTRAL_PERCENTILE}",
    above="sites standing where there were already more injury crashes than at the typical street point of the "
          "borough",
    below="sites standing where there were fewer injury crashes than at the typical street point of the borough")
say(rq7_sentence)
say(f"The mean percentile of the new-construction sites is {NEUTRAL_PERCENTILE + rq7_new['estimate']:.1f}, and "
    f"{PERCENT * (rq7_new_sites['borough_percentile'] > NEUTRAL_PERCENTILE).mean():.0f}% of them stand above the "
    f"median street point of their borough. The median new-construction site had "
    f"{rq7_new_sites['baseline_crashes'].median():.0f} injury crashes within {BUFFER_M} m in "
    f"{BASELINE_YEARS[0]} to {BASELINE_YEARS[1]}, against {rq7_reference['baseline_crashes'].median():.0f} at the "
    f"median of the {len(rq7_reference):,} reference points. With clusters resampled instead of blocks the same "
    f"estimate has an interval of {rq7_new['cluster_low']:+.1f} to {rq7_new['cluster_high']:+.1f} "
    f"({int(rq7_new['clusters_resampled']):,} clusters).")
say(f"Preservation sites, which already stood in the baseline years: {rq7_text(RQ7_PRESERVED)}. Both kinds "
    f"pooled: {rq7_text(RQ7_POOLED)}. Ranked inside their own ZIP code, for ZIP codes with at least "
    f"{MIN_GROUP_N} reference points, new-construction sites: {rq7_text(RQ7_WITHIN_ZIP)}; preservation sites: "
    f"{rq7_text(RQ7_PRESERVED_WITHIN_ZIP)}.")
say(f"New construction minus preservation, across the city: {rq7_text(RQ7_GAP)}. Inside boroughs: "
    f"{rq7_text(RQ7_GAP_IN_BOROUGH)}. With every site ranked in its own ZIP code: {rq7_text(RQ7_GAP_WITHIN_ZIP)}.")
store_result("RQ7", "Before the project starts, are new-construction sites on streets with more injury crashes "
             "than typical street locations of the borough?", rq7_new["estimate"], rq7_new["low"],
             rq7_new["high"], rq7_new["n_sites"], rq7_sentence,
             "percentile points above the typical street point of the borough")

# %% [markdown]
# New-construction sites are not spread like typical street locations. Their line in the right
# panel climbs from the quietest tenth of street points to the busiest, and their mean
# percentile is well above the neutral value. The interval stays far from zero, also when
# clusters are resampled instead of blocks, which widens it on the lower side. On the map the
# high percentiles gather in the dense parts of the Bronx and Brooklyn. The borough table, which
# pools both kinds of project, shows where the pattern is strong and where it is not: the excess
# is large in Brooklyn, the Bronx and Queens, while in Manhattan and on Staten Island the sites
# rank close to the typical street point. Part of the pattern is about neighbourhoods and part
# is about streets within them: ranked inside their own ZIP code, new-construction sites still
# stand above the middle, by a smaller margin.
#
# Most of this is a statement about apartment buildings and dense blocks, for the reason given
# with the method. The comparison of like with like is the gap between the two kinds of project.
# Preservation sites, which are existing multi-family buildings, rank above the neutral value
# too, although with clusters resampled the interval of their excess in the borough ranking
# reaches zero. New-construction sites rank higher than preservation sites, across the city and
# inside the same borough. The last table shows why the two versions differ: a much larger share
# of the preservation sites is in Manhattan, where sites of both kinds rank lower than in the
# Bronx and Brooklyn.
# With every site ranked inside its own ZIP code the gap is small and its interval includes
# zero. The difference between the two kinds is therefore mostly one of neighbourhood. New
# buildings go up in parts of a borough with more injury crashes than the parts where the
# programme renovates, and inside one ZIP code the streets of the two kinds cannot be told
# apart. The two groups may differ in more than the kind of project, so this is an association
# as well.
#
# None of this says that these streets are more dangerous for each person who uses them. The
# count is not divided by traffic or by the number of people walking, and affordable housing is
# built on dense blocks near avenues and transit, where more people and vehicles meet. For the
# people who move in, the practical meaning is the same either way: the streets outside the door
# already had more injury crashes than most street points of the borough before the project
# started.
#
# Requiring a crash within `STREET_SNAP_M` metres leaves out the quietest streets, which raises
# the benchmark, while points at the edge of the street network have part of their buffer over
# water or parks, which lowers it. The net direction is unknown. Only sites that started after
# the baseline years can be used, so the result speaks for the later part of the programme. The
# block interval keeps buildings of the same project together but ignores any similarity between
# neighbouring blocks, and the cluster interval printed above is the more cautious of the two.

# %% [markdown]
# ### 8.8 RQ8. Does the hour-of-day mix of injury crashes near sites shift after completion?
#
# Owner: M4
#
# A block of offices and workshops is busy at the rush hours and empty at night, while a block
# where several hundred people live has traffic, deliveries and people on foot in the evening
# and at weekends. If completions are followed by a move of injury crashes towards the evening
# or the night, lighting, crossings and enforcement at those hours matter near the building. We
# measure the change in the share of injury crashes in each hour, against the change at
# comparison sites, and test it on the four parts of the day defined in Section 4.
#
# The design is the one of RQ6 with the hour of the crash as the category: shares before and
# after at clean completed sites, the same at not-yet-completed sites over the same calendar
# years, and the gap between the two changes with a cluster bootstrap. The comparison matters
# even more here than for RQ6, because the pandemic removed commuting trips and changed the daily
# profile of crashes everywhere. Twenty-four hourly shares are too many and too noisy to test
# one by one with this sample, so the formal test is made on the four bands, with the p-values
# adjusted for four tests, and the hourly values are shown for the pattern. As in RQ6, completed
# sites of both kinds are pooled, and the gap is printed again for new construction alone.
#
# The headline is the largest of the four band gaps. A gap that is picked because it is the
# largest is biased away from zero, so it is stored with the wider interval that allows for four
# tests (the joint interval of `composition_shift`) and is labelled as the largest of four.

# %%
def hour_mix_shift(units, pairs, crash_frame):
    """RQ8 statistic: gap in the share of injury crashes per hour and per part of the day.

    Returns (hour_table, band_table, hour_summary, band_summary) in the format of
    `composition_shift`, which is run once on the 24 hours and once on the bands.
    """
    hour_codes = crash_frame["hour"].fillna(-1).to_numpy(dtype=np.int64)   # a missing hour is left out
    before, after = period_category_counts(units, pairs, crash_frame, hour_codes, HOURS_PER_DAY)
    hour_table, hour_summary = composition_shift(units, before, after, list(range(HOURS_PER_DAY)))
    # The bands are sums of whole hours, so the hourly counts are added up instead of recounted.
    band_names = list(HOUR_BANDS)
    band_hours = [np.arange(first, last + 1) for first, last in HOUR_BANDS.values()]
    assert sorted(np.concatenate(band_hours)) == list(range(HOURS_PER_DAY)), "the bands must cover each hour once"
    band_before = np.column_stack([before[:, hours].sum(axis=1) for hours in band_hours])
    band_after = np.column_stack([after[:, hours].sum(axis=1) for hours in band_hours])
    band_table, band_summary = composition_shift(units, band_before, band_after, band_names)
    return hour_table, band_table, hour_summary, band_summary


rq8_hours, rq8_bands, rq8_sample, rq8_band_sample = hour_mix_shift(event_units, site_pairs, crashes)
_, rq8_new_bands, _, rq8_new_sample = hour_mix_shift(event_units[is_new_construction_unit].reset_index(drop=True),
                                                     site_pairs, crashes)
rq8_band_ranges = {name: f"{first:02d}:00 to {last:02d}:59" for name, (first, last) in HOUR_BANDS.items()}
display(rq8_bands.rename(index=lambda name: f"{name} ({rq8_band_ranges[name]})").drop(columns=["flagged"]).round(2))

# %% [markdown]
# The heat map has one column per hour and three rows: the change at completed sites, the change
# at comparison sites over the same calendar years, and the gap between them. The colour scale
# is centred on zero, with red for a share that rose and blue for one that fell. Colour carries
# the values. A number is printed only in the gap row, and only for an hour whose own interval
# excludes zero, so that the eye is not drawn to cells that are within noise.

# %%
def plot_hour_shift_heatmap(hour_table, summary):
    """RQ8 figure: change in the hourly share of injury crashes at completed and comparison sites, and the gap."""
    rows = {
        f"Completed sites\n({summary['n_treated_sites']:,})": hour_table["treated_change"],
        f"Comparison sites\n({summary['n_comparison_sites']:,})": hour_table["comparison_change"],
        "Gap: completed\nminus comparison": hour_table["gap"],
    }
    values = np.vstack([series.to_numpy() for series in rows.values()])
    limit = float(np.abs(values).max())
    gap_excludes_zero = ((hour_table["low"] > 0) | (hour_table["high"] < 0)).to_numpy()
    annotations = np.full(values.shape, "", dtype=object)
    annotations[-1, gap_excludes_zero] = [f"{gap:+.1f}" for gap in hour_table["gap"].to_numpy()[gap_excludes_zero]]
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    sns.heatmap(values, ax=ax, cmap=DIVERGING_CMAP, center=0, vmin=-limit, vmax=limit, annot=annotations, fmt="",
                annot_kws={"fontsize": 9, "fontweight": "bold"}, linewidths=2, linecolor="white",
                cbar_kws={"label": "Change in share of injury crashes\n(percentage points)", "pad": 0.015})
    ax.set_xticklabels([f"{hour:02d}" for hour in hour_table.index], rotation=0)
    ax.set_yticklabels(rows.keys(), rotation=0)
    # A heavier line sets the gap row apart, and thin ticks mark where one part of the day ends.
    ax.axhline(2, color=PALETTE["neutral"], linewidth=2.5)
    for name, (first, last) in HOUR_BANDS.items():
        ax.text((first + last + 1) / 2, -0.12, name, ha="center", va="bottom", fontsize=9, color=PALETTE["neutral"])
        if first > 0:
            ax.axvline(first, color=PALETTE["neutral"], linewidth=1.2)
    ax.set_title(f"RQ8: hourly share of injury crashes within {BUFFER_M} m, {EVENT_WINDOW} years after completion "
                 f"against {EVENT_WINDOW} years before\nGap row: a number marks an hour whose {CI_LEVEL:.0%} "
                 f"interval excludes zero ({int(gap_excludes_zero.sum())} of {HOURS_PER_DAY} hours; "
                 f"{int(hour_table['flagged'].sum())} after adjustment for {HOURS_PER_DAY} tests)", pad=24)
    ax.set_xlabel("Hour of day in which the crash happened (24-hour clock)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


plot_hour_shift_heatmap(rq8_hours, rq8_sample)

# %%
rq8_top_band = rq8_bands["gap"].abs().idxmax()
rq8_top = rq8_bands.loc[rq8_top_band]
rq8_sentence = describe_effect(
    rq8_top["gap"], rq8_top["low_joint"], rq8_top["high_joint"], "percentage points", rq8_sample["n_treated_sites"],
    n_unit=f"completed sites, against {rq8_sample['n_comparison_sites']:,} comparison sites, in "
    f"{rq8_sample['n_clusters']:,} clusters", level=rq8_band_sample["joint_level"],
    subject=f"The largest gap among the {len(rq8_bands)} parts of the day, the one for the share of injury "
            f"crashes in the {rq8_top_band.lower()} ({rq8_band_ranges[rq8_top_band]}), with its interval widened "
            f"for {len(rq8_bands)} tests,",
    above=f"relatively more injury crashes in the {rq8_top_band.lower()} around completed buildings",
    below=f"relatively fewer injury crashes in the {rq8_top_band.lower()} around completed buildings")
say(rq8_sentence)
rq8_widest_hour = rq8_hours["gap"].abs().idxmax()
rq8_hours_excluding = int(((rq8_hours["low"] > 0) | (rq8_hours["high"] < 0)).sum())
say(f"Taken alone, the {rq8_top_band.lower()} gap has a {CI_LEVEL:.0%} interval of {rq8_top['low']:+.2f} to "
    f"{rq8_top['high']:+.2f} points. After adjustment for {len(rq8_bands)} parts of the day, "
    f"{int(rq8_bands['flagged'].sum())} of them keep an interval away from zero (adjusted p for the "
    f"{rq8_top_band.lower()}: {rq8_top['p_adjusted']:.2f}). "
    f"The hourly gaps lie between {rq8_hours['gap'].min():+.1f} and {rq8_hours['gap'].max():+.1f} points; the "
    f"widest is at {rq8_widest_hour:02d}:00, with an interval of {rq8_hours.loc[rq8_widest_hour, 'low']:+.1f} to "
    f"{rq8_hours.loc[rq8_widest_hour, 'high']:+.1f}. {rq8_hours_excluding} of the {HOURS_PER_DAY} hourly intervals "
    f"exclude zero before adjustment and {int(rq8_hours['flagged'].sum())} remain after it. The comparison "
    f"sites themselves moved by up to {rq8_hours['comparison_change'].abs().max():.1f} points in a single hour. "
    f"The counts behind the shares are {rq8_sample['treated_crashes']:,} site-level injury crash counts around "
    f"completed sites and {rq8_sample['comparison_crashes']:,} around comparison sites.")
rq8_new = rq8_new_bands.loc[rq8_top_band]
say(f"With the treated group limited to new construction ({rq8_new_sample['n_treated_sites']:,} sites) the "
    f"{rq8_top_band.lower()} gap is {rq8_new['gap']:+.2f} points ({CI_LEVEL:.0%} interval {rq8_new['low']:+.2f} to "
    f"{rq8_new['high']:+.2f}).")
store_result("RQ8", "Does the hour-of-day mix of injury crashes near sites shift after completion?",
             rq8_top["gap"], rq8_top["low_joint"], rq8_top["high_joint"], rq8_sample["n_treated_sites"],
             rq8_sentence, f"percentage points; largest of {len(rq8_bands)} gaps, interval widened for "
                           f"{len(rq8_bands)} tests",
             level=rq8_band_sample["joint_level"], adjusted_p=rq8_top["p_adjusted"])

# %% [markdown]
# No shift in the daily profile can be detected around completed sites of either kind. The
# largest gap among the four parts of the day is a rise in the afternoon share, and its interval
# includes zero, with or without the widening for four tests. The bottom row of the heat map shows why the bands move so little:
# positive and negative cells alternate from one hour to the next (early afternoon up, late
# afternoon down, one evening hour up and the next down). That is what noise looks like. A move
# of crashes towards the evening or the night would show as a run of neighbouring hours with
# the same sign. A few single hours have an interval that
# excludes zero when taken alone, and they are the ones with a number in the figure. With
# twenty-four hours examined, one or two such hours are expected by chance. Two of the hours
# marked here are neighbouring evening hours with opposite signs, and none is kept after the
# adjustment.
#
# The comparison row is paler than the row for completed sites, because it rests on several
# times more crashes, and it shows that the calendar years alone moved the hourly shares very
# little.
#
# Few crashes surround the completed sites, so the interval for a six-hour band is a few
# percentage points wide and a smaller shift would go unseen, while single hours are far
# noisier. Limiting the treated group to new construction gives a smaller afternoon gap with a
# somewhat wider interval. The hour is the one the officer wrote down. Weekdays and weekends
# are pooled, so a change that goes in opposite directions on the two would cancel.

# %% [markdown]
# ### 8.9 RQ9. Across ZIP codes, is the share of deeply affordable units associated with the severity of injury crashes or with the pedestrian and cyclist share of harm?
#
# Owner: M5
#
# Affordable units are counted in income bands, and the two lowest (extremely low and very low
# income) go to the tenants with the lowest incomes. If the neighbourhoods where the programme
# concentrates those units are also the ones where crashes hit people outside vehicles, or kill
# more often, the burden of unsafe streets falls on tenants with the fewest alternatives. We
# estimate the correlation across ZIP codes, with an interval, between the deep-affordable share
# of the units and each of two crash outcomes.
#
# The unit is the ZIP code, taken from the panel of Section 6. The housing variable is the share
# of deeply affordable units among all affordable units completed up to the last study year.
# ZIP codes with fewer than `MIN_ZIP_AFFORDABLE_UNITS` such units are left out, because their
# share depends on one or two small buildings. The two crash outcomes are computed over all ten
# years: the share of injury crashes that hurt a pedestrian or cyclist, and deaths per
# `PER_THOUSAND` injury crashes (the constant of Section 6) as a measure of severity. Both are ratios within injury
# crashes, so the collapse of property-damage reports in the break year does not enter them.
#
# Boroughs differ in everything at once: Manhattan has few drivers and many pedestrians, Staten
# Island the reverse. A raw correlation across the city would mostly compare boroughs. The
# borough means are therefore removed from both variables before ranking, so the coefficient
# measures whether, inside a borough, ZIP codes with a deeper mix differ from their neighbours.
# Spearman's rank correlation is used because neither relationship has to be a straight line and
# a few ZIP codes have extreme values. Two outcomes are tested, so the two p-values are adjusted
# together.

# %%
def zip_equity_table(panel, min_units=MIN_ZIP_AFFORDABLE_UNITS):
    """One row per ZIP code: deep-affordable share of the stock and the two crash outcomes.

    Shares are in percent and deaths are per PER_THOUSAND injury crashes. Only ZIP codes with at
    least `min_units` affordable units completed by the last study year are returned.
    """
    stock = panel[panel["year"] == YEAR_END].set_index("zip")
    totals = panel.groupby("zip")[["injury_crashes", "ped_cyc_crashes", "persons_killed"]].sum()
    table = pd.DataFrame({
        "borough": stock["borough"].astype(str),
        "affordable_units": stock["cum_affordable_units"],
        "deep_share_pct": PERCENT * stock["cum_deep_affordable_units"] / stock["cum_affordable_units"].where(
            stock["cum_affordable_units"] > 0),
        "injury_crashes": totals["injury_crashes"],
        "ped_cyc_share_pct": PERCENT * totals["ped_cyc_crashes"] / totals["injury_crashes"],
        "deaths_per_1000": PER_THOUSAND * totals["persons_killed"] / totals["injury_crashes"],
    })
    assert (table["injury_crashes"] > 0).all()
    return table[table["affordable_units"] >= min_units]


# Outcome column -> (panel title, unit of a difference in that outcome).
RQ9_OUTCOMES = {
    "ped_cyc_share_pct": ("Who is hurt: pedestrian and cyclist share of injury crashes", "percentage points"),
    "deaths_per_1000": (f"How severe: deaths per {PER_THOUSAND:,} injury crashes", "deaths"),
}


def deep_share_associations(zip_table):
    """RQ9 statistic: borough-adjusted Spearman correlation of the deep share with each outcome."""
    rows = {}
    for column in RQ9_OUTCOMES:
        adjusted = group_adjusted_spearman_ci(zip_table["deep_share_pct"].to_numpy(), zip_table[column].to_numpy(),
                                              zip_table["borough"].to_numpy())
        raw = spearman_ci(zip_table["deep_share_pct"].to_numpy(), zip_table[column].to_numpy())
        rows[column] = {"rho_within_borough": adjusted["estimate"], "low": adjusted["low"],
                        "high": adjusted["high"], "p_value": adjusted["p_value"], "n": adjusted["n"],
                        "n_boroughs": adjusted["n_groups"], "rho_unadjusted": raw["estimate"],
                        "unadjusted_low": raw["low"], "unadjusted_high": raw["high"]}
    result = pd.DataFrame(rows).T
    result["p_adjusted"] = benjamini_hochberg(result["p_value"].to_numpy(dtype=float))
    return result


rq9_zips = zip_equity_table(zip_year)
rq9_result = deep_share_associations(rq9_zips)
display(rq9_result.astype(float).round(3))

# %% [markdown]
# The figure shows what the coefficient ranks. Each point is a ZIP code, and both axes are
# differences from the mean of the ZIP code's own borough, so the offsets between boroughs are
# gone and a point to the right is a ZIP code with a deeper mix than its borough average. The
# dashed line is a least-squares line through the points and only guides the eye, since the
# test uses ranks. Colour and marker shape both give the borough.

# %%
def plot_deep_share_scatter(zip_table, result):
    """RQ9 figure: deep-affordable share against each crash outcome, both as differences from the borough mean."""
    borough_size = zip_table.groupby("borough")["borough"].transform("size")
    shown = zip_table[borough_size >= 2]     # a borough with one ZIP code has no within-borough variation
    by_borough = shown.groupby("borough")
    deep_within = shown["deep_share_pct"] - by_borough["deep_share_pct"].transform("mean")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, (column, (title, unit)) in zip(axes, RQ9_OUTCOMES.items()):
        outcome_within = shown[column] - by_borough[column].transform("mean")
        row = result.loc[column]
        # The plotted values are the ones the coefficient is computed from.
        assert np.isclose(spearman_rho(deep_within.to_numpy(), outcome_within.to_numpy()), row["rho_within_borough"])
        ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8)
        ax.axvline(0, color=PALETTE["neutral"], linewidth=0.8)
        for borough in BOROUGHS:
            in_borough = (shown["borough"] == borough).to_numpy()
            if not in_borough.any():
                continue
            ax.scatter(deep_within[in_borough], outcome_within[in_borough], s=38, color=BOROUGH_COLORS[borough],
                       marker=BOROUGH_MARKERS[borough], edgecolor=PALETTE["neutral"], linewidth=0.5,
                       label=f"{borough} ({int(in_borough.sum())})")
        slope = linear_slope(deep_within.to_numpy(), outcome_within.to_numpy())
        ends = np.array([deep_within.min(), deep_within.max()])
        ax.plot(ends, slope * ends, color=PALETTE["neutral"], linewidth=1.5, linestyle="--",
                label="Least-squares line (guide only)")
        ax.text(0.98, 0.97, f"within-borough rank correlation {row['rho_within_borough']:+.3f}\n"
                            f"{CI_LEVEL:.0%} interval {row['low']:+.3f} to {row['high']:+.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=PALETTE["neutral"],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
        reach = float(outcome_within.abs().max())
        ax.set_ylim(-1.15 * reach, 1.45 * reach)     # headroom for the text box
        finish(ax, title, "Deeply affordable share of affordable units,\ndifference from the borough mean "
                          "(percentage points)", f"Difference from the borough mean ({unit})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Borough (number of ZIP codes)", loc="lower center", ncol=len(handles),
               columnspacing=1.5, handletextpad=0.3)
    fig.suptitle(f"RQ9: depth of affordability and injury crashes by ZIP code, inside boroughs, {YEAR_START} to "
                 f"{YEAR_END}", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    # Leave a strip at the bottom of the figure for the shared legend.
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    plt.show()


plot_deep_share_scatter(rq9_zips, rq9_result)

# %%
rq9_ped = rq9_result.loc["ped_cyc_share_pct"]
rq9_deaths = rq9_result.loc["deaths_per_1000"]
# The sentence is written out here and not by describe_effect, because the interval ends so close
# to zero that the adjusted p-value has to be read with it.
rq9_excludes_zero = bool(rq9_ped["low"] > 0 or rq9_ped["high"] < 0)
rq9_kept = bool(rq9_ped["p_adjusted"] < FALSE_DISCOVERY_RATE)
rq9_sentence = (
    f"The within-borough rank correlation between the deep-affordable share and the pedestrian and cyclist "
    f"share of injury crashes is {rq9_ped['rho_within_borough']:+.3f} ({CI_LEVEL:.0%} interval "
    f"{rq9_ped['low']:+.3f} to {rq9_ped['high']:+.3f}, n = {int(rq9_ped['n'])} ZIP codes in "
    f"{int(rq9_ped['n_boroughs'])} boroughs). The interval {'excludes' if rq9_excludes_zero else 'includes'} zero"
    f"{f' by {min(abs(rq9_ped['low']), abs(rq9_ped['high'])):.3f}' if rq9_excludes_zero else ''}, the p-value is "
    f"{rq9_ped['p_value']:.3f} and {rq9_ped['p_adjusted']:.3f} after adjustment for the two outcomes, so this is "
    f"{'counted' if rq9_kept else 'not counted'} as a difference from zero.")
say(rq9_sentence)
say(describe_effect(
    rq9_deaths["rho_within_borough"], rq9_deaths["low"], rq9_deaths["high"], "", int(rq9_deaths["n"]),
    n_unit="ZIP codes", digits=3,
    subject=f"For deaths per {PER_THOUSAND:,} injury crashes the within-borough rank correlation",
    above="more deaths per injury crash where the affordable stock is deeper",
    below="fewer deaths per injury crash where the affordable stock is deeper"))
say(f"Adjusted for two outcomes, the p-values are {rq9_ped['p_adjusted']:.3f} (pedestrian and cyclist share) and "
    f"{rq9_deaths['p_adjusted']:.3f} (deaths). Without the borough adjustment the correlations would be "
    f"{rq9_ped['rho_unadjusted']:+.2f} ({rq9_ped['unadjusted_low']:+.2f} to {rq9_ped['unadjusted_high']:+.2f}) and "
    f"{rq9_deaths['rho_unadjusted']:+.2f} ({rq9_deaths['unadjusted_low']:+.2f} to "
    f"{rq9_deaths['unadjusted_high']:+.2f}). The {len(rq9_zips)} ZIP codes are those with at least "
    f"{MIN_ZIP_AFFORDABLE_UNITS} affordable units out of {zip_year['zip'].nunique()} in the panel; the median "
    f"ZIP has a deep share of {rq9_zips['deep_share_pct'].median():.0f}%, a pedestrian and cyclist share of "
    f"{rq9_zips['ped_cyc_share_pct'].median():.0f}% and {rq9_zips['deaths_per_1000'].median():.1f} deaths per "
    f"{PER_THOUSAND:,} injury crashes.")
store_result("RQ9", "Across ZIP codes, is the deep-affordable share associated with the pedestrian and cyclist "
             "share of harm?", rq9_ped["rho_within_borough"], rq9_ped["low"], rq9_ped["high"], rq9_ped["n"],
             rq9_sentence, "rank correlation within borough", adjusted_p=rq9_ped["p_adjusted"])

# %% [markdown]
# The estimate points against the concern that motivated the question: inside a borough, ZIP
# codes whose affordable stock is deeper tend to have a slightly lower pedestrian and cyclist
# share of injury crashes, where the concern would predict a higher one. It is not counted as a
# difference. The upper end of
# its interval lies a hair from zero, the p-value is just above the threshold, and after the two
# outcomes are adjusted for together it is clearly above it. In the left panel the dashed line
# slopes down gently through a wide cloud. One possible reading is geographic, but these data
# cannot test it: the notebook has no variable for distance from the centre or for how people
# travel. For deaths per injury crash there is no association in either direction, with or
# without the borough adjustment.
#
# These are area data. The housing variable describes the units this programme financed, not
# the people who live in the ZIP code, and nothing here says who was in the crashes.
# Neighbouring ZIP codes resemble each other, which the interval does not account for, so it is
# probably too narrow. Deaths are few in most ZIP codes, which makes the severity measure noisy
# and an absent correlation unsurprising. Staten Island contributes only a handful of ZIP codes
# after the minimum-units rule.

# %% [markdown]
# ### 8.10 RQ10. Which contributing-factor groups are over- or under-represented near completed housing, after against before?
#
# Owner: M5
#
# Every crash report names what the officer judged to be the main contributing factor:
# distraction, failure to yield, speed and so on. The mix of factors says what kind of street
# problem a place has, and each kind has a different remedy (signal timing for failure to yield,
# street design for speed). If the mix shifts around buildings once people live in them, that
# tells the transport department what to look for on those blocks. For each group of factors
# from Section 4 we report the change in its share of injury crashes relative to comparison
# sites, and keep the groups that survive a correction for testing several at once.
#
# This is again the design of RQ6, with the factor group as the category. Two choices are
# specific to it. Crashes whose factor is unrecorded are left out of the shares, because that
# group is large, says nothing about the street, and its size depends on reporting habits; its
# own shift is reported separately as a check and drawn as a separate row of the figure. And
# several groups are tested side by side, so the p-values are adjusted with the
# Benjamini-Hochberg procedure of Section 7. Without the adjustment, one group in twenty would
# look unusual by chance alone. Comparing crashes near housing with crashes elsewhere in the
# city would be simpler, but it would compare dense neighbourhoods with highways, which have a
# different mix of factors for reasons unrelated to housing.
#
# As in RQ8, the headline is the largest gap among the groups, stored with the interval widened
# for the number of groups. Completed sites of both kinds are pooled again. The largest gap and
# the check on unrecorded factors are therefore repeated with the treated group limited to one
# kind of project at a time. Preservation brings no new residents (RQ5), so a shift that appears
# around preservation sites only is not a sign of what new housing does.

# %%
def unrecorded_share_shift(units, pairs, crash_frame):
    """Gap in the share of injury crashes with no factor recorded, among all injury crashes.

    Returns (row, before): the UNRECORDED_FACTOR row in the format of `composition_shift`, with
    `n_treated_sites` added, and the (units, 2) counts before completion in the order of
    UNRECORDED_LABELS.
    """
    is_unrecorded = (crash_frame["factor_group"] == UNRECORDED_FACTOR).to_numpy()
    before, after = period_category_counts(units, pairs, crash_frame, np.where(is_unrecorded, 0, 1),
                                           len(UNRECORDED_LABELS))
    table, summary = composition_shift(units, before, after, UNRECORDED_LABELS)
    row = table.loc[UNRECORDED_FACTOR].copy()
    row["n_treated_sites"] = summary["n_treated_sites"]
    return row, before


def factor_mix_shift(units, pairs, crash_frame):
    """RQ10 statistic: gap in the share of each recorded factor group, and in the unrecorded share.

    Returns (factor_table, unrecorded_row, summary). The factor table follows `composition_shift`,
    with shares taken among injury crashes that have a recorded factor.
    """
    factor_names = [name for name in crash_frame["factor_group"].cat.categories if name != UNRECORDED_FACTOR]
    recorded_codes = pd.Categorical(crash_frame["factor_group"], categories=factor_names).codes.astype(np.int64)
    before, after = period_category_counts(units, pairs, crash_frame, recorded_codes, len(factor_names))
    factor_table, summary = composition_shift(units, before, after, factor_names)

    unrecorded_row, unrecorded_before = unrecorded_share_shift(units, pairs, crash_frame)
    assert (unrecorded_before[:, 1] == before.sum(axis=1)).all(), "recorded crashes must add up to the factor table"
    return factor_table.sort_values("gap", ascending=False), unrecorded_row, summary


rq10_factors, rq10_unrecorded, rq10_sample = factor_mix_shift(event_units, site_pairs, crashes)
rq10_new_factors, rq10_new_unrecorded, rq10_new_sample = factor_mix_shift(
    event_units[is_new_construction_unit].reset_index(drop=True), site_pairs, crashes)
rq10_preserved_unrecorded, _ = unrecorded_share_shift(
    event_units[is_preservation_unit].reset_index(drop=True), site_pairs, crashes)
display(rq10_factors.round(2))

# %% [markdown]
# Each row of the figure is one factor group, ordered from the largest relative rise to the
# largest relative fall. The label on the right gives the group's share of injury crashes around
# completed sites before completion, so that a shift can be judged against the size of the group.
# The last row, below the line, is the check on crashes with no factor recorded. Its share is
# taken among all injury crashes and it is not one of the tested groups.

# %%
def plot_factor_gaps(factor_table, unrecorded_row):
    """RQ10 figure: gap in the share of each contributing-factor group, and the unrecorded share as a check."""
    plotted = ["gap", "low", "high", "treated_before"]
    table = pd.concat([factor_table[plotted], unrecorded_row[plotted].to_frame(UNRECORDED_ROW).T]).astype(float)
    table["flagged"] = [*factor_table["flagged"], False]
    is_check = table.index == UNRECORDED_ROW
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    value_labels = [f"{row.gap:+.1f} points  ({row.treated_before:.0f}%)" for row in table.itertuples()]
    draw_gap_rows(ax, table, value_labels, "gap  (share before completion)", is_check=is_check,
                  check_label=f"Check, not one of the {len(factor_table)} tested groups: share of all injury "
                              f"crashes with no factor recorded")
    ax.axhline(len(factor_table) - 0.5, color=PALETTE["neutral"], linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    finish(ax, f"RQ10: contributing factor of injury crashes within {BUFFER_M} m, change in share at completed\n"
               f"sites minus change at comparison sites ({int(factor_table['flagged'].sum())} of "
               f"{len(factor_table)} group gaps kept after adjustment)",
           f"Gap in percentage points, {EVENT_WINDOW} years after against {EVENT_WINDOW} years before "
           f"(dot = estimate, line = {CI_LEVEL:.0%} interval)", "")
    plt.tight_layout()
    plt.show()


plot_factor_gaps(rq10_factors, rq10_unrecorded)

# %%
rq10_top_name = rq10_factors["gap"].abs().idxmax()
rq10_top = rq10_factors.loc[rq10_top_name]
rq10_sentence = describe_effect(
    rq10_top["gap"], rq10_top["low_joint"], rq10_top["high_joint"], "percentage points",
    rq10_sample["n_treated_sites"],
    n_unit=f"completed sites, against {rq10_sample['n_comparison_sites']:,} comparison sites, in "
    f"{rq10_sample['n_clusters']:,} clusters", level=rq10_sample["joint_level"],
    subject=f"The largest gap among the {len(rq10_factors)} factor groups, the one for the share of "
            f"\"{rq10_top_name}\" among injury crashes with a recorded factor, with its interval widened for "
            f"{len(rq10_factors)} tests,",
    above=f"a growing share of \"{rq10_top_name}\" around completed buildings",
    below=f"a shrinking share of \"{rq10_top_name}\" around completed buildings")
say(rq10_sentence)
rq10_flagged = rq10_factors.index[rq10_factors["flagged"]].tolist()
rq10_unadjusted = rq10_factors.index[rq10_factors["p_value"] < FALSE_DISCOVERY_RATE].tolist()
say(f"Taken alone, that gap has a {CI_LEVEL:.0%} interval of {rq10_top['low']:+.2f} to {rq10_top['high']:+.2f} "
    f"points. Before adjustment, {len(rq10_unadjusted)} of the {len(rq10_factors)} groups have p below "
    f"{FALSE_DISCOVERY_RATE} ({', '.join(rq10_unadjusted) if rq10_unadjusted else 'none'}). After the "
    f"Benjamini-Hochberg adjustment {len(rq10_flagged)} remain "
    f"({', '.join(rq10_flagged) if rq10_flagged else 'none'}); the adjusted p of \"{rq10_top_name}\" is "
    f"{rq10_top['p_adjusted']:.2f}. The shares rest on {rq10_sample['treated_crashes']:,} site-level injury "
    f"crash counts with a recorded factor around completed sites and {rq10_sample['comparison_crashes']:,} around "
    f"comparison sites.")
say(f"The unrecorded share went from {rq10_unrecorded['treated_before']:.1f}% to "
    f"{rq10_unrecorded['treated_after']:.1f}% around completed sites and from "
    f"{rq10_unrecorded['comparison_before']:.1f}% to {rq10_unrecorded['comparison_after']:.1f}% around comparison "
    f"sites, a gap of {rq10_unrecorded['gap']:+.1f} points (interval {rq10_unrecorded['low']:+.1f} to "
    f"{rq10_unrecorded['high']:+.1f}).")
rq10_new = rq10_new_factors.loc[rq10_top_name]
say(f"With the treated group limited to new construction ({rq10_new_sample['n_treated_sites']:,} sites) the gap "
    f"for \"{rq10_top_name}\" is {rq10_new['gap']:+.2f} points ({CI_LEVEL:.0%} interval {rq10_new['low']:+.2f} "
    f"to {rq10_new['high']:+.2f}).")
say("The gap in the unrecorded share by kind of project, each against the same comparison sites: "
    + "; ".join(f"{label} {row['gap']:+.1f} points (interval {row['low']:+.1f} to {row['high']:+.1f}, "
                f"{int(row['n_treated_sites']):,} sites, share {row['treated_before']:.1f}% before and "
                f"{row['treated_after']:.1f}% after)"
                for label, row in (("new construction", rq10_new_unrecorded),
                                   ("preservation", rq10_preserved_unrecorded))) + ".")
store_result("RQ10", "Which contributing-factor groups are over- or under-represented near completed housing, "
             "after against before?", rq10_top["gap"], rq10_top["low_joint"], rq10_top["high_joint"],
             rq10_sample["n_treated_sites"], rq10_sentence,
             f"percentage points; largest of {len(rq10_factors)} gaps, interval widened for "
             f"{len(rq10_factors)} tests",
             level=rq10_sample["joint_level"], adjusted_p=rq10_top["p_adjusted"])

# %% [markdown]
# No factor group is over- or under-represented around completed sites of either kind in a way
# these data can separate from chance. The interval of every tested group crosses zero, and no group reaches
# the threshold even before the adjustment for testing several of them. The two ends of the
# figure, a rise in improper manoeuvres and a fall in impairment or medical causes, are both
# small next to their intervals, and with the treated group limited to new construction the
# largest gap moves close to zero. The widest intervals belong to the largest groups,
# distraction and speed, where a shift of a few points in either direction cannot be excluded.
#
# The check on unrecorded factors, the last row of the figure, does show a difference. The share
# of injury crashes with no factor named fell around completed sites and hardly moved around
# comparison sites, and the interval of that gap excludes zero. Split by kind of project, the
# gap belongs to preservation. Around preservation sites it is large and its interval excludes
# zero, and around new construction it is small with an interval that includes zero.
# Preservation is the placebo of RQ5: nobody new moves in, so this gap is not a sign of anything
# that new residents bring. It concerns how completely reports were filled in around the
# preservation sites of these cohorts, and we do not know what produces it. It is one test
# outside the adjusted set, so it is reported and not counted as a finding. It still matters for
# the pooled result, because the shares above are taken among crashes with a recorded factor: if
# the newly recorded crashes are of a particular kind, they can hide or imitate a shift in the
# mix. With the treated group limited to new construction, where the unrecorded share moves
# much as it does at comparison sites, the largest gap is close to zero.
#
# The factor is the judgement of the reporting officer about the first vehicle only, the nine
# groups come from the keyword rules of Section 4, and the smallest groups (vehicle defects,
# pedestrian or cyclist error) hold only a few percent of crashes, so their shares rest on small
# counts.
