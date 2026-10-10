# %% [markdown]
# ## 8. Research questions
#
# The ten questions below all relate a housing variable to a crash variable. They share an
# outcome, a way of computing intervals and a way of reading them.
#
# The outcome is the **injury crash**, a crash in which at least one person was injured or killed.
# Reports of crashes with property damage only collapsed in 2020 (Section 3), so a count of all
# crashes is not comparable across that year and is not used here.
#
# Every interval is a bootstrap interval at the level `CI_LEVEL`, and the bootstrap resamples whole
# groups of observations that belong together: a ZIP code with all its years, or a cluster of
# neighbouring sites with all its buildings. Each result is printed with the number of units it
# rests on.
#
# The data are observational. Nobody assigned buildings to streets at random, so a difference
# reported here is an association. When an interval includes zero the reading is that the data
# cannot tell a difference from no difference, which is not the same as showing that there is none.
#
# Each question keeps its headline numbers in `RESULTS`, which Section 10 turns into the summary
# table.

# %% [markdown]
# The next cell holds the constants of questions 1 to 5 and two small helpers that the questions
# of this section and Section 9 share: the function that stores a headline in `RESULTS`, and the
# function that turns the event table of Section 5 into one row per site and cohort.

# %%
RESULTS = {}                 # one entry per research question, filled as the questions are answered

# --- Forecast (RQ1) ----------------------------------------------------------------------------
FORECAST_LAGS = 2            # years of history used as features; a third lag would cost one more
                             # of only ten years of training data
# --- Panel regression (RQ2) --------------------------------------------------------------------
UNITS_PER_STEP = 100         # affordable units; slopes are reported per 100 units, about one
                             # mid-sized apartment building
LEAD_LAG_SHIFTS = [-2, -1, 0, 1, 2]   # years; negative = completions that have not happened yet
                                      # (placebo), positive = completions that many years ago
LARGEST_COMPLETION_ZIPS = 5  # ZIP codes with the largest single-year completions, left out in a check
# --- Event study (RQ3 to RQ5) ------------------------------------------------------------------
PRE_YEARS = list(range(-EVENT_WINDOW, 0))        # years before the completion year
POST_YEARS = list(range(1, EVENT_WINDOW + 1))    # years after it; the completion year itself is
                                                 # left out because the building opens part-way through
REL_YEARS = PRE_YEARS + [0] + POST_YEARS
PRE_PERIOD_INDEX = INDEX_BASE    # the mean of the pre-completion years is set to the base of the
                                 # citywide index (Section 6.5) in the event chart
NEW_CONSTRUCTION = "New Construction"
PRESERVATION = "Preservation"
DOSE_GRID_POINTS = 40        # points at which the fitted dose line and its band are drawn
SMALL_BUILDING_UNITS = 20    # units; below this a building is in the two lowest size bands of
                             # Section 4 (a house or a small walk-up)
RATIO_TICKS = [0.125, 0.25, 0.5, 1, 2, 4, 8]     # after / before ratios labelled on the log axis


def store_result(key, question, estimate, low, high, n, sentence, unit, level=CI_LEVEL, adjusted_p=None):
    """Keep the headline of one research question for the summary table in Section 10.

    level: coverage of the stored interval, when it is not CI_LEVEL. adjusted_p: the p-value of
    the headline after adjustment for the tests made together with it, when there are several.
    """
    RESULTS[key] = {"question": question, "estimate": float(estimate), "ci_low": float(low),
                    "ci_high": float(high), "n": int(n), "unit": unit, "sentence": sentence,
                    "level": float(level), "adjusted_p": None if adjusted_p is None else float(adjusted_p)}


def cohort_unit_table(panel, columns, keep_contaminated=False):
    """One row per (site, cohort) of an event table: `site_id`, `cohort`, `group`, `treated` and `columns`.

    The completion-year row of each site and cohort carries its attributes. Unless
    `keep_contaminated` is set, rows that fail the screen of Section 5.4 (another completion
    inside the buffer) are left out. A comparison site appears once for every cohort it serves.
    """
    is_unit = (panel["rel_year"] == 0).to_numpy()
    if not keep_contaminated:
        is_unit &= ~panel["contaminated"].to_numpy()
    units = panel.loc[is_unit, ["site_id", "cohort", "group", *columns]].reset_index(drop=True)
    units["treated"] = (units["group"] == "treated").to_numpy()
    assert not units.duplicated(["site_id", "cohort"]).any()
    return units


# %% [markdown]
# ### 8.1 RQ1. Does knowing a ZIP code's affordable-housing completions improve a forecast of its injury crashes?
#
# Owner: M1
#
# Planners who allocate street-safety money start from last year's injury crashes, and housing
# completions are published by the city in advance. If new buildings changed how dangerous the
# surrounding streets are, adding them to the forecast should make it more accurate. The result
# is the change in forecast error, in injury crashes per ZIP and year, when housing is added.
#
# The test is out of time. Each model is a linear regression fitted on the years before
# `FIRST_TEST_YEAR` and scored on the years from then on, which it has never seen. A linear
# regression is used because only a few hundred ZIP-years are available for fitting, which
# leaves little room to tune a more flexible model without overfitting, and because its housing
# coefficients can be read directly. A tree-based model was the alternative. It was left out
# because the question is whether housing adds information, and a linear baseline makes that
# comparison plain. `FORECAST_LAGS` is two because every further lag costs one more of the ten
# study years. The lags cost the first two years of the window, so the training years start
# two years after `YEAR_START` and straddle the reporting break. `FIRST_TEST_YEAR` holds out the
# last three study years, which all come after the break. The
# baseline uses only the ZIP's own injury crashes in the previous `FORECAST_LAGS` years. Three
# richer models add housing information that would have been known at the end of the previous
# year: the affordable units completed in each of the previous two years, the stock of
# affordable units completed since the housing table begins, or both. The score is the mean
# absolute error (MAE), in injury crashes per ZIP and year. Because all models are scored on the
# same ZIP-years, the difference in MAE is a paired difference, and its interval comes from
# resampling whole ZIP codes.
#
# A correlation between completions and crashes across ZIPs would be simpler, but it would mostly
# show that large, busy ZIPs have more of both. A forecast that already knows the ZIP's crash
# history asks the narrower question of whether housing adds anything on top of it. A random
# split of ZIP-years was rejected because it would let the model learn from the last year to
# predict an earlier one.
#
# The training years contain the citywide drop of the break year and the test years do not. To
# see whether the answer depends on that, the comparison is run a second time with every count
# divided by the citywide level of its own year (the injury-crash index of Section 6.5), so the
# models only have to predict where a ZIP stands relative to the city. That second version is a
# check and not a usable forecast, because the citywide level of a future year is not known in
# advance.

# %%
HISTORY_FEATURES = [f"injury_crashes_lag{lag}" for lag in range(1, FORECAST_LAGS + 1)]
COMPLETION_FEATURES = [f"affordable_units_lag{lag}" for lag in range(1, FORECAST_LAGS + 1)]
STOCK_FEATURES = ["affordable_stock_lag1"]
FORECAST_FEATURE_SETS = {
    "Crash history only": HISTORY_FEATURES,
    "History + completions": HISTORY_FEATURES + COMPLETION_FEATURES,
    "History + stock": HISTORY_FEATURES + STOCK_FEATURES,
    "History + completions + stock": HISTORY_FEATURES + COMPLETION_FEATURES + STOCK_FEATURES,
}
FORECAST_BASELINE = next(iter(FORECAST_FEATURE_SETS))
FORECAST_FULL = list(FORECAST_FEATURE_SETS)[-1]
PERSISTENCE_NAME = "Same as last year (no model)"


def build_forecast_frame(panel, n_lags=FORECAST_LAGS):
    """One row per ZIP and year with the target and features that use only earlier years.

    Every feature is shifted inside its ZIP, so the row for year t holds values from t - 1 and
    t - 2. The first `n_lags` years of each ZIP have no history and are dropped.
    """
    ordered = panel.sort_values(["zip", "year"]).reset_index(drop=True)
    by_zip = ordered.groupby("zip", sort=False)
    frame = ordered[["zip", "year", "borough", "injury_crashes"]].copy()
    for lag in range(1, n_lags + 1):
        frame[f"injury_crashes_lag{lag}"] = by_zip["injury_crashes"].shift(lag)
        frame[f"affordable_units_lag{lag}"] = by_zip["affordable_units_completed"].shift(lag)
    frame["affordable_stock_lag1"] = by_zip["cum_affordable_units"].shift(1)
    frame = frame.dropna().reset_index(drop=True)
    assert len(frame) == panel["zip"].nunique() * (panel["year"].nunique() - n_lags)
    return frame


def citywide_adjusted_frame(frame, city):
    """The forecast frame with every injury-crash count divided by the citywide level of its own year.

    The citywide level is the injury-crash index of Section 6.5 divided by INDEX_BASE, so a
    count becomes "injury crashes at the citywide level of the base year". Lagged counts are
    divided by the level of the year they come from.
    """
    level = city.set_index("year")["injury_crash_index"] / INDEX_BASE
    years = frame["year"].astype(int)
    adjusted = frame.copy()
    adjusted["injury_crashes"] = frame["injury_crashes"] / years.map(level).to_numpy()
    for lag in range(1, FORECAST_LAGS + 1):
        column = f"injury_crashes_lag{lag}"
        adjusted[column] = frame[column] / (years - lag).map(level).to_numpy()
    assert adjusted[HISTORY_FEATURES + ["injury_crashes"]].notna().all().all()
    return adjusted


def forecast_comparison(frame):
    """Out-of-time MAE of each feature set, the paired difference from the baseline, and a no-model benchmark.

    Returns (summary, predictions, persistence). `persistence` describes the benchmark that needs
    no fitting, next year equals last year: its `mae` and the paired difference `gap` from the
    baseline model with the interval `low` to `high`. It shows how much of the accuracy comes
    from repeating the last count.
    """
    summary, predictions = time_split_evaluation(
        frame, "injury_crashes", FORECAST_FEATURE_SETS, time_column="year",
        first_test_period=FIRST_TEST_YEAR, cluster_column="zip")
    test = frame[frame["year"] >= FIRST_TEST_YEAR]
    actual = test["injury_crashes"].to_numpy(dtype=float)
    persistence_errors = np.abs(actual - test["injury_crashes_lag1"].to_numpy(dtype=float))
    baseline_errors = np.abs(actual - predictions[FORECAST_BASELINE].to_numpy())
    gap = bootstrap_ci(mean_gap, persistence_errors, baseline_errors, clusters=test["zip"].to_numpy())
    persistence = {"mae": float(persistence_errors.mean()), "gap": gap["estimate"], "low": gap["low"],
                   "high": gap["high"]}
    summary["difference_pct"] = PERCENT * summary["mae_minus_baseline"] / summary.loc[FORECAST_BASELINE, "mae"]
    return summary, predictions, persistence


def plot_forecast_comparison(summary, persistence):
    """Left: test MAE of every forecast. Right: its difference from crash history only, with the interval.

    The two panels share their rows, so each model name on the left labels the dot and the
    interval on the same line of both panels.
    """
    names = [PERSISTENCE_NAME] + list(summary.index)
    mae = np.array([persistence["mae"]] + list(summary["mae"]))
    gaps = pd.DataFrame({"gap": [persistence["gap"], *summary["mae_minus_baseline"]],
                         "low": [persistence["low"], *summary["low"]],
                         "high": [persistence["high"], *summary["high"]]}, index=names)
    positions = np.arange(len(names))
    colors = [PALETTE["control"], PALETTE["crash"]] + [PALETTE["housing"]] * (len(names) - 2)
    fig, (ax_level, ax_gap) = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True,
                                           gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.05})
    ax_level.scatter(mae, positions, s=60, color=colors, zorder=3, edgecolor="white", linewidth=1.2)
    for position, value in zip(positions, mae):
        ax_level.annotate(f"{value:.2f}", (value, position), xytext=(9, 0), textcoords="offset points",
                          va="center", fontsize=9.5, color=PALETTE["neutral"])
    spread = mae.max() - mae.min()
    ax_level.set_xlim(mae.min() - 0.4 * spread, mae.max() + 0.7 * spread)
    ax_level.set_yticks(positions)
    ax_level.set_yticklabels(names)
    ax_level.set_ylim(positions[-1] + 0.6, positions[0] - 0.6)
    ax_level.grid(axis="x", visible=False)
    finish(ax_level, f"Forecast error in {FIRST_TEST_YEAR}-{YEAR_END}",
           "Mean absolute error (injury crashes per ZIP\nper year); the axis does not start at zero", "")

    ax_gap.axvline(0, color=PALETTE["neutral"], linewidth=1)
    reach = float(np.nanmax(np.abs(gaps[["low", "high"]].to_numpy())))
    for position, color, (name, row) in zip(positions, colors, gaps.iterrows()):
        if name == FORECAST_BASELINE:
            ax_gap.scatter([0], [position], s=50, color=color, zorder=3, marker="D")
            label, label_x = "reference: crash history only", 0
        else:
            ax_gap.hlines(position, row["low"], row["high"], color=color, linewidth=2)
            ax_gap.scatter([row["gap"]], [position], s=55, color=color, zorder=3, edgecolor="white", linewidth=1.2)
            label, label_x = f"{row['gap']:+.2f}  [{row['low']:+.2f}, {row['high']:+.2f}]", row["high"]
        # The label starts to the right of the interval, so it never crosses the zero line.
        ax_gap.annotate(label, (label_x, position), xytext=(9, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=9, color=PALETTE["neutral"],
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax_gap.set_xlim(-1.15 * reach, 2.2 * reach)     # the right half holds the value labels
    ax_gap.grid(axis="x", visible=False)
    finish(ax_gap, "Difference from the error of crash history only",
           f"Difference in mean absolute error ({CI_LEVEL:.0%} interval, ZIPs resampled)\n"
           f"right of zero = less accurate than crash history only", "")
    fig.suptitle("RQ1: out-of-time forecast of injury crashes per ZIP, with and without housing features",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(top=0.84, left=0.22, bottom=0.2)
    plt.show()


forecast_frame = build_forecast_frame(zip_year)
forecast_summary, forecast_predictions, forecast_persistence = forecast_comparison(forecast_frame)
persistence_mae = forecast_persistence["mae"]
assert (forecast_predictions["year"] >= FIRST_TEST_YEAR).all(), "no test row may come from a training year"
display(forecast_summary[["mae", "mae_minus_baseline", "low", "high", "p_value", "difference_pct",
                          "n_features", "n_train", "n_test", "n_test_clusters"]].round(3))

# %%
plot_forecast_comparison(forecast_summary, forecast_persistence)

# %%
rq1 = forecast_summary.loc[FORECAST_FULL]
rq1_sentence = describe_effect(
    rq1["mae_minus_baseline"], rq1["low"], rq1["high"], "injury crashes per ZIP per year",
    int(rq1["n_test_clusters"]), n_unit="ZIP codes",
    subject="The change in out-of-time forecast error from adding affordable-housing completions and stock",
    above="a less accurate forecast", below="a more accurate forecast")
store_result("RQ1", "Does knowing a ZIP's affordable-housing completions improve an out-of-time forecast "
             "of its injury crashes?", rq1["mae_minus_baseline"], rq1["low"], rq1["high"],
             rq1["n_test_clusters"], rq1_sentence, "MAE difference (injury crashes per ZIP per year)")
mean_test_target = forecast_predictions["injury_crashes"].mean()
first_stock_year = int(housing.loc[housing["is_completed"], "completion_year"].min())
say(f"The models are fitted on {int(rq1['n_train']):,} ZIP-years ({forecast_frame['year'].min()} to "
    f"{FIRST_TEST_YEAR - 1}) and scored on {int(rq1['n_test']):,} ZIP-years ({FIRST_TEST_YEAR} to {YEAR_END}) "
    f"in {int(rq1['n_test_clusters'])} ZIP codes, where the average ZIP has {mean_test_target:.0f} injury "
    f"crashes a year. The stock counts the affordable units completed from {first_stock_year} on. Crash "
    f"history alone misses by {forecast_summary.loc[FORECAST_BASELINE, 'mae']:.2f} injury crashes on average, "
    f"against {persistence_mae:.2f} for repeating last year (difference {forecast_persistence['gap']:+.2f}, "
    f"interval {forecast_persistence['low']:+.2f} to {forecast_persistence['high']:+.2f}). "
    f"{rq1_sentence} In relative terms the change is {rq1['difference_pct']:+.1f}% of the baseline error.")

# %%
# The same comparison with the citywide level of each year divided out of every count.
adjusted_summary, _, adjusted_persistence = forecast_comparison(citywide_adjusted_frame(forecast_frame, city_year))
rq1_adjusted = adjusted_summary.loc[FORECAST_FULL]
display(adjusted_summary[["mae", "mae_minus_baseline", "low", "high", "p_value", "difference_pct"]].round(3))
say(f"With the citywide level divided out, crash history alone misses by "
    f"{adjusted_summary.loc[FORECAST_BASELINE, 'mae']:.2f} injury crashes (at the citywide level of "
    f"{INDEX_BASE_YEAR}) and repeating last year by {adjusted_persistence['mae']:.2f}. Adding completions and "
    f"stock changes the error by {rq1_adjusted['mae_minus_baseline']:+.2f} (interval "
    f"{rq1_adjusted['low']:+.2f} to {rq1_adjusted['high']:+.2f}, {rq1_adjusted['difference_pct']:+.1f}% of the "
    f"baseline error, n = {int(rq1_adjusted['n_test_clusters'])} ZIP codes), against "
    f"{rq1['mae_minus_baseline']:+.2f} ({rq1['low']:+.2f} to {rq1['high']:+.2f}) in the main version.")

# %% [markdown]
# Housing does not help the forecast. Each of the three models that know about completions or
# the housing stock has a slightly larger error on the held-out years than the model that knows
# only the ZIP's crash history, and the intervals of all three differences lie to the right of
# zero. The size of the loss is small next to the error itself. The fitted history model in turn
# cannot be told from repeating last year's count, since the interval of that difference
# includes zero.
#
# The second version shows that the loss does not come from the citywide drop inside the
# training years. With the citywide level divided out of every count, the models with housing
# features are still less accurate than crash history alone, by a similar margin. Why the
# housing features hurt is not tested here. One possible reason, not checked, is that with a few
# hundred ZIP-years the extra coefficients are fitted to noise.
#
# The interval resamples the test ZIPs but keeps the fitted models fixed, so it understates the
# uncertainty that comes from fitting on a few hundred ZIP-years. The result also rests on one
# train and test split and on a linear model, and Section 9.10 shows that the loss is absent
# when only crashes with a reported ZIP code are counted. The supported reading is that housing
# features did not improve the forecast in this split. It does not show that they make
# forecasts worse. A ZIP is a large area, and a forecast that does not improve at that scale says little about the
# blocks next to a building, which RQ3 looks at directly.

# %% [markdown]
# ### 8.2 RQ2. Within a ZIP code over time, do injury crashes move as its stock of completed affordable units grows?
#
# Owner: M1
#
# The forecast above compares ZIP codes with one another. This question follows each ZIP against
# itself: it compares a ZIP's injury crashes in the years when its stock of affordable units is
# above its own average with the years when the stock is below it. The result is reported as a
# percentage change in injury crashes for every `UNITS_PER_STEP` additional units, with an
# interval.
#
# The method is a two-way fixed-effects regression of the log of injury crashes on the
# cumulative number of affordable units completed since the housing table begins. The ZIP effect
# removes everything about a ZIP that does not change (its size and its road network), and the
# year effect removes whatever hit all ZIPs in the same year, which includes the drop of the
# break year. What is left is the movement of a ZIP around its own level and around the citywide
# year. The interval resamples whole ZIP codes, because the ten years of one ZIP are not ten
# independent observations.
#
# A single slope cannot show timing, so the figure uses a second regression in which the yearly
# additions to the stock are entered at several distances in time: the units completed one and
# two years ago, in the same year, and one and two years in the future. The stock itself is not
# shifted, because the stock in neighbouring years is almost the same number and the separate
# slopes could not be told apart. The future completions are a placebo. A building that does
# not exist yet cannot affect crashes, so a slope on a future completion that differs from zero
# would mean that crashes were already moving differently in the ZIPs that later received housing.
#
# A slope on yearly completions can rest on a handful of ZIP codes that received a very large
# project in one year. The timing regression is therefore repeated without the
# `LARGEST_COMPLETION_ZIPS` ZIP codes that have the largest single-year completions.
#
# The completions of a year mix two kinds of project. New construction brings new residents and
# preservation brings none (RQ5 uses it as a placebo for that reason). The timing regression is
# therefore also run on the units of each kind separately.

# %%
def panel_matrix(panel, column):
    """ZIP-by-year matrix of one panel column (rows are ZIP codes, columns are study years)."""
    matrix = panel.pivot(index="zip", columns="year", values=column)
    assert matrix.notna().all().all(), "the ZIP by year panel must be balanced"
    return matrix


def stock_regression(panel):
    """Two-way fixed-effects slopes of log injury crashes on affordable housing, per 100 units.

    Returns (stock_slope, timing): the single slope on the cumulative stock, and a table of
    jointly estimated slopes on the yearly completions at each lead and lag. Both are in
    percent change in injury crashes per UNITS_PER_STEP units.
    """
    injury = panel_matrix(panel, "injury_crashes").to_numpy(dtype=float)
    log_injury = np.log(injury + LOG_RATIO_OFFSET)
    stock = panel_matrix(panel, "cum_affordable_units").to_numpy(dtype=float) / UNITS_PER_STEP
    completions = panel_matrix(panel, "affordable_units_completed").to_numpy(dtype=float) / UNITS_PER_STEP

    stock_slope = within_slope(log_injury, {"cumulative stock": stock}).iloc[0]
    trimmed_outcome, shifted, kept_columns = lead_lag_panel(log_injury, completions, LEAD_LAG_SHIFTS)
    timing = within_slope(trimmed_outcome, shifted)
    timing["shift"] = sorted(LEAD_LAG_SHIFTS)
    timing["first_year"] = STUDY_YEARS[kept_columns[0]]
    timing["last_year"] = STUDY_YEARS[kept_columns[-1]]
    # Five slopes are read together, so their p-values are adjusted for multiple testing.
    timing["p_adjusted"] = benjamini_hochberg(timing["p_value"].to_numpy())
    for column in ["estimate", "low", "high"]:
        timing[f"{column}_pct"] = log_points_to_percent(timing[column])
    stock_slope = stock_slope.copy()
    for column in ["estimate", "low", "high"]:
        stock_slope[f"{column}_pct"] = log_points_to_percent(stock_slope[column])
    stock_slope["zips_with_growth"] = int((stock[:, -1] > stock[:, 0]).sum())
    return stock_slope, timing


def same_year_slope_by_kind(panel, unit_columns):
    """Completion-year slope of the timing regression, run separately on each column of completed units.

    unit_columns: {label: panel column}. Each column replaces the yearly completions in the lead
    and lag regression of `stock_regression`. Returns one row per label with the same-year slope
    in percent per UNITS_PER_STEP units, its interval and its unadjusted p-value.
    """
    log_injury = np.log(panel_matrix(panel, "injury_crashes").to_numpy(dtype=float) + LOG_RATIO_OFFSET)
    rows = {}
    for label, column in unit_columns.items():
        completions = panel_matrix(panel, column).to_numpy(dtype=float) / UNITS_PER_STEP
        trimmed_outcome, shifted, _ = lead_lag_panel(log_injury, completions, LEAD_LAG_SHIFTS)
        same_year = within_slope(trimmed_outcome, shifted).loc[shift_name(0)]
        rows[label] = {"estimate_pct": float(log_points_to_percent(same_year["estimate"])),
                       "low_pct": float(log_points_to_percent(same_year["low"])),
                       "high_pct": float(log_points_to_percent(same_year["high"])),
                       "p_value": float(same_year["p_value"]), "n_units": int(same_year["n_units"])}
    return pd.DataFrame(rows).T


def plot_timing_coefficients(timing):
    """Coefficient plot: slope on completions at each lead (placebo) and lag, with intervals."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    is_lead = (timing["shift"] < 0).to_numpy()
    ax.axhline(0, color=PALETTE["neutral"], linewidth=1)
    ax.axvspan(timing["shift"].min() - 0.5, -0.5, color=PALETTE["placebo"], alpha=0.08, linewidth=0)
    for mask, color, label in [(is_lead, PALETTE["placebo"], "Units not completed yet (placebo)"),
                               (~is_lead, PALETTE["housing"], "Units completed that year or earlier")]:
        rows = timing[mask]
        ax.vlines(rows["shift"], rows["low_pct"], rows["high_pct"], color=color, linewidth=2)
        ax.scatter(rows["shift"], rows["estimate_pct"], s=60, color=color, zorder=3, edgecolor="white",
                   linewidth=1.2, label=label)
    # The value sits above the upper end of its interval, clear of the marker and of the zero line.
    for _, row in timing.iterrows():
        ax.annotate(f"{row['estimate_pct']:+.2f}%", (row["shift"], row["high_pct"]), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=PALETTE["neutral"])
    ax.set_xticks(timing["shift"])
    ax.set_xticklabels([f"{-shift} yr before\ncompletion" if shift < 0 else
                        ("completion\nyear" if shift == 0 else f"{shift} yr after\ncompletion")
                        for shift in timing["shift"]])
    ax.set_xlim(timing["shift"].min() - 0.5, timing["shift"].max() + 0.5)
    low, high = timing["low_pct"].min(), timing["high_pct"].max()
    ax.set_ylim(low - 0.08 * (high - low), high + 0.22 * (high - low))     # headroom for the value labels
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:+.2f}%"))
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([], [], color=PALETTE["neutral"], linewidth=2))
    labels.append(f"{CI_LEVEL:.0%} interval, ZIPs resampled")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3)
    finish(ax, f"RQ2: injury crashes in a ZIP before and after affordable units are completed\n"
               f"{int(timing['n_units'].iloc[0])} ZIP codes, {int(timing['first_year'].iloc[0])}-"
               f"{int(timing['last_year'].iloc[0])}, ZIP and year effects removed",
           "Timing of the injury crashes relative to the completion of the units",
           f"Change in injury crashes per {UNITS_PER_STEP} units (%)")
    plt.tight_layout()
    plt.show()


stock_slope, timing_slopes = stock_regression(zip_year)
display(timing_slopes[["shift", "estimate_pct", "low_pct", "high_pct", "p_value", "p_adjusted",
                       "n_units", "n_periods"]].round(3))

# %%
plot_timing_coefficients(timing_slopes)

# %%
rq2_sentence = describe_effect(
    stock_slope["estimate_pct"], stock_slope["low_pct"], stock_slope["high_pct"],
    f"percent per {UNITS_PER_STEP} units", int(stock_slope["n_units"]), n_unit="ZIP codes",
    subject="The within-ZIP change in injury crashes as the stock of completed affordable units grows",
    above="more injury crashes in the years with a larger stock",
    below="fewer injury crashes in the years with a larger stock")
store_result("RQ2", "Within a ZIP over time, do injury crashes move as the cumulative stock of completed "
             "affordable units grows?", stock_slope["estimate_pct"], stock_slope["low_pct"],
             stock_slope["high_pct"], stock_slope["n_units"], rq2_sentence,
             f"% change in injury crashes per {UNITS_PER_STEP} units")
lead_rows = timing_slopes[timing_slopes["shift"] < 0]
lag_rows = timing_slopes[timing_slopes["shift"] >= 0]
n_leads_flagged = int((lead_rows["p_adjusted"] < FALSE_DISCOVERY_RATE).sum())
n_lags_flagged = int((lag_rows["p_adjusted"] < FALSE_DISCOVERY_RATE).sum())
say(f"{rq2_sentence} The panel has {int(stock_slope['n_units'])} ZIP codes over "
    f"{int(stock_slope['n_periods'])} years, and the stock grew in {int(stock_slope['zips_with_growth'])} "
    f"of them. In the timing regression ({int(timing_slopes['first_year'].iloc[0])} to "
    f"{int(timing_slopes['last_year'].iloc[0])}), {n_leads_flagged} of {len(lead_rows)} placebo slopes and "
    f"{n_lags_flagged} of {len(lag_rows)} slopes on completed units differ from zero after the "
    f"Benjamini-Hochberg adjustment at {FALSE_DISCOVERY_RATE:.2f}. The slopes range from "
    f"{timing_slopes['estimate_pct'].min():+.2f}% to {timing_slopes['estimate_pct'].max():+.2f}% per "
    f"{UNITS_PER_STEP} units.")

# %%
# The timing regression without the ZIP codes that received the largest completions in one year.
largest_completions = (zip_year.groupby("zip", observed=True)["affordable_units_completed"].max()
                       .nlargest(LARGEST_COMPLETION_ZIPS))
_, timing_without_largest = stock_regression(zip_year[~zip_year["zip"].isin(largest_completions.index)])
same_year = timing_slopes[timing_slopes["shift"] == 0].iloc[0]
same_year_without = timing_without_largest[timing_without_largest["shift"] == 0].iloc[0]
say(f"The completion-year slope is {same_year['estimate_pct']:+.2f}% per {UNITS_PER_STEP} units (interval "
    f"{same_year['low_pct']:+.2f}% to {same_year['high_pct']:+.2f}%, adjusted p = {same_year['p_adjusted']:.3f}, "
    f"n = {int(same_year['n_units'])} ZIP codes). Without the {LARGEST_COMPLETION_ZIPS} ZIP codes with the "
    f"largest single-year completions ({int(largest_completions.min()):,} to {int(largest_completions.max()):,} "
    f"affordable units in one year) it is {same_year_without['estimate_pct']:+.2f}% (interval "
    f"{same_year_without['low_pct']:+.2f}% to {same_year_without['high_pct']:+.2f}%, adjusted p = "
    f"{same_year_without['p_adjusted']:.3f}, n = {int(same_year_without['n_units'])} ZIP codes).")

# %%
# The same regression with the completed units of one kind of project at a time.
KIND_UNIT_COLUMNS = {"new construction": "new_construction_units", "preservation": "preservation_units"}
same_year_by_kind = same_year_slope_by_kind(zip_year, KIND_UNIT_COLUMNS)
display(same_year_by_kind.round(3))
say(f"Completion-year slope by kind of project, per {UNITS_PER_STEP} units completed in projects of that kind: "
    + "; ".join(f"{label} {row['estimate_pct']:+.2f}% (interval {row['low_pct']:+.2f}% to {row['high_pct']:+.2f}%, "
                f"p = {row['p_value']:.3f})" for label, row in same_year_by_kind.iterrows())
    + f". Both rest on {int(same_year_by_kind['n_units'].iloc[0])} ZIP codes, and the p-values are not adjusted.")

# %% [markdown]
# Within a ZIP, a growing stock of affordable units is not accompanied by a detectable movement
# in injury crashes. The slope on the cumulative stock is close to zero and its interval
# includes zero.
#
# The timing figure agrees with that for four of its five points. The two placebo slopes, on
# units that were not completed yet, sit on the zero line, so there is no sign that crashes were
# already moving differently in ZIPs that were about to receive housing. The slopes one and two
# years after completion have opposite signs and intervals that include zero.
#
# The completion-year slope is the exception. It is negative with an interval that excludes zero,
# here and without the ZIP codes with the largest single-year completions, where the interval
# widens and the adjusted p-value rises above the threshold. Split by kind of project, it
# belongs to preservation: the slope for units in new construction is close to zero with an
# interval that includes it, and the slope for preserved units is negative. A preservation
# completion is a financing date and brings no new residents, so by the reasoning of RQ5 this
# dip is not a sign of what new housing does to a street. We do not know what produces it. It
# does not carry into the following years, and the notebook runs many more tests than these
# five without an adjustment across all of them, so it is reported and not counted as a finding.
#
# A ZIP code is a coarse unit. A hundred units are a small change for an area in which
# tens of thousands of people live, so only a very large association could show up here. The
# stock also counts preserved units together with new ones, although preservation brings no new
# residents (RQ5 separates the two). The timing regression loses two years at each end of the
# window and does not estimate anything beyond two years after completion.

# %% [markdown]
# ### 8.3 RQ3. Do injury crashes within 250 m of a new building change after it is completed?
#
# Owner: M2
#
# A ZIP code is large, and a building of a hundred apartments is a small part of it. If new
# residents change anything on the street, the place to look is the few blocks around the front
# door. This question compares injury crashes within `BUFFER_M` metres of new-construction
# sites in the `EVENT_WINDOW` years before completion with the `EVENT_WINDOW` years after. The
# estimate is the percentage by which that change differs from the change at similar places
# where nothing was completed.
#
# The second half of that sentence is the reason for the design. Injury crashes fell citywide
# in the break year and only partly came back, so a plain before and after comparison at sites
# completed around then would show a drop that has nothing to do with the buildings. Each
# completion cohort is therefore compared with sites of the same programme that were not
# completed until after the cohort's window ended, observed over the same calendar years.
#
# The two groups pass different screens. A treated site is used only if no other site within
# its buffer is completed in a different year of its window; neighbours finished in the same
# year or before the window are allowed. A comparison site is used only if no site within its
# buffer is completed by the end of the window (Section 5.4). Section 9.5 applies the stricter
# rule to both groups.
#
# The estimate is a difference-in-differences on the log scale: the log of the after to before
# ratio at treated sites minus the same at comparison sites. Logs are used so that a busy avenue
# and a quiet side street count equally; in levels the few busiest sites would decide the
# result. The comparison is made inside each cohort and the cohorts are then averaged with the
# treated counts as weights. Whole clusters of neighbouring sites are resampled for the interval.
#
# Sites that pass the screen are few in each cohort, so the cohorts are pooled into one
# estimate instead of several noisy ones.

# %%
REL_COLUMNS = [f"rel_{rel_year:+d}" for rel_year in REL_YEARS]
PRE_COLUMNS = [f"rel_{rel_year:+d}" for rel_year in PRE_YEARS]
POST_COLUMNS = [f"rel_{rel_year:+d}" for rel_year in POST_YEARS]
EVENT_SITE_COLUMNS = ["cluster_id", "block_id", "borough", "construction_type", "total_units"]


def build_event_sites(panel, count_column="injury_crashes"):
    """One row per clean site and cohort: yearly counts by relative year, and before/after totals.

    Contaminated rows (another completion inside the buffer) are left out. A comparison site
    appears once for every cohort it serves; its `cluster_id` keeps those copies together in
    the bootstrap.
    """
    clean = panel[~panel["contaminated"]]
    counts = clean.pivot(index=["site_id", "cohort"], columns="rel_year", values=count_column)
    counts.columns = [f"rel_{rel_year:+d}" for rel_year in counts.columns]
    attributes = cohort_unit_table(panel, EVENT_SITE_COLUMNS).set_index(["site_id", "cohort"])
    table = attributes.join(counts[REL_COLUMNS], validate="one_to_one").reset_index()
    table["before"] = table[PRE_COLUMNS].sum(axis=1)
    table["after"] = table[POST_COLUMNS].sum(axis=1)
    assert table[REL_COLUMNS].notna().all().all() and len(table) == len(counts)
    return table


def select_event_sample(event_table, treated_types):
    """Treated sites of the given construction types plus every clean comparison site."""
    is_wanted_treated = event_table["treated"] & event_table["construction_type"].isin(treated_types)
    return event_table[is_wanted_treated | ~event_table["treated"]].reset_index(drop=True)


def event_profile(sample):
    """Mean injury crashes per site by relative year, indexed to the pre-completion mean, per group.

    Both groups are averaged cohort by cohort and the cohorts are weighted by the treated
    counts, so the two lines describe the same mix of calendar years. Returns a tidy DataFrame
    with `group`, `rel_year`, `index`, `low`, `high`.
    """
    cohort_codes, cohort_labels = pd.factorize(sample["cohort"])
    treated = sample["treated"].to_numpy()
    cohort_weights = np.bincount(cohort_codes[treated], minlength=len(cohort_labels)).astype(float)
    n_pre = len(PRE_YEARS)

    def indexed_means(counts, codes):
        """Cohort-weighted yearly means of one resample, divided by their pre-period mean."""
        sizes = np.bincount(codes, minlength=len(cohort_labels))
        present = (sizes > 0) & (cohort_weights > 0)
        if not present.any():
            return np.full(counts.shape[1], np.nan)
        sums = np.column_stack([np.bincount(codes, weights=counts[:, column], minlength=len(cohort_labels))
                                for column in range(counts.shape[1])])
        cohort_means = sums[present] / sizes[present, None]
        yearly = (cohort_weights[present] / cohort_weights[present].sum()) @ cohort_means
        baseline = yearly[:n_pre].mean()
        # A resample with no crash before completion has no base for the index. NaN (and not
        # the infinity a division would give) lets the bootstrap count the replicate as failed.
        if baseline <= 0:
            return np.full(counts.shape[1], np.nan)
        return PRE_PERIOD_INDEX * yearly / baseline

    parts = []
    for is_treated, group in [(True, "treated"), (False, "comparison")]:
        rows = sample[sample["treated"] == is_treated]
        boot = bootstrap_ci(indexed_means, rows[REL_COLUMNS].to_numpy(dtype=float),
                            cohort_codes[(treated == is_treated)], clusters=rows["cluster_id"].to_numpy())
        parts.append(pd.DataFrame({"group": group, "rel_year": REL_YEARS, "index": boot["estimate"],
                                   "low": boot["low"], "high": boot["high"], "n_sites": len(rows)}))
    return pd.concat(parts, ignore_index=True)


def event_did(sample, before="before", after="after"):
    """Cohort-stratified difference-in-differences of the log after/before ratio, clusters resampled."""
    return did_log_ratio(sample[before].to_numpy(), sample[after].to_numpy(), sample["treated"].to_numpy(),
                         strata=sample["cohort"].to_numpy(), clusters=sample["cluster_id"].to_numpy())


def plot_event_profile(profile, did):
    """Two indexed lines with interval bands: treated new-construction sites and comparison sites."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(PRE_PERIOD_INDEX, color=PALETTE["neutral"], linewidth=1)
    ax.axvspan(-0.5, 0.5, color=PALETTE["before"], alpha=0.25, linewidth=0)
    styles = {"treated": (PALETTE["treated"], "o", "New-construction sites, completed in year 0"),
              "comparison": (PALETTE["control"], "s", "Comparison sites, not completed in the window")}
    handles = []
    for group, (color, marker, label) in styles.items():
        rows = profile[profile["group"] == group]
        # The treated band is a light fill and the comparison band a hatched outline, so the
        # part where they overlap can still be read.
        if group == "treated":
            ax.fill_between(rows["rel_year"], rows["low"], rows["high"], color=color, alpha=0.15, linewidth=0)
        else:
            ax.fill_between(rows["rel_year"], rows["low"], rows["high"], facecolor="none", edgecolor=color,
                            hatch="///", linewidth=0.6, alpha=0.45)
        handles += ax.plot(rows["rel_year"], rows["index"], color=color, linewidth=2, marker=marker,
                           markersize=7, markeredgecolor="white",
                           label=f"{label} (n = {int(rows['n_sites'].iloc[0]):,})")
    handles.append(mpl.patches.Patch(facecolor=PALETTE["treated"], alpha=0.15, edgecolor=PALETTE["control"],
                                     hatch="///", label=f"Bands: {CI_LEVEL:.0%} interval, site clusters resampled"))
    # End labels: the higher line is labelled above its last point and the lower one below it,
    # so the two numbers cannot sit on top of each other.
    last_points = profile[profile["rel_year"] == REL_YEARS[-1]].sort_values("index")
    for (_, last), (offset, anchor) in zip(last_points.iterrows(), [(-7, "top"), (7, "bottom")]):
        ax.annotate(f"{last['index']:.0f}", (last["rel_year"], last["index"]), xytext=(8, offset),
                    textcoords="offset points", va=anchor, fontsize=9.5, color=PALETTE["neutral"])
    ax.text(0, 0.97, "completion\nyear", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9,
            color=PALETTE["neutral"])
    ax.set_xticks(REL_YEARS)
    ax.set_xticklabels([f"{rel_year:+d}" if rel_year else "0" for rel_year in REL_YEARS])
    ax.set_xlim(REL_YEARS[0] - 0.2, REL_YEARS[-1] + 0.45)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    finish(ax, f"RQ3: injury crashes within {BUFFER_M} m of new buildings, before and after completion\n"
               f"Relative to comparison sites: {did['percent_change']:+.1f}% "
               f"[{did['percent_low']:+.1f}%, {did['percent_high']:+.1f}%], n = {did['n_treated']} sites in "
               f"{did['n_strata']} cohorts",
           "Years relative to the completion year (the vertical axis does not start at zero)",
           f"Injury crashes per site (index, years -{EVENT_WINDOW} to -1 = {PRE_PERIOD_INDEX})")
    plt.tight_layout()
    plt.show()


event_site_table = build_event_sites(event_panel)
new_build_sample = select_event_sample(event_site_table, [NEW_CONSTRUCTION])
new_build_did = event_did(new_build_sample)
new_build_profile = event_profile(new_build_sample)
# Placebo inside the pre-period: the change from the earliest pre-year to the last one. Nothing was
# completed between them, so a difference here would point to diverging trends before completion.
pretrend_did = event_did(new_build_sample, before=PRE_COLUMNS[0], after=PRE_COLUMNS[-1])
assert new_build_did["n_treated_dropped"] == 0 and new_build_did["n_control_dropped"] == 0, \
    "every cohort must have both treated and comparison sites"
display(new_build_sample.groupby(["cohort", "group"], observed=True)
        .agg(sites=("site_id", "size"), clusters=("cluster_id", "nunique"),
             injury_crashes_before=("before", "mean"), injury_crashes_after=("after", "mean"))
        .round(1).unstack("group"))

# %%
plot_event_profile(new_build_profile, new_build_did)

# %%
rq3_sentence = describe_effect(
    new_build_did["percent_change"], new_build_did["percent_low"], new_build_did["percent_high"], "percent",
    new_build_did["n_treated"], n_unit="new-construction sites", digits=1,
    subject=f"The after-against-before change in injury crashes within {BUFFER_M} m of new buildings, "
            f"relative to comparison sites,",
    above="more injury crashes after completion than the comparison sites would suggest",
    below="fewer injury crashes after completion than the comparison sites would suggest")
store_result("RQ3", f"Do injury crashes within {BUFFER_M} m of new-construction sites change after completion, "
             "relative to not-yet-completed sites?", new_build_did["percent_change"],
             new_build_did["percent_low"], new_build_did["percent_high"], new_build_did["n_treated"],
             rq3_sentence, "% change relative to comparison sites")
say(f"{rq3_sentence} The comparison group has {new_build_did['n_control']:,} site-cohort rows and the "
    f"bootstrap resamples {new_build_did['n_clusters']:,} clusters. On their own, treated sites changed by "
    f"{log_points_to_percent(new_build_did['treated_change']):+.1f}% and comparison sites by "
    f"{log_points_to_percent(new_build_did['control_change']):+.1f}% (means of site log ratios, converted to percent). "
    f"The placebo change inside the pre-period, from year {PRE_YEARS[0]} to year {PRE_YEARS[-1]}, is "
    f"{pretrend_did['percent_change']:+.1f}% (interval {pretrend_did['percent_low']:+.1f}% to "
    f"{pretrend_did['percent_high']:+.1f}%).")

# %% [markdown]
# The two lines move together. At treated and comparison sites alike, injury crashes are lower
# at the end of the three years before completion than at the start, reach their lowest point in
# the year after completion and recover part of the way. The fall at the new buildings is
# therefore mostly the fall that happened at similar streets where nothing was completed, and a
# plain before and after comparison would have credited it to the buildings. Once the comparison
# sites are taken into account, the remaining difference is slightly negative and its interval
# includes zero.
#
# The placebo inside the pre-period also includes zero, but its interval is wider than the main
# one (both are printed above). It cannot rule out a difference in earlier trends as large as
# the main estimate, and Section 9.8 gives a narrower placebo.
#
# The answer is that no change is detectable. The interval is wide enough to hold a moderate
# decrease as well as a small increase, so this is not evidence that nothing changes.
#
# For a new building, the years before completion are largely years of construction, with
# fences and closed pavements, so "before" does not describe an undisturbed street. Some
# comparison sites were themselves under construction during the window. The sites that pass
# the screen are those with no other completion within the buffer in a different year of their
# window, which are the less densely built parts of the programme. The result describes those
# places, and Section 9.5 looks at the dense clusters where most buildings stand. The
# comparison is stratified by cohort only, and the comparison sites are spread differently
# across the boroughs than the treated sites. Section 9.9 repeats it inside boroughs.

# %% [markdown]
# ### 8.4 RQ4. Does the change around a new building grow with the size of the building?
#
# Owner: M2
#
# The new-construction sites in this sample range from a single house to a tower with several
# hundred apartments. If the people who move in are what changes the street, a tower should
# leave a bigger mark than a house, and a relation of that kind would be more convincing than
# one average over all sizes. The estimate is a slope: the change in the after to before ratio
# of injury crashes for each doubling of the number of units.
#
# Each treated site's log ratio is first compared with the mean log ratio of the comparison
# sites of its cohort, which takes out the calendar years as in RQ3. The adjusted change is then
# regressed on the base-2 logarithm of the size, so the slope reads per doubling. The size is
# the number of units completed in that year within the buffer, the site's own building
# included, because neighbouring buildings finished together are one event (Section 5.4). A row
# of ten small houses is then entered as one event of ten households on the same streets, and
# not as ten small ones. The slope on the building's own units is reported next to it. The
# comparison means are recomputed inside every bootstrap resample, together with the slope, so
# the interval carries the uncertainty of both steps.
#
# One slope is used instead of size bins because it uses every site once, does not depend on
# where the bin edges are put, and is a single test.

# %%
def cohort_adjusted_change(change, treated, cohort_codes, n_cohorts):
    """Each site's log ratio minus the mean log ratio of the comparison sites of its cohort."""
    comparison_n = np.bincount(cohort_codes[~treated], minlength=n_cohorts)
    comparison_sum = np.bincount(cohort_codes[~treated], weights=change[~treated], minlength=n_cohorts)
    # A resample can leave a cohort without comparison sites; its treated sites then get NaN
    # and are left out of the slope for that resample.
    comparison_mean = np.where(comparison_n > 0, comparison_sum / np.maximum(comparison_n, 1), np.nan)
    return change - comparison_mean[cohort_codes]


def add_event_units(sample, site_frame, radius_m=BUFFER_M):
    """Add `event_units` and `event_buildings` to the treated rows of an event sample.

    `event_units` is the sum of `total_units` over the sites within `radius_m` metres whose
    completion year equals the row's cohort, the site itself included; `event_buildings` counts
    those sites. Comparison rows have no completion and get NaN and zero.
    """
    first, second = site_neighbour_pairs(site_frame, radius_m)     # includes each site with itself
    neighbours = pd.DataFrame({
        "site_id": first,
        "cohort": site_frame["completion_year"].astype("float64").to_numpy()[second],
        "units": site_frame["total_units"].to_numpy(dtype=float)[second]})
    per_event = (neighbours.dropna(subset=["cohort"]).groupby(["site_id", "cohort"])["units"]
                 .agg(event_units="sum", event_buildings="size").reset_index())
    is_treated = sample["treated"].to_numpy()
    treated_keys = sample.loc[is_treated, ["site_id", "cohort"]].astype({"site_id": "int64", "cohort": "float64"})
    matched = treated_keys.merge(per_event.astype({"site_id": "int64"}), on=["site_id", "cohort"], how="left",
                                 validate="one_to_one")
    event_units = np.full(len(sample), np.nan)
    event_buildings = np.zeros(len(sample), dtype=np.int64)
    event_units[is_treated] = matched["event_units"].to_numpy()
    event_buildings[is_treated] = matched["event_buildings"].to_numpy(dtype=np.int64)
    with_events = sample.assign(event_units=event_units, event_buildings=event_buildings)
    treated_rows = with_events[is_treated]
    # A treated site is completed in its cohort year, so its own building is always in the sum.
    assert (treated_rows["event_units"] >= treated_rows["total_units"]).all()
    assert (treated_rows["event_buildings"] >= 1).all()
    return with_events


def dose_response(sample, n_grid=DOSE_GRID_POINTS, units_column="event_units"):
    """Slope of the comparison-adjusted log ratio on log2(units) among treated sites, with a fitted line.

    `units_column` names the size measure of the treated rows. Returns a dict with the slope per
    doubling (log points and percent), its interval, the treated points (`units`,
    `adjusted_change`, `cluster_id`) and the fitted line with its band on a grid.
    """
    change = log_ratio(sample["before"].to_numpy(), sample["after"].to_numpy())
    treated = sample["treated"].to_numpy()
    cohort_codes, cohort_labels = pd.factorize(sample["cohort"])
    # Comparison rows get a placeholder size of one unit; they never enter the slope.
    log_units = np.log2(np.where(treated, sample[units_column].to_numpy(dtype=float), 1.0))
    grid = np.linspace(log_units[treated].min(), log_units[treated].max(), n_grid)

    def slope_and_line(change_rows, treated_rows, cohort_rows, log_unit_rows):
        """[slope, fitted value at every grid point] for one resample."""
        adjusted = cohort_adjusted_change(change_rows, treated_rows, cohort_rows, len(cohort_labels))
        usable = treated_rows & ~np.isnan(adjusted)
        if usable.sum() < MIN_GROUP_N:
            return np.full(1 + n_grid, np.nan)
        slope = linear_slope(log_unit_rows[usable], adjusted[usable])
        intercept = adjusted[usable].mean() - slope * log_unit_rows[usable].mean()
        return np.r_[slope, intercept + slope * grid]

    boot = bootstrap_ci(slope_and_line, change, treated, cohort_codes, log_units,
                        clusters=sample["cluster_id"].to_numpy())
    adjusted = cohort_adjusted_change(change, treated, cohort_codes, len(cohort_labels))
    return {"estimate": float(boot["estimate"][0]), "low": float(boot["low"][0]), "high": float(boot["high"][0]),
            "p_value": float(boot["p_value"][0]), "n_treated": int(treated.sum()),
            "n_clusters": boot["n_clusters"],
            "points": pd.DataFrame({"units": sample.loc[treated, units_column].to_numpy(dtype=float),
                                    "adjusted_change": adjusted[treated],
                                    "cluster_id": sample.loc[treated, "cluster_id"].to_numpy()}),
            "line": pd.DataFrame({"units": 2.0 ** grid, "fit": boot["estimate"][1:],
                                  "low": boot["low"][1:], "high": boot["high"][1:]})}


def plot_dose_response(dose):
    """Scatter of treated sites (units against adjusted change) with the fitted line and its band."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    points, line = dose["points"], dose["line"]
    ax.axhline(0, color=PALETTE["neutral"], linewidth=1)
    ax.scatter(points["units"], points["adjusted_change"], s=34, color=PALETTE["treated"], alpha=0.5,
               edgecolor="white", linewidth=0.6, label=f"One new-construction site (n = {len(points)})")
    ax.fill_between(line["units"], line["low"], line["high"], color=PALETTE["crash"], alpha=0.18, linewidth=0,
                    label=f"{CI_LEVEL:.0%} interval of the line, site clusters resampled")
    ax.plot(line["units"], line["fit"], color=PALETTE["crash"], linewidth=2, label="Fitted line")
    ax.set_xscale("log")
    thousands(ax, axis="x")
    shown_ticks = [tick for tick in RATIO_TICKS
                   if points["adjusted_change"].min() - 0.3 <= np.log(tick) <= points["adjusted_change"].max() + 0.3]
    ax.set_yticks(np.log(shown_ticks))
    ax.set_yticklabels([f"x{tick:g}" for tick in shown_ticks])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    finish(ax, f"RQ4: change in injury crashes around a new building against the units completed there\n"
               f"Per doubling of units: {log_points_to_percent(dose['estimate']):+.1f}% "
               f"[{log_points_to_percent(dose['low']):+.1f}%, {log_points_to_percent(dose['high']):+.1f}%]",
           f"Units completed in that year within {BUFFER_M} m of the site, own building included (log scale)",
           "Injury crashes after / before,\nrelative to comparison sites (ratio, log scale)")
    plt.tight_layout()
    plt.show()


dose_sample = add_event_units(new_build_sample, sites)
dose = dose_response(dose_sample)
dose_own_units = dose_response(dose_sample, units_column="total_units")
assert dose["points"]["adjusted_change"].notna().all()
# Rank version of the same relation, as a check that a few very large events do not decide it.
# Sites of one cluster share streets, so whole clusters are resampled here as well.
dose_rank = spearman_ci(np.log2(dose["points"]["units"]), dose["points"]["adjusted_change"],
                        clusters=dose["points"]["cluster_id"].to_numpy())
display(dose["points"].assign(units_completed_nearby=pd.cut(dose["points"]["units"], SIZE_BAND_EDGES,
                                                            labels=SIZE_BAND_LABELS))
        .groupby("units_completed_nearby", observed=True)["adjusted_change"]
        .agg(sites="size", mean_adjusted_log_ratio="mean", median_adjusted_log_ratio="median").round(3))

# %%
plot_dose_response(dose)

# %%
rq4_sentence = describe_effect(
    log_points_to_percent(dose["estimate"]), log_points_to_percent(dose["low"]), log_points_to_percent(dose["high"]), "percent per doubling of units",
    dose["n_treated"], n_unit="new-construction sites", digits=1,
    subject=f"The slope of the after-against-before change in injury crashes on the units completed in "
            f"that year within {BUFFER_M} m",
    above="a larger rise (or smaller fall) in injury crashes where more units were completed",
    below="a larger fall (or smaller rise) in injury crashes where more units were completed")
store_result("RQ4", "Does the before/after change in injury crashes at a new-construction site scale with "
             "the number of units?", log_points_to_percent(dose["estimate"]), log_points_to_percent(dose["low"]),
             log_points_to_percent(dose["high"]), dose["n_treated"], rq4_sentence, "% change per doubling of units")
dose_treated = dose_sample[dose_sample["treated"]]
n_dose_sites = len(dose_treated)
n_shared_events = int((dose_treated["event_buildings"] > 1).sum())
n_small_buildings = int((dose_treated["total_units"] < SMALL_BUILDING_UNITS).sum())
say(f"{rq4_sentence} The units completed nearby range from {int(dose['points']['units'].min())} to "
    f"{int(dose['points']['units'].max()):,} (median {dose['points']['units'].median():.0f}), and "
    f"{dose['n_clusters']:,} clusters are resampled. The rank correlation between size and adjusted change "
    f"is {dose_rank['estimate']:+.2f} (interval {dose_rank['low']:+.2f} to {dose_rank['high']:+.2f}, "
    f"n = {dose_rank['n']} sites in {dose_rank['n_clusters']} clusters, clusters resampled).")
say(f"{n_shared_events} of the {n_dose_sites} sites ({pct(n_shared_events, n_dose_sites):.0f}%) have at least one "
    f"other building completed in the same year within {BUFFER_M} m, up to "
    f"{int(dose_treated['event_buildings'].max())} buildings in one event. The median site has "
    f"{dose_treated['total_units'].median():.0f} units in its own building, and "
    f"{pct(n_small_buildings, n_dose_sites):.0f}% of the buildings have fewer than {SMALL_BUILDING_UNITS} units. "
    f"With the building's own units as the size, the slope is "
    f"{log_points_to_percent(dose_own_units['estimate']):+.1f}% per doubling (interval "
    f"{log_points_to_percent(dose_own_units['low']):+.1f}% to {log_points_to_percent(dose_own_units['high']):+.1f}%).")

# %% [markdown]
# No relation with size can be detected. The fitted line slopes down slightly and its interval
# includes zero, and the rank correlation, which a few very large events cannot dominate, says
# the same. The scatter shows why: at every size the sites are spread widely above and below
# the line of no change, and that spread is far larger than any tilt of the line. Using the
# building's own units instead of the units of the whole event gives a flatter slope with the
# same reading.
#
# An interval per doubling that looks narrow still leaves room for a sizeable difference between
# the smallest and the largest events, which lie several doublings apart, so this is not
# evidence that size makes no difference.
#
# Two things limit this test: the sample is small and units only stand in for residents. Many of
# the buildings are small (the median and the share of small buildings are printed above), where
# the crash count within the buffer is decided by the street and not by a handful of new
# households, so they add noise at the low end. The data have no count of residents, trips or
# cars. The screen for nearby completions is also more likely to remove large projects, which
# tend to be built in groups, so the large buildings that remain may not be typical of large
# buildings in general.

# %% [markdown]
# ### 8.5 RQ5. Is the change different for new construction than for preservation?
#
# Owner: M3
#
# The programme counts two very different things as a completed building. New construction puts
# a building, and its residents, where there was none. Preservation refinances or renovates a
# building whose tenants already live there, so the completion date marks paperwork and repairs
# and nobody new arrives on the street. That makes preservation a placebo: a change in injury
# crashes that comes with new residents should appear around new construction and not around
# preservation. If both types show the same change, the change belongs to something else, such
# as the kind of street the programme works on or the years in which it finished buildings.
#
# Both types are compared with the same comparison sites, using the cohort-stratified
# difference-in-differences of RQ3. The headline is the difference between the two estimates.
# All three numbers come out of one bootstrap in which every resample draws the clusters once
# and recomputes both estimates, so the comparison sites that the two estimates share are the
# same in both and the interval of the difference accounts for that. Subtracting two separately
# bootstrapped estimates would treat them as independent and give an interval that is too wide.

# %%
CONTRAST_NAMES = [NEW_CONSTRUCTION, PRESERVATION, "New construction relative to preservation"]


def construction_type_contrast(event_table):
    """DiD of the log ratio for each construction type against the shared comparison sites, and their difference.

    Returns a DataFrame indexed by CONTRAST_NAMES with the estimate and interval in log points
    and percent, the p-value and the number of treated sites behind each row.
    """
    change = log_ratio(event_table["before"].to_numpy(), event_table["after"].to_numpy())
    treated = event_table["treated"].to_numpy()
    construction = event_table["construction_type"].astype(str).to_numpy()
    # Arm 0 = comparison (any type, since nothing was completed there), 1 and 2 = the treated types.
    arm = np.where(~treated, 0, np.where(construction == NEW_CONSTRUCTION, 1, 2))
    assert set(construction[treated]) <= {NEW_CONSTRUCTION, PRESERVATION}
    cohort_codes, cohort_labels = pd.factorize(event_table["cohort"])

    def type_estimates(change_rows, arm_rows, cohort_rows):
        """[new construction, preservation, difference] recomputed on one resample."""
        by_type = []
        for treated_arm in (1, 2):
            rows = (arm_rows == 0) | (arm_rows == treated_arm)
            by_type.append(stratified_difference(change_rows[rows], arm_rows[rows] == treated_arm,
                                                 cohort_rows[rows], len(cohort_labels))["estimate"])
        return np.array([by_type[0], by_type[1], by_type[0] - by_type[1]])

    boot = bootstrap_ci(type_estimates, change, arm, cohort_codes, clusters=event_table["cluster_id"].to_numpy())
    n_new, n_preserved = int((arm == 1).sum()), int((arm == 2).sum())
    table = pd.DataFrame({"estimate": boot["estimate"], "low": boot["low"], "high": boot["high"],
                          "p_value": boot["p_value"], "n_treated": [n_new, n_preserved, n_new + n_preserved],
                          "n_comparison": int((arm == 0).sum()), "n_clusters": boot["n_clusters"],
                          "n_failed": boot["n_failed"]},
                         index=pd.Index(CONTRAST_NAMES, name="contrast"))
    for column in ["estimate", "low", "high"]:
        table[f"{column}_pct"] = log_points_to_percent(table[column])
    return table


def plot_type_contrast(contrast, note):
    """Dot and interval for each construction type and for their difference; `note` goes under the chart."""
    fig, ax = plt.subplots(figsize=(10, 4.3))
    colors = [PALETTE["treated"], PALETTE["placebo"], PALETTE["neutral"]]
    positions = np.array([0, 1, 2.4])
    ax.axvline(0, color=PALETTE["neutral"], linewidth=1)
    ax.axhline(1.7, color=PALETTE["before"], linewidth=0.8, linestyle=":")
    ax.hlines(positions, contrast["low_pct"], contrast["high_pct"], color=colors, linewidth=2)
    ax.scatter(contrast["estimate_pct"], positions, s=70, color=colors, zorder=3, edgecolor="white",
               linewidth=1.2)
    reach = float(np.abs(contrast[["low_pct", "high_pct"]].to_numpy()).max())
    ax.set_xlim(-1.15 * reach, 1.15 * reach)
    # Every value label starts at the left edge of the plot, above its row, so none of them
    # is written across the zero line.
    for position, (_, row) in zip(positions, contrast.iterrows()):
        ax.text(0.01, position - 0.33,
                f"{row['estimate_pct']:+.1f}%  [{row['low_pct']:+.1f}%, {row['high_pct']:+.1f}%]",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5, color=PALETTE["neutral"])
    labels = [f"New construction\n(n = {contrast['n_treated'].iloc[0]} sites)",
              f"Preservation, the placebo\n(n = {contrast['n_treated'].iloc[1]} sites)",
              "New construction relative\nto preservation"]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(positions[-1] + 0.5, positions[0] - 0.7)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:+.0f}%"))
    ax.grid(axis="y", visible=False)
    finish(ax, "RQ5: change in injury crashes near a site after completion, by project type",
           f"Injury crashes within {BUFFER_M} m, after against before, relative to comparison sites\n"
           f"({CI_LEVEL:.0%} interval, site clusters resampled); right of zero = more after completion", "")
    ax.annotate(note, xy=(0, -0.36), xycoords="axes fraction", fontsize=8.5, color=PALETTE["neutral"], va="top")
    plt.tight_layout()
    plt.show()


type_contrast = construction_type_contrast(event_site_table)
assert np.isclose(type_contrast.loc[NEW_CONSTRUCTION, "estimate"], new_build_did["estimate"]), \
    "the new-construction row must reproduce RQ3"
display(type_contrast[["estimate_pct", "low_pct", "high_pct", "p_value", "n_treated", "n_comparison",
                       "n_clusters"]].round(3))

# %%
preserved_by_cohort = event_site_table[event_site_table["treated"]
                                       & (event_site_table["construction_type"] == PRESERVATION)].groupby("cohort").size()
preservation_cohort_share = pct(preserved_by_cohort.max(), preserved_by_cohort.sum())
plot_type_contrast(type_contrast, f"Preservation: {preservation_cohort_share:.0f}% of its sites belong to the "
                                  f"{preserved_by_cohort.idxmax()} completion cohort.")

# %%
rq5 = type_contrast.loc[CONTRAST_NAMES[2]]
rq5_sentence = describe_effect(
    rq5["estimate_pct"], rq5["low_pct"], rq5["high_pct"], "percent", int(rq5["n_treated"]),
    n_unit="completed sites", digits=1,
    subject="The after-to-before ratio of injury crashes around new construction, relative to the same "
            "ratio around preservation,",
    above="a more positive change around new construction than around preservation",
    below="a more negative change around new construction than around preservation")
store_result("RQ5", "Is the before/after change in injury crashes different near New Construction than near "
             "Preservation (the placebo)?", rq5["estimate_pct"], rq5["low_pct"], rq5["high_pct"],
             rq5["n_treated"], rq5_sentence, "% difference between the two after/before ratios")
preserved = type_contrast.loc[PRESERVATION]
new_built = type_contrast.loc[NEW_CONSTRUCTION]
say(f"{rq5_sentence} Taken separately, the change relative to comparison sites is "
    f"{new_built['estimate_pct']:+.1f}% (interval {new_built['low_pct']:+.1f}% to {new_built['high_pct']:+.1f}%, "
    f"n = {int(new_built['n_treated'])} sites) for new construction and {preserved['estimate_pct']:+.1f}% "
    f"(interval {preserved['low_pct']:+.1f}% to {preserved['high_pct']:+.1f}%, n = {int(preserved['n_treated'])} "
    f"sites) for preservation, both against {int(rq5['n_comparison']):,} comparison rows and with "
    f"{int(rq5['n_clusters']):,} clusters resampled. Of the preservation sites, "
    f"{preservation_cohort_share:.0f}% belong to the {preserved_by_cohort.idxmax()} cohort.")
say(f"The new-construction estimate is the one of RQ3, whose interval was {new_build_did['percent_low']:+.1f}% to "
    f"{new_build_did['percent_high']:+.1f}% with {new_build_did['n_clusters']:,} clusters. The interval of the "
    f"same estimate inside this joint bootstrap is slightly different because the {int(rq5['n_clusters']):,} "
    f"clusters drawn here include those that contain only preservation sites.")

# %% [markdown]
# New construction and preservation cannot be told apart. The estimate for new construction is
# below zero and the one for preservation above it, but each interval includes zero and so does
# the interval of the contrast between them. The preservation estimate is wide (its interval is
# printed above) and most of its sites belong to one cohort, so it is too imprecise to confirm
# that a completion without new residents leaves crashes unchanged. New construction does not
# differ from it by more than the noise.
#
# The placebo has its own weaknesses. For many preservation projects the recorded completion
# date equals the start date (Section 4.4), so the line between before and after is an
# administrative date and not the end of any work on the street. Most preservation sites in the
# sample come from one cohort, so their estimate depends heavily on the calendar years of that
# cohort. Preserved buildings are also older buildings on established streets, which differ
# from the streets where new buildings go up, and the two estimates share one pool of
# comparison sites. The contrast is the least precise number in this group of questions.
