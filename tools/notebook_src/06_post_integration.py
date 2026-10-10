# %% [markdown]
# ## 6. Post-integration cleaning
#
# Section 4 cleaned each table by itself. Some problems only exist once the two tables meet, and
# the naive panel of 5.2 showed them: cells that became missing because one side had no row, a
# housing ZIP the crash table never uses, and a large group of crashes absent because they carry
# no ZIP. This section repairs what can be repaired, counts what cannot, and fixes the form of the
# integrated tables. It follows the same policy as Section 4 and records each rule in the same
# log with `log_rule`.
#
# | Problem created by integration | Where it is handled |
# |---|---|
# | New missing values from the joins | 6.1 |
# | Inconsistent or redundant columns | 6.2 |
# | Data type mismatches between the tables | 6.3 |
# | Conservation of rows and totals | 6.4 |
# | A citywide reference series | 6.5 |
# | The rules logged and the final tables | 6.6 |

# %%
# --- Constants for Section 6 ------------------------------------------------------------------
# Side of the grid cells used to recover a ZIP from coordinates, in degrees, tried in this order.
# The fine cell is about one short block; the coarse cell is the fallback when no crash with a
# reported ZIP falls in the fine cell. Both values are justified with a comparison in 6.1.
ZIP_CELL_DEGREES = (0.001, 0.004)
THREE_LEVEL_CASCADE = (0.001, 0.004, 0.016)   # the chosen pair plus a still coarser level
# Cell sizes and cascades compared before settling on the pair above.
ZIP_CELL_ALTERNATIVES = [(0.0005,), (0.001,), (0.002,), (0.004,), ZIP_CELL_DEGREES, THREE_LEVEL_CASCADE]
ZIP_HOLDOUT_SHARE = 0.10       # share of crashes with a known ZIP hidden from the lookup to test it

# A ZIP needs this many crashes over the ten study years (50 a year) to enter the panel. Smaller
# codes are single office towers or post-office codes, not neighbourhoods.
MIN_ZIP_CRASHES = 500

INDEX_BASE = 100               # the index equals this value in INDEX_BASE_YEAR (Section 1)
PER_THOUSAND = 1000            # deaths are expressed per 1,000 injury crashes

LABEL_OFFSET_POINTS = 9        # distance between a marker and its value label in the line charts
ACCURACY_AXIS_MAX = 100.8      # right edge of the accuracy chart: a little past 100 so the labels fit
ACCURACY_TICK_STEP = 2         # percentage points between ticks on that axis

POST_LOG_START = len(CLEANING_LOG)   # rules logged from here on belong to this section

# %% [markdown]
# ### 6.1 New missing values created by the joins
#
# #### 6.1.1 Three kinds of gap
#
# The cell below restates the gaps of the naive panel as counts, so that the same table can be
# shown again after the repair.

# %%
def panel_gap_report(panel, crash_frame, zip_column):
    """Counts that describe how complete a ZIP x year table is against the crash table."""
    measure_columns = [column for column in HOUSING_PANEL_MEASURES + CRASH_PANEL_MEASURES if column in panel]
    n_zips = panel["zip"].nunique()
    crashes_placed = int(panel["crashes"].sum())
    return {
        "rows": len(panel),
        "ZIP codes": n_zips,
        "rows missing from a balanced panel": n_zips * N_STUDY_YEARS - len(panel),
        "missing cells": int(panel[measure_columns].isna().sum().sum()),
        "ZIP-years with crashes but missing housing values": int(panel["units_completed"].isna().sum()),
        "ZIP-years with housing but missing crash values": int(panel["crashes"].isna().sum()),
        "crashes placed in a row": crashes_placed,
        "% of all crashes placed": round(pct(crashes_placed, len(crash_frame)), 1),
        "crashes with no ZIP": int(crash_frame[zip_column].isna().sum()),
    }


gaps_before = panel_gap_report(naive_panel, crashes, "zip")
# dtype=object keeps the counts as whole numbers; a mixed Series would turn them into decimals.
display(pd.Series(gaps_before, name="naive panel (5.2)", dtype=object).to_frame())
naive_no_zip_count = gaps_before["crashes with no ZIP"]
naive_crash_count = len(crashes)
say(f"The naive panel leaves out {pct(naive_no_zip_count, naive_crash_count):.1f}% of all crashes ({naive_no_zip_count:,} of {naive_crash_count:,}) "
    f"because they have no ZIP.")

# %% [markdown]
# Most ZIP-years with crashes have no housing row. No completion in that ZIP and year means zero
# buildings and zero units, so the value is known and is filled with 0. Leaving it missing would
# drop most of the panel from any calculation, and filling with a mean would invent completions.
#
# A housing ZIP with no crash row at all gets no zero, as argued in 5.2. Such a ZIP cannot be
# compared with anything, so it is left out of the panel and its units are counted.
#
# The third case is crashes with no ZIP. The naive panel dropped them without saying so. The
# sentence above gives the size of the loss, and the loss differs from year to year. Most of
# these crashes have coordinates, and a coordinate determines a ZIP, so the ZIP is recovered from
# the location.
#
# #### 6.1.2 Recovering the ZIP of a crash from its coordinates
#
# No boundary file is used in this notebook, but the crash table can teach the boundaries by
# itself, because most crashes report both a ZIP and coordinates. The city is divided
# into small square cells, and each cell is given the ZIP reported most often by the crashes
# inside it. A crash without a ZIP then takes the ZIP of the cell it falls in. If its fine cell
# contains no crash with a known ZIP, a coarser cell around it is tried. If that is empty as
# well, the ZIP stays missing.
#
# The method is deterministic and its accuracy can be measured: a random tenth of the crashes
# that do have a ZIP is hidden, the lookup is learned from the other nine tenths, and the
# prediction for the hidden rows is compared with what was reported.
#
# One alternative is to assign the ZIP of the single nearest crash. It would look better on a
# random holdout, because crashes are recorded at intersections and a hidden crash almost always
# has a twin at exactly the same point, and one mislabelled crash would decide the ZIP of
# everything around it. A cell vote uses all the crashes in the cell. Another alternative was to
# leave the ZIP missing and work only at the spatial level. That was rejected because half of the
# research questions are about neighbourhoods.

# %%
def grid_cells(lat, lon, cell_degrees):
    """Integer row and column of the square cell of side `cell_degrees` that each point falls in."""
    cell_row = np.floor(np.asarray(lat, dtype="float64") / cell_degrees).astype("int64")
    cell_column = np.floor(np.asarray(lon, dtype="float64") / cell_degrees).astype("int64")
    return cell_row, cell_column


def learn_cell_zips(lat, lon, zip_codes, cell_degrees):
    """Most frequent ZIP of every grid cell: one row per cell with its ZIP and the votes behind it.

    Ties are broken by the lower ZIP code so that the lookup does not depend on row order.
    """
    cell_row, cell_column = grid_cells(lat, lon, cell_degrees)
    votes = (pd.DataFrame({"cell_row": cell_row, "cell_column": cell_column,
                           "zip": np.asarray(zip_codes, dtype=object)})
             .groupby(["cell_row", "cell_column", "zip"]).size().rename("votes").reset_index())
    votes = votes.sort_values(["cell_row", "cell_column", "votes", "zip"], ascending=[True, True, False, True])
    return votes.drop_duplicates(["cell_row", "cell_column"]).reset_index(drop=True)


def learn_zip_lookups(lat, lon, zip_codes, cell_sizes=ZIP_CELL_DEGREES):
    """One cell-to-ZIP lookup per cell size, finest first."""
    return [(cell_degrees, learn_cell_zips(lat, lon, zip_codes, cell_degrees)) for cell_degrees in cell_sizes]


def zip_from_cells(lat, lon, lookups):
    """ZIP for each point from the first lookup that covers it.

    Returns the ZIP (None where no lookup has the cell) and the level that supplied it
    (1 = finest cell, 2 = next, 0 = none).
    """
    recovered = np.full(len(lat), None, dtype=object)
    level = np.zeros(len(lat), dtype="int8")
    for level_number, (cell_degrees, cell_zips) in enumerate(lookups, start=1):
        cell_row, cell_column = grid_cells(lat, lon, cell_degrees)
        # A left merge keeps the points in their original order, so the result lines up by position.
        matched = pd.DataFrame({"cell_row": cell_row, "cell_column": cell_column}).merge(
            cell_zips[["cell_row", "cell_column", "zip"]], on=["cell_row", "cell_column"], how="left")["zip"]
        fill = (level == 0) & matched.notna().to_numpy()
        recovered[fill] = matched.to_numpy(dtype=object)[fill]
        level[fill] = level_number
    return recovered, level


def predict_hidden_zips(train, test, cell_sizes):
    """Learn the cell lookups from `train` and predict the ZIP of every `test` row.

    Returns the lookups, the predicted ZIPs (None where no cell covers a point) and the level
    that supplied each one (0 = none). The caller scores them against the reported ZIPs.
    """
    lookups = learn_zip_lookups(train["lat"].to_numpy(), train["lon"].to_numpy(), train["zip"], cell_sizes)
    predicted, level = zip_from_cells(test["lat"].to_numpy(), test["lon"].to_numpy(), lookups)
    return lookups, predicted, level


def cascade_label(cell_sizes):
    """Row label of a cell-size cascade in the comparison table, for example '0.001 then 0.004'."""
    return " then ".join(f"{size:g}" for size in cell_sizes)


# The recovery overwrites `zip`, so a second run would treat the recovered values as reported ones and
# find nothing left to recover. Stop with a clear message instead.
assert "zip_imputed" not in crashes.columns, "6.1 has already run on this `crashes`; restart the kernel and run all"
labelled = crashes.loc[crashes["is_located"] & crashes["zip"].notna(), ["lat", "lon", "zip", "borough"]]
zip_targets = crashes["is_located"].to_numpy() & crashes["zip"].isna().to_numpy()
target_lat = crashes["lat"].to_numpy()[zip_targets]
target_lon = crashes["lon"].to_numpy()[zip_targets]

holdout_rng = np.random.default_rng(SEED)
is_hidden = holdout_rng.random(len(labelled)) < ZIP_HOLDOUT_SHARE
train_rows, hidden_rows = labelled[~is_hidden], labelled[is_hidden]
hidden_truth = hidden_rows["zip"].to_numpy(dtype=object)

cell_trial_rows = []
for cell_sizes in ZIP_CELL_ALTERNATIVES:
    trial_lookups, trial_predicted, trial_level = predict_hidden_zips(train_rows, hidden_rows, cell_sizes)
    _, target_level = zip_from_cells(target_lat, target_lon, trial_lookups)
    covered = trial_level > 0
    cell_trial_rows.append({
        "cell sizes (degrees)": cascade_label(cell_sizes),
        "hidden rows covered (%)": 100 * covered.mean(),
        "accuracy where covered (%)": 100 * (trial_predicted[covered] == hidden_truth[covered]).mean(),
        "accuracy of the last level alone (%)": 100 * (
            trial_predicted[trial_level == len(cell_sizes)] == hidden_truth[trial_level == len(cell_sizes)]).mean(),
        "crashes without ZIP covered (%)": 100 * (target_level > 0).mean(),
    })
cell_comparison = pd.DataFrame(cell_trial_rows).set_index("cell sizes (degrees)")
display(cell_comparison.round(1))

# The commentary below reads rows of this table by name, so the rows it needs must exist.
fine_cell_label = cascade_label(ZIP_CELL_DEGREES[:1])
chosen_label = cascade_label(ZIP_CELL_DEGREES)
three_level_label = cascade_label(THREE_LEVEL_CASCADE)
assert {fine_cell_label, chosen_label, three_level_label} <= set(cell_comparison.index)

fine_cell_m = (ZIP_CELL_DEGREES[0] * METRES_PER_DEGREE,
               ZIP_CELL_DEGREES[0] * METRES_PER_DEGREE * math.cos(math.radians(MID_LATITUDE_DEG)))
say(f"A cell of {ZIP_CELL_DEGREES[0]:g} degrees measures about {fine_cell_m[0]:.0f} m north to south and "
    f"{fine_cell_m[1]:.0f} m east to west. The table shows the trade-off: the smaller the cell, the more "
    f"often its vote is right and the fewer crashes it covers. The cascade '{chosen_label}' keeps the accuracy "
    f"of the fine cell ({cell_comparison.loc[chosen_label, 'accuracy where covered (%)']:.1f}%) and covers "
    f"{cell_comparison.loc[chosen_label, 'crashes without ZIP covered (%)']:.1f}% of the crashes that need a "
    f"ZIP, against {cell_comparison.loc[fine_cell_label, 'crashes without ZIP covered (%)']:.1f}% "
    f"for the fine cell alone. A third, still coarser level would cover a little more, but its own votes are "
    f"right only {cell_comparison.loc[three_level_label, 'accuracy of the last level alone (%)']:.0f}% of the "
    f"time, so it is not used.")

# %% [markdown]
# The chosen cascade is now examined on the hidden rows in more detail: overall, by the level
# that supplied the answer, and by borough.

# %%
validation_lookups, hidden_predicted, hidden_level = predict_hidden_zips(train_rows, hidden_rows, ZIP_CELL_DEGREES)
hidden_result = pd.DataFrame({
    "borough": hidden_rows["borough"].to_numpy(),
    "level": hidden_level,
    "covered": hidden_level > 0,
    "correct": hidden_predicted == hidden_truth,
})
hidden_covered = hidden_result[hidden_result["covered"]]
holdout_accuracy = 100 * hidden_covered["correct"].mean()
accuracy_by_level = hidden_covered.groupby("level")["correct"].agg(["size", "mean"])
accuracy_by_borough = (hidden_covered.groupby("borough", observed=True)["correct"].agg(["size", "mean"])
                       .rename(columns={"size": "hidden rows", "mean": "accuracy"}))
accuracy_by_borough["accuracy"] *= 100

def plot_accuracy_by_borough(accuracy_by_borough, overall_accuracy):
    """Lollipop chart of the hold-out accuracy per borough against the overall accuracy."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    borough_positions = np.arange(len(accuracy_by_borough))
    ax.hlines(borough_positions, overall_accuracy, accuracy_by_borough["accuracy"], color=PALETTE["before"],
              linewidth=2, zorder=1)
    ax.scatter(accuracy_by_borough["accuracy"], borough_positions, s=90, zorder=2,
               color=[BOROUGH_COLORS[name] for name in accuracy_by_borough.index])
    for position, (borough_name, row) in enumerate(accuracy_by_borough.iterrows()):
        label_side = 1 if row["accuracy"] >= overall_accuracy else -1
        ax.annotate(f"{row['accuracy']:.1f}%", (row["accuracy"], position),
                    xytext=(10 * label_side, 0), textcoords="offset points", va="center",
                    ha="left" if label_side == 1 else "right", fontsize=9.5)
    ax.axvline(overall_accuracy, color=PALETTE["neutral"], linestyle=":", linewidth=1.2)
    ax.annotate(f"All boroughs: {overall_accuracy:.1f}%", (overall_accuracy, -0.75), xytext=(5, 0),
                textcoords="offset points", fontsize=9, color=PALETTE["neutral"], va="center")
    ax.set_yticks(borough_positions)
    ax.set_yticklabels([f"{name} (n = {int(row['hidden rows']):,})" for name, row in accuracy_by_borough.iterrows()])
    ax.set_ylim(len(accuracy_by_borough) - 0.5, -1.1)
    ax.set_xlim(accuracy_by_borough["accuracy"].min() - 3, ACCURACY_AXIS_MAX)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(ACCURACY_TICK_STEP))
    ax.grid(axis="y", visible=False)
    finish(ax, "Recovered ZIP against reported ZIP on the hidden crashes, by borough",
           "Hidden crashes whose recovered ZIP equals the reported ZIP (%); axis does not start at zero", "")
    plt.tight_layout()
    plt.show()


plot_accuracy_by_borough(accuracy_by_borough, holdout_accuracy)

# %%
weakest_borough = accuracy_by_borough["accuracy"].idxmin()
say(f"{int(is_hidden.sum()):,} crashes with a known ZIP were hidden. The lookup learned from the other "
    f"{int((~is_hidden).sum()):,} covers {100 * hidden_result['covered'].mean():.2f}% of them and gives the "
    f"reported ZIP for {holdout_accuracy:.1f}% of those. The fine cell answers "
    f"{pct(accuracy_by_level.loc[1, 'size'], len(hidden_covered)):.1f}% of the covered rows with "
    f"{100 * accuracy_by_level.loc[1, 'mean']:.1f}% accuracy, and the coarse cell the rest with "
    f"{100 * accuracy_by_level.loc[2, 'mean']:.1f}%. By borough the accuracy runs from "
    f"{accuracy_by_borough['accuracy'].min():.1f}% in {weakest_borough} to "
    f"{accuracy_by_borough['accuracy'].max():.1f}% in {accuracy_by_borough['accuracy'].idxmax()}.")

# %% [markdown]
# The errors that remain are mostly not errors of location. A crash at an intersection on a ZIP
# boundary can be reported with either code, so two crashes at the same corner can disagree and
# no rule based on position can match both. That also explains the borough ranking: accuracy is
# lowest where ZIP areas are smallest and boundaries are closest together.
#
# The holdout has one limit that should be stated. The hidden rows are crashes that had a ZIP,
# and the rows that need one may differ from them, for example by lying more often on highways
# and bridges where few crashes report a ZIP. More of them are answered by the coarse cell, whose
# accuracy is lower. The expected accuracy on the rows actually filled is therefore computed
# below from the mix of levels, and it is somewhat below the holdout figure.
#
# The lookup is now learned again from all crashes with a known ZIP, including the hidden ones,
# and applied. The reported value is kept in a temporary column, and `zip_imputed` marks every
# filled row so that any later result can be recomputed without them.

# %%
zip_lookups = learn_zip_lookups(labelled["lat"].to_numpy(), labelled["lon"].to_numpy(), labelled["zip"])
recovered_zip, recovered_level = zip_from_cells(target_lat, target_lon, zip_lookups)

zip_missing_before = crashes["zip"].isna().to_numpy()
missing_by_year_before = 100 * pd.Series(zip_missing_before).groupby(crashes["year"].to_numpy()).mean()

crashes["zip_reported"] = crashes["zip"]
filled_zip = crashes["zip"].to_numpy(dtype=object)
filled_zip[np.flatnonzero(zip_targets)] = recovered_zip
filled_zip = pd.Series(filled_zip, index=crashes.index)
crashes["zip"] = filled_zip.where(filled_zip.notna(), pd.NA).astype(crashes["zip_reported"].dtype)
crashes["zip_imputed"] = crashes["zip_reported"].isna() & crashes["zip"].notna()

zip_missing_after = crashes["zip"].isna().to_numpy()
missing_by_year_after = 100 * pd.Series(zip_missing_after).groupby(crashes["year"].to_numpy()).mean()
n_imputed = int(crashes["zip_imputed"].sum())
n_no_cell = int((zip_targets & zip_missing_after).sum())
n_no_location = int((~crashes["is_located"].to_numpy() & zip_missing_after).sum())
# Share of the crashes that needed a ZIP, by the level that supplied it (1 = fine cell, 2 = coarse cell).
level_counts = np.bincount(recovered_level, minlength=len(ZIP_CELL_DEGREES) + 1)
level_share_on_targets = {level: level_counts[level] / len(recovered_level)
                          for level in range(1, len(ZIP_CELL_DEGREES) + 1)}
expected_accuracy = 100 * sum(level_share_on_targets[level] * accuracy_by_level.loc[level, "mean"]
                              for level in accuracy_by_level.index) / sum(level_share_on_targets.values())
# The commentary below names the fine and the coarse level, so a third level would need new wording.
assert set(level_share_on_targets) == {1, 2}, "the commentary describes exactly two recovery levels"

# Reported ZIPs are untouched, every filled value is a ZIP the crash table already uses, and the
# flag marks exactly the rows that changed.
assert (crashes.loc[~crashes["zip_imputed"], "zip"].isna()
        == crashes.loc[~crashes["zip_imputed"], "zip_reported"].isna()).all()
assert crashes.loc[crashes["zip_reported"].notna(), "zip"].eq(
    crashes.loc[crashes["zip_reported"].notna(), "zip_reported"]).all()
assert set(crashes.loc[crashes["zip_imputed"], "zip"].unique()) <= set(labelled["zip"].unique())
assert n_imputed + n_no_cell + n_no_location == int(zip_missing_before.sum())
assert crashes["zip"].dtype == housing["zip"].dtype

log_rule("crashes", f"missing zip recovered from coordinates (modal ZIP of a {ZIP_CELL_DEGREES[0]:g} degree "
         f"cell, else of a {ZIP_CELL_DEGREES[1]:g} degree cell); flagged zip_imputed", "impute",
         len(crashes), len(crashes), n_imputed,
         f"A coordinate determines a ZIP; holdout accuracy {holdout_accuracy:.1f}%. Dropping the rows would "
         f"lose {100 * zip_missing_before.mean():.0f}% of the crashes, unevenly by year.")
log_rule("crashes", "zip still missing after recovery (no coordinates, or no labelled crash in either cell)",
         "flag", len(crashes), len(crashes), n_no_cell + n_no_location,
         "No deterministic value exists; kept for city totals and spatial work, excluded from the ZIP panel.")

def plot_missing_zip_by_year(missing_before, missing_after):
    """Share of each year's crashes without a ZIP, before and after the recovery."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    # Labels sit above their markers on a white backing, so a label never covers a marker. The two
    # lines are far apart at every labelled year, so the labels do not meet.
    series_specs = [(missing_before, "ZIP as reported", PALETTE["crash"]),
                    (missing_after, "After recovery from coordinates", PALETTE["after"])]
    for series, label, color in series_specs:
        ax.plot(series.index, series.to_numpy(), color=color, linewidth=2, marker="o", markersize=6, label=label)
        for year in sorted({series.index[0], series.idxmax(), series.index[-1]}):
            ax.annotate(f"{series.loc[year]:.0f}%", (year, series.loc[year]), xytext=(0, LABEL_OFFSET_POINTS),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9,
                        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax.set_xticks(missing_before.index)
    ax.set_ylim(0, missing_before.max() * LINE_HEADROOM)
    ax.legend(loc="upper right")
    finish(ax, "Crashes without a ZIP code, by year, before and after the recovery",
           "Crash year", "Crashes with no ZIP (% of the year's crashes)")
    plt.tight_layout()
    plt.show()


plot_missing_zip_by_year(missing_by_year_before, missing_by_year_after)

# %%
say(f"{n_imputed:,} crashes received a ZIP ({pct(n_imputed, int(zip_missing_before.sum())):.1f}% of those "
    f"that had none): {100 * level_share_on_targets[1]:.1f}% of the candidates from the fine cell and "
    f"{100 * level_share_on_targets[2]:.1f}% from the coarse cell, which puts the expected accuracy of the "
    f"filled values near {expected_accuracy:.1f}%. The share of crashes without a ZIP falls from "
    f"{100 * zip_missing_before.mean():.1f}% to {100 * zip_missing_after.mean():.1f}%. Before the recovery it "
    f"ranged from {missing_by_year_before.min():.1f}% ({missing_by_year_before.idxmin()}) to "
    f"{missing_by_year_before.max():.1f}% ({missing_by_year_before.idxmax()}), a spread of "
    f"{missing_by_year_before.max() - missing_by_year_before.min():.1f} points; afterwards from "
    f"{missing_by_year_after.min():.1f}% ({missing_by_year_after.idxmin()}) to "
    f"{missing_by_year_after.max():.1f}% ({missing_by_year_after.idxmax()}), a spread of "
    f"{missing_by_year_after.max() - missing_by_year_after.min():.1f} points. {n_no_location:,} crashes have "
    f"neither a ZIP nor coordinates and {n_no_cell:,} have coordinates in a cell where no crash reports a "
    f"ZIP; these {n_no_location + n_no_cell:,} crashes ({pct(n_no_location + n_no_cell, len(crashes)):.1f}%) "
    f"stay without a ZIP and are excluded from the ZIP panel.")

# %% [markdown]
# The upper line is the artificial trend that the naive panel would have carried: the share of
# crashes missing from it moved by many points between years for reasons that have nothing to do
# with safety, so ZIP counts built on reported ZIPs alone would rise and fall with record
# keeping. After the recovery the gap is much smaller in every year. It is not flat, though. What
# remains comes mainly from crashes that have no coordinates at all, and that share was higher in
# the first study year than later. A ZIP-level comparison that includes the first year should be
# read with that in mind, and the panel carries the number of imputed crashes per row so that a
# result can be checked against reported ZIPs only.
#
# #### 6.1.3 The final ZIP by year panel
#
# The panel is rebuilt with the two aggregation functions of 5.2, now on the recovered ZIPs. Three
# choices make it complete. Its ZIP codes are those with at least `MIN_ZIP_CRASHES` crashes over
# the study years; the codes below that are single buildings and post-office codes, and a housing
# ZIP that the crash table does not use falls below it too. Every such ZIP is crossed with every
# study year, so the panel is balanced. Cells without a source row are zeros on both sides: no
# completion that year, or no crash that year in a ZIP that otherwise has hundreds.

# %%
def build_zip_year_panel(housing_aggregate, crash_aggregate, panel_zips):
    """Balanced ZIP x study-year table: every ZIP in `panel_zips` in every study year, zeros filled in.

    Both aggregates are unique on (zip, year), so aligning them on the full grid is one row to
    one row. The key columns are converted to plain text and integers first, because the two
    aggregates arrive with different key dtypes (6.3).
    """
    grid = pd.MultiIndex.from_product([sorted(panel_zips), STUDY_YEARS], names=["zip", "year"])

    def on_grid(aggregate, measures):
        keyed = aggregate.assign(zip=aggregate["zip"].astype(object), year=aggregate["year"].astype("int64"))
        return keyed.set_index(["zip", "year"])[measures].reindex(grid, fill_value=0)

    return pd.concat([on_grid(housing_aggregate, HOUSING_PANEL_MEASURES),
                      on_grid(crash_aggregate, CRASH_PANEL_MEASURES)], axis=1).reset_index()


crashes_by_zip_year = aggregate_crashes_to_zip_year(crashes)
crashes_per_zip = crashes_by_zip_year.groupby("zip")["crashes"].sum()
panel_zips = sorted(crashes_per_zip.index[crashes_per_zip >= MIN_ZIP_CRASHES])
small_zips = crashes_per_zip.index[crashes_per_zip < MIN_ZIP_CRASHES]

zip_year = build_zip_year_panel(housing_by_zip_year, crashes_by_zip_year, panel_zips)

# How many of each row's crashes rest on a recovered ZIP (for sensitivity checks on reported ZIPs).
imputed_counts = (crashes[crashes["zip_imputed"]].groupby(["zip", "year"])
                  .agg(crashes_zip_imputed=("collision_id", "size"),
                       injury_crashes_zip_imputed=("is_injury_crash", "sum")).reset_index())
imputed_counts = imputed_counts.assign(zip=imputed_counts["zip"].astype(object),
                                       year=imputed_counts["year"].astype("int64"))
zip_year = zip_year.merge(imputed_counts, on=["zip", "year"], how="left", validate="one_to_one")
zip_year[["crashes_zip_imputed", "injury_crashes_zip_imputed"]] = (
    zip_year[["crashes_zip_imputed", "injury_crashes_zip_imputed"]].fillna(0))

crashes["in_zip_panel"] = crashes["zip"].isin(panel_zips).to_numpy()

structural_zero_rows = int((zip_year["buildings_completed"] == 0).sum())
housing_zips_all = set(housing.loc[housing["zip"].notna(), "zip"])
housing_zips_excluded = sorted(housing_zips_all - set(panel_zips))
completed_in_window = housing[housing["is_completed"] & housing["zip"].notna()
                              & housing["completion_year"].between(YEAR_START, YEAR_END)]
excluded_housing = completed_in_window[completed_in_window["zip"].isin(housing_zips_excluded)]
crashes_in_small_zips = int((crashes["zip"].notna() & ~crashes["in_zip_panel"]).sum())

log_rule("zip panel", "ZIP-years with crashes and no completion: housing measures set to 0", "impute",
         len(zip_year), len(zip_year), structural_zero_rows,
         "Structural zero: no row in the housing aggregate means nothing was completed there that year.")
log_rule("zip panel", f"ZIP codes with fewer than {MIN_ZIP_CRASHES} crashes in {N_STUDY_YEARS} years left out",
         "drop", (len(panel_zips) + len(small_zips)) * N_STUDY_YEARS, len(zip_year), len(small_zips),
         "Single-building and post-office codes, and one housing ZIP the crash table never uses; a zero "
         "crash count there would be false. Rows are ZIP-years, affected counts ZIP codes.")

gaps_after = panel_gap_report(zip_year, crashes, "zip")
gap_comparison = pd.DataFrame({"naive panel (5.2)": gaps_before, "final panel": gaps_after}, dtype=object)
display(gap_comparison)

# %%
say(f"The final panel has {len(zip_year):,} rows: {len(panel_zips)} ZIP codes in each of {N_STUDY_YEARS} "
    f"years. It places {gaps_after['crashes placed in a row']:,} crashes "
    f"({gaps_after['% of all crashes placed']:.1f}% of all), against "
    f"{gaps_before['crashes placed in a row']:,} ({gaps_before['% of all crashes placed']:.1f}%) for the naive "
    f"panel, and has no missing cell. {structural_zero_rows:,} ZIP-years "
    f"({pct(structural_zero_rows, len(zip_year)):.0f}%) had no completion and carry zeros on the housing side. "
    f"{len(small_zips)} low-volume ZIP codes with {crashes_in_small_zips:,} crashes between them were left "
    f"out. On the housing side, {len(housing_zips_excluded)} housing ZIP codes are not in the panel "
    f"({', '.join(housing_zips_excluded)}); they hold {len(excluded_housing)} completed rows and "
    f"{int(excluded_housing['total_units'].sum()):,} units "
    f"({pct(excluded_housing['total_units'].sum(), completed_in_window['total_units'].sum()):.1f}% of the "
    f"units completed in the study years in rows that have a ZIP).")

# %% [markdown]
# The panel now covers nearly all crashes that can be given a ZIP, every ZIP has all ten years,
# and the only housing left out of it sits in codes that the crash table barely uses. Whether a
# value of zero is correct was decided case by case: zero completions is a fact, while zero
# crashes in a ZIP that never appears in the crash table would have been an artefact, which is
# why that ZIP was removed instead of filled.

# %% [markdown]
# ### 6.2 Inconsistent or redundant columns
#
# #### 6.2.1 Borough: one attribute, two sources
#
# Both tables carry a borough. After integration a ZIP therefore has two candidate boroughs, the
# one its crashes report and the one its housing rows report, and a crash with a recovered ZIP
# has a ZIP but no borough at all, because borough and ZIP are missing together in the crash
# table (Section 4.8). A ZIP can also cross a borough line. The next cell measures the
# disagreement and builds a single ZIP-to-borough lookup.

# %%
def modal_borough_by_zip(frame, zip_column):
    """Per ZIP: the most frequent borough, its row count and its share of the ZIP's rows with a borough."""
    counted = (frame[frame[zip_column].notna() & frame["borough"].notna()]
               .groupby([zip_column, "borough"], observed=True).size().rename("rows").reset_index()
               .rename(columns={zip_column: "zip"}))
    counted["zip"] = counted["zip"].astype(object)
    counted["share"] = counted["rows"] / counted.groupby("zip")["rows"].transform("sum")
    counted["borough"] = counted["borough"].astype(object)
    top = counted.sort_values(["zip", "rows", "borough"], ascending=[True, False, True]).drop_duplicates("zip")
    return top.set_index("zip")[["borough", "share"]]


borough_from_crashes = modal_borough_by_zip(crashes, "zip_reported")
borough_from_housing = modal_borough_by_zip(housing, "zip")
zip_geography = borough_from_crashes.rename(columns={"borough": "borough_crashes", "share": "share_crashes"}).join(
    borough_from_housing.rename(columns={"borough": "borough_housing", "share": "share_housing"}), how="outer")

in_both = zip_geography["borough_crashes"].notna() & zip_geography["borough_housing"].notna()
sources_disagree = in_both & (zip_geography["borough_crashes"] != zip_geography["borough_housing"])
split_zips = zip_geography[(zip_geography["share_crashes"] < 1) | (zip_geography["share_housing"] < 1)]
reported_both = crashes["zip_reported"].notna() & crashes["borough"].notna()
crash_zip_borough = crashes.loc[reported_both, "zip_reported"].astype(object).map(zip_geography["borough_crashes"])
crashes_off_modal = int((crash_zip_borough.to_numpy() != crashes.loc[reported_both, "borough"].astype(object).to_numpy()).sum())

# Resolution: the crash table decides (it has thousands of rows per ZIP), housing fills the rest.
zip_geography["borough"] = zip_geography["borough_crashes"].fillna(zip_geography["borough_housing"])
zip_borough = zip_geography["borough"]
assert zip_borough.notna().all() and set(panel_zips) <= set(zip_borough.index)

display(split_zips.sort_values("share_crashes").head(6).round(3))
say(f"{int(in_both.sum())} ZIP codes have a borough from both tables and the two sources name a different "
    f"borough for {int(sources_disagree.sum())} of them. {len(split_zips)} ZIP codes have rows in more than "
    f"one borough in at least one table; the table above lists those with the lowest share in their main "
    f"borough. Across all crashes with a reported ZIP and borough, {crashes_off_modal:,} "
    f"({pct(crashes_off_modal, int(reported_both.sum())):.2f}%) are recorded in a different borough from the "
    f"main borough of their ZIP. {int(zip_geography['borough_crashes'].isna().sum())} ZIP code gets its "
    f"borough from housing because no crash reports it.")

# %% [markdown]
# The two tables agree on the borough of every ZIP they share, and the ZIPs that cross a borough
# line do so for a very small share of their crashes. One borough per ZIP is therefore a sound
# simplification for the panel. The decisions are these.
#
# The panel takes the ZIP's main borough from the lookup. A crash keeps the borough it reported,
# even in the few cases where that differs from the main borough of its ZIP, because the report
# is the observation and the lookup is a summary. A crash with no reported borough and a ZIP,
# reported or recovered, takes the borough of its ZIP and is flagged `borough_imputed`. Giving
# every crash the borough of its ZIP was the alternative; it would have overwritten reported
# values to gain tidiness.
#
# The crashes that reported a borough but had no usable ZIP give a second, independent check of
# the ZIP recovery, this time on rows that really lacked a ZIP: the borough of the recovered ZIP
# should match the borough the officer wrote down.

# %%
borough_missing_before = int(crashes["borough"].isna().sum())
borough_of_zip = crashes["zip"].astype(object).map(zip_borough)

zip_check_rows = (crashes["zip_imputed"] & crashes["borough"].notna()).to_numpy()
recovery_agrees = (borough_of_zip.to_numpy()[zip_check_rows]
                   == crashes["borough"].astype(object).to_numpy()[zip_check_rows])

fill_borough = crashes["borough"].isna().to_numpy() & borough_of_zip.notna().to_numpy()
filled_borough = crashes["borough"].astype(object).to_numpy()
filled_borough[fill_borough] = borough_of_zip.to_numpy()[fill_borough]
crashes["borough"] = pd.Categorical(filled_borough, categories=BOROUGHS)
crashes["borough_imputed"] = fill_borough
borough_missing_after = int(crashes["borough"].isna().sum())

zip_year["borough"] = zip_year["zip"].map(zip_borough)
assert zip_year["borough"].notna().all()
assert zip_year.groupby("zip")["borough"].nunique().eq(1).all(), "one borough per ZIP"
assert crashes["borough"].isna().sum() == (crashes["zip"].isna() & crashes["borough"].isna()).sum()

log_rule("crashes", "missing borough filled from the main borough of the crash's ZIP; flagged borough_imputed",
         "impute", len(crashes), len(crashes), int(fill_borough.sum()),
         "Borough and ZIP are missing together; once the ZIP is recovered the borough follows from it.")
log_rule("zip panel", "two borough sources (crash reports, housing rows) resolved into one borough per ZIP",
         "standardise", len(zip_year), len(zip_year), int(sources_disagree.sum()),
         "Crash-modal borough, housing as fallback. Affected counts ZIPs where the two sources disagreed.")

say(f"Borough was missing in {borough_missing_before:,} crashes before and is missing in "
    f"{borough_missing_after:,} after ({pct(borough_missing_after, len(crashes)):.1f}%); "
    f"{int(fill_borough.sum()):,} values were filled. In the {int(zip_check_rows.sum())} crashes that reported a "
    f"borough and received a recovered ZIP, the recovered ZIP lies in the reported borough "
    f"{100 * recovery_agrees.mean():.1f}% of the time.")

# %% [markdown]
# #### 6.2.2 Duplicate and leftover columns
#
# Integration left several columns that say the same thing twice or that only served a step
# above. Each was reviewed; the table lists what is dropped and what is knowingly kept.

# %%
# Every claim of redundancy is checked before the column goes.
restored = crashes["zip"].where(~crashes["zip_imputed"])
assert (restored.isna() == crashes["zip_reported"].isna()).all()
assert restored.dropna().eq(crashes["zip_reported"].dropna()).all(), "zip_reported = zip where not imputed"
assert (sites["is_completed"] == sites["completion_year"].notna()).all()
assert (crashes["is_located"] == crashes["lat"].notna()).all()
assert (zip_year["new_construction_units"] + zip_year["preservation_units"] == zip_year["units_completed"]).all()

column_review = pd.DataFrame([
    ("crashes", "zip_reported", "drop", "Equal to zip wherever zip_imputed is False, and missing elsewhere."),
    ("zip_geography", "borough_crashes, borough_housing, share_crashes, share_housing", "drop",
     "Resolved into one borough per ZIP (zip_borough)."),
    ("naive_panel", "whole table, with its match indicator", "drop", "Replaced by zip_year."),
    ("crashes_by_zip_year_reported", "whole table", "drop", "Crash counts on reported ZIPs only; replaced."),
    ("crashes", "is_located", "keep", "Same as lat not missing, but used as a filter in every spatial step."),
    ("crashes", "near_cluster_id", "keep", "Derivable from near_site_id; saves a join in every bootstrap."),
    ("sites", "is_completed", "keep", "Same as completion_year not missing; kept for readable filters."),
    ("sites / zip_year", "borough and zip in both", "keep",
     "Not duplicates: one is the building's own borough and ZIP, the other the panel key."),
    ("zip_year", "new_construction_units, preservation_units", "keep",
     "They add up to units_completed, but each is a variable of its own in Section 8."),
], columns=["table", "column", "decision", "reason"])
with pd.option_context("display.max_colwidth", None):
    display(column_review)

crash_columns_before = crashes.shape[1]
crashes = crashes.drop(columns=["zip_reported"])
log_rule("crashes", "zip_reported dropped after the recovery was verified", "drop", len(crashes), len(crashes),
         crash_columns_before - crashes.shape[1],
         "Redundant with zip and zip_imputed. Affected counts columns, not rows.")
release("naive_panel", "crashes_by_zip_year_reported", "restored", "zip_geography")

# %% [markdown]
# #### 6.2.3 Cumulative units
#
# The panel records what was completed in each year. Several questions need the stock instead:
# how many subsidised units stand in the ZIP by a given year. Summing by hand over the unbalanced
# housing aggregate would be inconsistent in two ways. Years without a completion have no row
# there, so a running sum would skip them, and completions before the first study year would be
# lost although those buildings are standing throughout the study.
#
# The cumulative columns are therefore computed on a complete grid that starts at the first
# completion year in the housing table and ends at the last study year, and only then cut to the
# study years. The stock is counted from the start of the housing table. Subsidised housing built
# before that is not in the data, so `cum_units` is the stock added since then and not the total
# stock. Completions after the last study year are not counted.

# %%
CUMULATIVE_COLUMNS = {"cum_units": "units_completed", "cum_affordable_units": "affordable_units_completed",
                      "cum_deep_affordable_units": "deep_affordable_units",
                      "cum_new_construction_units": "new_construction_units",
                      "cum_preservation_units": "preservation_units"}


def cumulative_housing(housing_aggregate, panel_zips):
    """Running totals of the unit measures per ZIP, from the first year in the housing table."""
    first_year = int(housing_aggregate["year"].min())
    history = pd.MultiIndex.from_product([sorted(panel_zips), range(first_year, YEAR_END + 1)],
                                         names=["zip", "year"])
    keyed = housing_aggregate.assign(zip=housing_aggregate["zip"].astype(object),
                                     year=housing_aggregate["year"].astype("int64"))
    yearly = keyed.set_index(["zip", "year"])[list(CUMULATIVE_COLUMNS.values())].reindex(history, fill_value=0)
    running = yearly.groupby(level="zip").cumsum()
    running.columns = list(CUMULATIVE_COLUMNS)
    return running.reset_index()


housing_first_year = int(housing_by_zip_year["year"].min())
cumulative = cumulative_housing(housing_by_zip_year, panel_zips)
zip_year = zip_year.merge(cumulative, on=["zip", "year"], how="left", validate="one_to_one")

# Consistency between the yearly and the cumulative columns.
previous_stock = zip_year.groupby("zip")["cum_units"].shift(1)
after_first_year = zip_year["year"] > YEAR_START
assert (zip_year.loc[after_first_year, "cum_units"] - previous_stock[after_first_year]
        == zip_year.loc[after_first_year, "units_completed"]).all(), "stock change = completions of the year"
assert (zip_year["cum_units"] >= zip_year["units_completed"]).all()
assert (zip_year["cum_new_construction_units"] + zip_year["cum_preservation_units"] == zip_year["cum_units"]).all()
in_panel_history = housing_by_zip_year[housing_by_zip_year["zip"].isin(panel_zips)
                                       & (housing_by_zip_year["year"] <= YEAR_END)]
assert zip_year.loc[zip_year["year"] == YEAR_END, "cum_units"].sum() == in_panel_history["units_completed"].sum()

stock_before_study = int(zip_year.loc[zip_year["year"] == YEAR_START, "cum_units"].sum()
                         - zip_year.loc[zip_year["year"] == YEAR_START, "units_completed"].sum())
stock_end = int(zip_year.loc[zip_year["year"] == YEAR_END, "cum_units"].sum())
log_rule("zip panel", "cumulative unit columns computed on a complete grid from the first housing year",
         "standardise", len(zip_year), len(zip_year), len(CUMULATIVE_COLUMNS),
         "A running sum over the unbalanced aggregate would skip empty years and lose earlier completions. "
         "Affected counts columns.")
say(f"The housing table starts in {housing_first_year}. The panel's ZIP codes enter {YEAR_START} with "
    f"{stock_before_study:,} units completed in {housing_first_year} to {YEAR_START - 1} and end {YEAR_END} "
    f"with a stock of {stock_end:,} units. In every ZIP and year the change in `cum_units` equals "
    f"`units_completed`, and the new-construction and preservation stocks add up to the total.")

# %% [markdown]
# ### 6.3 Data type mismatches
#
# A join compares key values exactly, so two keys that look the same and are stored differently
# do not match. Three mismatches arose, or would have arisen, between these tables.
#
# The first is the ZIP code. Section 2 loaded every column as text for this reason. With a
# default load, pandas reads the housing postcode as a decimal number, because the column has
# blanks, while crash ZIP codes are text. The cell below reads the housing file once more with
# default types to show what a join on those keys would have found.
#
# The second is the year. The housing completion year is a nullable integer, since unfinished
# buildings have none, and the crash year is a plain integer. The panel builder converts both.
#
# The third is what merges and grids do to counts and labels: an outer merge turns integer
# counts into decimals as soon as one cell is missing, a grid built from Python values comes back
# as generic objects and 64-bit integers, and a category is lost when a label passes through a
# lookup. The final tables are cast to one declared schema and the schema is asserted.

# %%
default_housing_keys = pd.read_csv(HOUSING_FILE, usecols=["postcode"])
default_zip_text = set(default_housing_keys["postcode"].dropna().astype(str))
crash_zip_text = set(crashes["zip"].dropna().astype(object))
clean_housing_zip_text = set(housing["zip"].dropna().astype(object))
key_match = pd.DataFrame({
    "dtype of the housing key": [str(default_housing_keys["postcode"].dtype), str(housing["zip"].dtype)],
    "example value": [sorted(default_zip_text)[0], sorted(clean_housing_zip_text)[0]],
    "distinct housing ZIPs": [len(default_zip_text), len(clean_housing_zip_text)],
    "found among crash ZIPs": [len(default_zip_text & crash_zip_text), len(clean_housing_zip_text & crash_zip_text)],
}, index=["default load (number)", "as cleaned (text)"])
display(key_match)

year_types = pd.DataFrame({
    "dtype": [str(housing["completion_year"].dtype), str(housing_by_zip_year["year"].dtype),
              str(crashes["year"].dtype), str(crashes_by_zip_year["year"].dtype)],
    "can hold a missing value": [True, True, False, False],
}, index=["housing.completion_year", "housing aggregate: year", "crashes.year", "crash aggregate: year"])
display(year_types)

# %%
COUNT_DTYPE = "int32"
ZIP_DTYPE = housing["zip"].dtype     # the text dtype both cleaned tables use for ZIP codes
CATEGORY_LEVELS = {"borough": BOROUGHS}   # levels for a label column that arrives as plain text
ZIP_YEAR_SCHEMA = {
    "zip": ZIP_DTYPE, "year": "int16", "borough": "category",
    **{column: COUNT_DTYPE for column in HOUSING_PANEL_MEASURES},
    **{column: COUNT_DTYPE for column in CUMULATIVE_COLUMNS},
    **{column: COUNT_DTYPE for column in CRASH_PANEL_MEASURES},
    "crashes_zip_imputed": COUNT_DTYPE, "injury_crashes_zip_imputed": COUNT_DTYPE,
}
SITES_SCHEMA = {
    "site_id": "int32", "cluster_id": "int32", "block_id": "int32", "housing_id": "int32",
    "project_id": "int64", "building_id": "int64", "project_name": housing["project_name"].dtype,
    "borough": "category", "zip": ZIP_DTYPE, "lat": "float64", "lon": "float64",
    "start_year": "int16", "completion_year": "Int16", "is_completed": "bool", "same_day_completion": "bool",
    "completion_before_start": "bool", "construction_type": "category", "total_units": "int16",
    "all_counted_units": "int16", "deep_affordable_units": "int16", "deep_affordable_share": "float64",
    "size_band": "category", "is_size_outlier": "bool", "cluster_size": "int32",
    "n_sites_within_buffer": "int32",
}
SITE_YEAR_SCHEMA = {"site_id": "int32", "year": "int16", **{column: COUNT_DTYPE for column in SITE_COUNT_COLUMNS}}
EVENT_PANEL_SCHEMA = {
    "site_id": "int32", "cluster_id": "int32", "block_id": "int32", "cohort": "int16", "group": "category",
    "rel_year": "int8", "year": "int16", **{column: COUNT_DTYPE for column in SITE_COUNT_COLUMNS},
    "contaminated": "bool", "near_completion_in_window": "bool", "near_earlier_completion": "bool",
    "completion_year": "Int16", "construction_type": "category", "borough": "category",
    "size_band": "category", "total_units": "int16",
}
CRASH_NEW_COLUMNS_SCHEMA = {
    "zip": ZIP_DTYPE, "zip_imputed": "bool", "borough": "category", "borough_imputed": "bool",
    "in_zip_panel": "bool", "housing_relation": "category", "near_site_id": "Int32", "near_cluster_id": "Int32",
    "near_site_distance_m": "float32", "near_completion_year": "Int16", "rel_year": "Int16",
    "n_sites_near": "int16", "near_completed_site": "bool",
}


def cast_to_schema(frame, schema):
    """Return the frame with exactly the schema's columns, in its order and with its dtypes.

    Count columns must already hold whole numbers and no missing value, so a cast that would
    silently truncate or fail on a gap is caught by the checks instead.
    """
    assert set(schema) == set(frame.columns), set(schema) ^ set(frame.columns)
    result = frame[list(schema)].copy()
    for column, dtype in schema.items():
        if dtype == "category":
            if result[column].dtype != "category":
                result[column] = pd.Categorical(result[column], categories=CATEGORY_LEVELS[column])
                assert result[column].notna().all(), f"{column}: label outside the declared levels"
            continue
        if dtype == COUNT_DTYPE:
            assert result[column].notna().all(), f"{column}: missing value in a count column"
            assert (result[column] == np.floor(result[column].astype("float64"))).all(), f"{column}: not whole"
        result[column] = result[column].astype(dtype)
    return result


def dtype_changes(before, after):
    """Side-by-side dtypes of the columns whose dtype changed."""
    changed = [column for column in after.columns if before[column].dtype != after[column].dtype]
    return pd.DataFrame({"before": [repr(before[column].dtype) for column in changed],
                         "after": [repr(after[column].dtype) for column in changed]}, index=changed)


# sites: building_id is nullable in housing because confidential rows have none; every site has one.
assert sites["building_id"].notna().all()
typed = {"zip_year": cast_to_schema(zip_year, ZIP_YEAR_SCHEMA), "sites": cast_to_schema(sites, SITES_SCHEMA),
         "site_year": cast_to_schema(site_year, SITE_YEAR_SCHEMA),
         "event_panel": cast_to_schema(event_panel, EVENT_PANEL_SCHEMA)}
untyped = {"zip_year": zip_year, "sites": sites, "site_year": site_year, "event_panel": event_panel}
all_changes = pd.concat({name: dtype_changes(untyped[name], typed[name]) for name in typed})
n_dtype_changes = len(all_changes)
display(all_changes.rename_axis(["table", "column"]))

zip_year, sites, site_year, event_panel = (typed["zip_year"], typed["sites"], typed["site_year"],
                                           typed["event_panel"])
release("typed", "untyped")

# The final schema, asserted.
for name, frame, schema in [("zip_year", zip_year, ZIP_YEAR_SCHEMA), ("sites", sites, SITES_SCHEMA),
                            ("site_year", site_year, SITE_YEAR_SCHEMA),
                            ("event_panel", event_panel, EVENT_PANEL_SCHEMA)]:
    assert list(frame.columns) == list(schema), name
    for column, dtype in schema.items():
        assert frame[column].dtype == dtype, (name, column, str(frame[column].dtype))
for column, dtype in CRASH_NEW_COLUMNS_SCHEMA.items():
    assert crashes[column].dtype == dtype, (column, str(crashes[column].dtype))

# Join keys have one dtype everywhere, and category columns share their categories.
assert zip_year["zip"].dtype == crashes["zip"].dtype == housing["zip"].dtype == sites["zip"].dtype
assert zip_year["year"].dtype == crashes["year"].dtype == site_year["year"].dtype == event_panel["year"].dtype
assert sites["site_id"].dtype == site_year["site_id"].dtype == event_panel["site_id"].dtype
for frame in (zip_year, sites, event_panel, crashes, housing):
    assert list(frame["borough"].cat.categories) == BOROUGHS
assert list(event_panel["group"].cat.categories) == EVENT_GROUPS
assert list(crashes["housing_relation"].cat.categories) == HOUSING_RELATIONS
# The keys really join: every tagged crash finds its site, every panel ZIP its crashes.
assert crashes["near_site_id"].dropna().isin(sites["site_id"]).all()
assert set(zip_year["zip"]) <= set(crashes["zip"].dropna())

log_rule("zip panel", "integrated tables cast to a declared schema (keys, counts, categories)", "standardise",
         len(zip_year), len(zip_year), n_dtype_changes,
         "Merges and grids returned object keys, 64-bit and decimal counts and plain-text labels. "
         "Affected counts columns across the four tables.")
# The sentence below says that none of the default-load values matches; this is what makes it true.
assert key_match.loc["default load (number)", "found among crash ZIPs"] == 0
say(f"Read with default types, the housing postcode is a decimal number and none of its "
    f"{key_match.loc['default load (number)', 'distinct housing ZIPs']} values equals a crash ZIP as text; as cleaned, "
    f"{key_match.loc['as cleaned (text)', 'found among crash ZIPs']} of "
    f"{key_match.loc['as cleaned (text)', 'distinct housing ZIPs']} do. {n_dtype_changes} columns across the four "
    f"integrated tables changed dtype in the cast. The panel is {memory_mb(zip_year):.2f} "
    f"MB and the tagged crash table {memory_mb(crashes):.0f} MB.")

# %% [markdown]
# The first table is the reason the loading step read everything as text: with default types the
# two ZIP columns would have had no value in common, and a merge would have returned an empty
# match without raising an error. The second table shows the year stored in two ways on the two
# sides. The third lists what the cast changed. The identifiers that can be missing keep a
# nullable integer type on purpose: a crash that is not near any site has no `near_site_id`, and
# a site that is not finished has no `completion_year`. A zero or a minus one in those cells
# would be a value that looks real.

# %% [markdown]
# ### 6.4 Conservation checks
#
# Integration should not create or lose anything without it being counted. The checks below
# are assertions, so the notebook stops if one of them fails. The first table accounts for every
# crash: it is either in a row of the panel or in one of the excluded groups.

# %%
crash_measure_sources = {"crashes": None, "injury_crashes": "is_injury_crash", "persons_injured": "injured",
                         "persons_killed": "killed", "ped_cyc_crashes": "ped_cyc_harm"}
exclusion_groups = {
    "In the ZIP panel": crashes["in_zip_panel"].to_numpy(),
    f"Excluded: ZIP with fewer than {MIN_ZIP_CRASHES} crashes": (crashes["zip"].notna() & ~crashes["in_zip_panel"]).to_numpy(),
    "Excluded: no ZIP, coordinates in an unlabelled cell": (crashes["zip"].isna() & crashes["is_located"]).to_numpy(),
    "Excluded: no ZIP and no coordinates": (crashes["zip"].isna() & ~crashes["is_located"]).to_numpy(),
}
conservation = pd.DataFrame({
    measure: {group: int(mask.sum()) if source is None else int(crashes[source].to_numpy()[mask].sum())
              for group, mask in exclusion_groups.items()}
    for measure, source in crash_measure_sources.items()})
conservation.loc["Total"] = conservation.sum()
conservation["% of crashes"] = (100 * conservation["crashes"] / len(crashes)).round(2)
display(conservation)

# Crashes: the panel plus the excluded groups equal the cleaned table, measure by measure.
group_masks = np.column_stack(list(exclusion_groups.values()))
assert (group_masks.sum(axis=1) == 1).all(), "every crash is in exactly one group"
assert conservation.loc["Total", "crashes"] == len(crashes) == crashes["collision_id"].nunique()
for measure, source in crash_measure_sources.items():
    cleaned_total = len(crashes) if source is None else int(crashes[source].sum())
    assert conservation.loc["Total", measure] == cleaned_total, measure
    assert zip_year[measure].sum() == conservation.loc["In the ZIP panel", measure], measure

# Panel: unique on its key, balanced, and no negative count.
panel_count_columns = [column for column, dtype in ZIP_YEAR_SCHEMA.items() if dtype == COUNT_DTYPE]
assert not zip_year.duplicated(["zip", "year"]).any()
assert len(zip_year) == zip_year["zip"].nunique() * N_STUDY_YEARS
assert zip_year.groupby("zip")["year"].apply(list).map(lambda years: years == STUDY_YEARS).all()
assert zip_year.notna().all().all()
assert (zip_year[panel_count_columns] >= 0).all().all()
assert (zip_year["injury_crashes"] <= zip_year["crashes"]).all()
assert (zip_year["ped_cyc_crashes"] <= zip_year["injury_crashes"]).all()
assert (zip_year["crashes_zip_imputed"] <= zip_year["crashes"]).all()
assert (zip_year["deep_affordable_units"] <= zip_year["affordable_units_completed"]).all()
assert (zip_year["affordable_units_completed"] <= zip_year["units_completed"]).all()

# Housing: every unit of the cleaned table is either in the panel or in a counted exclusion.
housing_groups = {
    "In the ZIP panel": (housing["is_completed"] & housing["zip"].isin(panel_zips)
                         & housing["completion_year"].between(YEAR_START, YEAR_END)),
    f"Completed before {YEAR_START} (in cum_units only)": (
        housing["is_completed"] & housing["zip"].isin(panel_zips) & (housing["completion_year"] < YEAR_START)),
    f"Completed after {YEAR_END}": (housing["is_completed"] & housing["zip"].isin(panel_zips)
                                    & (housing["completion_year"] > YEAR_END)),
    "Completed, ZIP not in the panel": (housing["is_completed"] & housing["zip"].notna()
                                        & ~housing["zip"].isin(panel_zips)),
    "Completed, no ZIP (confidential)": housing["is_completed"] & housing["zip"].isna(),
    "No completion date": ~housing["is_completed"],
}
housing_conservation = pd.DataFrame({
    "rows": {group: int(mask.sum()) for group, mask in housing_groups.items()},
    "total_units": {group: int(housing.loc[mask, "total_units"].sum()) for group, mask in housing_groups.items()},
})
housing_conservation.loc["Total"] = housing_conservation.sum()
housing_conservation["% of units"] = (100 * housing_conservation["total_units"] / housing["total_units"].sum()).round(1)
display(housing_conservation)
assert housing_conservation.loc["Total", "rows"] == len(housing)
assert housing_conservation.loc["Total", "total_units"] == housing["total_units"].sum()
assert zip_year["units_completed"].sum() == housing_conservation.loc["In the ZIP panel", "total_units"]
assert zip_year["buildings_completed"].sum() == housing_conservation.loc["In the ZIP panel", "rows"]

# Spatial tables: each crash once per site, the tables balanced, no negative count.
assert not site_pairs.duplicated(["site_id", "crash_row"]).any(), "each crash is counted once per site"
assert site_year["crashes"].sum() == len(site_pairs)
assert len(site_year) == len(sites) * N_STUDY_YEARS and not site_year.duplicated(["site_id", "year"]).any()
assert (site_year[SITE_COUNT_COLUMNS] >= 0).all().all() and (event_panel[SITE_COUNT_COLUMNS] >= 0).all().all()
assert not event_panel.duplicated(["site_id", "cohort", "rel_year"]).any()
assert len(sites) == int(housing["is_located"].sum()) and sites["site_id"].is_unique
# The nearest-site tag is a partition: a crash has at most one site, and the per-site totals of
# tagged crashes add up to the number of distinct crashes near any site.
assert crashes["near_site_id"].notna().sum() == site_pairs["crash_row"].nunique()
assert crashes.groupby("near_site_id").size().sum() == site_pairs["crash_row"].nunique()

say(f"All checks pass. Of {len(crashes):,} crashes, {int(conservation.loc['In the ZIP panel', 'crashes']):,} "
    f"({conservation.loc['In the ZIP panel', '% of crashes']:.1f}%) are in the panel and "
    f"{len(crashes) - int(conservation.loc['In the ZIP panel', 'crashes']):,} are excluded in three counted "
    f"groups. For injury crashes the panel holds "
    f"{pct(conservation.loc['In the ZIP panel', 'injury_crashes'], conservation.loc['Total', 'injury_crashes']):.1f}%. "
    f"On the housing side the panel's yearly columns hold "
    f"{housing_conservation.loc['In the ZIP panel', '% of units']:.1f}% of all units in the cleaned table; the "
    f"largest group outside it is the buildings with no completion date "
    f"({housing_conservation.loc['No completion date', '% of units']:.1f}% of units).")

# %% [markdown]
# Nothing was lost without a count. The crashes outside the panel remain in `crashes`, flagged
# by `in_zip_panel`, and still take part in the spatial tables when they have coordinates and in
# the citywide series below. The housing rows outside the panel's yearly columns are mostly
# buildings that are not finished, which the ZIP level cannot use by design and the spatial level
# uses as comparison sites.

# %% [markdown]
# ### 6.5 A citywide yearly index for reference
#
# Every comparison over time in this notebook has to be read against what happened in the city as
# a whole. The last table of this section gives that reference: yearly totals over all cleaned
# crashes, with or without a ZIP or coordinates, and an index that sets `INDEX_BASE_YEAR` to 100.
# The two series are drawn on one axis as indices, since their levels differ several times over.

# %%
city_year = (crashes.groupby("year")
             .agg(crashes=("collision_id", "size"), injury_crashes=("is_injury_crash", "sum"),
                  persons_injured=("injured", "sum"), persons_killed=("killed", "sum"),
                  ped_cyc_crashes=("ped_cyc_harm", "sum"))
             .astype(COUNT_DTYPE).reset_index())
city_year["year"] = city_year["year"].astype("int16")
city_year["injury_share"] = city_year["injury_crashes"] / city_year["crashes"]
# Deaths are recorded whatever the severity of the crash, so this ratio shows whether the injury
# crashes that remain after the break are more severe, or whether minor ones went missing.
city_year["deaths_per_1000_injury_crashes"] = PER_THOUSAND * city_year["persons_killed"] / city_year["injury_crashes"]
base_row = city_year.set_index("year").loc[INDEX_BASE_YEAR]
city_year["crash_index"] = INDEX_BASE * city_year["crashes"] / base_row["crashes"]
city_year["injury_crash_index"] = INDEX_BASE * city_year["injury_crashes"] / base_row["injury_crashes"]
break_row = city_year.set_index("year").loc[BREAK_YEAR]
assert list(city_year["year"]) == STUDY_YEARS
assert city_year["crashes"].sum() == len(crashes)
display(city_year.round(3))

def plot_city_index(city_year, break_row):
    """Index of all crashes and of injury crashes against the base year, with the break year marked."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    index_specs = [("crash_index", "All reported crashes", PALETTE["before"]),
                   ("injury_crash_index", "Injury crashes (primary outcome)", PALETTE["crash"])]
    for column, label, color in index_specs:
        ax.plot(city_year["year"], city_year[column], color=color, linewidth=2.2, marker="o", markersize=6,
                label=label)
        ax.annotate(f"{city_year[column].iloc[-1]:.0f}", (YEAR_END, city_year[column].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5)
    ax.axhline(INDEX_BASE, color=PALETTE["neutral"], linewidth=0.8, linestyle=":")
    ax.axvspan(BREAK_YEAR - 0.5, BREAK_YEAR + 0.5, color=PALETTE["before"], alpha=0.25, linewidth=0)
    ax.annotate(f"{BREAK_YEAR}: reporting of\nminor crashes drops", (BREAK_YEAR + 0.55, INDEX_BASE * 1.07),
                fontsize=9, color=PALETTE["neutral"], va="center")
    ax.set_xticks(STUDY_YEARS)
    ax.set_xlim(YEAR_START - 0.3, YEAR_END + 0.8)
    ax.set_ylim(0, city_year[["crash_index", "injury_crash_index"]].max().max() * AXIS_HEADROOM)
    ax.legend(loc="lower left")
    finish(ax, f"In {BREAK_YEAR} all crashes fall {INDEX_BASE - break_row['crash_index']:.0f}% and injury crashes "
               f"{INDEX_BASE - break_row['injury_crash_index']:.0f}%",
           "Crash year", f"Index ({INDEX_BASE_YEAR} = {INDEX_BASE})")
    plt.tight_layout()
    plt.show()


plot_city_index(city_year, break_row)

# %%
city_by_year = city_year.set_index("year")
after_break = city_by_year.loc[BREAK_YEAR + 1:YEAR_END - 1, "injury_crash_index"]
deaths_rate = city_by_year["deaths_per_1000_injury_crashes"]
deaths_before = deaths_rate.loc[:INDEX_BASE_YEAR]
deaths_after = deaths_rate.loc[BREAK_YEAR:]
say(f"Between {INDEX_BASE_YEAR} and {BREAK_YEAR} all reported crashes fall from "
    f"{int(city_by_year.loc[INDEX_BASE_YEAR, 'crashes']):,} to {int(city_by_year.loc[BREAK_YEAR, 'crashes']):,} "
    f"(index {city_by_year.loc[BREAK_YEAR, 'crash_index']:.0f}) and reach index "
    f"{city_by_year.loc[YEAR_END, 'crash_index']:.0f} in {YEAR_END}. Injury crashes go from "
    f"{int(city_by_year.loc[INDEX_BASE_YEAR, 'injury_crashes']):,} to "
    f"{int(city_by_year.loc[BREAK_YEAR, 'injury_crashes']):,} (index "
    f"{city_by_year.loc[BREAK_YEAR, 'injury_crash_index']:.0f}), stand between index {after_break.min():.0f} and "
    f"{after_break.max():.0f} in {BREAK_YEAR + 1} to {YEAR_END - 1} and at index "
    f"{city_by_year.loc[YEAR_END, 'injury_crash_index']:.0f} in {YEAR_END}. The share of crashes with an injury "
    f"or death rises from {100 * city_by_year.loc[INDEX_BASE_YEAR, 'injury_share']:.0f}% to "
    f"{100 * city_by_year.loc[YEAR_END, 'injury_share']:.0f}%. Deaths per 1,000 injury crashes run from "
    f"{deaths_before.min():.1f} to {deaths_before.max():.1f} in {YEAR_START} to {INDEX_BASE_YEAR} and from "
    f"{deaths_after.min():.1f} to {deaths_after.max():.1f} in {BREAK_YEAR} to {YEAR_END}.")

# %% [markdown]
# The two lines part in the year after the base year and never rejoin. A count of all crashes
# would show a large drop at every site and in every ZIP after that year, whatever was built
# there, because fewer minor collisions were written up. Injury crashes dip in the same year,
# recover part of the way and stay below the base level, so the break reaches them too, only
# less. They are the outcome used in Section 8 because they were far less affected than the
# count of all crashes. Deaths are recorded whatever the severity of the crash, and the deaths
# per 1,000 injury crashes are higher after the break than before. That fits more severe crashes
# among those that remain, but it also fits some minor-injury crashes having gone unreported, and
# the data cannot tell the two apart. This is another reason that every before and after
# comparison in Section 8 is made against comparison sites or within the same calendar years.
# `city_year` is kept so that later sections can express a local count relative to the citywide
# level of its year.

# %% [markdown]
# ### 6.6 Rules logged in this section and final outputs
#
# The log below lists every rule recorded in Section 6, and the table after it lists the six
# tables that the analysis sections use.

# %%
post_integration_log = pd.DataFrame(CLEANING_LOG[POST_LOG_START:])
with pd.option_context("display.max_colwidth", None):
    display(post_integration_log)
assert set(post_integration_log["action"]) <= {"drop", "impute", "flag", "standardise"}

final_outputs = pd.DataFrame([
    {"table": "zip_year", "grain": "ZIP x year (balanced)", "rows": len(zip_year), "columns": zip_year.shape[1]},
    {"table": "sites", "grain": "located building in a project", "rows": len(sites), "columns": sites.shape[1]},
    {"table": "site_year", "grain": "site x year (balanced)", "rows": len(site_year), "columns": site_year.shape[1]},
    {"table": "event_panel", "grain": "site x cohort x relative year", "rows": len(event_panel),
     "columns": event_panel.shape[1]},
    {"table": "crashes", "grain": "collision, with ZIP and housing relation", "rows": len(crashes),
     "columns": crashes.shape[1]},
    {"table": "city_year", "grain": "year", "rows": len(city_year), "columns": city_year.shape[1]},
]).set_index("table")
display(final_outputs)
display(zip_year[zip_year["units_completed"] > 0].head(5))

# %% [markdown]
# These six tables are what the analysis sections use. The functions of Sections 5 and 6 stay
# defined, together with the crash tree and the ZIP lookup, so that Section 9 can rebuild the
# spatial tables at another radius or the panel on reported ZIPs only.

# %%
release("labelled", "train_rows", "hidden_rows", "hidden_result", "hidden_covered", "filled_zip",
        "recovered_zip", "borough_of_zip", "filled_borough", "crash_zip_borough", "cumulative",
        "imputed_counts", "group_masks", "default_housing_keys", "completed_in_window", "excluded_housing",
        "in_panel_history")
