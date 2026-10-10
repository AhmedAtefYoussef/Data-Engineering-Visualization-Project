# %% [markdown]
# ## 3. Exploratory data analysis
#
# This section looks at the two tables exactly as they were loaded, before any cleaning. The
# aim is to find out what is in them, what is wrong with them and what that means for joining
# them, so that every cleaning rule in Section 4 answers a problem that was seen here first.
#
# Both raw frames hold text only. Statistics and charts need numbers and dates, so each table
# gets a temporary numeric view built inside a function. The raw frames are never modified,
# the views are deleted at the end of the section, and nothing defined here is used later.
# A value that cannot be converted becomes missing in the view and is counted, which is itself
# a finding.

# %%
# --- Constants used only in this section (the ones shared with Section 4 are defined in 2.1) ----
DAYS_PER_YEAR = 365.25                     # converts a duration in days to years
DURATION_BIN_YEARS = 0.25                  # bar width (one quarter) in the construction-time histogram
DURATION_MAX_YEARS = 8                     # right edge of that histogram; longer projects are counted in the text
HEXBIN_GRIDSIZE = 140                      # hexagons across the crash map, about 400 m each
ZIP_PATTERN = r"^\d{5}$"                   # a US ZIP code is exactly five digits
TOP_FACTORS = 15                           # contributing factors shown in the bar chart
SPELLING_SIMILARITY = 0.88                 # difflib ratio above which two factor labels count as near-duplicates
LABEL_MARGIN = (0, 0, 0.86, 1)              # tight_layout rectangle that leaves room for end-of-line labels
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CONSTRUCTION_COLORS = {"New Construction": PALETTE["housing"], "Preservation": PALETTE["placebo"]}

HOUSING_DATE_COLUMNS = ["project_start_date", "project_completion_date", "building_completion_date"]
HOUSING_PLACE_COLUMNS = ["building_id", "postcode", "bbl", "bin", "census_tract",
                         "neighborhood_tabulation_area", "latitude", "longitude",
                         "latitude_internal", "longitude_internal"]
CRASH_CASUALTY_COLUMNS = [
    "number_of_persons_injured", "number_of_persons_killed",
    "number_of_pedestrians_injured", "number_of_pedestrians_killed",
    "number_of_cyclist_injured", "number_of_cyclist_killed",
    "number_of_motorist_injured", "number_of_motorist_killed",
]
CRASH_INJURY_PARTS = ["number_of_pedestrians_injured", "number_of_cyclist_injured",
                      "number_of_motorist_injured"]

EDA_FACTS = {}   # numbers collected along the way for the summary table in 3.3
EDA_RAW_SHAPES = (raw_housing.shape, raw_crashes.shape)   # checked again at the end of the section


# Section 2.1 defines `to_number`. Exploration calls it with strict=False, so that text which
# does not convert becomes NaN and is counted as a finding instead of stopping the notebook.


def to_date(text):
    """Parse a raw ISO timestamp column; anything that does not match the format becomes NaT."""
    return pd.to_datetime(text, format=DATE_FORMAT, errors="coerce")


def as_flags(condition):
    """Turn a nullable boolean Series into a plain numpy bool array, treating unknown as False."""
    return condition.fillna(False).to_numpy(dtype=bool)


def raw_profile(frame):
    """Profile every column of an all-text frame: completeness, cardinality and text hygiene."""
    rows = []
    for column in frame.columns:
        text = frame[column]
        present = text.dropna()
        lengths = present.str.len()
        padded = int((lengths != present.str.strip().str.len()).sum())
        rows.append({
            "column": column, "dtype as loaded": str(text.dtype),
            "missing": int(text.isna().sum()), "missing_pct": pct(int(text.isna().sum()), len(frame)),
            "distinct": int(present.nunique()),
            "min length": int(lengths.min()) if len(present) else 0,
            "max length": int(lengths.max()) if len(present) else 0,
            "padded with spaces": padded,
            "example": present.iloc[0] if len(present) else "",
        })
    return pd.DataFrame(rows).set_index("column")


def end_label(ax, x, y, text, shift_points=0):
    """Write a series name next to the last point of its line, in text colour, not series colour.

    `shift_points` moves the label up or down when two lines end close together.
    """
    ax.annotate(text, xy=(x, y), xytext=(7, shift_points), textcoords="offset points", va="center",
                fontsize=9, color=PALETTE["neutral"], annotation_clip=False)


def color_legend(ax, colors, **legend_options):
    """Legend built from a {label: colour} mapping, for charts whose bars are coloured one by one."""
    handles = [mpl.patches.Patch(color=color, label=label) for label, color in colors.items()]
    return ax.legend(handles=handles, **legend_options)


def describe_table(frame, percentiles):
    """describe() turned sideways, with the non-missing count shown as a whole number."""
    table = frame.describe(percentiles=percentiles).T.round(2)
    table.insert(0, "rows with a value", table.pop("count").astype(int))
    return table


# %% [markdown]
# ### 3.1 Housing: Affordable Housing Production by Building
#
# #### 3.1.1 Shape, types and a first look
#
# The profile below lists every column as loaded: how many values are missing, how many distinct
# values there are, the shortest and longest text and one example. Length and padding are
# checked because identifiers of uneven length or values with stray spaces break joins without
# raising an error.

# %%
def housing_view(raw):
    """Numeric and datetime view of the raw housing table for exploration only."""
    view = pd.DataFrame(index=raw.index)
    view["is_confidential"] = as_flags(raw["project_name"].str.strip().str.upper() == CONFIDENTIAL_NAME)
    view["borough"] = raw["borough"].astype(object)
    view["construction_type"] = raw["reporting_construction_type"].astype(object)
    for column in UNIT_COLUMNS + ["latitude", "longitude", "postcode"]:
        view[column] = to_number(raw[column], strict=False)
    for column in HOUSING_DATE_COLUMNS:
        view[column] = to_date(raw[column])
    # The building date is the precise one; the project date stands in when it is absent.
    view["completion_date"] = view["building_completion_date"].fillna(view["project_completion_date"])
    return view


eda_housing = housing_view(raw_housing)
eda_housing_profile = raw_profile(raw_housing)

say(f"The housing table has **{raw_housing.shape[0]:,} rows and {raw_housing.shape[1]} columns**, all "
    f"loaded as `{raw_housing.dtypes.astype(str).unique()[0]}`. "
    f"{int((eda_housing_profile['missing'] > 0).sum())} of the columns have at least one missing value.")
with pd.option_context("display.max_rows", 60, "display.max_colwidth", 38):
    display(eda_housing_profile.round(2))
display(raw_housing.head(3).T)

# %%
def housing_text_checks(raw, view):
    """Count the format problems a silent type conversion would hide."""
    unparsed_dates = {column: int((raw[column].notna() & view[column].isna()).sum())
                      for column in HOUSING_DATE_COLUMNS}
    non_numeric_units = {column: int((raw[column].notna() & view[column].isna()).sum())
                         for column in UNIT_COLUMNS}
    has_time_of_day = int((view[HOUSING_DATE_COLUMNS].apply(lambda dates: dates.dt.normalize() != dates)
                           & view[HOUSING_DATE_COLUMNS].notna()).to_numpy().sum())
    postcode_ok = as_flags(raw["postcode"].str.match(ZIP_PATTERN))
    return {
        "unparsed_dates": sum(unparsed_dates.values()),
        "non_numeric_units": sum(non_numeric_units.values()),
        "dates_with_time": has_time_of_day,
        "postcode_present": int(raw["postcode"].notna().sum()),
        "postcode_five_digits": int(postcode_ok.sum()),
        "first_date": view[HOUSING_DATE_COLUMNS].min().min(),
        "last_date": view[HOUSING_DATE_COLUMNS].max().max(),
    }


eda_housing_text = housing_text_checks(raw_housing, eda_housing)
say(f"Text that fails to convert: **{eda_housing_text['unparsed_dates']}** date values and "
    f"**{eda_housing_text['non_numeric_units']}** unit values. {eda_housing_text['dates_with_time']} dates "
    f"carry a time of day. Of {eda_housing_text['postcode_present']:,} ZIP codes present, "
    f"{eda_housing_text['postcode_five_digits']:,} are exactly five digits. The dates run from "
    f"{eda_housing_text['first_date']:%d %B %Y} to {eda_housing_text['last_date']:%d %B %Y}.")

# %% [markdown]
# The table is small and its text is clean: every date follows one format with no time part,
# every unit count is a whole number and every ZIP code present is five digits. The last date
# lies in the future relative to the study window, which means some completion dates are
# planned or very recent and those buildings have no "after" period yet.
#
# The profile also shows the cost of reading everything as text. Dates, coordinates and counts
# all have to be converted explicitly in Section 4. The alternative, letting pandas guess the
# types, was rejected because it turns ZIP codes and building identifiers into decimals as soon
# as one value is missing, and a ZIP code stored as a decimal no longer matches the same ZIP
# code stored as text in the crash table.

# %% [markdown]
# #### 3.1.2 Descriptive statistics
#
# Summary statistics of the unit counts, taken over the values that are present. The count
# column is the number of rows with a value, so it also shows how sparse each column is.

# %%
display(describe_table(eda_housing[UNIT_COLUMNS], [0.25, 0.5, 0.75, 0.95, 0.99]))

# %% [markdown]
# No unit column has a minimum of zero, although a building without, say, three-bedroom flats
# must be common. The mean of `total_units` also sits far above its median, with a maximum
# several times the 99th percentile, so the size distribution has a long right tail. Both points
# are followed up below.

# %% [markdown]
# #### 3.1.3 Missing values, and why unit columns are blank and not zero
#
# The chart shows the share of rows missing in every column that has any gap. Columns are
# coloured by what they hold, because the three kinds of gap have different causes and will
# need different treatment.

# %%
def housing_missing_groups(profile):
    """Label each column with a gap by the kind of information it holds."""
    gaps = profile.loc[profile["missing"] > 0, ["missing", "missing_pct"]].copy()
    gaps["kind"] = np.select(
        [gaps.index.isin(UNIT_COLUMNS), gaps.index.isin(HOUSING_DATE_COLUMNS)],
        ["unit count", "date"], default="location or identifier")
    return gaps.sort_values("missing_pct")


def plot_housing_missing(gaps):
    """Horizontal bars of missing share per housing column, coloured by kind of column."""
    kind_colors = {"unit count": PALETTE["placebo"], "location or identifier": PALETTE["housing"],
                   "date": PALETTE["after"]}
    fig, ax = plt.subplots(figsize=(9, 0.27 * len(gaps) + 1.4))
    bars = ax.barh(gaps.index, gaps["missing_pct"], color=gaps["kind"].map(kind_colors), height=0.72)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8.5, color=PALETTE["neutral"])
    ax.set_xlim(0, 108)
    ax.grid(axis="y", visible=False)
    color_legend(ax, kind_colors, title="column holds", loc="lower right")
    finish(ax, "Housing: unit columns are mostly blank, location fields share one gap",
           "rows with no value (%)", "")
    plt.tight_layout()
    plt.show()


eda_housing_gaps = housing_missing_groups(eda_housing_profile)
plot_housing_missing(eda_housing_gaps)

# %%
def unit_blank_report(raw, view):
    """Check whether blank unit cells stand for zero: literal zeros, and totals after a zero fill."""
    filled = view[UNIT_COLUMNS].fillna(0)
    report = {
        "blank_cells": int(raw[UNIT_COLUMNS].isna().to_numpy().sum()),
        "literal_zeros": int((view[UNIT_COLUMNS] == 0).to_numpy().sum()),
        "cells": raw[UNIT_COLUMNS].size,
    }
    for name, columns in [("income", INCOME_UNIT_COLUMNS), ("bedroom", BEDROOM_UNIT_COLUMNS),
                          ("tenure", TENURE_UNIT_COLUMNS)]:
        report[f"{name}_sum_matches"] = int((filled[columns].sum(axis=1) == filled["all_counted_units"]).sum())
    report["counted_above_total"] = int((filled["all_counted_units"] > filled["total_units"]).sum())
    report["counted_below_total"] = int((filled["all_counted_units"] < filled["total_units"]).sum())
    report["counted_blank"] = int(view["all_counted_units"].isna().sum())
    return report


eda_units = unit_blank_report(raw_housing, eda_housing)
EDA_FACTS["unit_blank_cells"] = eda_units["blank_cells"]
say(f"Across the {len(UNIT_COLUMNS)} unit columns, **{eda_units['blank_cells']:,} of "
    f"{eda_units['cells']:,} cells are blank ({pct(eda_units['blank_cells'], eda_units['cells']):.1f}%) and "
    f"{eda_units['literal_zeros']} contain a written zero**. If each blank is read as zero, the six income "
    f"bands add up to `all_counted_units` in {eda_units['income_sum_matches']:,} of {len(raw_housing):,} rows, "
    f"the eight bedroom sizes in {eda_units['bedroom_sum_matches']:,} and rental plus homeownership in "
    f"{eda_units['tenure_sum_matches']:,}. `all_counted_units` is above `total_units` in "
    f"{eda_units['counted_above_total']} rows, below it in {eda_units['counted_below_total']:,} "
    f"(buildings that also hold units outside the programme) and blank in {eda_units['counted_blank']}.")

# %% [markdown]
# The unit columns are not missing in the usual sense. The source never writes a zero, so a
# blank is the only way a zero can appear, and the totals confirm it: with blanks read as zero
# the parts add up to the published total. Section 4 therefore fills these blanks with zero.
# Treating them as unknown was the alternative, and it would be wrong twice over: it would
# discard most rows from any analysis of income mix, and it would bias averages upward because
# only non-zero values would be averaged.
#
# The location columns are a different case. They show the same missing share, which suggests
# one group of rows lacks all of them at once. The next chart checks that.

# %% [markdown]
# #### 3.1.4 The confidential rows
#
# Some rows have the project name `CONFIDENTIAL` and a masked address. The heatmap compares
# how often each location, identifier and date field is missing in those rows and in all the
# others.

# %%
def confidential_missing(raw, view):
    """Missing share (%) of each place and date column, split by confidential flag."""
    columns = HOUSING_PLACE_COLUMNS + ["project_completion_date", "building_completion_date"]
    missing = raw[columns].isna()
    table = missing.groupby(view["is_confidential"].to_numpy()).mean().mul(100)
    counts = view["is_confidential"].value_counts()
    table.index = [f"{'confidential' if flag else 'address published'} ({counts[flag]:,} rows)"
                   for flag in table.index]
    return table.sort_index(ascending=False)


def plot_confidential_missing(table):
    """Annotated heatmap: which fields disappear in the confidential rows."""
    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    sns.heatmap(table, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100, linewidths=2,
                linecolor="white", cbar_kws={"label": "rows with no value (%)", "aspect": 8}, ax=ax)
    ax.set_xticklabels([label.get_text().replace("neighborhood_tabulation_area", "nta")
                        for label in ax.get_xticklabels()], rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    finish(ax, "Housing: confidential rows lose every field that could place them on a map", "", "")
    plt.tight_layout()
    plt.show()


def confidential_summary(raw, view):
    """Size of the confidential group in rows and in units, and how its address is masked."""
    hidden = view["is_confidential"]
    located = view["latitude"].notna() & view["longitude"].notna()
    return {
        "rows": int(hidden.sum()),
        "units": int(view.loc[hidden, "total_units"].sum()),
        "all_units": int(view["total_units"].sum()),
        "median_units": float(view.loc[hidden, "total_units"].median()),
        "median_units_other": float(view.loc[~hidden, "total_units"].median()),
        "masked_address": int(as_flags(raw["street_name"].str.strip() == WITHHELD_ADDRESS).sum()),
        "with_borough": int(raw.loc[hidden.to_numpy(), "borough"].notna().sum()),
        "other_unlocated": int((~hidden & ~located).sum()),
        "by_type": view.loc[hidden, "construction_type"].value_counts(),
    }


plot_confidential_missing(confidential_missing(raw_housing, eda_housing))
eda_confidential = confidential_summary(raw_housing, eda_housing)
EDA_FACTS["confidential_rows"] = eda_confidential["rows"]
EDA_FACTS["confidential_unit_pct"] = pct(eda_confidential["units"], eda_confidential["all_units"])
EDA_FACTS["other_unlocated"] = eda_confidential["other_unlocated"]
say(f"**{eda_confidential['rows']:,} rows ({pct(eda_confidential['rows'], len(raw_housing)):.1f}%) are "
    f"confidential.** {eda_confidential['masked_address']:,} rows have the street name `{WITHHELD_ADDRESS}`. "
    f"All {eda_confidential['with_borough']:,} confidential rows still name a borough. They hold "
    f"{eda_confidential['units']:,} of {eda_confidential['all_units']:,} units "
    f"(**{EDA_FACTS['confidential_unit_pct']:.2f}%**), with a median of "
    f"{eda_confidential['median_units']:.0f} unit per row against {eda_confidential['median_units_other']:.0f} "
    f"elsewhere. By type, "
    + join_and(f"{count:,} are {name}" for name, count in eda_confidential["by_type"].items())
    + f". Outside this group, "
    f"{eda_confidential['other_unlocated']} rows with a published address have no coordinates.")

# %% [markdown]
# The gap in the location columns is one block of rows, not scattered omissions. In a
# confidential row the building identifier, ZIP code, tax lot, census tract and both pairs of
# coordinates are all absent, while borough and the dates survive. The second pair of
# coordinates (`latitude_internal`, `longitude_internal`) is missing in the same rows, so it
# cannot serve as a fallback.
#
# This matters for integration because both join levels need a place. A confidential row has
# no ZIP code for the ZIP by year panel and no point for the distance join, and nothing else in
# the row narrows its position below the borough. Imputing a location, for example the borough
# centroid, was considered and rejected: it would stack all of these rows on five points and
# invent crashes "near" buildings that are somewhere else. The rows are therefore flagged and
# kept for borough totals only. The next chart shows how little housing that sets aside.

# %%
def plot_confidential_sizes(view):
    """Cumulative distribution of building size for confidential and published rows (log x)."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    groups = [("address published", ~view["is_confidential"], PALETTE["housing"]),
              ("confidential", view["is_confidential"], PALETTE["accent"])]
    for label, mask, color in groups:
        units = np.sort(view.loc[mask, "total_units"].dropna().to_numpy())
        share = 100 * np.arange(1, len(units) + 1) / len(units)
        ax.step(units, share, where="post", color=color, linewidth=2, label=f"{label} (n = {len(units):,})")
    ax.set_xscale("log")
    thousands(ax, axis="x")
    ax.set_ylim(0, 101)
    ax.legend(loc="lower right")
    finish(ax, "Housing: confidential rows are almost all one- to three-unit buildings",
           "total units in the building (log scale)", "rows with at most this many units (%)")
    plt.tight_layout()
    plt.show()


plot_confidential_sizes(eda_housing)

# %% [markdown]
# The confidential curve rises almost vertically at the left edge: most of these rows are a
# single home, while buildings with a published address range from single homes to very large
# towers. Setting the confidential rows aside from the location joins costs a large share of
# the rows but a very small share of the units, so the located buildings still describe nearly
# all of the housing the programme produced.

# %% [markdown]
# #### 3.1.5 Building size and its outliers
#
# `total_units` is the housing variable several research questions rely on, so its shape
# matters. On a plain axis almost every building would fall into the first bar and the tail
# would be invisible, so the histogram uses a log axis. It marks the median and the usual upper
# outlier fence (third quartile plus 1.5 times the interquartile range).

# %%
def size_summary(view):
    """Quartiles, Tukey upper fence and tail counts of total_units."""
    units = view["total_units"].dropna()
    first_quartile, median, third_quartile = units.quantile([0.25, 0.5, 0.75])
    fence = tukey_upper_fence(units)
    return {"n": len(units), "median": median, "mean": units.mean(), "q1": first_quartile,
            "q3": third_quartile, "fence": fence, "above_fence": int((units > fence).sum()),
            "max": units.max(), "p99": units.quantile(0.99), "skew": stats.skew(units),
            "top_decile_unit_pct": pct(units[units >= units.quantile(0.9)].sum(), units.sum())}


def plot_building_sizes(view, summary):
    """Histogram of total_units on a log axis with the median and the outlier fence."""
    units = view["total_units"].dropna()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.hist(units, bins=integer_log_bins(summary["max"]), color=PALETTE["housing"], edgecolor="white",
            linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(summary["median"], color=PALETTE["neutral"], linestyle="-", linewidth=1.4, zorder=5,
               label=f"median ({summary['median']:.0f} units)")
    ax.axvline(summary["fence"], color=PALETTE["neutral"], linestyle="--", linewidth=1.4, zorder=5,
               label=f"upper IQR fence ({summary['fence']:.0f} units)")
    ax.legend(loc="upper right")
    thousands(ax, axis="x")
    thousands(ax)
    finish(ax, "Housing: building size is strongly right-skewed", "total units in the building (log scale)",
           "number of rows per bin")
    plt.tight_layout()
    plt.show()


eda_sizes = size_summary(eda_housing)
EDA_FACTS["size_above_fence"] = eda_sizes["above_fence"]
EDA_FACTS["size_above_fence_pct"] = pct(eda_sizes["above_fence"], eda_sizes["n"])
plot_building_sizes(eda_housing, eda_sizes)
say(f"Half of the {eda_sizes['n']:,} rows have at most **{eda_sizes['median']:.0f} units**, the mean is "
    f"{eda_sizes['mean']:.1f}, the 99th percentile {eda_sizes['p99']:.0f} and the maximum "
    f"{eda_sizes['max']:,.0f} (skewness {eda_sizes['skew']:.1f}). The IQR fence lies at "
    f"{eda_sizes['fence']:.0f} units and would flag **{eda_sizes['above_fence']:,} rows "
    f"({EDA_FACTS['size_above_fence_pct']:.1f}%)**. The largest tenth of the rows holds "
    f"{eda_sizes['top_decile_unit_pct']:.0f}% of all units.")
display(raw_housing.loc[eda_housing["total_units"].nlargest(5).index,
                        ["project_name", "borough", "reporting_construction_type", "total_units",
                         "all_counted_units"]])

# %% [markdown]
# The histogram shows two groups: a spike of single-unit rows, and a broad spread of apartment
# buildings from small walk-ups to towers. The IQR fence falls inside the second group, so it
# flags ordinary apartment buildings, not errors. The largest rows are real, named developments
# in which only part of the building counts towards the programme.
#
# Section 4 therefore keeps every row, uses medians, log scales or size bands, and flags the
# very large buildings so an analysis can be repeated without them. Dropping rows above the
# fence was rejected because it would remove exactly the buildings that add the most residents
# to a street, and the large majority of units with them.

# %% [markdown]
# #### 3.1.6 Categories and their spellings
#
# The table counts every value of the categorical columns exactly as spelled, including the
# split by confidential flag for borough. Variants such as trailing spaces or mixed case would
# appear as separate rows.

# %%
def housing_category_counts(raw, columns):
    """Value counts of several raw categorical columns stacked into one table."""
    parts = [raw[column].value_counts(dropna=False).rename("rows").rename_axis("value").reset_index()
             .assign(column=column) for column in columns]
    table = pd.concat(parts, ignore_index=True)
    table["share_pct"] = 100 * table["rows"] / len(raw)
    return table.set_index(["column", "value"])


display(housing_category_counts(raw_housing, ["reporting_construction_type", "extended_affordability_status",
                                               "prevailing_wage_status"]).round(1))
display(pd.crosstab(raw_housing["borough"].astype(object),
                    [eda_housing["construction_type"], eda_housing["is_confidential"].map(
                        {True: "confidential", False: "published"})], margins=True))
say(f"`community_board` has {raw_housing['community_board'].nunique()} distinct values in the form "
    f"`{raw_housing['community_board'].iloc[0]}`, and `borough` has {raw_housing['borough'].nunique()} "
    f"values: {join_and(sorted(raw_housing['borough'].dropna().unique()))}.")

# %% [markdown]
# The categories are clean: every value has a single spelling and none is blank. New
# construction and preservation are close to evenly split, which is convenient because several
# research questions compare the two. The cross-tabulation also shows that the confidential
# rows are not spread evenly over the boroughs, so borough totals that left them out would
# understate some boroughs far more than others. Borough is written in title case here;
# Section 3.2.7 shows that the crash table writes it differently.

# %% [markdown]
# #### 3.1.7 Time: when buildings were completed and how long they took
#
# Completion is the event the before and after comparisons hang on, so its distribution over
# the years decides how many buildings each comparison can use. The completion date used here
# is the building date, with the project date standing in where the building date is blank.

# %%
def completion_year_counts(view):
    """Rows per completion year and construction type; rows without any completion date apart."""
    years = view["completion_date"].dt.year
    table = (view.assign(completion_year=years).dropna(subset=["completion_year"])
             .groupby(["completion_year", "construction_type"]).size().unstack(fill_value=0))
    table.index = table.index.astype(int)
    return table, int(years.isna().sum())


def plot_completion_years(table, no_date_rows):
    """Stacked bars of completed rows per year by construction type."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bottom = np.zeros(len(table))
    for construction_type in CONSTRUCTION_COLORS:
        bars = ax.bar(table.index, table[construction_type], bottom=bottom, width=0.74, edgecolor="white",
                      linewidth=1.5, color=CONSTRUCTION_COLORS[construction_type], label=construction_type)
        for bar, year in zip(bars, table.index):
            if year > YEAR_END:      # years after the crash window hold planned or very recent dates
                bar.set_hatch("//")
                bar.set_alpha(0.6)
        bottom += table[construction_type].to_numpy()
    ax.axvspan(YEAR_START - 0.5, YEAR_END + 0.5, color=PALETTE["before"], alpha=0.18, zorder=0,
               label=f"crash data loaded ({YEAR_START} to {YEAR_END})")
    ax.set_xticks(table.index)
    ax.grid(axis="x", visible=False)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor=PALETTE["neutral"], hatch="//",
                                     label=f"after {YEAR_END}: partial or planned"))
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    ax.set_ylim(0, bottom.max() * 1.3)
    finish(ax, "Housing: completions per year, by construction type", "completion year",
           "rows (buildings) completed",
           note=f"{no_date_rows:,} rows have no completion date and are not shown.")
    plt.tight_layout()
    plt.show()


eda_completions, eda_no_completion = completion_year_counts(eda_housing)
EDA_FACTS["no_completion_rows"] = eda_no_completion
plot_completion_years(eda_completions, eda_no_completion)
display(eda_completions.assign(total=eda_completions.sum(axis=1)).T)
say(f"**{eda_no_completion:,} rows ({pct(eda_no_completion, len(eda_housing)):.1f}%) have no completion "
    f"date at all.** The project date fills in for "
    f"{int((eda_housing['building_completion_date'].isna() & eda_housing['project_completion_date'].notna()).sum())} "
    f"rows whose building date is blank. "
    f"{int(eda_completions.loc[eda_completions.index > YEAR_END].to_numpy().sum()):,} rows are completed after "
    f"{YEAR_END}, and {int(eda_completions.loc[eda_completions.index < YEAR_START].to_numpy().sum()):,} "
    f"before {YEAR_START}.")

# %% [markdown]
# Every year of the crash window has many completions, with both construction types present
# in each, so before and after comparisons are not carried by one unusual cohort. The mix
# shifts over time, from mostly preservation in the early years to mostly new construction in
# the later ones, which is a reason to compare the two types within the same calendar years.
# The dip in 2020 coincides with the pandemic. Rows without a completion date are projects still
# under way. They are not errors, and Section 4 keeps them flagged as not yet completed. They
# cannot enter a before and after comparison, and dropping them outright was rejected because
# buildings that are not finished yet are useful as a comparison group.
#
# Buildings completed in the first or last years of the window have little crash history on
# one side, which limits the cohorts with complete windows to the middle years named in
# Section 2.5.

# %%
def construction_durations(view):
    """Years between project start and completion, with counts of the odd cases."""
    done = view.dropna(subset=["completion_date", "project_start_date"])
    years = (done["completion_date"] - done["project_start_date"]).dt.days / DAYS_PER_YEAR
    return pd.DataFrame({"years": years, "construction_type": done["construction_type"]})


def plot_durations(durations):
    """Small multiples: time from project start to completion for each construction type.

    Rows completed on their start date get a bar of their own on the left, so that the height of
    that bar is exactly the count in the text. Rows that finish before they start are not drawn.
    """
    bins = np.arange(0, DURATION_MAX_YEARS + DURATION_BIN_YEARS, DURATION_BIN_YEARS)
    same_day_x = -DURATION_BIN_YEARS
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, construction_type in zip(axes, CONSTRUCTION_COLORS):
        years = durations.loc[durations["construction_type"] == construction_type, "years"]
        color = CONSTRUCTION_COLORS[construction_type]
        ax.hist(years[years > 0], bins=bins, color=color, edgecolor="white", linewidth=0.4)
        same_day = int((years == 0).sum())
        ax.bar(same_day_x, same_day, width=DURATION_BIN_YEARS * 0.8, color=color, edgecolor="white")
        ax.annotate(f"{same_day:,}", xy=(same_day_x, same_day), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color=PALETTE["neutral"])
        ticks = [same_day_x, *range(2, DURATION_MAX_YEARS + 1, 2)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["same day"] + [str(tick) for tick in ticks[1:]])
        ax.set_xlim(same_day_x - DURATION_BIN_YEARS, DURATION_MAX_YEARS)
        ax.margins(y=0.1)
        thousands(ax)
        finish(ax, f"{construction_type} (n = {len(years):,})", "years from project start to completion",
               "rows (buildings)" if ax is axes[0] else "")
    fig.suptitle("Housing: many rows are completed on the day the project starts", x=0.01, ha="left",
                 fontweight="bold")
    plt.tight_layout()
    plt.show()


eda_durations = construction_durations(eda_housing)
EDA_FACTS["completion_before_start"] = int((eda_durations["years"] < 0).sum())
EDA_FACTS["completion_on_start"] = int((eda_durations["years"] == 0).sum())
plot_durations(eda_durations)
say(f"Of {len(eda_durations):,} completed rows, **{EDA_FACTS['completion_on_start']:,} "
    f"({pct(EDA_FACTS['completion_on_start'], len(eda_durations)):.1f}%) have a completion date equal to "
    f"the start date** and {EDA_FACTS['completion_before_start']} have a completion date before the start "
    f"date. {int((eda_durations['years'] > DURATION_MAX_YEARS).sum())} took longer than {DURATION_MAX_YEARS} "
    f"years and fall outside the chart. Median duration when it is above zero: "
    + ", ".join(f"{name} {group.loc[group['years'] > 0, 'years'].median():.1f} years"
                for name, group in eda_durations.groupby("construction_type")) + ".")

# %% [markdown]
# The spike at zero is the most notable feature. For those rows the "project start" is the day the
# financing closed on a building that was already standing or already finished, so the table
# shows no construction period. It is common in both types, not only in preservation. Rows
# where completion precedes the start are date-entry inconsistencies and are few.
#
# For the analysis this means the start date is not a dependable marker of when building work
# began, so the before and after comparisons use the completion date alone. Section 4 flags
# rows whose completion precedes their start and leaves the dates as published. The two dates
# are not swapped, because there is no way to know which of them is wrong.

# %% [markdown]
# #### 3.1.8 Duplicates and the grain of the table
#
# Before any join the grain has to be known: what one row stands for and which columns identify
# it. The check below looks for full-row duplicates and tests two candidate keys.

# %%
def housing_key_checks(raw, view):
    """Duplicate counts for the full row and for the candidate keys, among rows with a building id."""
    with_id = raw.loc[raw["building_id"].notna()]
    repeated_buildings = with_id["building_id"].value_counts()
    repeated_buildings = repeated_buildings[repeated_buildings > 1]
    extra_rows = with_id.loc[with_id["building_id"].isin(repeated_buildings.index)]
    return {
        "full_row_duplicates": int(raw.duplicated().sum()),
        "rows_with_id": len(with_id),
        "pair_duplicates": int(with_id.duplicated(["project_id", "building_id"]).sum()),
        "repeated_buildings": len(repeated_buildings),
        "extra_rows": int(repeated_buildings.sum() - len(repeated_buildings)),
        "distinct_start_dates_pct": pct(
            int((extra_rows.groupby("building_id")["project_start_date"].nunique() > 1).sum()),
            len(repeated_buildings)),
        "projects": raw["project_id"].nunique(),
        "max_buildings_per_project": int(raw["project_id"].value_counts().max()),
    }


eda_keys = housing_key_checks(raw_housing, eda_housing)
EDA_FACTS["repeated_building_rows"] = eda_keys["extra_rows"]
say(f"There are **{eda_keys['full_row_duplicates']} full-row duplicates**. Among the "
    f"{eda_keys['rows_with_id']:,} rows with a `building_id`, the pair (`project_id`, `building_id`) repeats "
    f"**{eda_keys['pair_duplicates']} times**, while `building_id` alone repeats on "
    f"{eda_keys['repeated_buildings']} buildings ({eda_keys['extra_rows']} extra rows). In "
    f"{eda_keys['distinct_start_dates_pct']:.0f}% of those buildings the repeated rows have different project "
    f"start dates. The table covers {eda_keys['projects']:,} projects, the largest with "
    f"{eda_keys['max_buildings_per_project']} buildings.")

# %% [markdown]
# One row is one building within one project. A building that appears twice was financed twice,
# usually years apart, for example built under one project and preserved under a later one.
# Those are separate events, so the key for Section 4 is the pair of project and building, and
# de-duplicating on the building alone would delete real events. The confidential rows have no
# building identifier, so they can only be told apart by their row position and are counted,
# not keyed.

# %% [markdown]
# #### 3.1.9 Where the buildings are
#
# A plain scatter of longitude against latitude, coloured by the borough written in the row.
# It checks two things at once: that the coordinates fall inside the city, and that the borough
# label agrees with the position.

# %%
def housing_coordinate_checks(view):
    """Count located rows and rows whose coordinates fall outside the city's bounding box."""
    located = view.dropna(subset=["latitude", "longitude"])
    inside = (located["latitude"].between(*NYC_BOUNDS["lat"])
              & located["longitude"].between(*NYC_BOUNDS["lon"]))
    zero = (located["latitude"] == 0) | (located["longitude"] == 0)
    return located, int((~inside).sum()), int(zero.sum())


def plot_housing_map(located):
    """Scatter of building coordinates coloured by the borough named in the row."""
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    for borough in BOROUGHS:
        points = located.loc[located["borough"] == borough]
        ax.scatter(points["longitude"], points["latitude"], s=7, alpha=0.55, linewidths=0,
                   color=BOROUGH_COLORS[borough], label=f"{borough} ({len(points):,})")
        ax.annotate(borough, xy=(points["longitude"].median(), points["latitude"].median()),
                    ha="center", fontsize=9.5, fontweight="bold", color=PALETTE["neutral"],
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    map_axes(ax)
    ax.legend(title="borough in the row (located rows)", loc="upper left", markerscale=2.6)
    finish(ax, "Housing: located buildings cluster in the Bronx, upper Manhattan and central Brooklyn",
           "longitude (degrees east)", "latitude (degrees north)")
    plt.tight_layout()
    plt.show()


eda_housing_located, eda_housing_outside, eda_housing_zero = housing_coordinate_checks(eda_housing)
plot_housing_map(eda_housing_located)
say(f"**{len(eda_housing_located):,} of {len(eda_housing):,} rows have coordinates.** "
    f"{eda_housing_outside} fall outside the bounding box of the city and {eda_housing_zero} sit at zero. "
    f"The located rows cover {eda_housing_located['postcode'].nunique()} ZIP codes.")

# %% [markdown]
# The coordinates are trustworthy: none falls outside the city and the colours form five clean
# regions with no stray points, so borough and position agree. The spatial pattern is far from
# even. Buildings concentrate in the South Bronx, Harlem and East Harlem, and central and
# eastern Brooklyn, with little in Staten Island and most of Queens. Any comparison of streets
# near affordable housing with "other" streets therefore has to compare within a borough,
# since the places where this housing is built are not a random sample of the city.

# %% [markdown]
# ### 3.2 Crashes: Motor Vehicle Collisions
#
# #### 3.2.1 Shape, types and a first look
#
# The same profile for the crash table. With this many rows the padding and length checks matter
# more than for housing, because these fields were typed by many different officers over ten
# years.

# %%
def crash_view(raw):
    """Compact numeric and datetime view of the raw crash table for exploration only."""
    view = pd.DataFrame(index=raw.index)
    crash_date = to_date(raw["crash_date"])
    assert crash_date.notna().all(), "a crash date does not follow the API date format"
    view["crash_date"] = crash_date
    view["year"] = crash_date.dt.year.astype("int16")
    view["month"] = crash_date.dt.month.astype("int8")
    view["weekday"] = crash_date.dt.dayofweek.astype("int8")
    # crash_time is written H:MM without a leading zero, so the hour is whatever precedes the colon.
    view["hour"] = parse_hour(raw["crash_time"])[0].astype("float32")
    for column in CRASH_CASUALTY_COLUMNS:
        view[column] = to_number(raw[column], strict=False).astype("float32")
    view["latitude"] = to_number(raw["latitude"], strict=False)
    view["longitude"] = to_number(raw["longitude"], strict=False)
    view["zip_missing"] = raw["zip_code"].isna().to_numpy()
    view["borough_missing"] = raw["borough"].isna().to_numpy()
    view["coordinate_status"] = coordinate_status(view["latitude"], view["longitude"])
    # An injury crash has at least one person injured or killed in any of the eight casualty columns.
    view["is_injury_crash"] = view[CRASH_CASUALTY_COLUMNS].fillna(0).sum(axis=1).gt(0).to_numpy()
    return view


def coordinate_status(latitude, longitude):
    """Classify each coordinate pair as missing, at (0, 0), outside the city box or usable."""
    missing = latitude.isna() | longitude.isna()
    at_zero = (latitude == 0) & (longitude == 0)
    in_box = latitude.between(*NYC_BOUNDS["lat"]) & longitude.between(*NYC_BOUNDS["lon"])
    status = np.select([missing, at_zero, ~in_box], ["missing", "at (0, 0)", "outside the city box"],
                       default="usable")
    return pd.Categorical(status, categories=["usable", "missing", "at (0, 0)", "outside the city box"])


eda_crashes = crash_view(raw_crashes)
eda_crash_profile = raw_profile(raw_crashes)
say(f"The crash table has **{raw_crashes.shape[0]:,} rows and {raw_crashes.shape[1]} columns**, all loaded "
    f"as `{raw_crashes.dtypes.astype(str).unique()[0]}`. Crash dates run from "
    f"{eda_crashes['crash_date'].min():%d %B %Y} to {eda_crashes['crash_date'].max():%d %B %Y}, and "
    f"`collision_id` has {eda_crash_profile.loc['collision_id', 'distinct']:,} distinct values.")
with pd.option_context("display.max_colwidth", 34):
    display(eda_crash_profile.round(2))
display(raw_crashes.head(3).T)

# %%
def crash_text_checks(raw, view):
    """Format findings in the raw crash text: time layout, padding, unconverted numbers."""
    time_lengths = raw["crash_time"].str.len().value_counts()
    non_numeric = {column: int((raw[column].notna() & view[column].isna()).sum())
                   for column in CRASH_CASUALTY_COLUMNS + ["latitude", "longitude"]}
    return {
        "time_unparsed": int(view["hour"].isna().sum()),
        "time_short": int(time_lengths.get(4, 0)),
        "time_midnight": int(as_flags(raw["crash_time"] == "0:00").sum()),
        "non_numeric": sum(non_numeric.values()),
        "casualty_missing": int(raw[CRASH_CASUALTY_COLUMNS].isna().to_numpy().sum()),
        "negative_counts": int((view[CRASH_CASUALTY_COLUMNS] < 0).to_numpy().sum()),
    }


eda_crash_text = crash_text_checks(raw_crashes, eda_crashes)
EDA_FACTS["street_padded"] = int(eda_crash_profile.loc["on_street_name", "padded with spaces"])
EDA_FACTS["time_short"] = eda_crash_text["time_short"]
say(f"`crash_time` has no leading zero in **{eda_crash_text['time_short']:,} rows "
    f"({pct(eda_crash_text['time_short'], len(raw_crashes)):.0f}%)**, for example `{raw_crashes['crash_time'].iloc[0]}`, "
    f"and {eda_crash_text['time_unparsed']} values do not fit the hour and minute layout. "
    f"{eda_crash_text['time_midnight']:,} crashes are timed at exactly midnight. `on_street_name` is padded with "
    f"spaces in **{EDA_FACTS['street_padded']:,} rows**. The casualty and coordinate columns contain "
    f"{eda_crash_text['non_numeric']} values that are not numbers, {eda_crash_text['casualty_missing']:,} "
    f"blank casualty cells and {eda_crash_text['negative_counts']} negative counts.")

# %% [markdown]
# `collision_id` is unique, so one row is one collision and the table has a ready key. The
# numeric fields convert without loss, apart from a handful of blank totals that Section 3.2.9
# comes back to. The time is written without a leading zero, so it cannot be sorted or sliced
# as text and Section 4 parses it into an hour. Street names carry trailing spaces, so the same
# street could count as different values if compared as loaded.
# The analysis locates crashes by coordinates and ZIP code and never by street name, so that
# column is not carried forward.

# %% [markdown]
# #### 3.2.2 Descriptive statistics
#
# Summary statistics of the eight casualty columns and the coordinates, over all loaded rows.

# %%
display(describe_table(eda_crashes[CRASH_CASUALTY_COLUMNS + ["latitude", "longitude"]], [0.5, 0.9, 0.99, 0.999]))

# %% [markdown]
# The casualty counts are zero at the median and still small at the 99th percentile, with
# maxima far above that, which is the shape of rare-event data. The coordinate rows give away
# a problem before any map is drawn: the minimum latitude and the maximum longitude are zero,
# values that lie on the equator off the coast of Africa and not in New York.

# %% [markdown]
# #### 3.2.3 Missing values that travel together
#
# The table gives the missing share of every crash column. The second table counts the most
# common combinations of four gaps (ZIP code, borough, usable coordinates and street name),
# because whether a crash can be placed depends on the combination, not on any single column.

# %%
def crash_missing_patterns(raw, view):
    """Share of crashes in each combination of missing ZIP, borough, coordinates and street."""
    flags = pd.DataFrame({
        "ZIP code": np.where(view["zip_missing"], "missing", "present"),
        "borough": np.where(view["borough_missing"], "missing", "present"),
        "coordinates": np.where(view["coordinate_status"] == "usable", "usable", "not usable"),
        "on_street_name": np.where(raw["on_street_name"].isna().to_numpy(), "missing", "present"),
    })
    patterns = flags.value_counts().rename("crashes").reset_index()
    patterns["share_pct"] = 100 * patterns["crashes"] / len(view)
    return patterns


display(missing_table(raw_crashes).query("missing > 0").round(2))
eda_patterns = crash_missing_patterns(raw_crashes, eda_crashes)
display(eda_patterns.round(2))

eda_zip_only = int((eda_crashes["zip_missing"] & ~eda_crashes["borough_missing"]).sum())
eda_borough_only = int((~eda_crashes["zip_missing"] & eda_crashes["borough_missing"]).sum())
eda_zip_missing = int(eda_crashes["zip_missing"].sum())
eda_recoverable = int((eda_crashes["zip_missing"] & (eda_crashes["coordinate_status"] == "usable")).sum())
EDA_FACTS["zip_missing"] = eda_zip_missing
EDA_FACTS["zip_recoverable"] = eda_recoverable
say(f"**{eda_zip_missing:,} crashes ({pct(eda_zip_missing, len(eda_crashes)):.1f}%) have no ZIP code.** "
    f"Borough is missing in the same rows: only {eda_zip_only:,} crashes have a borough without a ZIP code "
    f"and {eda_borough_only:,} a ZIP code without a borough. Of the crashes without a ZIP code, "
    f"**{eda_recoverable:,} ({pct(eda_recoverable, eda_zip_missing):.1f}%) have usable coordinates**, and "
    f"{eda_zip_missing - eda_recoverable:,} have neither.")

# %% [markdown]
# ZIP code and borough are filled in together or not at all, which points to one geocoding
# step (the step that assigns an address and ZIP code from the reported position) that either
# succeeds or fails for a crash. The largest incomplete pattern is a crash
# with no ZIP code and no borough but with good coordinates. That is the important case for
# integration. A join on ZIP code and year would lose a large share of all crashes, and the
# distance join would keep nearly all of them, so the two joins would describe different sets
# of crashes. It also shows the way out: where the coordinates are good, the ZIP code can be
# worked out from them. That recovery is done in Section 6, after the tables meet, because the
# size of the loss only becomes visible there.
#
# A constant loss would be a nuisance. The next chart checks whether it is constant.

# %%
def crash_gaps_by_year(view):
    """Per crash year: share with no ZIP, with unusable coordinates, and the overlap of the two."""
    usable = view["coordinate_status"] == "usable"
    table = pd.DataFrame({
        "crashes": view.groupby("year").size(),
        "no ZIP code (%)": view.groupby("year")["zip_missing"].mean() * 100,
        "no borough (%)": view.groupby("year")["borough_missing"].mean() * 100,
        "no usable coordinates (%)": (~usable).groupby(view["year"]).mean() * 100,
        "no ZIP code, usable coordinates (%)": (view["zip_missing"] & usable).groupby(view["year"]).mean() * 100,
        "no ZIP code, no usable coordinates (%)": (view["zip_missing"] & ~usable).groupby(view["year"]).mean() * 100,
        "coordinates at (0, 0) (%)": (view["coordinate_status"] == "at (0, 0)").groupby(view["year"]).mean() * 100,
    })
    return table


def plot_crash_gaps(table):
    """Lines of the location gaps per year; borough is left out because it coincides with ZIP."""
    label_shift_points = 7   # the two upper lines end close together, so their labels are pushed apart
    series = [("no ZIP code (%)", "no ZIP code or borough", PALETTE["crash"], "-", label_shift_points),
              ("no ZIP code, usable coordinates (%)", "no ZIP code, but usable coordinates", PALETTE["after"],
               "-", -label_shift_points),
              ("no usable coordinates (%)", "no usable coordinates", PALETTE["neutral"], "--", 0)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for column, label, color, style, shift in series:
        ax.plot(table.index, table[column], color=color, linestyle=style, linewidth=2, marker="o",
                markersize=5, label=label)
        end_label(ax, table.index[-1], table[column].iloc[-1], f"{table[column].iloc[-1]:.0f}%", shift)
    ax.set_xticks(table.index)
    ax.set_ylim(0, table["no ZIP code (%)"].max() * 1.25)
    ax.legend(loc="upper right", ncol=1)
    finish(ax, "Crashes: the share without a ZIP code changes from year to year",
           "crash year", "share of that year's crashes (%)")
    plt.tight_layout()
    plt.show()


eda_gaps_by_year = crash_gaps_by_year(eda_crashes)
EDA_FACTS["zip_missing_min_pct"] = eda_gaps_by_year["no ZIP code (%)"].min()
EDA_FACTS["zip_missing_max_pct"] = eda_gaps_by_year["no ZIP code (%)"].max()
plot_crash_gaps(eda_gaps_by_year)
display(eda_gaps_by_year.round(2))
say(f"The share of crashes without a ZIP code ranges from **{EDA_FACTS['zip_missing_min_pct']:.1f}% "
    f"({eda_gaps_by_year['no ZIP code (%)'].idxmin()}) to {EDA_FACTS['zip_missing_max_pct']:.1f}% "
    f"({eda_gaps_by_year['no ZIP code (%)'].idxmax()})**. The largest difference between the ZIP and borough "
    f"shares in any year is "
    f"{(eda_gaps_by_year['no ZIP code (%)'] - eda_gaps_by_year['no borough (%)']).abs().max():.2f} "
    f"percentage points.")

# %% [markdown]
# The gap is not constant. It moves by many percentage points across the ten years, so a count
# of crashes per ZIP code taken from the raw column would rise and fall with the geocoding
# rate and not only with what happened on the street. A ZIP by year panel built that way would
# carry an artificial time trend into every model that uses it.
#
# The green line is the part of the gap that coordinates can close, and it follows the red line
# closely. What remains, crashes with neither a ZIP code nor a usable position, is a much
# smaller group in every year. The first year stands apart: far more of its crashes lack
# usable coordinates than in any later year, so both joins see that year less completely.
# Dropping all crashes without a ZIP code was the simple alternative and is rejected for the
# reason just given: the loss is large and its size depends on the year.

# %% [markdown]
# #### 3.2.4 The 2020 reporting break
#
# The chart compares three yearly totals on one axis: all reported crashes, crashes in which at
# least one person was injured or killed, and the number of people injured. If the streets had
# simply become safer or emptier, the three lines would fall together.

# %%
def crashes_by_year(view):
    """Yearly totals of all crashes, injury crashes, persons injured and persons killed."""
    return pd.DataFrame({
        "all reported crashes": view.groupby("year").size(),
        "injury crashes": view.groupby("year")["is_injury_crash"].sum(),
        "persons injured": view.groupby("year")["number_of_persons_injured"].sum(),
        "persons killed": view.groupby("year")["number_of_persons_killed"].sum(),
    }).astype(int)


def plot_reporting_break(yearly):
    """Three yearly lines on one count axis, with the break year marked."""
    series = [("all reported crashes", PALETTE["neutral"]), ("persons injured", PALETTE["accent"]),
              ("injury crashes", PALETTE["crash"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    for column, color in series:
        ax.plot(yearly.index, yearly[column], color=color, linewidth=2, marker="o", markersize=5, label=column)
        end_label(ax, yearly.index[-1], yearly[column].iloc[-1], column)
    ax.axvline(BREAK_YEAR - 0.5, color=PALETTE["neutral"], linestyle=":", linewidth=1.2)
    ax.annotate(f"from {BREAK_YEAR}: far fewer reports\nof crashes without injury",
                xy=(BREAK_YEAR - 0.45, yearly["all reported crashes"].max() * 0.82), fontsize=9,
                color=PALETTE["neutral"])
    ax.set_xticks(yearly.index)
    ax.set_xlim(yearly.index[0] - 0.3, yearly.index[-1] + 0.3)
    ax.set_ylim(0, yearly["all reported crashes"].max() * 1.08)
    thousands(ax)
    ax.legend(loc="upper right")
    finish(ax, "Crashes: total reports halve in 2020, injury crashes fall far less",
           "crash year", "count per year")
    plt.tight_layout(rect=LABEL_MARGIN)
    plt.show()


eda_yearly = crashes_by_year(eda_crashes)
eda_yearly["injury crashes (% of all)"] = 100 * eda_yearly["injury crashes"] / eda_yearly["all reported crashes"]
plot_reporting_break(eda_yearly)
display(eda_yearly.round(1))
EDA_FACTS["all_change_pct"] = 100 * (eda_yearly.loc[YEAR_END, "all reported crashes"]
                                     / eda_yearly.loc[BREAK_YEAR - 1, "all reported crashes"] - 1)
EDA_FACTS["injury_change_pct"] = 100 * (eda_yearly.loc[YEAR_END, "injury crashes"]
                                        / eda_yearly.loc[BREAK_YEAR - 1, "injury crashes"] - 1)
say(f"Between {BREAK_YEAR - 1} and {YEAR_END}, all reported crashes change by "
    f"**{EDA_FACTS['all_change_pct']:+.0f}%** "
    f"({eda_yearly.loc[BREAK_YEAR - 1, 'all reported crashes']:,} to {eda_yearly.loc[YEAR_END, 'all reported crashes']:,}), "
    f"injury crashes by **{EDA_FACTS['injury_change_pct']:+.0f}%** "
    f"({eda_yearly.loc[BREAK_YEAR - 1, 'injury crashes']:,} to {eda_yearly.loc[YEAR_END, 'injury crashes']:,}). "
    f"The share of crashes with an injury or death goes from "
    f"{eda_yearly.loc[BREAK_YEAR - 1, 'injury crashes (% of all)']:.0f}% to "
    f"{eda_yearly.loc[YEAR_END, 'injury crashes (% of all)']:.0f}%.")

# %% [markdown]
# The lines do not fall together. Total reports drop sharply in 2020 and keep sliding, while
# injury crashes and persons injured dip in 2020, recover most of the way and then stay roughly
# flat. As a result the share of crashes with an injury rises sharply. A city in which crashes
# fell by far more than injuries is hard to believe. The simpler
# reading is that crashes with property damage only stopped being recorded at the earlier rate
# from 2020 on, so the total counts two different things before and after that year. The data
# cannot prove that reading, but it is the one consistent with all three lines.
#
# Any comparison of total crashes across 2020 would mostly measure the change in reporting, and
# a building completed in 2019 would appear to be followed by a large fall in crashes that has
# nothing to do with it. The decision is to use the injury crash as the outcome in every
# research question, since a crash in which someone was hurt is far less affected by the change in reporting. Total
# crashes appear again only as a robustness check. Restricting the study to 2016 to 2019 was the
# alternative. It would avoid the break but leave too few years after most completions.

# %% [markdown]
# #### 3.2.5 Time patterns: months, weekdays and hours
#
# The monthly series puts the break on a finer time scale and shows the seasonal rhythm that a
# before and after comparison must not mistake for an effect.

# %%
def crashes_by_month(view):
    """Monthly counts of all crashes and injury crashes."""
    month_start = view["crash_date"].dt.to_period("M").dt.to_timestamp()
    return pd.DataFrame({
        "all reported crashes": view.groupby(month_start).size(),
        "injury crashes": view.groupby(month_start)["is_injury_crash"].sum(),
    })


def plot_monthly_series(monthly):
    """Two monthly lines on one axis."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for column, color in [("all reported crashes", PALETTE["neutral"]), ("injury crashes", PALETTE["crash"])]:
        ax.plot(monthly.index, monthly[column], color=color, linewidth=1.8, label=column)
        end_label(ax, monthly.index[-1], monthly[column].iloc[-1], column)
    ax.set_ylim(0, monthly["all reported crashes"].max() * 1.08)
    ax.set_xlim(monthly.index[0], monthly.index[-1] + pd.DateOffset(months=1))
    thousands(ax)
    ax.legend(loc="upper right")
    finish(ax, "Crashes per month: the fall in total reports begins in spring 2020",
           "month of crash", "crashes per month")
    plt.tight_layout(rect=LABEL_MARGIN)
    plt.show()


eda_monthly = crashes_by_month(eda_crashes)
plot_monthly_series(eda_monthly)
eda_low_month = eda_monthly.loc[f"{BREAK_YEAR}", "all reported crashes"].idxmin()
say(f"The lowest month of {BREAK_YEAR} is {eda_low_month:%B} with "
    f"{eda_monthly.loc[eda_low_month, 'all reported crashes']:,} reported crashes, against "
    f"{eda_monthly.loc[eda_low_month - pd.DateOffset(years=1), 'all reported crashes']:,} in the same month a "
    f"year earlier. Injury crashes in those two months: "
    f"{eda_monthly.loc[eda_low_month, 'injury crashes']:,} and "
    f"{eda_monthly.loc[eda_low_month - pd.DateOffset(years=1), 'injury crashes']:,}.")

# %% [markdown]
# On the monthly scale the break is a cliff in spring 2020. Both series fall during the first
# lockdown, which is a real drop in traffic. Injury crashes then climb back within months and
# resume their yearly cycle, while total reports settle on a new, lower level and never return.
# A temporary shock would be expected to affect both lines and then fade; here only the total
# keeps its step.

# %%
def month_profile(view):
    """Average injury crashes per day in each calendar month, before and after the break year."""
    injury = view.loc[view["is_injury_crash"] & (view["year"] != BREAK_YEAR)]
    period = np.where(injury["year"] < BREAK_YEAR, f"{YEAR_START} to {BREAK_YEAR - 1}",
                      f"{BREAK_YEAR + 1} to {YEAR_END}")
    crashes = injury.groupby([period, injury["month"]]).size()
    days = injury.groupby([period, injury["month"]])["crash_date"].nunique()
    return (crashes / days).unstack(level=0)


def plot_month_profile(profile):
    """Seasonal profile: injury crashes per day by calendar month, one line per period."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for column, color in zip(profile.columns, [PALETTE["neutral"], PALETTE["crash"]]):
        ax.plot(profile.index, profile[column], color=color, linewidth=2, marker="o", markersize=5, label=column)
        end_label(ax, profile.index[-1], profile[column].iloc[-1], column)
    ax.set_xticks(profile.index)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_xlim(0.7, 12.3)
    ax.set_ylim(0, profile.to_numpy().max() * 1.15)
    finish(ax, "Injury crashes per day peak from late spring to early autumn",
           "calendar month", "injury crashes per day (average)",
           note=f"{BREAK_YEAR} is left out because of the lockdown months.")
    plt.tight_layout(rect=LABEL_MARGIN)
    plt.show()


eda_month_profile = month_profile(eda_crashes)
plot_month_profile(eda_month_profile)
say("Lowest and highest month by period: " + "; ".join(
    f"{period}: {MONTH_NAMES[column.idxmin() - 1]} {column.min():.0f} per day, "
    f"{MONTH_NAMES[column.idxmax() - 1]} {column.max():.0f} per day "
    f"({100 * (column.max() / column.min() - 1):.0f}% higher)"
    for period, column in eda_month_profile.items()) + ".")

# %% [markdown]
# Injury crashes follow the seasons in both periods: fewest in January and February, most from
# late spring to early autumn, when more people walk and cycle. The swing between the quietest
# and the busiest month is large enough to matter. A comparison of the six months before a completion with the six
# months after it would mix in this cycle, which is why the research questions compare whole
# years on each side.

# %%
def weekday_hour_table(view):
    """Average injury crashes in each weekday and hour slot, per calendar day of that weekday."""
    injury = view.loc[view["is_injury_crash"] & view["hour"].notna()]
    counts = injury.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    days_per_weekday = view.groupby("weekday")["crash_date"].nunique()
    table = counts.div(days_per_weekday, axis=0)
    table.index = WEEKDAY_NAMES
    table.columns = table.columns.astype(int)
    return table


def plot_weekday_hour(table):
    """Heatmap of injury crashes per hour slot, weekday by hour of day."""
    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    sns.heatmap(table, cmap=SEQUENTIAL_CMAP, linewidths=1.5, linecolor="white", ax=ax,
                cbar_kws={"label": "injury crashes per hour (average)"})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    finish(ax, "Injury crashes by weekday and hour: an afternoon peak, and late nights at the weekend",
           "hour of day the crash was reported to have happened (0 = midnight to 00:59)", "")
    plt.tight_layout()
    plt.show()


eda_weekday_hour = weekday_hour_table(eda_crashes)
plot_weekday_hour(eda_weekday_hour)
eda_hourly = eda_crashes.loc[eda_crashes["is_injury_crash"]].groupby("hour").size()
say(f"Across all days the busiest hour for injury crashes is **{int(eda_hourly.idxmax())}:00** "
    f"({pct(eda_hourly.max(), eda_hourly.sum()):.1f}% of injury crashes) and the quietest "
    f"{int(eda_hourly.idxmin())}:00 ({pct(eda_hourly.min(), eda_hourly.sum()):.1f}%). The busiest single slot "
    f"is {eda_weekday_hour.stack().idxmax()[0]} at {eda_weekday_hour.stack().idxmax()[1]}:00. The hour after "
    f"midnight holds {pct(eda_hourly.get(0.0, 0), eda_hourly.sum()):.1f}% of injury crashes, and "
    f"{pct(eda_crash_text['time_midnight'], len(eda_crashes)):.1f}% of all crashes are timed at exactly 0:00.")

# %% [markdown]
# Weekdays show a build-up through the day to a peak in the late afternoon and early evening,
# with a smaller morning step. Weekends lack the morning step and carry more crashes after
# midnight. The first column is slightly brighter than its neighbours, and part of that is an
# artefact: a crash whose time was not known tends to be entered as midnight. Section 4 keeps
# the hour as reported, and the research question on time of day works with broad bands of
# hours so that this heap at midnight cannot drive a result.

# %% [markdown]
# #### 3.2.6 Where crashes are, and coordinates that are not in New York
#
# The map is a hexbin density of the crashes inside the bounding box of the city, on a log
# colour scale because a few Manhattan hexagons hold far more crashes than most of the city.
# The coordinate pairs that fall outside the box cannot be drawn on a city map without
# stretching it across half the globe, so they are counted in the table and sentence below the
# map and drawn in Section 4.8, where they are removed.

# %%
def plot_crash_coordinates(view):
    """Hexbin density of the usable coordinates; returns the count of each coordinate status."""
    status_counts = view["coordinate_status"].value_counts()
    usable = view.loc[view["coordinate_status"] == "usable"]
    fig, ax = plt.subplots(figsize=(8, 6.6))
    hexes = ax.hexbin(usable["longitude"], usable["latitude"], gridsize=HEXBIN_GRIDSIZE, bins="log",
                      cmap=SEQUENTIAL_CMAP, mincnt=1, linewidths=0)
    map_axes(ax)
    fig.colorbar(hexes, ax=ax, label="crashes per hexagon (log scale)", shrink=0.8)
    finish(ax, "Crashes inside the city box trace the street network", "longitude (degrees east)",
           "latitude (degrees north)")
    plt.tight_layout()
    plt.show()
    return status_counts


eda_coordinate_counts = plot_crash_coordinates(eda_crashes)
EDA_FACTS["coords_missing"] = int(eda_coordinate_counts["missing"])
EDA_FACTS["coords_zero"] = int(eda_coordinate_counts["at (0, 0)"])
EDA_FACTS["coords_outside"] = int(eda_coordinate_counts["outside the city box"])
display(pd.DataFrame({"crashes": eda_coordinate_counts,
                      "share_pct": 100 * eda_coordinate_counts / len(eda_crashes)}).round(3))
say(f"**{EDA_FACTS['coords_missing']:,} crashes ({pct(EDA_FACTS['coords_missing'], len(eda_crashes)):.1f}%) "
    f"have no coordinates, {EDA_FACTS['coords_zero']:,} sit at exactly (0, 0)** and "
    f"{EDA_FACTS['coords_outside']:,} more fall outside the city box. The share at (0, 0) is highest in "
    f"{eda_gaps_by_year['coordinates at (0, 0) (%)'].idxmax()} "
    f"({eda_gaps_by_year['coordinates at (0, 0) (%)'].max():.2f}% of that year's crashes) and lowest in "
    f"{eda_gaps_by_year['coordinates at (0, 0) (%)'].idxmin()} "
    f"({eda_gaps_by_year['coordinates at (0, 0) (%)'].min():.2f}%).")

# %% [markdown]
# The point at (0, 0) is a placeholder for "location unknown" written as a number. It is more
# dangerous than a blank because it passes every check for missing values and would enter a
# distance calculation as a real place several thousand kilometres from any building. The other
# stray points are isolated typing or geocoding errors. Section 4 sets all of them to missing
# and flags the row. The row itself is kept, since a crash with a bad position can still have a
# valid ZIP code and still counts in the ZIP by year panel. Deleting such rows was rejected for
# that reason.
#
# Inside the box no cleaning is needed for the density map to draw the city: the bridges,
# avenues and expressways are all recognisable, and density is highest in Manhattan below
# Central Park, downtown Brooklyn and along the main arterial roads. The areas where the
# housing map was dense (the South Bronx, upper Manhattan, central Brooklyn) are also areas of
# high crash density. The overlap matters for the analysis. Affordable housing is built in
# dense neighbourhoods and dense neighbourhoods have more crashes, so a raw count near
# buildings says little without a comparison to similar streets.

# %% [markdown]
# #### 3.2.7 ZIP codes and boroughs as written
#
# The crash ZIP code is checked for values that are present but not usable, and the borough
# spellings of the two tables are put side by side, since ZIP code and borough are the keys of
# the coarser join.

# %%
def zip_status_table(raw):
    """Classify every raw crash ZIP code by what is wrong with it, if anything."""
    text = raw["zip_code"]
    stripped = text.str.strip()
    status = np.select(
        [text.isna().to_numpy(), as_flags(stripped == ""), as_flags(stripped == ZIP_PLACEHOLDER),
         ~as_flags(stripped.str.match(ZIP_PATTERN))],
        ["missing", "blank spaces", f"placeholder {ZIP_PLACEHOLDER}", "not five digits"],
        default="five digits")
    table = pd.Series(status).value_counts().rename("crashes").to_frame()
    table["share_pct"] = 100 * table["crashes"] / len(raw)
    return table


def borough_spellings(housing, crashes):
    """Every distinct borough value of the two raw tables, with its row count in each."""
    counts = pd.DataFrame({
        "housing rows": housing["borough"].astype(object).fillna("(missing)").value_counts(),
        "crash rows": crashes["borough"].astype(object).fillna("(missing)").value_counts(),
    }).fillna(0).astype(int)
    return counts.rename_axis("borough as written").sort_index(key=lambda labels: labels.str.lower())


eda_zip_status = zip_status_table(raw_crashes)
EDA_FACTS["zip_junk"] = int(eda_zip_status["crashes"].drop(["missing", "five digits"], errors="ignore").sum())
display(eda_zip_status.round(3))
eda_zip_counts = raw_crashes["zip_code"].str.strip().value_counts()
eda_housing_zips = set(raw_housing["postcode"].dropna())
display(eda_zip_counts.tail(8).rename("crashes").rename_axis("rarest ZIP code values").reset_index()
        .assign(in_housing_table=lambda table: table["rarest ZIP code values"].isin(eda_housing_zips)))
display(borough_spellings(raw_housing, raw_crashes))
eda_shared_boroughs = set(raw_housing["borough"].dropna()) & set(raw_crashes["borough"].dropna())
eda_shared_zips = eda_housing_zips & set(eda_zip_counts.index)
EDA_FACTS["shared_borough_spellings"] = len(eda_shared_boroughs)
eda_zip_junk = eda_zip_status["crashes"].drop(["missing", "five digits"], errors="ignore")
eda_zip_junk_kinds = (f"all of them classed as {eda_zip_junk.index[0]}" if len(eda_zip_junk) == 1
                      else "namely " + join_and(f"{count:,} {kind}" for kind, count in eda_zip_junk.items()))
say(f"**{EDA_FACTS['zip_junk']:,} crash ZIP codes are present but unusable**, {eda_zip_junk_kinds}. "
    f"The crash table has {len(eda_zip_counts):,} distinct ZIP values; "
    f"{len(eda_shared_zips)} of the {len(eda_housing_zips)} housing ZIP codes also occur there. Compared as "
    f"written, **{len(eda_shared_boroughs)} borough names match** between the two tables.")

# %% [markdown]
# A small number of ZIP codes are filled in with something that is not a ZIP code. In this copy
# they are all one placeholder value. It is awkward because it has five digits and starts like a
# Manhattan ZIP code, so a format check alone lets it through and it has to be named explicitly.
# The rarest values are worth a glance as well, because a ZIP code with a handful of crashes in
# ten years is either a border case outside the city or a typing error. Section 4 sets the unusable values to missing so that they
# join the group whose ZIP code is recovered from coordinates in Section 6.
#
# Borough needs no repair of content, only of form. The crash table writes it in capitals and
# the housing table in title case, so not one borough name matches as written. Without
# standardising the case a join on borough would return nothing and raise no error.

# %% [markdown]
# #### 3.2.8 Injuries per crash and the extreme values
#
# Persons injured per crash is a count with most of its mass at zero, so each count is drawn as a
# dot on a log axis. On a plain axis only the first two values would be visible and the long
# tail, which is where the outliers are, would disappear. Dots are used instead of bars because
# the length of a bar means nothing when the axis is logarithmic. The dotted line marks the
# threshold at which Section 4.10 flags a crash as extreme.

# %%
def plot_injuries_per_crash(view):
    """Dot per value of persons injured on a log count axis, with the extreme-crash threshold marked."""
    counts = view["number_of_persons_injured"].dropna().astype(int).value_counts().sort_index()
    log_baseline = 0.5    # a log axis needs a positive floor; stems start just below one crash
    colors = [PALETTE["placebo"] if value > EXTREME_INJURED else PALETTE["crash"] for value in counts.index]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.vlines(counts.index, log_baseline, counts.to_numpy(), color=colors, linewidth=1.5)
    ax.scatter(counts.index, counts.to_numpy(), s=30, color=colors, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(log_baseline, counts.max() * 2)
    thousands(ax)
    ax.axvline(EXTREME_INJURED + 0.5, color=PALETTE["neutral"], linestyle=":", linewidth=1.2)
    ax.annotate(f"flag threshold: more than {EXTREME_INJURED} injured\n"
                f"({int(counts[counts.index > EXTREME_INJURED].sum()):,} crashes)",
                xy=(EXTREME_INJURED + 1, 0.8), xycoords=("data", "axes fraction"), fontsize=9,
                color=PALETTE["neutral"])
    ax.grid(axis="x", visible=False)
    finish(ax, "Crashes: most injure nobody, a few injure dozens",
           "persons injured in the crash", "number of crashes (log scale)")
    plt.tight_layout()
    plt.show()
    return counts


eda_injury_counts = plot_injuries_per_crash(eda_crashes)
EDA_FACTS["many_injured"] = int(eda_injury_counts[eda_injury_counts.index > EXTREME_INJURED].sum())
eda_killed_counts = eda_crashes["number_of_persons_killed"].dropna().astype(int).value_counts().sort_index()
say(f"**{pct(eda_injury_counts.get(0, 0), eda_injury_counts.sum()):.1f}% of crashes injure nobody**, "
    f"{pct(eda_injury_counts.get(1, 0), eda_injury_counts.sum()):.1f}% injure one person and "
    f"{pct(eda_injury_counts[eda_injury_counts.index >= 2].sum(), eda_injury_counts.sum()):.1f}% two or more. "
    f"The maximum is {eda_injury_counts.index.max()} injured in one crash, and {EDA_FACTS['many_injured']:,} "
    f"crashes have more than {EXTREME_INJURED}. {int(eda_killed_counts[eda_killed_counts.index >= 1].sum()):,} "
    f"crashes involve a death, at most {eda_killed_counts.index.max()} in one crash. "
    f"{int(eda_crashes['is_injury_crash'].sum()):,} crashes "
    f"({pct(int(eda_crashes['is_injury_crash'].sum()), len(eda_crashes)):.1f}%) count as injury crashes.")
display(raw_crashes.loc[eda_crashes["number_of_persons_injured"].nlargest(5).index,
                        ["crash_date", "borough", "on_street_name", "number_of_persons_injured",
                         "number_of_motorist_injured", "vehicle_type_code1"]])

# %% [markdown]
# On the log axis the counts fall in a nearly straight line, the mark of a heavy-tailed count
# with no natural cut-off. A rule based on the interquartile range cannot separate errors from
# real events here. Most crashes injure nobody, so both quartiles sit at the bottom of the scale
# and the fence falls between two and three injured people, so it would flag ordinary
# multi-injury crashes, not errors (Section 4.10 gives the numbers). The extreme rows look like
# genuine events, typically a bus or several vehicles with many injured occupants.
#
# These crashes are kept. They barely affect the main outcome, because an injury crash counts
# once however many people were hurt in it. They do matter for sums of persons injured, where a
# single bus crash can change the total of a small area, so Section 4 flags crashes above a
# domain threshold and the analyses of severity can be repeated without them. Capping the value
# was the alternative, and it would replace a recorded number with an invented one.

# %% [markdown]
# #### 3.2.9 Does persons injured equal pedestrians plus cyclists plus motorists?
#
# The table has a total and three parts. If the total always equalled the sum of the parts, one
# of the four columns would be redundant and the pedestrian and cyclist share could be computed
# from either. The check below tests that row by row and by year.

# %%
def injury_sum_check(view):
    """Compare persons injured with the sum of its three parts, row by row.

    A few totals are blank; those rows cannot be compared and are counted separately.
    """
    parts = view[CRASH_INJURY_PARTS].sum(axis=1)
    total_known = view["number_of_persons_injured"].notna()
    difference = (view["number_of_persons_injured"] - parts).where(total_known)
    by_year = pd.DataFrame({
        "injury crashes": view.groupby("year")["is_injury_crash"].sum(),
        "total above parts": (difference > 0).groupby(view["year"]).sum(),
        "total below parts": (difference < 0).groupby(view["year"]).sum(),
    })
    by_year["mismatch_pct_of_injury_crashes"] = (
        100 * (by_year["total above parts"] + by_year["total below parts"]) / by_year["injury crashes"])
    return difference, by_year, int((~total_known).sum())


def mismatch_vehicle_table(raw, is_mismatch, is_injury_crash):
    """First vehicle types most common among mismatched crashes, against their share of all injury crashes."""
    vehicle = raw["vehicle_type_code1"].str.strip().str.lower().astype(object).fillna("(missing)")
    among_mismatch = vehicle[is_mismatch.to_numpy()].value_counts(normalize=True).mul(100)
    among_injury = vehicle[is_injury_crash.to_numpy()].value_counts(normalize=True).mul(100)
    table = pd.DataFrame({"share of mismatched crashes (%)": among_mismatch.head(6)})
    table["share of all injury crashes (%)"] = among_injury.reindex(table.index).fillna(0)
    return table.rename_axis("first vehicle type (lower-cased)")


eda_injury_difference, eda_mismatch_by_year, eda_blank_totals = injury_sum_check(eda_crashes)
eda_is_mismatch = eda_injury_difference.fillna(0).ne(0)
EDA_FACTS["injury_mismatch"] = int(eda_is_mismatch.sum())
EDA_FACTS["casualty_blank_cells"] = eda_crash_text["casualty_missing"]
display(eda_mismatch_by_year.round(2))
display(eda_injury_difference.dropna().astype(int).value_counts().sort_index().rename("crashes")
        .rename_axis("persons injured minus (pedestrians + cyclists + motorists)").to_frame().T)
display(mismatch_vehicle_table(raw_crashes, eda_is_mismatch, eda_crashes["is_injury_crash"]).round(1))
say(f"The total differs from the sum of its parts in **{EDA_FACTS['injury_mismatch']:,} crashes**: higher in "
    f"{int((eda_injury_difference > 0).sum()):,} and lower in {int((eda_injury_difference < 0).sum()):,}. "
    f"The yearly share among injury crashes runs from "
    f"{eda_mismatch_by_year['mismatch_pct_of_injury_crashes'].min():.2f}% "
    f"({eda_mismatch_by_year['mismatch_pct_of_injury_crashes'].idxmin()}) to "
    f"{eda_mismatch_by_year['mismatch_pct_of_injury_crashes'].max():.2f}% "
    f"({eda_mismatch_by_year['mismatch_pct_of_injury_crashes'].idxmax()}). The total is blank in "
    f"{eda_blank_totals} crashes, which are left out of this comparison.")

# %% [markdown]
# The identity does not hold, and the pattern has two parts (Section 4.9 charts the yearly
# counts). In the first years of the window there is a small number of mismatches in both
# directions, which look like entry errors. After two years with almost none, the mismatches
# return at a much higher rate and in one direction only: the total is above the parts, usually by one. E-scooters and e-bikes are the
# first vehicle in a far larger share of these crashes than of injury crashes in general. That
# is consistent with riders of those devices being counted as injured persons without fitting
# any of the three older categories, although the table does not say so directly.
#
# For cleaning this rules out the obvious repair. Overwriting the total with the sum of the
# parts would delete injured people from the recent years only, which would add a second
# artificial break to the one found in 3.2.4. Section 4 keeps a total that is above its parts
# and flags the row, raises the few totals that are below their parts, and defines an injury
# crash from all casualty columns together. The pedestrian and cyclist share in the research
# questions is computed from the parts, so it leaves these riders out. That is a stated limit of
# the analysis, because the data gives no way to assign them to a group.

# %% [markdown]
# #### 3.2.10 Contributing factors and their spellings
#
# The first contributing factor is free of typing variation only if it comes from a fixed list.
# The chart shows the most frequent values as shares of all crashes, with the values that carry
# no information in grey. The table after it lists labels that differ only by case, spacing or
# a letter or two, and labels that are bare numbers.

# %%
def factor_counts(raw):
    """Share of crashes for each raw value of the first contributing factor (missing included)."""
    counts = raw["contributing_factor_vehicle_1"].astype(object).fillna("(missing)").value_counts()
    return pd.DataFrame({"crashes": counts, "share_pct": 100 * counts / len(raw)})


def plot_top_factors(factors):
    """Horizontal bars of the most frequent contributing factors, uninformative ones in grey."""
    top = factors.head(TOP_FACTORS).iloc[::-1]
    uninformative = top.index.isin(["Unspecified", "(missing)"])
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.barh(top.index, top["share_pct"], height=0.72,
            color=np.where(uninformative, PALETTE["before"], PALETTE["crash"]))
    for label, value in top["share_pct"].items():
        ax.annotate(f"{value:.1f}%", xy=(value, label), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=PALETTE["neutral"])
    color_legend(ax, {"a stated factor": PALETTE["crash"], "no factor stated": PALETTE["before"]},
                 loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top["share_pct"].max() * 1.12)
    finish(ax, f"Crashes: the {TOP_FACTORS} most frequent first contributing factors",
           "share of all crashes (%)", "")
    plt.tight_layout()
    plt.show()


def factor_spelling_variants(factors):
    """Pairs of factor labels that are near-duplicates, and labels that are only digits."""
    labels = [label for label in factors.index if label != "(missing)"]
    rows = []
    for position, label in enumerate(labels):
        for other in labels[position + 1:]:
            similarity = difflib.SequenceMatcher(None, label.lower().strip(), other.lower().strip()).ratio()
            if similarity >= SPELLING_SIMILARITY:
                rows.append({"label": label, "crashes": int(factors.loc[label, "crashes"]),
                             "near-duplicate": other, "its crashes": int(factors.loc[other, "crashes"]),
                             "similarity": round(similarity, 2)})
    numeric = factors.loc[[label for label in labels if label.strip().isdigit()], "crashes"]
    return pd.DataFrame(rows), numeric


eda_factors = factor_counts(raw_crashes)
EDA_FACTS["factor_unspecified_pct"] = float(eda_factors["share_pct"].get("Unspecified", 0.0))
EDA_FACTS["factor_missing"] = int(eda_factors["crashes"].get("(missing)", 0))
plot_top_factors(eda_factors)
eda_factor_variants, eda_numeric_factors = factor_spelling_variants(eda_factors)
EDA_FACTS["factor_variant_pairs"] = len(eda_factor_variants)
display(eda_factor_variants)
display(eda_numeric_factors.rename("crashes").rename_axis("factor label that is only digits").reset_index())
eda_injury_unspecified = pct(
    int(as_flags(raw_crashes["contributing_factor_vehicle_1"] == "Unspecified")[eda_crashes["is_injury_crash"].to_numpy()].sum()),
    int(eda_crashes["is_injury_crash"].sum()))
say(f"The column has **{len(eda_factors) - int('(missing)' in eda_factors.index)} distinct labels**. "
    f"**\"Unspecified\" accounts for {EDA_FACTS['factor_unspecified_pct']:.1f}% of all crashes** and "
    f"{eda_injury_unspecified:.1f}% of injury crashes, and {EDA_FACTS['factor_missing']:,} crashes have no "
    f"value. The five most frequent stated factors cover "
    f"{eda_factors.drop(['Unspecified', '(missing)'], errors='ignore')['share_pct'].head(5).sum():.1f}% of "
    f"crashes. {len(eda_factor_variants)} pairs of labels are near-duplicates and "
    f"{len(eda_numeric_factors)} labels are bare numbers "
    f"({int(eda_numeric_factors.sum())} crashes).")

# %% [markdown]
# "Unspecified" is as frequent as the most common stated factor, so it is kept as a category of
# its own. Guessing a factor for those crashes from the other columns was considered and
# rejected, because whether an officer names a factor is likely related to how serious the
# crash was, so the guesses would be biased in an unknown direction. "Unspecified" therefore
# stays as its own group, and the research question on contributing factors works with shares
# among crashes that do state one.
#
# The stated factors come from a list, yet the same factor
# appears with different capitalisation, one label has a typing error, and a few crashes carry
# a numeric code where the label should be. Each variant is rare, yet left alone it would split
# one factor across two rows of every table. Section 4 standardises the spellings and folds the
# long tail of rare factors into a small number of groups, since the full list of labels is too
# long to compare around individual buildings.

# %% [markdown]
# #### 3.2.11 Duplicates
#
# `collision_id` is unique, but the same collision could still have been entered twice under two
# identifiers. The check counts rows that agree on date, time, position, street and every
# casualty count.

# %%
def crash_duplicate_checks(raw):
    """Count repeated collision ids and rows identical in everything that describes the event."""
    event_columns = ["crash_date", "crash_time", "latitude", "longitude", "on_street_name",
                     "contributing_factor_vehicle_1", "vehicle_type_code1"] + CRASH_CASUALTY_COLUMNS
    located = raw.loc[raw["latitude"].notna().to_numpy()]
    return {
        "repeated_ids": int(raw["collision_id"].duplicated().sum()),
        "located_rows": len(located),
        "same_event_rows": int(located.duplicated(event_columns).sum()),
    }


eda_crash_duplicates = crash_duplicate_checks(raw_crashes)
EDA_FACTS["crash_same_event"] = eda_crash_duplicates["same_event_rows"]
say(f"`collision_id` repeats **{eda_crash_duplicates['repeated_ids']} times**. Among the "
    f"{eda_crash_duplicates['located_rows']:,} crashes with coordinates, "
    f"**{eda_crash_duplicates['same_event_rows']:,} rows "
    f"({pct(eda_crash_duplicates['same_event_rows'], eda_crash_duplicates['located_rows']):.3f}%)** repeat an "
    f"earlier row in date, time, position, street, factor, vehicle type and all casualty counts.")

# %% [markdown]
# No identifier repeats, so there are no exact duplicates to remove. The rows that match on
# everything else are a tiny fraction and cannot be told apart from two separate minor crashes
# at a busy junction within the same minute, which is not unusual in ten years of data.
# Removing them would mean deleting records on a guess, so Section 4 flags them as possible
# duplicates and leaves them in.

# %% [markdown]
# ### 3.3 What the exploration tells us to do
#
# The table collects the problems found above, with their size as measured in this section, and
# names the cleaning action each one leads to. Sections 4 and 6 carry the actions out and
# record them in the cleaning log.

# %%
def exploration_summary(facts, housing_rows, crash_rows):
    """One row per observed problem: its measured size and the cleaning action it leads to."""
    rows = [
        ("Housing", "Unit columns are blank where the count is zero",
         f"{facts['unit_blank_cells']:,} blank cells, no written zero",
         "Impute: fill with 0 and check that the parts add up to the total", "Section 4"),
        ("Housing", "Confidential rows have no address, ZIP code or coordinates",
         f"{facts['confidential_rows']:,} rows ({pct(facts['confidential_rows'], housing_rows):.1f}%), "
         f"{facts['confidential_unit_pct']:.2f}% of units",
         "Flag: keep for borough totals, leave out of both joins, no location imputed", "Section 4"),
        ("Housing", "Published address but no coordinates",
         f"{facts['other_unlocated']} rows", "Flag: cannot enter the distance join", "Section 4"),
        ("Housing", "No completion date",
         f"{facts['no_completion_rows']:,} rows ({pct(facts['no_completion_rows'], housing_rows):.1f}%)",
         "Flag as not yet completed; not used in before and after comparisons", "Section 4"),
        ("Housing", "Completion date before the start date",
         f"{facts['completion_before_start']} rows", "Flag; dates left as published", "Section 4"),
        ("Housing", "Completion date equal to the start date",
         f"{facts['completion_on_start']:,} rows",
         "Flag; the completion date alone marks the event", "Section 4"),
        ("Housing", "Same building under more than one project",
         f"{facts['repeated_building_rows']} extra rows",
         "Keep; the key is (project_id, building_id), not building_id", "Section 4"),
        ("Housing", "Building size is right-skewed",
         f"{facts['size_above_fence']:,} rows ({facts['size_above_fence_pct']:.1f}%) above the IQR fence",
         "Keep all rows; flag very large buildings, use size bands and medians", "Section 4"),
        ("Both", "Borough written in different letter case",
         f"{facts['shared_borough_spellings']} names match as written", "Standardise to one spelling", "Section 4"),
        ("Both", "Dates, numbers and ZIP codes loaded as text",
         "every column", "Standardise: explicit types, ZIP code kept as five-character text", "Section 4"),
        ("Crashes", "ZIP code and borough missing together",
         f"{facts['zip_missing']:,} crashes ({pct(facts['zip_missing'], crash_rows):.1f}%), "
         f"{facts['zip_missing_min_pct']:.0f}% to {facts['zip_missing_max_pct']:.0f}% by year",
         f"Impute from coordinates where usable ({pct(facts['zip_recoverable'], facts['zip_missing']):.0f}% "
         f"of these rows), validated on a holdout", "Section 6"),
        ("Crashes", "ZIP code present but not a ZIP code",
         f"{facts['zip_junk']:,} crashes", "Set to missing, then treated like the rows above", "Section 4"),
        ("Crashes", "Coordinates missing",
         f"{facts['coords_missing']:,} crashes ({pct(facts['coords_missing'], crash_rows):.1f}%)",
         "Flag: kept for the ZIP level if a ZIP code exists, left out of the distance join", "Section 4"),
        ("Crashes", "Coordinates at (0, 0) or outside the city",
         f"{facts['coords_zero']:,} at (0, 0), {facts['coords_outside']:,} outside the box",
         "Set coordinates to missing and flag; the row is kept", "Section 4"),
        ("Crashes", f"Reports of crashes without injury fall from {BREAK_YEAR}",
         f"all crashes {facts['all_change_pct']:+.0f}%, injury crashes {facts['injury_change_pct']:+.0f}% "
         f"({BREAK_YEAR - 1} to {YEAR_END})",
         "Use the injury crash as the outcome; all crashes only as a robustness check", "Sections 4 to 9"),
        ("Crashes", "crash_time without a leading zero",
         f"{facts['time_short']:,} rows", "Standardise: parse to an hour of day", "Section 4"),
        ("Crashes", "Street names padded with spaces",
         f"{facts['street_padded']:,} rows",
         "Column not carried forward; crashes are located by coordinates and ZIP code", "Section 4"),
        ("Crashes", "Contributing factor not stated",
         f"\"Unspecified\" in {facts['factor_unspecified_pct']:.1f}% of crashes, {facts['factor_missing']:,} blank",
         "Keep \"Unspecified\" as its own group, blanks join it; no factor is guessed", "Section 4"),
        ("Crashes", "Contributing factor spelled in several ways",
         f"{facts['factor_variant_pairs']} near-duplicate pairs",
         "Standardise spelling and group the rare factors", "Section 4"),
        ("Crashes", "Persons injured differs from the sum of its parts",
         f"{facts['injury_mismatch']:,} crashes",
         "Flag; a total above its parts is kept, a total below them is raised to their sum", "Section 4"),
        ("Crashes", "Persons injured or persons killed blank",
         f"{facts['casualty_blank_cells']} cells; the six part columns are never blank",
         "Impute the total from its parts", "Section 4"),
        ("Crashes", f"More than {EXTREME_INJURED} persons injured in one crash",
         f"{facts['many_injured']:,} crashes", "Flag as extreme; keep the recorded value", "Section 4"),
        ("Crashes", "Rows identical except for the identifier",
         f"{facts['crash_same_event']:,} crashes", "Flag as possible duplicates; not removed", "Section 4"),
    ]
    return pd.DataFrame(rows, columns=["dataset", "problem observed", "size in the raw data",
                                       "cleaning action", "handled in"])


with pd.option_context("display.max_colwidth", None, "display.max_rows", 40):
    display(exploration_summary(EDA_FACTS, len(raw_housing), len(raw_crashes)))

# %% [markdown]
# Most of the later design follows from three of these rows. The outcome is the injury crash,
# because total crash counts stop being comparable in 2020. The ZIP code of a
# crash has to be recovered from its coordinates, because the share without one is large and
# changes from year to year. And the confidential housing rows can only be used at borough
# level, although they hold very little of the housing.

# %%
# The exploration views are temporary: free them so that no later section can depend on them.
# Names are found by their prefix, so renaming a view cannot break this cell and running it twice is safe.
release(*[name for name in list(globals()) if name.startswith("eda_")])
assert (raw_housing.shape, raw_crashes.shape) == EDA_RAW_SHAPES, "a raw frame was modified during exploration"
