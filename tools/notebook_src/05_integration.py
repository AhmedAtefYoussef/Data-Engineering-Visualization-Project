# %% [markdown]
# ## 5. Integration
#
# The two cleaned tables describe different things. A housing row is a building in a subsidised
# project and a crash row is one collision. This section builds the tables that let a housing
# variable and a crash variable be read side by side. It does so at two levels: a ZIP by year
# panel, where both tables are first reduced to the same grain, and a spatial level, where a crash
# is linked to the buildings within `BUFFER_M` metres of it.
#
# ### 5.1 The problem: no shared key
#
# `collision_id` identifies a crash and (`project_id`, `building_id`) identifies a housing row.
# Neither identifier appears in the other table, so there is nothing to join row to row. What the
# tables do share is where and when. Four bridges are available, and each gives a different grain.
#
# | Bridge | Grain it gives | What it can answer | What it cannot answer |
# |---|---|---|---|
# | Time (year) | city by year | whether the two series move together over ten years | anything about place: ten points, and both series follow the city as a whole |
# | Borough | borough by year | broad contrasts between five areas; the only level that includes confidential housing rows | neighbourhood differences: a borough holds hundreds of thousands of crashes and hundreds of buildings |
# | ZIP code | ZIP by year | change within a neighbourhood over time, comparison across neighbourhoods | what happens on the streets next to one building; needs a ZIP on both sides |
# | Coordinates | building by year, and crash by nearest building | crashes on the streets around a building before and after it is completed | anything for records without coordinates; neighbouring buildings overlap |
#
# Time and borough are too coarse to carry the analysis alone, so they are used as controls and
# for reference totals. The ZIP and the coordinates are the two working bridges. The cell below
# measures how much of each table every bridge can reach.

# %%
# --- Constants for Section 5 ------------------------------------------------------------------
N_STUDY_YEARS = len(STUDY_YEARS)
BYTES_PER_GB = 1_000_000_000
BYTES_PER_FLOAT = 8                 # one float64 distance
SQUARE_METRES_PER_HECTARE = 10_000
# The merge demo counts rows per (ZIP, year) with one integer key, zip * scale + year. A year is
# always below the scale, so every (ZIP, year) pair gets its own number, and counting integers is
# much cheaper than counting 1.5 million joined text keys.
YEAR_KEY_SCALE = 10_000
NO_YEAR_KEY = 0                     # stands for "no completion year" in that key

# Radii (metres) compared when judging the 250 m buffer; the same list is reused in Section 9.
RADIUS_CANDIDATES_M = [100, 150, 250, 400, 500]

# The many-to-many merge is run for real on one ZIP; it needs enough housing rows to show the effect.
DEMO_MIN_HOUSING_ROWS = 10

# Brute-force check of the spatial index: sample sizes small enough for a full distance matrix.
CHECK_SITE_SAMPLE = 300             # buildings
CHECK_CRASH_SAMPLE = 20_000         # crashes

# Alternative resampling unit for Section 9: buildings grouped by a fixed square of this side.
# One kilometre is four buffer radii, so two buildings in non-adjacent squares never share a crash.
BLOCK_SIDE_M = 1000

CLUSTER_SIZE_EDGES = [0, 1, 5, 20, 100, np.inf]          # buildings per cluster, upper edge inclusive
CLUSTER_SIZE_LABELS = ["1", "2-5", "6-20", "21-100", "101+"]

ROW_MERGE_AXIS_START = 1_000        # rows; left end of the log axis of the merge chart
ROW_MERGE_AXIS_HEADROOM = 20        # the log axis of the merge chart extends this factor past the largest value
ROW_MERGE_LABEL_GAP = 1.3           # a value label sits this factor to the right of its dot (log axis)
AXIS_HEADROOM = 1.15                # linear axes extend this factor past the largest bar so its label fits
LINE_HEADROOM = 1.18                # same, for the line charts of the radius comparison
LABEL_GAP = 0.015                   # gap between a bar and its value label, as a share of the largest bar

EVENT_GROUPS = ["treated", "comparison"]
# Relation of a crash to its nearest housing site, in the order used by every later table.
HOUSING_RELATIONS = ["No coordinates", "Not near a site", "Before completion", "Completion year",
                     "After completion"]
RELATION_COLORS = {"Not near a site": PALETTE["neutral"], "Before completion": PALETTE["control"],
                   "Completion year": PALETTE["accent"], "After completion": PALETTE["after"]}

SITE_ATTRIBUTES = ["housing_id", "project_id", "building_id", "project_name", "borough", "zip", "lat", "lon",
                   "start_year", "completion_year", "is_completed", "same_day_completion",
                   "completion_before_start", "construction_type", "total_units", "all_counted_units",
                   "deep_affordable_units", "deep_affordable_share", "size_band", "is_size_outlier"]
SITE_COUNT_COLUMNS = ["crashes", "injury_crashes", "ped_cyc_crashes", "persons_injured"]
HOUSING_PANEL_MEASURES = ["buildings_completed", "units_completed", "affordable_units_completed",
                          "deep_affordable_units", "new_construction_units", "preservation_units"]
CRASH_PANEL_MEASURES = ["crashes", "injury_crashes", "persons_injured", "persons_killed", "ped_cyc_crashes"]

# %%
bridge_coverage = pd.DataFrame({
    "housing rows with the key": [
        int(housing["completion_year"].notna().sum()), int(housing["borough"].notna().sum()),
        int(housing["zip"].notna().sum()), int(housing["is_located"].sum())],
    "crash rows with the key": [
        int(crashes["year"].notna().sum()), int(crashes["borough"].notna().sum()),
        int(crashes["zip"].notna().sum()), int(crashes["is_located"].sum())],
    "distinct values in housing": [
        housing["completion_year"].nunique(), housing["borough"].nunique(), housing["zip"].nunique(),
        int(housing.loc[housing["is_located"], ["lat", "lon"]].drop_duplicates().shape[0])],
    "distinct values in crashes": [
        crashes["year"].nunique(), crashes["borough"].nunique(), crashes["zip"].nunique(),
        int(crashes.loc[crashes["is_located"], ["lat", "lon"]].drop_duplicates().shape[0])],
}, index=["Time (completion year / crash year)", "Borough", "ZIP code", "Coordinates"])
bridge_coverage.insert(1, "% of housing", (100 * bridge_coverage["housing rows with the key"] / len(housing)).round(1))
bridge_coverage.insert(3, "% of crashes", (100 * bridge_coverage["crash rows with the key"] / len(crashes)).round(1))
display(bridge_coverage)

# %% [markdown]
# Every bridge loses something. The housing rows without a ZIP or coordinates are the confidential
# records. The crash rows without a ZIP are the larger gap, and most of them still have
# coordinates, which is why the spatial level reaches more crashes than the ZIP level does before
# any repair.
#
# #### 5.1.1 Why the tables are not merged row to row on ZIP
#
# A ZIP code is a key in neither table: one ZIP holds many buildings and thousands of crashes. A
# row-level merge on ZIP therefore pairs every building with every crash of the same ZIP. The
# result is not a list of crashes and not a list of buildings, and any sum taken over it counts
# each crash once per building in its ZIP. The size of that merge can be computed without running
# it, by multiplying the two row counts ZIP by ZIP. To show that the arithmetic matches what
# pandas would do, the merge is run for real on one ZIP, the one with the smallest product among
# ZIPs that have at least `DEMO_MIN_HOUSING_ROWS` housing rows.

# %%
def zip_year_key(zip_codes, years):
    """One integer per (ZIP, year); see YEAR_KEY_SCALE."""
    return zip_codes.astype("int64") * YEAR_KEY_SCALE + years.astype("int64")


def row_merge_size(housing_keys, crash_keys):
    """Rows a many-to-many merge would return: per key, housing rows times crash rows, summed."""
    housing_rows = housing_keys.value_counts()
    crash_rows = crash_keys.value_counts()
    paired = pd.concat([housing_rows.rename("housing_rows"), crash_rows.rename("crash_rows")],
                       axis=1, join="inner")
    paired["merged_rows"] = paired["housing_rows"].astype("int64") * paired["crash_rows"].astype("int64")
    return paired


housing_with_zip = housing[housing["zip"].notna()]
crashes_with_zip = crashes.loc[crashes["zip"].notna(), ["collision_id", "zip", "year", "injured"]]

zip_merge = row_merge_size(housing_with_zip["zip"], crashes_with_zip["zip"])
zip_merge["persons_injured"] = crashes_with_zip.groupby("zip")["injured"].sum().reindex(zip_merge.index)
zip_year_merge = row_merge_size(
    zip_year_key(housing_with_zip["zip"], housing_with_zip["completion_year"].fillna(NO_YEAR_KEY)),
    zip_year_key(crashes_with_zip["zip"], crashes_with_zip["year"]))

# The real merge, on one ZIP where it is small, to confirm the arithmetic.
smallest_zip = zip_merge.loc[zip_merge["housing_rows"] >= DEMO_MIN_HOUSING_ROWS, "merged_rows"].idxmin()
real_merge = housing_with_zip.loc[housing_with_zip["zip"] == smallest_zip, ["housing_id", "zip"]].merge(
    crashes_with_zip[crashes_with_zip["zip"] == smallest_zip], on="zip")
assert len(real_merge) == zip_merge.loc[smallest_zip, "merged_rows"]
assert real_merge["injured"].sum() == (zip_merge.loc[smallest_zip, "housing_rows"]
                                       * zip_merge.loc[smallest_zip, "persons_injured"])

merged_rows_total = int(zip_merge["merged_rows"].sum())
injured_true = int(zip_merge["persons_injured"].sum())
injured_after_merge = int((zip_merge["housing_rows"] * zip_merge["persons_injured"]).sum())
row_counts = pd.Series({
    "Housing rows": len(housing),
    "Crash rows": len(crashes),
    "Row merge on ZIP and year": int(zip_year_merge["merged_rows"].sum()),
    "Row merge on ZIP": merged_rows_total,
})

def plot_row_merge_sizes(row_counts):
    """Lollipop chart of the row counts of the two tables and of the two row-level merges (log axis)."""
    # Lollipops instead of bars: on a log axis the length of a bar depends on where the axis starts,
    # so only the position of the dot carries the value.
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    dot_colors = [PALETTE["housing"], PALETTE["crash"], PALETTE["neutral"], PALETTE["neutral"]]
    ax.set_xscale("log")
    ax.set_xlim(ROW_MERGE_AXIS_START, row_counts.max() * ROW_MERGE_AXIS_HEADROOM)
    ax.hlines(row_counts.index, ROW_MERGE_AXIS_START, row_counts.to_numpy(), color=dot_colors, linewidth=2)
    ax.scatter(row_counts.to_numpy(), row_counts.index, color=dot_colors, s=70, zorder=3)
    for position, value in enumerate(row_counts.to_numpy()):
        is_merge = position >= 2
        label = f"{value:,}" + (f"\n{value / len(crashes):,.0f} x the crash rows" if is_merge else "")
        ax.text(value * ROW_MERGE_LABEL_GAP, position, label, va="center", fontsize=9.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    finish(ax, "A row-level merge on ZIP multiplies rows instead of matching them",
           "Rows (log scale, each gridline is ten times the previous one)", "")
    plt.tight_layout()
    plt.show()


plot_row_merge_sizes(row_counts)

# %%
say(f"A row merge on ZIP would return {merged_rows_total:,} rows, {merged_rows_total / len(crashes):,.0f} times "
    f"the crash table, because the median ZIP with both pairs {zip_merge['housing_rows'].median():.0f} housing "
    f"rows with {zip_merge['crash_rows'].median():,.0f} crashes. Adding the year to the key still returns "
    f"{int(zip_year_merge['merged_rows'].sum()):,} rows. The crashes that have a ZIP shared with housing "
    f"injured {injured_true:,} people; summed over the merged table the same column gives "
    f"{injured_after_merge:,}, an overcount by a factor of {injured_after_merge / injured_true:.0f}. In ZIP "
    f"{smallest_zip}, where the merge was run, pandas returns {len(real_merge):,} rows from "
    f"{int(zip_merge.loc[smallest_zip, 'housing_rows'])} housing rows and "
    f"{int(zip_merge.loc[smallest_zip, 'crash_rows']):,} crashes, exactly the product, and the "
    f"{int(zip_merge.loc[smallest_zip, 'persons_injured']):,} people injured there become "
    f"{int(real_merge['injured'].sum()):,}.")

# %% [markdown]
# The merge would run without an error and look like a joined table, but each crash would be
# repeated once for every building in its ZIP, so ZIPs with many buildings would appear to have
# many more injuries. The rule used from here on is to aggregate
# each table to a common grain first and merge second, so that the merge is one row to one row.

# %% [markdown]
# ### 5.2 Level A: a ZIP by year panel
#
# The common grain for the first level is one row per ZIP code and calendar year. For crashes the
# year is the crash year. For housing it is the completion year, because a building can only add
# residents and street activity once it is finished. Buildings without a completion date
# contribute nothing at this level, and neither do confidential rows, which have no ZIP.
#
# Each table is aggregated separately, and only then are the two aggregates merged on
# (`zip`, `year`). Both aggregates are unique on that key, so the merge cannot multiply rows. The
# panel built in this subsection is deliberately naive: it uses the ZIP codes exactly as reported
# and an outer merge, and its purpose is to expose what does not match. Section 6 repairs those
# gaps and builds the final panel with the same two functions.
#
# Two unit totals are kept for housing. `units_completed` is every unit in the building,
# including market-rate units, and measures how many households the building adds.
# `affordable_units_completed` counts only the units that the programme counts as affordable.

# %%
def aggregate_housing_to_zip_year(housing_frame):
    """Completed housing rows with a ZIP, summed to one row per (zip, completion year).

    The year column keeps the nullable dtype of `completion_year`; fixing that is part of the
    post-integration cleaning, so it is not hidden here.
    """
    completed = housing_frame[housing_frame["is_completed"] & housing_frame["zip"].notna()]
    is_new = completed["construction_type"] == "New Construction"
    is_preservation = completed["construction_type"] == "Preservation"
    measures = pd.DataFrame({
        "zip": completed["zip"], "year": completed["completion_year"],
        "buildings_completed": 1,
        "units_completed": completed["total_units"].astype("int32"),
        "affordable_units_completed": completed["all_counted_units"].astype("int32"),
        "deep_affordable_units": completed["deep_affordable_units"].astype("int32"),
        "new_construction_units": completed["total_units"].where(is_new, 0).astype("int32"),
        "preservation_units": completed["total_units"].where(is_preservation, 0).astype("int32"),
    })
    return measures.groupby(["zip", "year"], as_index=False)[HOUSING_PANEL_MEASURES].sum()


def aggregate_crashes_to_zip_year(crash_frame, zip_column="zip"):
    """Crashes that have a value in `zip_column`, summed to one row per (zip, crash year)."""
    with_zip = crash_frame[crash_frame[zip_column].notna()]
    measures = pd.DataFrame({
        "zip": with_zip[zip_column], "year": with_zip["year"],
        "crashes": 1,
        "injury_crashes": with_zip["is_injury_crash"].astype("int32"),
        "persons_injured": with_zip["injured"].astype("int32"),
        "persons_killed": with_zip["killed"].astype("int32"),
        "ped_cyc_crashes": with_zip["ped_cyc_harm"].astype("int32"),
    })
    return measures.groupby(["zip", "year"], as_index=False)[CRASH_PANEL_MEASURES].sum()


housing_by_zip_year = aggregate_housing_to_zip_year(housing)
crashes_by_zip_year_reported = aggregate_crashes_to_zip_year(crashes)
assert not housing_by_zip_year.duplicated(["zip", "year"]).any()
assert not crashes_by_zip_year_reported.duplicated(["zip", "year"]).any()

housing_in_study_years = housing_by_zip_year[housing_by_zip_year["year"].between(YEAR_START, YEAR_END)]
naive_panel = housing_in_study_years.merge(crashes_by_zip_year_reported, on=["zip", "year"], how="outer",
                                           indicator="match")
# One row to one row: the merged table can be no longer than the two aggregates together.
assert len(naive_panel) <= len(housing_in_study_years) + len(crashes_by_zip_year_reported)
assert not naive_panel.duplicated(["zip", "year"]).any()

match_labels = {"both": "ZIP-year in both tables", "left_only": "housing only (no crash row)",
                "right_only": "crashes only (no housing row)"}
naive_match = (naive_panel["match"].astype(str).map(match_labels).value_counts()
               .reindex(list(match_labels.values()), fill_value=0).rename("ZIP-years").to_frame())
naive_match["% of naive panel"] = (100 * naive_match["ZIP-years"] / len(naive_panel)).round(1)
display(naive_match)

housing_zips = set(housing_by_zip_year["zip"])
crash_zips_reported = set(crashes_by_zip_year_reported["zip"])
housing_zips_without_crashes = sorted(housing_zips - crash_zips_reported)
naive_missing_cells = int(naive_panel[HOUSING_PANEL_MEASURES + CRASH_PANEL_MEASURES].isna().sum().sum())
naive_balanced_rows = naive_panel["zip"].nunique() * N_STUDY_YEARS

crashes_without_zip = int(crashes["zip"].isna().sum())
completed_rows = housing[housing["is_completed"]]
completed_without_zip = completed_rows[completed_rows["zip"].isna()]
completed_outside_years = completed_rows[completed_rows["zip"].notna()
                                         & ~completed_rows["completion_year"].between(YEAR_START, YEAR_END)]
not_completed = housing[~housing["is_completed"]]

left_out = pd.DataFrame({
    "rows": [crashes_without_zip, len(completed_without_zip), len(completed_outside_years), len(not_completed)],
    "% of its table": [pct(crashes_without_zip, len(crashes)), pct(len(completed_without_zip), len(housing)),
                       pct(len(completed_outside_years), len(housing)), pct(len(not_completed), len(housing))],
    "units": [np.nan, completed_without_zip["total_units"].sum(), completed_outside_years["total_units"].sum(),
              not_completed["total_units"].sum()],
}, index=["Crashes with no ZIP (never reach the panel)", "Completed housing rows with no ZIP (confidential)",
          f"Housing rows completed outside {YEAR_START}-{YEAR_END}", "Housing rows with no completion date"])
display(left_out.round(1))

# %%
say(f"The naive panel has {len(naive_panel):,} rows for {naive_panel['zip'].nunique()} ZIP codes. A balanced "
    f"panel of those ZIPs over {N_STUDY_YEARS} years would have {naive_balanced_rows:,} rows, so "
    f"{naive_balanced_rows - len(naive_panel):,} ZIP-years are absent altogether. Only "
    f"{int(naive_match.loc[match_labels['both'], 'ZIP-years']):,} rows "
    f"({naive_match.loc[match_labels['both'], '% of naive panel']:.1f}%) matched on both sides; "
    f"{int(naive_match.loc[match_labels['right_only'], 'ZIP-years']):,} have crashes and no housing row, and "
    f"{int(naive_match.loc[match_labels['left_only'], 'ZIP-years'])} have housing and no crash row. The merge created {naive_missing_cells:,} missing cells that neither input "
    f"had. Housing ZIP codes that never appear in the crash table: {len(housing_zips_without_crashes)} "
    f"({', '.join(housing_zips_without_crashes)}). Outside the panel, {crashes_without_zip:,} crashes "
    f"({pct(crashes_without_zip, len(crashes)):.1f}%) have no ZIP and could not be placed in any row.")

# %% [markdown]
# Three different things are hiding behind those numbers, and they need different treatment.
#
# A ZIP-year with crashes and no housing row is the normal case: most neighbourhoods complete no
# subsidised building in a given year. The missing housing values are zeros that the merge wrote
# as missing.
#
# A housing ZIP with no crash row at all is not a neighbourhood without crashes. Ten years with
# no reported collision is not believable for a residential ZIP, so the crash table simply does
# not use that code, and a zero there would be false.
#
# The crashes without a ZIP are the largest problem and they are invisible in the panel itself,
# because a row that has no key drops out of a grouped table without leaving a missing value
# behind. Their share also changes from year to year (Section 4.8), so the panel as built here
# would show a trend that comes from record keeping. Section 6.1 deals with all three.

# %% [markdown]
# ### 5.3 Level B: spatial proximity
#
# The second level ignores administrative areas and asks which crashes happened within `BUFFER_M`
# metres of each building. It needs a distance between two points given as latitude and longitude,
# and a way to find all pairs closer than the radius without comparing everything to everything.
#
# For distance, degrees are not a unit of length: at New York's latitude one degree of longitude
# is about three quarters of the length of one degree of latitude, so a circle drawn in degrees
# would be an ellipse on the ground. The haversine formula gives the great-circle distance on a
# sphere. With latitude $\varphi$ and longitude $\lambda$ in radians,
#
# $$d = 2R \arcsin\sqrt{\sin^2\frac{\varphi_2-\varphi_1}{2} + \cos\varphi_1\cos\varphi_2\sin^2\frac{\lambda_2-\lambda_1}{2}}$$
#
# where $R$ is `EARTH_RADIUS_M`. The formula works on the unit sphere, so coordinates are converted
# to radians on the way in, and a radius in metres is divided by $R$ before a query and distances
# are multiplied by $R$ on the way out. Treating the Earth as a sphere is wrong by a fraction of
# a percent, which is about a metre at this radius and smaller than the error in the geocoding.
#
# For the index, a ball tree groups the points into nested spheres. A radius query skips every sphere
# that lies entirely outside the search radius, so it looks at a small part of the points instead
# of all of them. The tree is built over the crashes, the larger table, and queried once per
# building; the result is the same set of pairs as querying the other way round, and it is much
# faster because there are far fewer buildings than crashes.

# %%
def to_radians(lat, lon):
    """Stack latitude and longitude (degrees) into the (n, 2) radian array the haversine tree expects."""
    return np.radians(np.column_stack([np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64")]))


def haversine_m(lat_a, lon_a, lat_b, lon_b):
    """Great-circle distance in metres between points given in degrees (arrays broadcast)."""
    phi_a, phi_b = np.radians(lat_a), np.radians(lat_b)
    half_dphi = (phi_b - phi_a) / 2
    half_dlambda = np.radians(np.asarray(lon_b) - np.asarray(lon_a)) / 2
    inside = np.sin(half_dphi) ** 2 + np.cos(phi_a) * np.cos(phi_b) * np.sin(half_dlambda) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(inside))


def build_haversine_tree(lat, lon):
    """Ball tree over points given in degrees; point i of the tree is row i of the inputs."""
    return BallTree(to_radians(lat, lon), metric="haversine")


def pairs_within_radius(tree, lat, lon, radius_m):
    """All (query point, tree point) pairs no farther apart than `radius_m` metres.

    Returns three flat arrays of equal length: the position of the query point in `lat`/`lon`,
    the position of the tree point, and their distance in metres. The tree answers with one
    ragged list per query point; flattening it into pair arrays lets everything downstream
    (counting, taking the nearest, linking clusters) be done with vectorised numpy.
    """
    neighbours, distances = tree.query_radius(to_radians(lat, lon), radius_m / EARTH_RADIUS_M,
                                              return_distance=True)
    sizes = np.fromiter((len(found) for found in neighbours), dtype=np.int64, count=len(neighbours))
    query_index = np.repeat(np.arange(len(neighbours)), sizes)
    if sizes.sum() == 0:
        return query_index, np.empty(0, dtype=np.int64), np.empty(0, dtype="float64")
    return query_index, np.concatenate(neighbours), np.concatenate(distances) * EARTH_RADIUS_M


def brute_force_pairs(query_lat, query_lon, point_lat, point_lon, radius_m):
    """The same pairs as `pairs_within_radius`, found by computing every distance (small inputs only)."""
    all_distances = haversine_m(np.asarray(query_lat)[:, None], np.asarray(query_lon)[:, None],
                                np.asarray(point_lat)[None, :], np.asarray(point_lon)[None, :])
    query_index, point_index = np.nonzero(all_distances <= radius_m)
    return query_index, point_index, all_distances[query_index, point_index]


# %% [markdown]
# A toy example that can be checked by hand comes first. Four points are placed around a centre in
# Midtown: one on the centre, one 100 m north, one 200 m east and one 300 m north. A query from
# the centre with a 250 m radius must return the first three and leave out the fourth. The
# offsets are converted to degrees with the fact that an arc of length $s$ on a sphere spans
# $s/R$ radians, and for the eastward point that angle is divided by the cosine of the latitude.

# %%
toy_centre_lat, toy_centre_lon = 40.7500, -73.9900
toy_offsets_m = {"On the centre": (0, 0), "100 m north": (100, 0), "200 m east": (0, 200),
                 "300 m north": (300, 0)}
toy_points = pd.DataFrame([
    {"point": name,
     "lat": toy_centre_lat + math.degrees(north_m / EARTH_RADIUS_M),
     "lon": toy_centre_lon + math.degrees(east_m / EARTH_RADIUS_M) / math.cos(math.radians(toy_centre_lat))}
    for name, (north_m, east_m) in toy_offsets_m.items()])
toy_points["haversine distance to centre (m)"] = haversine_m(
    toy_centre_lat, toy_centre_lon, toy_points["lat"].to_numpy(), toy_points["lon"].to_numpy()).round(2)

toy_tree = build_haversine_tree(toy_points["lat"], toy_points["lon"])
_, toy_found, toy_distance = pairs_within_radius(toy_tree, [toy_centre_lat], [toy_centre_lon], BUFFER_M)
toy_points[f"returned by the {BUFFER_M} m query"] = toy_points.index.isin(toy_found)
display(toy_points)

assert set(toy_points.loc[toy_found, "point"]) == {"On the centre", "100 m north", "200 m east"}
assert np.allclose(np.sort(toy_distance), [0, 100, 200], atol=0.01)

# %% [markdown]
# The distances come back as placed and the query keeps exactly the three points inside the
# radius. The next check uses real data. The tree over all located crashes is built, and for a
# random sample of buildings and crashes every distance is also computed directly. The two sets of
# pairs have to be identical.

# %%
def build_crash_tree(crash_frame):
    """Ball tree over the located crashes, plus the row position in `crash_frame` of each tree point."""
    crash_rows = np.flatnonzero(crash_frame["is_located"].to_numpy())
    tree = build_haversine_tree(crash_frame["lat"].to_numpy()[crash_rows],
                                crash_frame["lon"].to_numpy()[crash_rows])
    return tree, crash_rows


# Row positions are used as crash identifiers in this section, so the index must be 0..n-1.
crashes = crashes.reset_index(drop=True)
located_housing = housing[housing["is_located"]]
crash_tree, crash_tree_rows = build_crash_tree(crashes)

check_rng = np.random.default_rng(SEED)
check_sites = located_housing.sample(CHECK_SITE_SAMPLE, random_state=SEED)
check_crash_rows = check_rng.choice(crash_tree_rows, size=CHECK_CRASH_SAMPLE, replace=False)
check_crash_lat = crashes["lat"].to_numpy()[check_crash_rows]
check_crash_lon = crashes["lon"].to_numpy()[check_crash_rows]

check_tree = build_haversine_tree(check_crash_lat, check_crash_lon)
tree_query, tree_point, tree_distance = pairs_within_radius(
    check_tree, check_sites["lat"].to_numpy(), check_sites["lon"].to_numpy(), BUFFER_M)
brute_query, brute_point, brute_distance = brute_force_pairs(
    check_sites["lat"].to_numpy(), check_sites["lon"].to_numpy(), check_crash_lat, check_crash_lon, BUFFER_M)

tree_pairs = pd.DataFrame({"query": tree_query, "point": tree_point, "distance": tree_distance}
                          ).sort_values(["query", "point"]).reset_index(drop=True)
brute_pairs = pd.DataFrame({"query": brute_query, "point": brute_point, "distance": brute_distance}
                           ).sort_values(["query", "point"]).reset_index(drop=True)
assert len(tree_pairs) == len(brute_pairs)
assert (tree_pairs[["query", "point"]].to_numpy() == brute_pairs[["query", "point"]].to_numpy()).all()
assert np.allclose(tree_pairs["distance"], brute_pairs["distance"], atol=1e-6)

full_matrix_cells = len(crash_tree_rows) * len(located_housing)
say(f"The tree and the brute-force calculation return the same {len(tree_pairs):,} pairs out of "
    f"{CHECK_SITE_SAMPLE * CHECK_CRASH_SAMPLE:,} candidate pairs, and the largest difference between the two "
    f"distances is {np.abs(tree_pairs['distance'] - brute_pairs['distance']).max():.1e} m. For the full data a "
    f"distance matrix would need {len(crash_tree_rows):,} x {len(located_housing):,} = {full_matrix_cells:,} "
    f"values, about {full_matrix_cells * BYTES_PER_FLOAT / BYTES_PER_GB:.0f} GB as 8-byte numbers, which does "
    f"not fit in memory. The tree avoids ever forming it.")

# %% [markdown]
# #### 5.3.1 Why a 250 m buffer, and why not tracts or grid cells
#
# The buffer has to be large enough to catch the crashes that a building's residents could be
# part of on their way in and out, and to tolerate the way crash locations are recorded: most are
# snapped to the nearest intersection, so a radius shorter than a block would miss crashes that
# happened in front of the building. It has to be small enough that the area still belongs to the
# building, since every extra metre adds streets that it shares with its neighbours. The table
# below puts numbers on that trade-off for five radii.

# %%
def site_crash_pairs(site_frame, tree, tree_rows, radius_m=BUFFER_M):
    """One row per (site, crash) with the crash within `radius_m` of the site.

    `crash_row` is the row position of the crash in `crashes`. A crash near several sites appears
    once for each of them and never twice for the same site.
    """
    site_position, tree_position, distance_m = pairs_within_radius(
        tree, site_frame["lat"].to_numpy(), site_frame["lon"].to_numpy(), radius_m)
    return pd.DataFrame({
        "site_id": site_frame["site_id"].to_numpy()[site_position].astype("int32"),
        "crash_row": tree_rows[tree_position].astype("int32"),
        "distance_m": distance_m.astype("float32"),
    })


def radius_profile(site_frame, crash_frame, tree, tree_rows, radius_m):
    """Summary of what one buffer radius implies: reach, overlap and counts per site."""
    pairs = site_crash_pairs(site_frame, tree, tree_rows, radius_m)
    is_injury = crash_frame["is_injury_crash"].to_numpy()
    injury_pairs = pairs[is_injury[pairs["crash_row"].to_numpy()]]
    sites_per_crash = injury_pairs.groupby("crash_row").size()
    per_site_year = (np.bincount(injury_pairs["site_id"], minlength=len(site_frame)) / N_STUDY_YEARS)
    located_injury_crashes = int(is_injury[tree_rows].sum())
    return {
        "radius (m)": radius_m,
        "area of the buffer (hectares)": math.pi * radius_m ** 2 / SQUARE_METRES_PER_HECTARE,
        "% of located injury crashes near a site": pct(len(sites_per_crash), located_injury_crashes),
        "sites sharing each near crash (mean)": sites_per_crash.mean(),
        "injury crashes per site and year (median)": float(np.median(per_site_year)),
        "% of sites below one injury crash a year": 100 * float((per_site_year < 1).mean()),
    }


# A provisional site table (the full one is built in 5.4) is enough to profile the radii.
provisional_sites = located_housing[["lat", "lon"]].reset_index(drop=True)
provisional_sites["site_id"] = np.arange(len(provisional_sites), dtype="int32")
radius_table = pd.DataFrame([radius_profile(provisional_sites, crashes, crash_tree, crash_tree_rows, radius)
                             for radius in RADIUS_CANDIDATES_M]).set_index("radius (m)")
display(radius_table.round(1))

# %%
def plot_radius_profile(radius_table):
    """Two panels over the candidate radii: injury crashes per site and the overlap between sites."""
    fig, (ax_signal, ax_overlap) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    panel_specs = [
        (ax_signal, "injury crashes per site and year (median)", PALETTE["crash"],
         "Counts per site grow with the radius", "Injury crashes per site and year (median)"),
        (ax_overlap, "sites sharing each near crash (mean)", PALETTE["housing"],
         "So does the overlap between sites", "Mean number of sites per crash"),
    ]
    for ax, column, color, title, ylabel in panel_specs:
        ax.plot(radius_table.index, radius_table[column], color=color, linewidth=2, marker="o", markersize=7)
        for radius, value in radius_table[column].items():
            ax.annotate(f"{value:.1f}", (radius, value), textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=9, fontweight="bold" if radius == BUFFER_M else "normal")
        ax.axvline(BUFFER_M, color=PALETTE["neutral"], linestyle=":", linewidth=1.2)
        ax.set_xticks(RADIUS_CANDIDATES_M)
        ax.set_ylim(0, radius_table[column].max() * LINE_HEADROOM)
        finish(ax, title, "Buffer radius (metres)", ylabel)
    ax_signal.annotate(f"{BUFFER_M} m, the radius used", xy=(BUFFER_M, ax_signal.get_ylim()[1] * 0.93),
                       xytext=(6, 0), textcoords="offset points", fontsize=9, color=PALETTE["neutral"])
    plt.tight_layout()
    plt.show()


plot_radius_profile(radius_table)

# %%
chosen = radius_table.loc[BUFFER_M]
smallest_radius, largest_radius = RADIUS_CANDIDATES_M[0], RADIUS_CANDIDATES_M[-1]
say(f"At {smallest_radius} m the median site sees {radius_table.loc[smallest_radius, 'injury crashes per site and year (median)']:.1f} "
    f"injury crashes a year and {radius_table.loc[smallest_radius, '% of sites below one injury crash a year']:.0f}% "
    f"of sites see fewer than one, so a before and after comparison at one site would mostly compare small "
    f"counts. At {BUFFER_M} m the median is {chosen['injury crashes per site and year (median)']:.1f} and "
    f"{chosen['% of sites below one injury crash a year']:.1f}% of sites are below one a year. At "
    f"{largest_radius} m a crash near housing lies within the radius of "
    f"{radius_table.loc[largest_radius, 'sites sharing each near crash (mean)']:.0f} sites on average and "
    f"{radius_table.loc[largest_radius, '% of located injury crashes near a site']:.0f}% of all located injury "
    f"crashes count as near housing, against {chosen['sites sharing each near crash (mean)']:.1f} sites and "
    f"{chosen['% of located injury crashes near a site']:.0f}% at {BUFFER_M} m.")

# %% [markdown]
# No radius is free of both problems. A small one gives counts too low to compare, and a large
# one turns "near a building" into "anywhere in the built-up city". The 250 m radius is the
# smallest of the five at which nearly every site has at least one injury crash a year, and it
# corresponds to a walk of about three minutes. The overlap at 250 m is already large, and 5.4
# handles it explicitly instead of assuming that sites are independent. Section 9 repeats the
# main results at other radii with the functions defined here.
#
# Two alternatives were considered. Census tracts would give fixed, non-overlapping areas, and
# the raw housing table carries a tract code. The crash table does not, so assigning crashes to
# tracts would need a boundary file, and this notebook reads only its two data files. A fixed
# grid of square cells needs no outside file, and Section 6 uses one for a different purpose, but
# a grid is not centred on the buildings: a building at the edge of its cell would be compared
# with crashes up to a cell width away on one side and none on the other. A buffer is centred on
# each building by construction. Its cost is the overlap, which the next subsection measures.

# %% [markdown]
# ### 5.4 Sites, clusters and the site by year table
#
# A **site** is one located housing row, that is, one building in one project. Every located row
# becomes a site whether or not it is completed, because buildings that are not finished yet are
# the comparison group later on.
#
# Buildings in this programme stand close together, often as several addresses of one project.
# Two sites less than `BUFFER_M` apart have overlapping buffers and count many of the same
# crashes, so their counts are not independent observations. A confidence interval that treated
# each site as a separate draw would be too narrow. The sites are therefore grouped into
# **clusters**: two sites belong to the same cluster if they are within `BUFFER_M` of each other,
# directly or through a chain of other sites (the connected components of the graph that links
# every such pair). A bootstrap builds an interval by drawing, many times, a new sample of the
# same size from the data at hand, recomputing the estimate on each draw, and reading the spread
# of those estimates. Intervals at the spatial level resample whole clusters of linked sites, so
# sites that share crashes stay together in every draw. The exception is RQ7, which resamples
# one-kilometre blocks for the reason given there.
#
# Linking at one radius does not remove all sharing, since two sites up to twice the radius apart
# can both reach a crash that lies between them. The cell after the next one measures how much
# sharing is left between clusters. As a second, finer grouping for sensitivity checks, each site
# also gets a `block_id`, the one-kilometre square it falls in.

# %%
def connected_clusters(first, second, n_points):
    """Label the connected components of an undirected graph given as two arrays of linked points."""
    links = coo_matrix((np.ones(len(first), dtype=np.int8), (first, second)), shape=(n_points, n_points))
    _, labels = connected_components(links, directed=False)
    return labels.astype("int32")


def site_neighbour_pairs(site_frame, radius_m):
    """Pairs of sites within `radius_m` of each other, both directions, including each site with itself."""
    site_tree = build_haversine_tree(site_frame["lat"].to_numpy(), site_frame["lon"].to_numpy())
    first, second, _ = pairs_within_radius(site_tree, site_frame["lat"].to_numpy(),
                                           site_frame["lon"].to_numpy(), radius_m)
    return first, second


def build_sites(housing_frame, link_m=BUFFER_M):
    """One row per located housing row, with `site_id`, `cluster_id` and `block_id`.

    `site_id` is the row position (0..n-1), so it can index numpy arrays directly. `cluster_id`
    labels the connected components of sites within `link_m` metres of one another.
    """
    site_frame = housing_frame.loc[housing_frame["is_located"], SITE_ATTRIBUTES].reset_index(drop=True)
    site_frame.insert(0, "site_id", np.arange(len(site_frame), dtype="int32"))
    first, second = site_neighbour_pairs(site_frame, link_m)
    site_frame.insert(1, "cluster_id", connected_clusters(first, second, len(site_frame)))
    site_frame["cluster_size"] = site_frame.groupby("cluster_id")["site_id"].transform("size").astype("int32")
    # Other sites within the link distance (the pair list includes each site with itself).
    site_frame["n_sites_within_buffer"] = (np.bincount(first, minlength=len(site_frame)) - 1).astype("int32")

    # Square blocks: metres are converted to degrees separately for the two directions.
    block_lat_degrees = BLOCK_SIDE_M / METRES_PER_DEGREE
    block_lon_degrees = block_lat_degrees / math.cos(math.radians(MID_LATITUDE_DEG))
    block_keys = pd.DataFrame({"row": np.floor(site_frame["lat"] / block_lat_degrees).astype("int64"),
                               "column": np.floor(site_frame["lon"] / block_lon_degrees).astype("int64")})
    site_frame.insert(2, "block_id", block_keys.groupby(["row", "column"]).ngroup().astype("int32"))
    return site_frame


sites = build_sites(housing)
assert len(sites) == int(housing["is_located"].sum())
assert sites["site_id"].is_unique and not sites.duplicated(["project_id", "building_id"]).any()
assert (sites["site_id"].to_numpy() == np.arange(len(sites))).all()

cluster_sizes = sites.groupby("cluster_id").size()
cluster_bands = pd.cut(cluster_sizes, CLUSTER_SIZE_EDGES, labels=CLUSTER_SIZE_LABELS)
cluster_summary = pd.DataFrame({"clusters": cluster_bands.value_counts().reindex(CLUSTER_SIZE_LABELS),
                                "sites": cluster_sizes.groupby(cluster_bands, observed=False).sum()})
cluster_summary["% of sites"] = (100 * cluster_summary["sites"] / len(sites)).round(1)
display(cluster_summary.rename_axis("sites in the cluster"))


def plot_cluster_sizes(cluster_summary):
    """Number of clusters and number of sites by cluster-size band, side by side."""
    fig, (ax_clusters, ax_sites) = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, column, color, title in [
            (ax_clusters, "clusters", PALETTE["neutral"], "Most clusters are small ..."),
            (ax_sites, "sites", PALETTE["housing"], "... but most sites sit in a few very large ones")]:
        ax.barh(cluster_summary.index.astype(str), cluster_summary[column], color=color, height=0.62)
        for position, value in enumerate(cluster_summary[column]):
            ax.text(value + cluster_summary[column].max() * LABEL_GAP, position, f"{value:,}", va="center",
                    fontsize=9.5)
        ax.set_xlim(0, cluster_summary[column].max() * AXIS_HEADROOM)
        ax.grid(axis="y", visible=False)
        thousands(ax, "x")
        finish(ax, title, f"Number of {column}", "")
    ax_clusters.set_ylabel(f"Sites in the cluster (linked within {BUFFER_M} m)")
    ax_clusters.invert_yaxis()
    plt.tight_layout()
    plt.show()


plot_cluster_sizes(cluster_summary)

# %%
largest_clusters = cluster_sizes.sort_values(ascending=False)
largest_cluster_boroughs = sites.loc[sites["cluster_id"] == largest_clusters.index[0], "borough"].value_counts()
say(f"The {len(sites):,} sites form {len(cluster_sizes):,} clusters. {int((cluster_sizes == 1).sum())} clusters "
    f"are a single site with no other site within {BUFFER_M} m, but they hold only "
    f"{pct((cluster_sizes == 1).sum(), len(sites)):.1f}% of the sites. The three largest clusters contain "
    f"{', '.join(f'{size:,}' for size in largest_clusters.head(3))} sites, together "
    f"{pct(largest_clusters.head(3).sum(), len(sites)):.0f}% of all sites; the largest lies mostly in "
    f"{largest_cluster_boroughs.index[0]}. The median site has {sites['n_sites_within_buffer'].median():.0f} "
    f"other sites within {BUFFER_M} m. The finer grouping has {sites['block_id'].nunique():,} one-kilometre "
    f"blocks, the largest with {int(sites.groupby('block_id').size().max())} sites.")

# %% [markdown]
# Subsidised housing in New York is concentrated: in the neighbourhoods where most of it stands,
# nearly every building has another one within the buffer, and the chain of neighbours does not
# break for kilometres. The clusters are therefore very unequal. Resampling clusters is the
# appropriate way to measure uncertainty here, because sites in one chain share crashes. It also
# means that the effective number of independent units is far below the number of sites, so
# intervals built on clusters will be wide.
#
# #### 5.4.1 Counting crashes per site and calendar year
#
# For every site and every study year the next cell counts the crashes within the buffer, the
# injury crashes (the primary outcome, see Section 3), the injury crashes in which a pedestrian or
# cyclist was hurt, and the persons injured. Each (site, crash) pair exists once, so a crash is
# counted once per site.

# %%
def build_site_year(pairs, crash_frame, n_sites):
    """Balanced site x study-year table of crash counts from a (site, crash) pair list.

    One flat position per (site, year) lets a single `bincount` per measure do the counting,
    with no loop over sites.
    """
    crash_rows = pairs["crash_row"].to_numpy()
    year_offset = crash_frame["year"].to_numpy()[crash_rows].astype("int64") - YEAR_START
    flat_position = pairs["site_id"].to_numpy().astype("int64") * N_STUDY_YEARS + year_offset
    n_cells = n_sites * N_STUDY_YEARS
    weights = {
        "crashes": None,
        "injury_crashes": crash_frame["is_injury_crash"].to_numpy()[crash_rows].astype("float64"),
        "ped_cyc_crashes": crash_frame["ped_cyc_harm"].to_numpy()[crash_rows].astype("float64"),
        "persons_injured": crash_frame["injured"].to_numpy()[crash_rows].astype("float64"),
    }
    table = pd.DataFrame({
        "site_id": np.repeat(np.arange(n_sites, dtype="int32"), N_STUDY_YEARS),
        "year": np.tile(np.asarray(STUDY_YEARS, dtype="int16"), n_sites),
    })
    for column in SITE_COUNT_COLUMNS:
        table[column] = np.bincount(flat_position, weights=weights[column], minlength=n_cells).astype("int32")
    return table


site_pairs = site_crash_pairs(sites, crash_tree, crash_tree_rows, BUFFER_M)
site_year = build_site_year(site_pairs, crashes, len(sites))

# Each crash is counted once per site, and the table is balanced.
assert not site_pairs.duplicated(["site_id", "crash_row"]).any()
assert site_year["crashes"].sum() == len(site_pairs)
assert len(site_year) == len(sites) * N_STUDY_YEARS and not site_year.duplicated(["site_id", "year"]).any()
assert (site_year[SITE_COUNT_COLUMNS] >= 0).all().all()
assert (site_year["ped_cyc_crashes"] <= site_year["injury_crashes"]).all()
assert (site_year["injury_crashes"] <= site_year["crashes"]).all()

pair_is_injury = crashes["is_injury_crash"].to_numpy()[site_pairs["crash_row"].to_numpy()]
injury_pairs = site_pairs[pair_is_injury]
distinct_injury_crashes_near = injury_pairs["crash_row"].nunique()
clusters_per_crash = (injury_pairs.assign(cluster_id=sites["cluster_id"].to_numpy()[injury_pairs["site_id"]])
                      .groupby("crash_row")["cluster_id"].nunique())
say(f"There are {len(site_pairs):,} (site, crash) pairs within {BUFFER_M} m. The site counts of injury crashes "
    f"add up to {len(injury_pairs):,}, but only {distinct_injury_crashes_near:,} different injury crashes are "
    f"involved, so adding counts across sites would count the average crash "
    f"{len(injury_pairs) / distinct_injury_crashes_near:.1f} times. Counted by cluster instead, "
    f"{pct((clusters_per_crash == 1).sum(), len(clusters_per_crash)):.1f}% of those crashes belong to exactly "
    f"one cluster and {pct((clusters_per_crash > 1).sum(), len(clusters_per_crash)):.1f}% are still shared "
    f"between two or more clusters.")

# %% [markdown]
# Site counts are therefore never summed across sites to get a total. They are used as repeated
# measurements of the streets around each building, compared with the same site in other years,
# and uncertainty comes from resampling clusters. The small share of crashes still shared between
# clusters is the remaining dependence that the cluster bootstrap does not remove.
#
# #### 5.4.2 The event-time table
#
# To compare the years before and after a completion, calendar years are re-expressed relative to
# the completion year: `rel_year` runs from minus `EVENT_WINDOW` to plus `EVENT_WINDOW`, with zero
# for the completion year itself. A completion **cohort** is the set of sites completed in one
# calendar year. Only cohorts whose whole window lies inside the study years are used, so that
# every treated site is observed for the same number of years on both sides.
#
# Crash reporting changed during these years (Section 3), so a site's own before and after
# difference would mix the building with the citywide change. Each cohort therefore gets
# **comparison** sites observed over the same calendar years: located sites that are not completed
# by the end of the cohort's window, either because they are completed later or because they have
# no completion date yet. These are places chosen for the same programme, which makes them more
# similar to the treated sites than arbitrary street corners would be. A comparison site serves
# every cohort for which it qualifies, so it can appear under several cohorts; the cluster
# bootstrap accounts for that repetition because all copies of a site are resampled together.
#
# A comparison site is only a comparison if nothing was completed next to it. Because the sites
# are so densely packed, two screens are applied and recorded as flags.
#
# For a comparison site, `near_completion_in_window` marks another site within the buffer whose
# completion falls inside the cohort's window, and `near_earlier_completion` marks one completed
# before the window opens. Either one makes the site `contaminated`: its streets already received
# a completed building, so it is not an untreated place.
#
# For a treated site, `near_completion_in_window` marks another site within the buffer that is
# completed in a different year of the same window, which would blur the before and after
# periods. Neighbours completed in the same year are treated as part of the same event and are
# not flagged, since projects often finish several adjacent buildings together. A treated site is
# `contaminated` when that flag is set. `near_earlier_completion` is recorded for treated sites
# too, for information, but an earlier neighbour is present both before and after and does not
# blur the comparison.
#
# The rows stay in the table with their flags, so the cost of the screen is visible and Section 9
# can relax it.

# %%
def event_cohorts(completion_year, window=EVENT_WINDOW):
    """Completion years present in the data whose full window of `window` years on each side lies inside the study years."""
    years_in_data = set(np.unique(completion_year[~np.isnan(completion_year)]).astype(int).tolist())
    return [year for year in full_window_years(window) if year in years_in_data]


def cohort_members(site_frame, first, second, cohort, window):
    """Treated and comparison sites of one cohort, with their contamination flags (one row per site)."""
    completion = site_frame["completion_year"].astype("float64").to_numpy()
    n_sites = len(site_frame)
    is_treated = completion == cohort
    # Not completed by the end of the window: completed later, or no completion date at all.
    is_comparison = np.isnan(completion) | (completion > cohort + window)

    neighbour_completion = completion[second]
    is_other_site = first != second
    in_window = is_other_site & (neighbour_completion >= cohort - window) & (neighbour_completion <= cohort + window)
    # A neighbour completed in the site's own completion year is the same event, not a second one.
    # For comparison sites the own year is missing or later than the window, so every neighbour counts.
    separate_event = in_window & (neighbour_completion != completion[first])
    earlier = is_other_site & (neighbour_completion < cohort - window)

    near_in_window = np.bincount(first[separate_event], minlength=n_sites) > 0
    near_earlier = np.bincount(first[earlier], minlength=n_sites) > 0
    member = is_treated | is_comparison
    return pd.DataFrame({
        "site_id": site_frame["site_id"].to_numpy()[member],
        "cohort": np.int16(cohort),
        "group": pd.Categorical(np.where(is_treated[member], "treated", "comparison"), categories=EVENT_GROUPS),
        "near_completion_in_window": near_in_window[member],
        "near_earlier_completion": near_earlier[member],
        "contaminated": np.where(is_treated[member], near_in_window[member],
                                 near_in_window[member] | near_earlier[member]),
    })


def build_event_panel(site_frame, site_year_frame, radius_m=BUFFER_M, window=EVENT_WINDOW, neighbour_pairs=None):
    """Site x cohort x relative-year table with crash counts, group and contamination flags.

    `radius_m` is the distance within which another completion contaminates a site; it should be
    the radius that was used to count the crashes in `site_year_frame`. `neighbour_pairs` is the
    output of `site_neighbour_pairs(site_frame, radius_m)` when the caller already has it, which
    saves building the same tree again.
    """
    first, second = neighbour_pairs if neighbour_pairs is not None else site_neighbour_pairs(site_frame, radius_m)
    cohorts = event_cohorts(site_frame["completion_year"].astype("float64").to_numpy(), window)
    members = pd.concat([cohort_members(site_frame, first, second, cohort, window) for cohort in cohorts],
                        ignore_index=True)
    relative_years = np.arange(-window, window + 1, dtype="int8")
    panel = members.loc[members.index.repeat(len(relative_years))].reset_index(drop=True)
    panel["rel_year"] = np.tile(relative_years, len(members))
    panel["year"] = (panel["cohort"] + panel["rel_year"]).astype("int16")
    panel = panel.merge(site_year_frame, on=["site_id", "year"], how="left", validate="many_to_one")
    attributes = site_frame[["site_id", "cluster_id", "block_id", "completion_year", "construction_type",
                             "borough", "size_band", "total_units"]]
    panel = panel.merge(attributes, on="site_id", how="left", validate="many_to_one")
    ordered = ["site_id", "cluster_id", "block_id", "cohort", "group", "rel_year", "year", *SITE_COUNT_COLUMNS,
               "contaminated", "near_completion_in_window", "near_earlier_completion", "completion_year",
               "construction_type", "borough", "size_band", "total_units"]
    return panel[ordered]


# The 250 m neighbour pairs are needed again by the independent check below, so they are built once.
site_first, site_second = site_neighbour_pairs(sites, BUFFER_M)
event_panel = build_event_panel(sites, site_year, neighbour_pairs=(site_first, site_second))

assert not event_panel.duplicated(["site_id", "cohort", "rel_year"]).any()
assert event_panel["year"].between(YEAR_START, YEAR_END).all()
assert event_panel[SITE_COUNT_COLUMNS].notna().all().all()
assert (event_panel.groupby(["site_id", "cohort"]).size() == 2 * EVENT_WINDOW + 1).all()
treated_rows = event_panel[event_panel["group"] == "treated"]
comparison_rows = event_panel[event_panel["group"] == "comparison"]
assert (treated_rows["completion_year"] == treated_rows["cohort"]).all()
assert (comparison_rows["completion_year"].isna()
        | (comparison_rows["completion_year"] > comparison_rows["cohort"] + EVENT_WINDOW)).all()
assert treated_rows.groupby("site_id")["cohort"].nunique().eq(1).all(), "a treated site has one cohort"

# No completed site within one buffer radius, checked independently of the flags: no clean
# comparison site may have any other site within the buffer that is completed by the end of its
# cohort's window.
site_completion = sites["completion_year"].astype("float64").to_numpy()
earliest_neighbour = pd.Series(site_completion[site_second][site_first != site_second]).groupby(
    site_first[site_first != site_second]).min().reindex(range(len(sites)))
clean_comparison = comparison_rows[~comparison_rows["contaminated"]]
neighbour_year_of_clean = earliest_neighbour.reindex(clean_comparison["site_id"]).to_numpy()
assert not (neighbour_year_of_clean <= clean_comparison["cohort"].to_numpy() + EVENT_WINDOW).any()

event_site_table = event_panel[event_panel["rel_year"] == 0]
cohort_counts = (event_site_table.assign(clean=~event_site_table["contaminated"])
                 .groupby(["cohort", "group"], observed=True)
                 .agg(candidates=("site_id", "size"), clean=("clean", "sum")).unstack("group"))
cohort_counts.columns = [f"{group}: {kind}" for kind, group in cohort_counts.columns]
cohort_counts = cohort_counts[[f"{group}: {kind}" for group in EVENT_GROUPS for kind in ["candidates", "clean"]]]
clean_event_site_table = event_site_table[~event_site_table["contaminated"]]
cohort_counts["treated clean: new construction"] = (
    clean_event_site_table[(clean_event_site_table["group"] == "treated")
                      & (clean_event_site_table["construction_type"] == "New Construction")]
    .groupby("cohort").size().reindex(cohort_counts.index, fill_value=0))
cohort_counts["treated clean: clusters"] = (clean_event_site_table[clean_event_site_table["group"] == "treated"]
                                            .groupby("cohort")["cluster_id"].nunique())
cohort_counts["comparison clean: clusters"] = (clean_event_site_table[clean_event_site_table["group"] == "comparison"]
                                               .groupby("cohort")["cluster_id"].nunique())
display(cohort_counts)

# %%
def plot_cohort_counts(cohort_counts):
    """Candidate and clean site counts per completion cohort, one panel per group."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    bar_height = 0.38
    cohort_positions = np.arange(len(cohort_counts))
    largest_candidate_count = max(cohort_counts[f"{group}: candidates"].max() for group in EVENT_GROUPS)
    for ax, group, color in zip(axes, EVENT_GROUPS, [PALETTE["treated"], PALETTE["control"]]):
        candidates = cohort_counts[f"{group}: candidates"].to_numpy()
        clean = cohort_counts[f"{group}: clean"].to_numpy()
        ax.barh(cohort_positions - bar_height / 2, candidates, height=bar_height, color=PALETTE["before"],
                label="All candidates")
        ax.barh(cohort_positions + bar_height / 2, clean, height=bar_height, color=color,
                label="Clean (no other completion nearby)")
        for position, (n_candidates, n_clean) in enumerate(zip(candidates, clean)):
            ax.text(n_candidates + largest_candidate_count * LABEL_GAP, position - bar_height / 2,
                    f"{n_candidates:,}", va="center", fontsize=9)
            ax.text(n_clean + largest_candidate_count * LABEL_GAP, position + bar_height / 2, f"{n_clean:,}",
                    va="center", fontsize=9, fontweight="bold")
        ax.set_yticks(cohort_positions)
        ax.set_yticklabels(cohort_counts.index)
        ax.set_xlim(0, largest_candidate_count * AXIS_HEADROOM)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        thousands(ax, "x")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        finish(ax, f"{group.capitalize()} sites per cohort", "Number of sites", "Completion cohort (year)")
    plt.tight_layout()
    plt.show()


plot_cohort_counts(cohort_counts)

# %%
treated_total = int(cohort_counts["treated: candidates"].sum())
treated_clean_total = int(cohort_counts["treated: clean"].sum())
comparison_distinct = event_site_table.loc[event_site_table["group"] == "comparison", "site_id"].nunique()
comparison_clean_distinct = clean_event_site_table.loc[clean_event_site_table["group"] == "comparison", "site_id"].nunique()
thinnest_cohort = cohort_counts["comparison: clean"].idxmin()
say(f"The cohorts with a full window are {', '.join(str(cohort) for cohort in cohort_counts.index)}. They hold "
    f"{treated_total:,} treated sites, of which {treated_clean_total:,} "
    f"({pct(treated_clean_total, treated_total):.0f}%) have no other completion nearby in a different year of "
    f"their window; {int(cohort_counts['treated clean: new construction'].sum())} of those are new "
    f"construction. Of {comparison_distinct:,} different comparison candidates, {comparison_clean_distinct:,} "
    f"are clean for at least one cohort. The thinnest comparison group is cohort {thinnest_cohort} with "
    f"{int(cohort_counts.loc[thinnest_cohort, 'comparison: clean'])} clean sites in "
    f"{int(cohort_counts.loc[thinnest_cohort, 'comparison clean: clusters'])} clusters. The event table has "
    f"{len(event_panel):,} rows.")

# %% [markdown]
# The screen removes most candidates on both sides. Without it, the large majority of comparison
# sites would have had a completed building within their buffer during or before the window, and
# the comparison would have been between treated streets and other treated streets. What is left
# is a much smaller sample, and the later cohorts have fewer comparison sites because fewer
# buildings remain unfinished that late. The clean sites are also not a random subset: a site
# with no completed neighbour lies, by definition, in a less dense part of the programme. Results
# from the event table describe those places and are stated with their sample sizes.
#
# The alternative of keeping contaminated sites and adding the number of nearby completions as a
# control was rejected for the main analysis, because it replaces a clear rule with a modelling
# assumption about how effects add up.
#
# #### 5.4.3 What the screen does not rule out
#
# The screen looks for completed sites within one buffer radius of a comparison site. Two
# buffers of that radius still overlap while their centres are up to twice the radius apart, so a
# comparison site that passes the screen can have a completed building 250 to 500 metres away,
# and a crash between the two lies in both buffers. Such a crash is counted for the comparison
# site and for the treated one. Crashes shared this way make the comparison sites partly
# resemble the treated ones, which would pull a difference between the two toward zero. The next
# cell measures how many clean comparison sites have such a neighbour, and how many of their
# injury crashes are shared with it.

# %%
def other_site_completion_per_pair(pairs, site_frame):
    """For each (site, crash) pair, the earliest completion year of any other site that reaches the same crash.

    Sites that are not completed count as infinitely late. Sorting the pairs by crash and then
    by completion year puts the earliest site first in each crash's group, so a pair takes the
    group's first completion, or the second one when the pair is itself the first.
    """
    completion = site_frame["completion_year"].astype("float64").to_numpy()
    pair_completion = np.where(np.isnan(completion), np.inf, completion)[pairs["site_id"].to_numpy()]
    crash_rows = pairs["crash_row"].to_numpy()
    order = np.lexsort((pair_completion, crash_rows))
    sorted_rows = crash_rows[order]
    sorted_completion = pair_completion[order]
    is_first = np.r_[True, sorted_rows[1:] != sorted_rows[:-1]]
    group_start = np.flatnonzero(is_first)
    group_size = np.diff(np.r_[group_start, len(order)])
    group_id = np.cumsum(is_first) - 1
    start, size = group_start[group_id], group_size[group_id]
    second_completion = np.where(size > 1, sorted_completion[np.minimum(start + 1, len(order) - 1)], np.inf)
    other_sorted = np.where(np.arange(len(order)) == start, second_completion, sorted_completion[start])
    other_completion = np.empty(len(order))
    other_completion[order] = other_sorted
    return other_completion


clean_comparison_units = clean_event_site_table.loc[clean_event_site_table["group"] == "comparison", ["site_id", "cohort"]]
wide_first, wide_second = site_neighbour_pairs(sites, 2 * BUFFER_M)
is_other_wide = wide_first != wide_second
earliest_wide_neighbour = pd.Series(site_completion[wide_second][is_other_wide]).groupby(
    wide_first[is_other_wide]).min().reindex(range(len(sites)))
window_end = clean_comparison_units["cohort"].to_numpy() + EVENT_WINDOW
has_wide_neighbour = earliest_wide_neighbour.reindex(clean_comparison_units["site_id"]).to_numpy() <= window_end

injury_pair_context = injury_pairs.assign(
    other_completion=other_site_completion_per_pair(injury_pairs, sites),
    crash_year=crashes["year"].to_numpy()[injury_pairs["crash_row"].to_numpy()])
comparison_injury_pairs = clean_comparison_units.merge(
    injury_pair_context[["site_id", "crash_year", "other_completion"]], on="site_id")
in_cohort_window = comparison_injury_pairs["crash_year"].between(
    comparison_injury_pairs["cohort"] - EVENT_WINDOW, comparison_injury_pairs["cohort"] + EVENT_WINDOW)
comparison_injury_pairs = comparison_injury_pairs[in_cohort_window]
is_shared_with_completed = (comparison_injury_pairs["other_completion"]
                            <= comparison_injury_pairs["cohort"] + EVENT_WINDOW)
say(f"Of {len(clean_comparison_units):,} clean comparison units (a site in one cohort), "
    f"{has_wide_neighbour.sum():,} ({100 * has_wide_neighbour.mean():.1f}%) have another site within "
    f"{2 * BUFFER_M} m that is completed by the end of the cohort's window. Of the "
    f"{len(comparison_injury_pairs):,} injury crashes in their windows, {is_shared_with_completed.sum():,} "
    f"({100 * is_shared_with_completed.mean():.1f}%) also lie within {BUFFER_M} m of such a site, so they are "
    f"counted for the comparison site and for a completed site.")

# The same overlap on the treated side: another site within twice the radius that is completed in a
# different year of the treated site's own window, so that its buffer can share crashes with
# the treated one. Neighbours completed in the same year are one event and are not counted.
own_completion = site_completion[wide_first]
neighbour_completion_wide = site_completion[wide_second]
shares_window_with_other_event = (is_other_wide
                                  & (neighbour_completion_wide >= own_completion - EVENT_WINDOW)
                                  & (neighbour_completion_wide <= own_completion + EVENT_WINDOW)
                                  & (neighbour_completion_wide != own_completion))
overlaps_other_completion = np.bincount(wide_first[shares_window_with_other_event], minlength=len(sites)) > 0
clean_treated_units = clean_event_site_table[clean_event_site_table["group"] == "treated"]
clean_treated_overlap = overlaps_other_completion[clean_treated_units["site_id"].to_numpy()]
is_new_construction = (clean_treated_units["construction_type"] == "New Construction").to_numpy()
say(f"On the treated side, {clean_treated_overlap.sum():,} of {len(clean_treated_units):,} clean treated sites "
    f"({100 * clean_treated_overlap.mean():.0f}%) have another site within {2 * BUFFER_M} m that is completed "
    f"in a different year of their own window; among the {is_new_construction.sum():,} clean new-construction "
    f"sites it is {clean_treated_overlap[is_new_construction].sum():,} "
    f"({100 * clean_treated_overlap[is_new_construction].mean():.0f}%). Most treated sites that pass the "
    f"screen therefore still share part of their buffer with another completion.")

# %% [markdown]
# The share is large on both sides, so Section 9.5 repeats the comparison with only those sites,
# on either side, whose buffer does not overlap the buffer of another completion. Its size is
# stated here so that a contrast between treated and comparison sites is read with it in mind.

# %% [markdown]
# ### 5.5 Tagging every crash with its relation to housing
#
# This table describes how the crashes relate to housing and is used for the overview below. The
# composition questions (RQ6, RQ8, RQ10) work from the pair list and the clean sites of 5.4,
# where a crash near two clean sites counts for both. The pair list holds every crash within the
# buffer of every site, and this subsection reduces it to one answer per crash.
#
# A crash near several sites is assigned to the **nearest** one. That makes the assignment a
# partition: each crash belongs to at most one site, the same place always belongs to the same
# site whatever the year, and comparing the crashes of one site before and after its completion
# compares the same streets. When two sites are at the same distance, which happens when one
# building appears in two projects, the earlier completion is taken, then the lower `site_id`, so
# the rule is deterministic.
#
# The relation is defined on calendar years, as in the event table. A crash in a year before the
# nearest site's completion year, or near a site with no completion date, is "Before completion".
# A crash in a later year is "After completion". A crash in the completion year itself gets its
# own label, "Completion year", and belongs to neither side: the building was finished for part
# of that year only, and many completion dates in this table are administrative (Section 4.4
# found many buildings completed on their start date), so splitting the year at the recorded day
# would claim a precision that the dates do not have.
#
# Because the nearest site can be unfinished while another site inside the buffer is already
# completed, a second column, `near_completed_site`, records whether any site within the buffer
# was completed in an earlier year than the crash.

# %%
def tag_crashes_with_housing(crash_frame, site_frame, pairs):
    """Add the housing-relation columns to the crash frame (one nearest site per crash) and return it."""
    n_crashes = len(crash_frame)
    crash_rows = pairs["crash_row"].to_numpy()
    site_ids = pairs["site_id"].to_numpy()
    completion = site_frame["completion_year"].astype("float64").to_numpy()
    pair_completion = completion[site_ids]
    crash_year = crash_frame["year"].to_numpy()

    # Sort the pairs by crash, then distance, then completion year (missing last), then site id.
    # The first pair of each crash is then its nearest site under the tie-break described above.
    completion_sort_key = np.where(np.isnan(pair_completion), np.inf, pair_completion)
    order = np.lexsort((site_ids, completion_sort_key, pairs["distance_m"].to_numpy(), crash_rows))
    sorted_rows = crash_rows[order]
    is_first = np.r_[True, sorted_rows[1:] != sorted_rows[:-1]]
    nearest_pairs = order[is_first]
    near_rows = crash_rows[nearest_pairs]

    near_site = pd.array(np.full(n_crashes, pd.NA), dtype="Int32")
    near_site[near_rows] = site_ids[nearest_pairs]
    near_cluster = pd.array(np.full(n_crashes, pd.NA), dtype="Int32")
    near_cluster[near_rows] = site_frame["cluster_id"].to_numpy()[site_ids[nearest_pairs]]
    near_distance = np.full(n_crashes, np.nan, dtype="float32")
    near_distance[near_rows] = pairs["distance_m"].to_numpy()[nearest_pairs]
    near_completion = np.full(n_crashes, np.nan)
    near_completion[near_rows] = pair_completion[nearest_pairs]
    relative_year = crash_year - near_completion

    is_located = crash_frame["is_located"].to_numpy()
    has_site = np.zeros(n_crashes, dtype=bool)
    has_site[near_rows] = True
    relation = np.select(
        [~is_located, ~has_site, relative_year > 0, relative_year == 0],
        ["No coordinates", "Not near a site", "After completion", "Completion year"],
        default="Before completion")

    completed_earlier = pair_completion < crash_year[crash_rows]
    crash_frame["housing_relation"] = pd.Categorical(relation, categories=HOUSING_RELATIONS)
    crash_frame["near_site_id"] = near_site
    crash_frame["near_cluster_id"] = near_cluster
    crash_frame["near_site_distance_m"] = near_distance
    crash_frame["near_completion_year"] = pd.array(near_completion, dtype="Int16")
    crash_frame["rel_year"] = pd.array(relative_year, dtype="Int16")
    crash_frame["n_sites_near"] = np.bincount(crash_rows, minlength=n_crashes).astype("int16")
    crash_frame["near_completed_site"] = np.bincount(crash_rows[completed_earlier], minlength=n_crashes) > 0
    return crash_frame


crashes = tag_crashes_with_housing(crashes, sites, site_pairs)

assert crashes["housing_relation"].notna().all(), "every crash gets exactly one relation to housing"
assert crashes["near_site_id"].notna().sum() == site_pairs["crash_row"].nunique()
assert (crashes["near_site_id"].notna() == (crashes["n_sites_near"] > 0)).all()
assert crashes.loc[crashes["near_site_id"].notna(), "near_site_distance_m"].le(BUFFER_M).all()
assert (crashes["housing_relation"] == "No coordinates").sum() == (~crashes["is_located"]).sum()
assert crashes.loc[crashes["housing_relation"] == "After completion", "near_completed_site"].all()

injury_crash_rows = crashes[crashes["is_injury_crash"]]
relation_table = pd.DataFrame({
    "all crashes": crashes["housing_relation"].value_counts().reindex(HOUSING_RELATIONS),
    "injury crashes": injury_crash_rows["housing_relation"].value_counts().reindex(HOUSING_RELATIONS),
})
relation_table["% of injury crashes"] = (100 * relation_table["injury crashes"] / len(injury_crash_rows)).round(1)
display(relation_table)

before_rows = injury_crash_rows[injury_crash_rows["housing_relation"] == "Before completion"]
relation_by_year = (pd.crosstab(injury_crash_rows["year"], injury_crash_rows["housing_relation"])
                    .reindex(columns=HOUSING_RELATIONS, fill_value=0).drop(columns="No coordinates"))
relation_share_by_year = 100 * relation_by_year.div(relation_by_year.sum(axis=1), axis=0)


def plot_relation_shares(relation_share_by_year):
    """One line per housing relation: share of located injury crashes by crash year."""
    # One line per state with its name at the right end: in a stacked bar the middle bands float,
    # so their changes over the years could not be compared.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for relation_name in relation_share_by_year.columns:
        shares = relation_share_by_year[relation_name]
        color = RELATION_COLORS[relation_name]
        ax.plot(shares.index, shares.to_numpy(), color=color, linewidth=2.2, marker="o", markersize=6,
                linestyle="--" if relation_name == "Not near a site" else "-")
        ax.annotate(f"{relation_name} ({shares.iloc[-1]:.0f}%)", (YEAR_END, shares.iloc[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=color)
    ax.set_xticks(relation_share_by_year.index)
    ax.set_xlim(YEAR_START - 0.4, YEAR_END + 3.2)
    ax.set_ylim(0, relation_share_by_year.to_numpy().max() * 1.1)
    finish(ax, f"Located injury crashes by relation to the nearest housing site within {BUFFER_M} m",
           "Crash year", "Share of located injury crashes (%)")
    plt.tight_layout()
    plt.show()


plot_relation_shares(relation_share_by_year)

# %%
first_year_share = relation_share_by_year.loc[YEAR_START]
last_year_share = relation_share_by_year.loc[YEAR_END]
say(f"Of {len(injury_crash_rows):,} injury crashes, {relation_table.loc['No coordinates', '% of injury crashes']:.1f}% "
    f"have no coordinates and cannot be tagged, and "
    f"{relation_table.loc['Not near a site', '% of injury crashes']:.1f}% are located but not within {BUFFER_M} m "
    f"of any site. The rest are near a site: {int(relation_table.loc['Before completion', 'injury crashes']):,} "
    f"before its completion, {int(relation_table.loc['Completion year', 'injury crashes']):,} in the completion "
    f"year and {int(relation_table.loc['After completion', 'injury crashes']):,} after. Among located injury "
    f"crashes the share that falls after the nearest site's completion rises from "
    f"{first_year_share['After completion']:.0f}% in {YEAR_START} to {last_year_share['After completion']:.0f}% "
    f"in {YEAR_END}, while the share not near any site is {first_year_share['Not near a site']:.0f}% in "
    f"{YEAR_START}, {last_year_share['Not near a site']:.0f}% in {YEAR_END} and between "
    f"{relation_share_by_year['Not near a site'].min():.0f}% and "
    f"{relation_share_by_year['Not near a site'].max():.0f}% in every year. "
    f"Of the crashes tagged as before completion, {100 * before_rows['near_completed_site'].mean():.0f}% have "
    f"another, already completed site within the buffer.")

# %% [markdown]
# The share of crashes near any site hardly moves over the ten years, which is expected because
# the set of sites is fixed: it contains every located building of the programme whatever its
# completion year. What changes is the split between before and after, simply because more
# buildings are finished as time passes. A comparison of "after" crashes with "before" crashes
# is therefore also a comparison of later years with earlier years, and any crash-level question
# has to compare shares within the same sites or the same years, not raw counts. The last number
# above shows why `near_completed_site` is kept: many crashes that precede their nearest site's
# completion already had a finished building within the buffer, so "before completion" does not
# mean that no housing was nearby.

# %% [markdown]
# ### 5.6 What the integrated tables look like
#
# Four tables now exist besides the two cleaned ones. The ZIP by year panel is still the naive
# version and is replaced in Section 6; the three spatial tables are final.

# %%
integrated_overview = pd.DataFrame([
    {"table": "sites", "grain": "one located building in one project",
     "rows": len(sites), "columns": sites.shape[1], "unique key": "site_id"},
    {"table": "site_year", "grain": "site x calendar year",
     "rows": len(site_year), "columns": site_year.shape[1], "unique key": "site_id, year"},
    {"table": "event_panel", "grain": "site x cohort x relative year",
     "rows": len(event_panel), "columns": event_panel.shape[1], "unique key": "site_id, cohort, rel_year"},
    {"table": "crashes (tagged)", "grain": "one collision",
     "rows": len(crashes), "columns": crashes.shape[1], "unique key": "collision_id"},
    {"table": "naive_panel (replaced in 6.1)", "grain": "ZIP x year, unbalanced",
     "rows": len(naive_panel), "columns": naive_panel.shape[1], "unique key": "zip, year"},
]).set_index("table")
display(integrated_overview)

# %%
display(sites.head(4))
display(site_year.head(4))
display(event_panel[event_panel["rel_year"].isin([-1, 0, 1])].head(6))
display(crashes.loc[crashes["near_site_id"].notna(),
                    ["collision_id", "crash_date", "year", "is_injury_crash", "housing_relation", "near_site_id",
                     "near_cluster_id", "near_site_distance_m", "near_completion_year", "rel_year",
                     "n_sites_near", "near_completed_site"]].head(5))

# %% [markdown]
# Each table is unique on its key, and none of them was produced by a row-level key join between
# housing and crashes. The spatial tables were built by counting crashes inside buffers, and the
# ZIP panel by aggregating first. The working objects that are no longer needed are released; the
# crash tree, the pair list and the functions stay, because Section 9 re-runs this section's steps
# at other radii.

# %%
release("real_merge", "crashes_with_zip", "housing_with_zip", "zip_year_merge", "tree_pairs", "brute_pairs",
        "check_tree", "provisional_sites", "injury_pairs", "pair_is_injury", "clusters_per_crash",
        "injury_crash_rows", "before_rows", "treated_rows", "comparison_rows", "clean_comparison",
        "event_site_table", "clean_event_site_table", "injury_pair_context", "comparison_injury_pairs",
        "earliest_wide_neighbour", "wide_first", "wide_second", "own_completion",
        "neighbour_completion_wide", "shares_window_with_other_event", "clean_treated_units")
