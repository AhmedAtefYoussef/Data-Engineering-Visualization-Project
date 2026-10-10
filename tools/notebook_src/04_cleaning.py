# %% [markdown]
# ## 4. Pre-integration cleaning
#
# ### 4.0 Policy, and how each decision is recorded
#
# Both raw tables were loaded as text, so nothing has been converted yet. This section turns them
# into two typed frames, `housing` and `crashes`, and every change follows one of four actions.
#
# A row is **dropped** only when it cannot be used for anything: it has no key, it repeats another
# row, or it lies outside the study window. Dropping is the last choice because a crash row that
# lacks a ZIP code still counts toward a borough total and may still have coordinates.
#
# A value is **imputed** only when the correct value follows from other fields of the same row
# with no guessing. Two cases qualify: a blank unit count that the publisher uses to mean zero,
# and a missing casualty total that equals the sum of its parts. Statistical imputation (a mean, a
# model) is not used anywhere in this section. A missing coordinate or a missing completion date
# has no deterministic answer, so it stays missing.
#
# A value or row is **flagged** when it is unusual but plausible, or wrong in a way that cannot be
# repaired. The row is kept and a boolean column lets a later section include or exclude it. Large
# buildings, crashes with many injured people, confidential housing records and completion dates
# that precede the start date are all handled this way.
#
# A value is **standardised** when only its form changes: text to dates, upper case to title case,
# free text to a fixed list of categories, a placeholder code to a missing value.
#
# Every rule calls `log_rule(dataset, rule, action, rows_before, rows_after, affected, reason)`.
# `rows_before` and `rows_after` show whether rows were lost, and `affected` counts the rows or
# cells the rule touched. Rules that find nothing to fix are logged too, with zero affected, so the
# log also shows what was checked. The full log is printed in 4.13.
#
# The five cleaning tasks of the assignment map to the subsections like this.
#
# | Task | Housing | Crashes |
# |---|---|---|
# | Missing values (drop against impute) | 4.1, 4.3, 4.4 | 4.7, 4.8, 4.9 |
# | Outliers (IQR and domain rules) | 4.6 | 4.8 (coordinates), 4.10 |
# | Inconsistent data | 4.4, 4.5 | 4.8, 4.9 |
# | Format standardisation | 4.1, 4.3, 4.4 | 4.7, 4.8, 4.11 |
# | Duplicates | 4.2 | 4.7 |
#
# The constants below are used only in this section. The ones shared with Section 3 (date layout,
# placeholder values, the unit-column lists, the outlier multiplier and the parsing helpers) are
# defined once in 2.1.

# %%
# --- Constants for Section 4 ------------------------------------------------------------------
# ZIP code ranges the Postal Service assigns to each borough (inclusive, as integers). A code is
# accepted as a New York City ZIP only if it falls in one of them. 11001 (Floral Park) and 11040
# (New Hyde Park) straddle the Queens / Nassau County line, so they are listed under Queens.
NYC_ZIP_RANGES = {
    "Manhattan": [(10001, 10282)],
    "Staten Island": [(10301, 10314)],
    "Bronx": [(10451, 10475)],
    "Queens": [(11001, 11001), (11004, 11005), (11040, 11040), (11101, 11120), (11351, 11697)],
    "Brooklyn": [(11201, 11256)],
}
DEEP_AFFORDABLE_COLUMNS = ["extremely_low_income_units", "very_low_income_units"]

# Housing columns that are not carried into the clean frame: finer geographic identifiers that the
# crash table has no counterpart for, and a second coordinate pair that is missing in the same rows.
HOUSING_COLUMNS_NOT_KEPT = ["bbl", "bin", "council_district", "census_tract",
                            "neighborhood_tabulation_area", "latitude_internal", "longitude_internal"]

SIZE_BAND_EDGES = [0, 4, 19, 49, 149, np.inf]              # total units, upper edge inclusive
SIZE_BAND_LABELS = ["1-4", "5-19", "20-49", "50-149", "150+"]   # from row houses to towers

# Crash columns: raw name -> short name used from here on.
CASUALTY_COLUMNS = {
    "number_of_persons_injured": "injured", "number_of_persons_killed": "killed",
    "number_of_pedestrians_injured": "ped_injured", "number_of_pedestrians_killed": "ped_killed",
    "number_of_cyclist_injured": "cyc_injured", "number_of_cyclist_killed": "cyc_killed",
    "number_of_motorist_injured": "mot_injured", "number_of_motorist_killed": "mot_killed",
}
ROAD_USER_PREFIXES = ["ped", "cyc", "mot"]   # the three road-user roles the table breaks out

# Time-of-day bands as (first hour, last hour), inclusive, in 24-hour clock.
HOUR_BANDS = {"Night": (0, 5), "Morning": (6, 11), "Afternoon": (12, 17), "Evening": (18, 23)}

# Contributing-factor groups. Each raw value is lower-cased and trimmed, then given the group of
# the FIRST pattern it matches, so the order below matters and is explained line by line.
FACTOR_RULES = [
    # the explicit "no factor recorded" label and bare numeric codes ("80", "1") carry no information
    ("Unspecified", r"^unspecified$|^\d+$"),
    # the one factor that blames a person outside a vehicle; kept apart because RQ6 looks at them
    ("Pedestrian or cyclist error", r"pedestrian|bicyclist"),
    # before "Vehicle defect" (pavement / shoulders "defective") and before "Improper manoeuvre"
    # ("lane marking"), and before "Failure to yield or obey" ("traffic control device")
    ("Road or environment", r"pavement|shoulders|lane marking|traffic control device|obstruct|debris"
                            r"|glare|animals"),
    # named parts, so that "tire failure" is not caught by "failure to yield" further down
    ("Vehicle defect", r"defective|defects|steering|tire|windshield|tinted"),
    # "illnes" absorbs the misspelling "Illnes"; "drugs" absorbs "(illegal)" and "(Illegal)"
    ("Impairment or medical", r"alcohol|drugs|medication|asleep|fatigued|drowsy|consciousness"
                              r"|illnes|disability"),
    # "cell phone" absorbs "(hand-held)", "(hand-Held)" and "(hands-free)"
    ("Distraction", r"distraction|inattention|cell phone|electronic|texting|headphones|eating"
                    r"|navigation"),
    ("Speed or aggressive driving", r"speed|aggressive|following too closely"),
    ("Failure to yield or obey", r"yield|disregarded|keep right"),
    # after "Road or environment", so that "lane" here means lane use, not lane markings
    ("Improper manoeuvre", r"passing|lane|turning|backing|inexperience"),
    ("Other", r"other vehicular|uninvolved|vandalism|driverless|oversized"),
]
FACTOR_GROUPS = [group for group, _ in FACTOR_RULES]
FACTOR_FALLBACK = "Other"   # group for a value that no pattern recognises

MAP_SAMPLE_ROWS = 60_000    # crashes drawn in the coordinate figure; enough to show the street grid
MINUTES_PER_HOUR = 60       # a crash minute recorded evenly would be on the hour in 1 of these
ROUNDING_STEP_MIN = 5       # people round minutes to multiples of five; 1 in 5 minutes is one anyway


# %% [markdown]
# Two small helpers are shared by both tables: the ZIP validator and the coordinate validator.
# Using the same functions on both sides guarantees that a ZIP or a point accepted in one table
# would also be accepted in the other, which the integration step relies on. Numbers are parsed
# with the strict `to_number` from 2.1, which stops the notebook on a value that is not a number.

# %%
def standardise_zip(zip_text):
    """Return (zip, implied_borough): a 5-character NYC ZIP or missing, and the borough of its range.

    The text is trimmed and a trailing '.0' is removed (the form a ZIP takes when a file is read
    with default types). Anything that is not five digits inside NYC_ZIP_RANGES becomes missing;
    this removes blanks, the 10000 placeholder and codes from outside the city in one test.
    """
    text = zip_text.str.strip().str.replace(r"\.0$", "", regex=True)
    is_five_digits = text.str.fullmatch(r"\d{5}").fillna(False).astype(bool)
    number = pd.to_numeric(text.where(is_five_digits), errors="coerce").astype("float64")
    implied_borough = pd.Series(np.nan, index=zip_text.index, dtype=object)
    for borough_name, ranges in NYC_ZIP_RANGES.items():
        for low, high in ranges:
            implied_borough[number.between(low, high)] = borough_name
    clean_zip = text.where(implied_borough.notna()).astype("string[pyarrow]")
    return clean_zip, implied_borough


def validate_coordinates(latitude_text, longitude_text):
    """Return (lat, lon, n_zero, n_outside): float64 coordinates, missing unless inside NYC_BOUNDS.

    (0, 0) is the placeholder an automatic address lookup (geocoder) writes when it fails. It is counted apart
    from other out-of-box points because the two have different causes.
    """
    lat, lon = to_number(latitude_text), to_number(longitude_text)
    is_zero = (lat == 0) & (lon == 0)
    in_box = lat.between(*NYC_BOUNDS["lat"]) & lon.between(*NYC_BOUNDS["lon"])
    is_outside = lat.notna() & lon.notna() & ~in_box & ~is_zero
    keep = in_box & ~is_zero
    return lat.where(keep), lon.where(keep), int(is_zero.sum()), int(is_outside.sum())


# %% [markdown]
# ### 4.1 Housing: types, and blank unit counts as structural zeros
#
# The housing table counts units in many columns (by income band, by bedroom count, by tenure,
# and two totals). The raw profile shows these columns are mostly blank, and that no cell
# in any of them contains a zero. A count column with no zeros and many blanks means the blank is
# how the publisher writes zero. The decision is to impute zero. Dropping rows with blanks would
# remove almost the whole table, and leaving them missing would make every sum over income bands
# missing too. The claim is tested rather than assumed: if blank means zero, then after the fill
# the income bands, the bedroom counts and the two tenure counts must each add up to
# `all_counted_units` in every row. Section 4.5 asserts exactly that.
#
# Identifiers become integers (`building_id` is a nullable integer because withheld records have
# none), and the two yes/no status columns become booleans.

# %%
def type_housing(raw):
    """Build the typed housing frame: keys, names, status flags and zero-filled unit counts."""
    frame = pd.DataFrame(index=raw.index)
    frame["housing_id"] = np.arange(len(raw), dtype="int32")       # surrogate row key for later joins
    frame["project_id"] = to_number(raw["project_id"]).astype("int64")
    frame["building_id"] = to_number(raw["building_id"]).astype("Int64")
    frame["project_name"] = raw["project_name"].str.strip()

    status_values = {"extended_affordability_status": {"Yes", "No"},
                     "prevailing_wage_status": {"Prevailing Wage", "Non Prevailing Wage"}}
    for column, allowed in status_values.items():
        assert set(raw[column].dropna().unique()) <= allowed, f"unexpected value in {column}"
        assert raw[column].notna().all(), f"missing value in {column}"
    frame["extended_affordability"] = (raw["extended_affordability_status"] == "Yes").astype(bool)
    frame["prevailing_wage"] = (raw["prevailing_wage_status"] == "Prevailing Wage").astype(bool)

    for column in UNIT_COLUMNS:
        units = to_number(raw[column])
        assert ((units.dropna() % 1) == 0).all() and (units.dropna() >= 0).all(), column
        frame[column] = units.fillna(0).astype("int16")
    return frame


raw_unit_numbers = raw_housing[UNIT_COLUMNS].apply(to_number)
unit_blank_cells = int(raw_unit_numbers.isna().sum().sum())
unit_zero_cells_before = int((raw_unit_numbers == 0).sum().sum())

housing = type_housing(raw_housing)
log_rule("housing", f"blank unit counts set to 0 ({len(UNIT_COLUMNS)} unit columns)", "impute", len(raw_housing),
         len(housing), unit_blank_cells,
         "The publisher writes zero as blank; sums over bands need a number. Cells, not rows.")

unit_fill_table = pd.DataFrame({
    "blank before (%)": 100 * raw_unit_numbers.isna().mean(),
    "zero before (%)": 100 * (raw_unit_numbers == 0).mean(),
    "zero after (%)": 100 * (housing[UNIT_COLUMNS] == 0).mean(),
}).round(1)
display(unit_fill_table)
# The fill must not change any total: total_units is never blank, so its sum is the same before and after.
assert housing["total_units"].sum() == to_number(raw_housing["total_units"]).sum()
assert (housing[UNIT_COLUMNS] >= 0).all().all()
say(f"Before the fill the {len(UNIT_COLUMNS)} unit columns held {unit_blank_cells:,} blank cells and "
    f"{unit_zero_cells_before:,} cells equal to zero. Afterwards no cell is blank and the share of "
    f"zeros in each column equals the share that was blank, so nothing else changed.")

# %% [markdown]
# The blank share and the zero share swap places column by column, which is the expected result of
# a pure relabelling. `total_units` was never blank, so it is the one column the rule leaves alone.

# %% [markdown]
# ### 4.2 Housing: duplicates on the true key
#
# A building can appear in more than one project, for example when it is financed once for
# construction and again years later for preservation. Those are separate public investments with
# their own dates and unit counts, so one `building_id` on several rows is not a duplicate. The
# grain of the table is the pair (`project_id`, `building_id`), and that pair is what the
# duplicate check uses, together with a check for rows that are identical in every column.
#
# The alternative, keeping one row per `building_id`, was rejected after measuring what it would
# remove: the code below counts those rows and the units they carry. Withheld records have no
# `building_id` at all, so for them the check is that `project_id` alone is unique.

# %%
def repeats_key(frame):
    """True for rows that repeat an earlier (project_id, building_id) pair.

    Pandas treats two missing values as equal, so two withheld rows of one project (both without a
    building_id) would look like a repeat. Only rows that have a building_id are compared.
    """
    return (frame["building_id"].notna() & frame.duplicated(["project_id", "building_id"])).to_numpy()


def housing_duplicate_report(frame, raw):
    """Count duplicates under three definitions: full row, true key, and building_id alone."""
    has_building = frame["building_id"].notna()
    repeated_building = has_building & frame["building_id"].duplicated(keep="first")
    return {
        "full_row": int(raw.duplicated().sum()),
        "true_key": int(repeats_key(frame).sum()),
        "no_building_project_repeat": int(frame.loc[~has_building, "project_id"].duplicated().sum()),
        "building_only": int(repeated_building.sum()),
        "building_only_units": int(frame.loc[repeated_building, "total_units"].sum()),
        "building_only_distinct_start": int(
            raw.loc[has_building & frame["building_id"].duplicated(keep=False)]
            .groupby("building_id")["project_start_date"].nunique().gt(1).sum()),
        "buildings_repeated": int(frame.loc[repeated_building, "building_id"].nunique()),
    }


duplicate_counts = housing_duplicate_report(housing, raw_housing)
rows_before_dedup = len(housing)
is_duplicate_row = pd.Series(repeats_key(housing), index=housing.index) | raw_housing.duplicated()
# Withheld rows have no building_id, so only a row identical in every column could be dropped
# among them. Nothing is expected to be, and a refreshed download that did would stop here.
dropped_withheld_rows = int((is_duplicate_row & housing["building_id"].isna()).sum())
assert dropped_withheld_rows == 0, "a withheld (confidential) row would be dropped as a duplicate"
housing = housing.loc[~is_duplicate_row].reset_index(drop=True)
housing_source = raw_housing.loc[~is_duplicate_row.to_numpy()].reset_index(drop=True)
housing["housing_id"] = np.arange(len(housing), dtype="int32")
log_rule("housing", "duplicate rows on (project_id, building_id) or on every column", "drop",
         rows_before_dedup, len(housing), rows_before_dedup - len(housing),
         "A repeated key would double count units. building_id alone is not the key.")

assert not repeats_key(housing).any()
assert housing["housing_id"].is_unique
say(f"Full-row duplicates: {duplicate_counts['full_row']}. Duplicates on (project_id, building_id): "
    f"{duplicate_counts['true_key']}. Repeated project_id among rows without a building_id: "
    f"{duplicate_counts['no_building_project_repeat']}. Rows dropped: "
    f"{rows_before_dedup - len(housing)}, leaving {len(housing):,}.")
say(f"Deduplicating on building_id alone would have removed {duplicate_counts['building_only']} rows "
    f"on {duplicate_counts['buildings_repeated']} buildings, carrying "
    f"{duplicate_counts['building_only_units']:,} units. "
    f"{duplicate_counts['building_only_distinct_start']} of those {duplicate_counts['buildings_repeated']} "
    f"buildings have different project start dates on their rows, which marks them as separate "
    f"projects and not as repeated reports.")

# %% [markdown]
# The table is already unique on its true key, so the rule removes nothing and stays in place as a
# guard for a future download. The rows that share a building are kept. Any later step that counts
# places and not investments has to group by `building_id` or by coordinates, which is why the
# integration section works with sites.

# %% [markdown]
# ### 4.3 Housing: geographic identifiers, and records that cannot be located
#
# Three geographic fields are used for integration: borough, ZIP code and the coordinate pair.
# Borough is turned into a category with the fixed spelling of `BOROUGHS`. The ZIP code is kept as
# a five-character string, because it is a label and not a quantity, and is checked against the
# borough ZIP ranges. Coordinates become numbers and must fall inside `NYC_BOUNDS`.
#
# Part of the table is withheld by the publisher: the project name reads CONFIDENTIAL and the
# address, building identifier, ZIP and coordinates are all blank. These rows cannot be imputed,
# because nothing in the row narrows the location below the borough, and assigning a random ZIP
# inside the borough would invent a relationship with crashes. Dropping them would lose their
# units from borough totals. They are kept and flagged with `is_confidential`. A second flag,
# `is_located`, marks every row that has usable coordinates, and it is the filter the spatial
# analyses use.

# %%
def clean_housing_geography(frame, raw):
    """Add borough, zip, lat, lon, address and the two location flags; return (frame, report)."""
    frame = frame.copy()
    borough_text = raw["borough"].str.strip().str.title()
    frame["borough"] = pd.Categorical(borough_text, categories=BOROUGHS)
    frame["community_board"] = raw["community_board"].str.strip().astype("category")

    frame["zip"], implied_borough = standardise_zip(raw["postcode"])
    lat, lon, n_zero, n_outside = validate_coordinates(raw["latitude"], raw["longitude"])
    frame["lat"], frame["lon"] = lat, lon

    frame["is_confidential"] = (raw["project_name"].str.strip().str.upper() == CONFIDENTIAL_NAME).to_numpy()
    frame["is_located"] = frame["lat"].notna() & frame["lon"].notna()
    is_withheld_address = raw["house_number"].str.strip() == WITHHELD_ADDRESS
    address = raw["house_number"].str.strip() + " " + raw["street_name"].str.strip()
    frame["address"] = address.where(~is_withheld_address)

    stated = frame["borough"].astype(object)
    report = {
        "borough_unrecognised": int(frame["borough"].isna().sum()),
        "zip_raw_present": int(raw["postcode"].notna().sum()),
        "zip_rejected": int(raw["postcode"].notna().sum() - frame["zip"].notna().sum()),
        "coordinates_zero": n_zero, "coordinates_outside": n_outside,
        "zip_borough_disagree": int((implied_borough.notna() & (implied_borough != stated)).sum()),
        "disagree_zips": sorted(frame.loc[implied_borough.notna() & (implied_borough != stated), "zip"].unique()),
    }
    return frame, report


housing, housing_geo_report = clean_housing_geography(housing, housing_source)
confidential = housing["is_confidential"]
unlocated_not_confidential = int((~housing["is_located"] & ~confidential).sum())

log_rule("housing", "borough to category with the BOROUGHS spelling", "standardise", len(housing),
         len(housing), housing_geo_report["borough_unrecognised"],
         "One spelling on both tables; affected counts values outside the five boroughs.")
log_rule("housing", "postcode to 5-character string, checked against NYC ZIP ranges", "standardise",
         len(housing), len(housing), housing_geo_report["zip_rejected"],
         "ZIP is a label and a join key; affected counts codes rejected as not NYC.")
log_rule("housing", "coordinates at (0,0) or outside NYC_BOUNDS set to missing", "standardise",
         len(housing), len(housing),
         housing_geo_report["coordinates_zero"] + housing_geo_report["coordinates_outside"],
         "A point outside the city would corrupt distance joins.")
log_rule("housing", "confidential records kept and flagged (is_confidential)", "flag", len(housing),
         len(housing), int(confidential.sum()),
         "No address, ZIP or coordinates, so not imputable; still valid for borough totals.")
log_rule("housing", "rows without usable coordinates flagged (is_located = False)", "flag",
         len(housing), len(housing), int((~housing["is_located"]).sum()),
         "Spatial analyses need a point; ZIP and borough analyses do not.")
log_rule("housing", "ZIP range borough differs from stated borough (stated borough kept)", "flag",
         len(housing), len(housing), housing_geo_report["zip_borough_disagree"],
         "Checked only; a few ZIPs legitimately cross a borough line.")

location_table = (housing.groupby(["is_confidential", "is_located"], observed=True)
                  .agg(rows=("housing_id", "size"), total_units=("total_units", "sum"),
                       with_zip=("zip", "count"), with_building_id=("building_id", "count"))
                  .reset_index())
location_table["share_of_rows_pct"] = (100 * location_table["rows"] / len(housing)).round(2)
location_table["share_of_units_pct"] = (100 * location_table["total_units"]
                                        / housing["total_units"].sum()).round(2)
display(location_table)

assert housing["borough"].notna().all()
assert not housing.loc[confidential, ["zip", "lat", "lon", "building_id", "address"]].notna().any().any()
assert housing.loc[housing["is_located"], "lat"].between(*NYC_BOUNDS["lat"]).all()
assert housing.loc[housing["is_located"], "lon"].between(*NYC_BOUNDS["lon"]).all()
assert housing["zip"].dropna().str.len().eq(5).all()

say(f"Borough values outside the five boroughs: {housing_geo_report['borough_unrecognised']}. "
    f"ZIP codes rejected: {housing_geo_report['zip_rejected']} of "
    f"{housing_geo_report['zip_raw_present']:,} present; {housing['zip'].nunique()} distinct ZIPs remain. "
    f"Coordinates at (0,0): {housing_geo_report['coordinates_zero']}; outside the city box: "
    f"{housing_geo_report['coordinates_outside']}.")
say(f"Confidential records: {int(confidential.sum()):,} rows "
    f"({pct(confidential.sum(), len(housing)):.1f}% of rows) holding "
    f"{int(housing.loc[confidential, 'total_units'].sum()):,} units "
    f"({pct(housing.loc[confidential, 'total_units'].sum(), housing['total_units'].sum()):.2f}% of all units). "
    f"A further {unlocated_not_confidential} named records also lack coordinates and a ZIP. "
    f"Located rows: {int(housing['is_located'].sum()):,}.")
say(f"The ZIP range and the stated borough disagree in {housing_geo_report['zip_borough_disagree']} rows, "
    f"all in ZIP {', '.join(housing_geo_report['disagree_zips'])}.")

# %% [markdown]
# The geographic fields of this table were already clean: no borough, ZIP or coordinate had to be
# rejected. The work in this step is therefore the typing and the two flags. The confidential
# records are a large share of the rows but a very small share of the units, because most of them
# are single homes. The handful of named records without coordinates also lack a ZIP and a tax
# lot number, so there is nothing in the row to look up an address from. Sending the addresses of
# the others to an outside geocoding service (a web service that turns an address into coordinates)
# was considered and rejected, because the notebook would then depend on a third source and a
# network call. The one ZIP where the range and the stated borough disagree
# is Marble Hill, which belongs to Manhattan but gets its mail through the Bronx, so the stated
# borough is right and is kept.

# %% [markdown]
# ### 4.4 Housing: dates, completion and date consistency
#
# The three date columns arrive as text in one fixed timestamp layout. They are parsed with that
# exact layout and without tolerance, so a value in any other form would stop the notebook instead
# of turning silently into a missing date.
#
# The analysis needs one completion date per row. The building's own completion date is the best
# measure of when residents could move in. Where it is blank, the project completion date is used,
# which is an imputation from the same row. Where both are blank the building is not finished:
# the date stays missing and `is_completed` is false. Filling those with a typical construction
# time was rejected because the completion year is the treatment date of the event study, and an
# invented date would assign crashes to the wrong side of it.
#
# Two consistency checks follow. A completion date earlier than the project start date is
# impossible for construction, but the start date in this table is the date the financing closed,
# which can come after the work for some programmes. These rows are flagged, not corrected,
# because there is no way to tell which of the two dates is wrong. Rows where completion and start
# fall on the same day are flagged too: they have no observable construction period.

# %%
def clean_housing_dates(frame, raw):
    """Parse the three dates and derive completion_date, completion_year and the date flags."""
    frame = frame.copy()
    for column in ["project_start_date", "project_completion_date", "building_completion_date"]:
        frame[column] = pd.to_datetime(raw[column], format=DATE_FORMAT, errors="raise")
        assert frame[column].notna().sum() == raw[column].notna().sum(), f"{column}: values lost in parsing"
        assert (frame[column].dropna().dt.normalize() == frame[column].dropna()).all(), "unexpected time of day"

    frame["completion_date"] = frame["building_completion_date"].fillna(frame["project_completion_date"])
    frame["completion_year"] = frame["completion_date"].dt.year.astype("Int16")
    frame["start_year"] = frame["project_start_date"].dt.year.astype("int16")
    frame["is_completed"] = frame["completion_date"].notna()
    frame["completion_before_start"] = (frame["completion_date"] < frame["project_start_date"]).fillna(False)
    frame["same_day_completion"] = (frame["completion_date"] == frame["project_start_date"]).fillna(False)
    return frame


housing = clean_housing_dates(housing, housing_source)
building_date_missing = housing["building_completion_date"].isna()
filled_from_project = int((building_date_missing & housing["is_completed"]).sum())
days_early = (housing["project_start_date"] - housing["completion_date"]).dt.days[housing["completion_before_start"]]

log_rule("housing", "three date columns parsed from text with one exact layout", "standardise",
         len(housing), len(housing), 0, "Dates must be comparable; affected counts unparseable values.")
log_rule("housing", "completion_date = building date, else project date", "impute", len(housing),
         len(housing), filled_from_project,
         "Deterministic fill from the same row; rows with neither date stay missing.")
log_rule("housing", "no completion date: is_completed = False, date left missing", "flag",
         len(housing), len(housing), int((~housing["is_completed"]).sum()),
         "Unfinished buildings have no treatment date; an invented one would misplace crashes.")
log_rule("housing", "completion before project start flagged (completion_before_start)", "flag",
         len(housing), len(housing), int(housing["completion_before_start"].sum()),
         "Impossible order, but which date is wrong is unknown, so neither is changed.")
log_rule("housing", "completion on the start date flagged (same_day_completion)", "flag",
         len(housing), len(housing), int(housing["same_day_completion"].sum()),
         "No observable construction period; relevant to before/after designs.")

assert housing["project_start_date"].notna().all()
assert housing.loc[housing["is_completed"], "completion_year"].notna().all()

say(f"Building completion date missing in {int(building_date_missing.sum()):,} rows; the project date "
    f"fills {filled_from_project} of them and {int((~housing['is_completed']).sum()):,} rows have neither "
    f"and count as not completed. Dates run from {housing['project_start_date'].min():%Y-%m-%d} to "
    f"{housing['completion_date'].max():%Y-%m-%d}.")
say(f"Completion precedes the start date in {int(housing['completion_before_start'].sum())} rows "
    f"(median gap {days_early.median():.0f} days, largest {days_early.max():.0f} days). Completion equals "
    f"the start date in {int(housing['same_day_completion'].sum()):,} rows, of which "
    f"{int((housing['same_day_completion'] & (housing_source['reporting_construction_type'] == 'New Construction').to_numpy()).sum()):,} "
    f"are new construction.")

# %% [markdown]
# The figure below checks the completion year that the rest of the notebook depends on, and shows
# how many completed buildings per year have a point on the map.

# %%
def plot_completions_by_year(completions):
    """Stacked bars of completed housing rows per year, split by whether the row has coordinates."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(completions.index, completions["located"], color=PALETTE["housing"],
           width=0.78, label="Has coordinates (usable for the 250 m analysis)")
    ax.bar(completions.index, completions["not located"],
           bottom=completions["located"], color=PALETTE["before"], width=0.78,
           label="No coordinates (confidential or unaddressed)")
    ax.set_ylim(0, completions.sum(axis=1).max() * 1.14)   # headroom for the window label
    ax.axvspan(YEAR_START - 0.5, YEAR_END + 0.5, color=PALETTE["neutral"], alpha=0.06, zorder=0)
    ax.text(YEAR_START - 0.4, ax.get_ylim()[1] * 0.97, f"crash study window {YEAR_START} to {YEAR_END}",
            fontsize=9, color=PALETTE["neutral"], va="top")
    ax.set_xticks(completions.index)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    finish(ax, "Housing rows by completion year after date cleaning", "Completion year",
           "Housing rows completed (buildings)")
    plt.show()


completed_housing = housing[housing["is_completed"]]
completions_by_year = (completed_housing.groupby(["completion_year", "is_located"]).size()
                       .unstack(fill_value=0).rename(columns={True: "located", False: "not located"}))
plot_completions_by_year(completions_by_year)

# %%
located_share_by_year = 100 * completions_by_year["located"] / completions_by_year.sum(axis=1)
in_window = completed_housing["completion_year"].between(YEAR_START, YEAR_END)
say(f"{int(housing['is_completed'].sum()):,} rows have a completion year, from "
    f"{int(completions_by_year.index.min())} to {int(completions_by_year.index.max())}. "
    f"{int(in_window.sum()):,} of them fall inside the crash window, and "
    f"{int((in_window & completed_housing['is_located']).sum()):,} of those are located. The located share "
    f"is lowest in {int(located_share_by_year.idxmin())} ({located_share_by_year.min():.0f}%) and highest in "
    f"{int(located_share_by_year.idxmax())} ({located_share_by_year.max():.0f}%).")

# %% [markdown]
# Completions are spread over every year of the crash window, so before/after comparisons are
# possible for many cohorts. The unlocated share is not constant: it is larger in the earliest
# years, when more records were withheld. The last year on the axis lies after the crash window
# and is incomplete, so those buildings can only serve as not-yet-completed comparison sites.

# %% [markdown]
# ### 4.5 Housing: unit consistency and derived unit measures
#
# The same units are counted three ways in each row (by income band, by bedroom count and by
# tenure), and each of the three should add up to `all_counted_units`. That total in turn cannot
# exceed `total_units`, which also includes market-rate units in the building. These identities
# test the zero fill of 4.1: if a blank had meant unknown, the sums would fall short.
#
# Two measures are derived for the research questions. `deep_affordable_units` is the number of
# extremely low and very low income units. `deep_affordable_share` divides it by the counted
# units. Where a row has no counted units the share is undefined and is left missing; writing zero
# there would claim that a building has no deeply affordable units when the truth is that it has
# no income-restricted units of any kind on record. The construction type becomes a two-level
# category.

# %%
def check_and_derive_units(frame, raw):
    """Verify the unit identities, then add construction_type and the deep-affordability measures."""
    frame = frame.copy()
    counted = frame["all_counted_units"]
    checks = {
        "income bands sum to counted units": int((frame[INCOME_UNIT_COLUMNS].sum(axis=1) != counted).sum()),
        "bedroom counts sum to counted units": int((frame[BEDROOM_UNIT_COLUMNS].sum(axis=1) != counted).sum()),
        "rental + homeownership = counted units": int((frame[TENURE_UNIT_COLUMNS].sum(axis=1) != counted).sum()),
        "counted units exceed total units": int((counted > frame["total_units"]).sum()),
        "total units below 1": int((frame["total_units"] < 1).sum()),
    }
    construction = raw["reporting_construction_type"].str.strip().str.title()
    frame["construction_type"] = pd.Categorical(construction, categories=["New Construction", "Preservation"])
    frame["deep_affordable_units"] = frame[DEEP_AFFORDABLE_COLUMNS].sum(axis=1).astype("int16")
    # A share needs a denominator: rows with zero counted units get NaN, not 0.
    frame["deep_affordable_share"] = (frame["deep_affordable_units"] / counted.where(counted > 0)).astype("float64")
    return frame, checks


housing, unit_checks = check_and_derive_units(housing, housing_source)
display(pd.DataFrame({"rows violating": unit_checks}))
for check_name, violations in unit_checks.items():
    log_rule("housing", f"unit consistency check: {check_name}", "flag", len(housing), len(housing),
             violations, "Identity that must hold if blank means zero; affected counts violations.")
    assert violations == 0, f"unit identity broken: {check_name}"

no_counted_units = int((housing["all_counted_units"] == 0).sum())
log_rule("housing", "deep_affordable_share left missing where counted units = 0", "flag",
         len(housing), len(housing), no_counted_units, "A share with a zero denominator is undefined.")
assert housing["construction_type"].notna().all()
assert housing["deep_affordable_share"].dropna().between(0, 1).all()
assert housing["deep_affordable_share"].isna().sum() == no_counted_units

construction_table = (housing.groupby("construction_type", observed=True)
                      .agg(rows=("housing_id", "size"), total_units=("total_units", "sum"),
                           counted_units=("all_counted_units", "sum"),
                           deep_affordable_units=("deep_affordable_units", "sum")))
display(construction_table)
say(f"All five identities hold in every one of the {len(housing):,} rows. Counted units are below total "
    f"units in {int((housing['all_counted_units'] < housing['total_units']).sum()):,} rows (buildings that "
    f"mix affordable and market-rate units), and {no_counted_units} rows have no counted units, so their "
    f"deep-affordable share is missing. Across the table "
    f"{pct(housing['deep_affordable_units'].sum(), housing['all_counted_units'].sum()):.1f}% of counted units "
    f"are deeply affordable.")

# %% [markdown]
# No row breaks an identity, which confirms that blank means zero and that the table is internally
# consistent. A repair rule for totals below their parts would have nothing to act on, so none is
# written. The two construction types have nearly the same number of rows, which matters for RQ5,
# where preservation serves as the comparison group.

# %% [markdown]
# ### 4.6 Housing: size outliers
#
# Building size is the explanatory variable of RQ4, which asks whether a larger building changes
# the street more, so extreme values need a decision. Tukey's rule marks
# a value as an outlier when it lies more than 1.5 interquartile ranges above the upper quartile.
# The rule assumes a roughly symmetric distribution. Building sizes are strongly right-skewed
# (many small houses, a few towers), so the rule is applied twice: on the unit count as it is, and
# on its base-10 logarithm, where the distribution is closer to symmetric.
#
# The decision is to flag and keep. The largest records were checked by name in the output below
# and are real, well-known developments, and removing or capping them would delete exactly the
# buildings where a change in street activity is most likely to be visible. Analyses that are
# sensitive to a long tail can use the flag or the `size_band` category, which cuts the size into
# five ordered bands.

# %%
def flag_size_outliers(frame):
    """Add is_size_outlier (raw-scale Tukey fence) and size_band; return (frame, fences)."""
    frame = frame.copy()
    units = frame["total_units"].astype("float64")
    raw_fence = tukey_upper_fence(units)
    # On the log scale the fence is computed in log units and converted back to a unit count.
    log_fence = 10 ** tukey_upper_fence(np.log10(units))
    frame["is_size_outlier"] = (units > raw_fence).to_numpy()
    frame["size_band"] = pd.cut(units, bins=SIZE_BAND_EDGES, labels=SIZE_BAND_LABELS, ordered=True)
    return frame, {"raw": raw_fence, "log": log_fence, "above_log": int((units > log_fence).sum())}


housing, size_fences = flag_size_outliers(housing)
log_rule("housing", "total_units above the Tukey fence flagged (is_size_outlier), not removed", "flag",
         len(housing), len(housing), int(housing["is_size_outlier"].sum()),
         "Large buildings are real and are the cases RQ4 needs; the fence marks the skewed tail.")
assert housing["size_band"].notna().all()

display(housing.nlargest(5, "total_units")[["project_name", "borough", "total_units", "all_counted_units",
                                             "construction_type", "completion_year"]])
display(housing.groupby("size_band", observed=True)
        .agg(rows=("housing_id", "size"), total_units=("total_units", "sum"),
             flagged=("is_size_outlier", "sum")))

# %%
def plot_size_outliers(frame, fences):
    """Log-binned histogram of total_units with the rows above the Tukey fence drawn in a flag colour."""
    size_bins = integer_log_bins(frame["total_units"].max())
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(frame.loc[~frame["is_size_outlier"], "total_units"], bins=size_bins,
            color=PALETTE["housing"], edgecolor="white", linewidth=0.5, label="Inside the fence")
    # A data-quality flag gets its own colour (not the amber of Preservation and the placebo).
    ax.hist(frame.loc[frame["is_size_outlier"], "total_units"], bins=size_bins,
            color=PALETTE["accent"], edgecolor="white", linewidth=0.5, label="Flagged: above the fence")
    ax.axvline(fences["raw"], color=PALETTE["neutral"], linestyle="--", linewidth=1.5)
    ax.text(fences["raw"] * 1.08, ax.get_ylim()[1] * 0.95,
            f"Tukey fence on raw units: {fences['raw']:.0f}", fontsize=9, color=PALETTE["neutral"], va="top")
    ax.set_xscale("log")
    thousands(ax, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    finish(ax, "Building size is right-skewed: the IQR fence cuts off a long, real tail",
           "Total units in the building (log scale)", "Rows")
    plt.show()


plot_size_outliers(housing, size_fences)

# %%
say(f"The raw-scale fence is {size_fences['raw']:.0f} units and flags "
    f"{int(housing['is_size_outlier'].sum()):,} rows ({pct(housing['is_size_outlier'].sum(), len(housing)):.1f}% "
    f"of the table), which together hold "
    f"{pct(housing.loc[housing['is_size_outlier'], 'total_units'].sum(), housing['total_units'].sum()):.0f}% of "
    f"all units. On the log scale the fence moves to {size_fences['log']:,.0f} units and flags "
    f"{size_fences['above_log']} rows; the largest building has {int(housing['total_units'].max()):,} units. "
    f"The median is {housing['total_units'].median():.0f} units.")

# %% [markdown]
# A rule that flags this large a share of a table is not finding errors. On the raw scale the IQR
# fence only measures how skewed building sizes are, and on the log scale, where the rule's
# assumption is closer to true, no building is extreme at all. The histogram shows a continuous
# tail with no isolated values beyond it. The flag is therefore read as "large building", and it
# is kept because those rows hold most of the units even though they are a small share of the rows.
# The histogram is the one drawn in 3.1.5 with the decision applied: the flagged rows are coloured.

# %% [markdown]
# ### 4.7 Crashes: key, duplicates, dates and time of day
#
# Rows are dropped from the crash table for three reasons only: no `collision_id`, a repeated
# `collision_id`, or a crash date that cannot be parsed or lies outside the study window. Each
# test is applied and logged even though the extraction already filtered on the date, so a changed
# download cannot slip through. A unique key also rules out full-row duplicates, since two
# identical rows would share their key.
#
# A weaker kind of duplicate has to be considered as well: the same crash filed twice under two
# identifiers. Such pairs would agree on date, time, coordinates, first vehicle type and number of
# injured. These are counted and flagged with `is_possible_duplicate`, but kept, because two
# separate crashes at a busy junction within the same reported minute would look exactly the same
# and the table has no field that tells the two cases apart.
#
# The crash time is text in hours and minutes without a leading zero. The hour is extracted as a
# nullable small integer and grouped into four named six-hour bands for the time-of-day question.

# %%
def select_crash_rows(raw):
    """Apply the three row-dropping rules; return (kept raw rows, collision_id, crash_date)."""
    collision_id = pd.to_numeric(raw["collision_id"], errors="coerce").astype("float64")
    has_id = collision_id.notna()
    log_rule("crashes", "missing or non-numeric collision_id", "drop", len(raw), int(has_id.sum()),
             int((~has_id).sum()), "A crash without a key cannot be counted once and only once.")

    is_repeat = has_id & collision_id.duplicated(keep="first")
    log_rule("crashes", "duplicate collision_id (first kept)", "drop", int(has_id.sum()),
             int((has_id & ~is_repeat).sum()), int(is_repeat.sum()),
             "One row per collision; a repeat would double count its casualties.")

    crash_date = pd.to_datetime(raw["crash_date"], format=DATE_FORMAT, errors="coerce")
    in_window = crash_date.dt.year.between(YEAR_START, YEAR_END)
    keep = has_id & ~is_repeat & in_window
    log_rule("crashes", f"crash_date unparseable or outside {YEAR_START}-{YEAR_END}", "drop",
             int((has_id & ~is_repeat).sum()), int(keep.sum()), int((has_id & ~is_repeat & ~in_window).sum()),
             "Only full calendar years are comparable.")

    if keep.all():      # the usual case: avoid copying 1.5 million text rows
        return raw, collision_id.astype("int64"), crash_date
    kept = raw.loc[keep].reset_index(drop=True)
    return kept, collision_id[keep].astype("int64").reset_index(drop=True), crash_date[keep].reset_index(drop=True)


def hour_to_band(hour):
    """Map an hour to its named band from HOUR_BANDS as an ordered category (missing stays missing)."""
    band = pd.Series(np.nan, index=hour.index, dtype=object)
    for band_name, (first_hour, last_hour) in HOUR_BANDS.items():
        band[hour.between(first_hour, last_hour).fillna(False).to_numpy()] = band_name
    return pd.Categorical(band, categories=list(HOUR_BANDS), ordered=True)


crash_rows_raw = len(raw_crashes)
crash_columns_raw = raw_crashes.shape[1]
crash_memory_raw = memory_mb(raw_crashes)

crash_source, crash_ids, crash_dates = select_crash_rows(raw_crashes)
crash_hour, crash_minute = parse_hour(crash_source["crash_time"])

crashes = pd.DataFrame({
    "collision_id": crash_ids.to_numpy(),
    "crash_date": crash_dates.to_numpy(),
    "year": crash_dates.dt.year.astype("int16").to_numpy(),
    "month": crash_dates.dt.month.astype("int8").to_numpy(),
})
crashes["hour"] = crash_hour
crashes["hour_band"] = hour_to_band(crashes["hour"])

log_rule("crashes", "crash_date parsed to datetime; year and month derived", "standardise",
         len(crashes), len(crashes), 0, "Text dates cannot be grouped by year.")
log_rule("crashes", "crash_time to hour (Int8) and four named hour bands; invalid time left missing",
         "standardise", len(crashes), len(crashes), int(crashes["hour"].isna().sum()),
         "Time of day is analysed in bands; affected counts invalid times.")

assert crashes["collision_id"].is_unique
assert crashes["year"].between(YEAR_START, YEAR_END).all()
assert crashes["hour"].dropna().between(0, 23).all()
assert crashes["hour_band"].isna().sum() == crashes["hour"].isna().sum()

say(f"Rows in: {crash_rows_raw:,}. Rows out: {len(crashes):,}. The three dropping rules removed "
    f"{crash_rows_raw - len(crashes)} rows: collision_id is present and unique in every row and every date "
    f"parses and lies between {crashes['crash_date'].min():%Y-%m-%d} and {crashes['crash_date'].max():%Y-%m-%d}.")
say(f"Invalid crash times: {int(crashes['hour'].isna().sum())}. Times are rounded by the people who record "
    f"them: {pct((crash_minute == 0).sum(), len(crashes)):.1f}% of crashes are logged exactly on the hour and "
    f"{pct((crash_minute % ROUNDING_STEP_MIN == 0).sum(), len(crashes)):.1f}% on a multiple of "
    f"{ROUNDING_STEP_MIN} minutes, against {100 / MINUTES_PER_HOUR:.1f}% and {100 / ROUNDING_STEP_MIN:.1f}% "
    f"if minutes were recorded evenly.")
display(crashes["hour_band"].value_counts(sort=False).rename("crashes").to_frame()
        .assign(share_pct=lambda table: (100 * table["crashes"] / len(crashes)).round(1)))

# %% [markdown]
# The crash table needs no row removed: the key is complete and unique, and the dates and times
# are all valid. The minute of the crash is clearly rounded, which is a reason to analyse time of
# day in six-hour bands (and at most by hour) and never by minute. The next cell runs the check for
# crashes that may have been filed twice.

# %%
def flag_possible_duplicates(raw):
    """Mark later copies of rows that agree on date, time, exact coordinates, vehicle and injuries.

    Only rows with a usable point (both coordinates inside the city, not at (0, 0)) are compared.
    Without a location, two different crashes in the same minute would match on almost nothing
    but the clock.
    """
    signature = ["crash_date", "crash_time", "latitude", "longitude", "vehicle_type_code1",
                 "number_of_persons_injured"]
    latitude, _, _, _ = validate_coordinates(raw["latitude"], raw["longitude"])
    has_point = latitude.notna().to_numpy()
    is_copy = np.zeros(len(raw), dtype=bool)
    is_copy[has_point] = raw.loc[has_point, signature].duplicated(keep="first").to_numpy()
    return is_copy


crashes["is_possible_duplicate"] = flag_possible_duplicates(crash_source)
log_rule("crashes", "same date, time, coordinates, vehicle type and injuries under another id "
         "(is_possible_duplicate)", "flag", len(crashes), len(crashes),
         int(crashes["is_possible_duplicate"].sum()),
         "May be a double filing or two real crashes at one junction; kept because they cannot be told apart.")
say(f"{int(crashes['is_possible_duplicate'].sum()):,} rows "
    f"({pct(crashes['is_possible_duplicate'].sum(), len(crashes)):.2f}%) repeat the date, time, coordinates, "
    f"first vehicle type and injury count of an earlier row under a different collision_id.")

# %% [markdown]
# The share is small enough that the choice between keeping and dropping these rows cannot move a
# result, so they stay in and the flag is available for a sensitivity check.

# %% [markdown]
# ### 4.8 Crashes: borough, ZIP code and coordinates
#
# Location is where the crash table is weakest, and it is the part that integration depends on.
#
# Borough is written in upper case here and in title case in the housing table. It is converted to
# the same category, so the two tables agree letter for letter.
#
# The ZIP code goes through the same validator as the housing postcode. Three kinds of value are
# set to missing: blanks, the placeholder 10000 (which is not a real ZIP and would otherwise form
# a fake neighbourhood of its own), and codes outside the city. The crash row itself is kept in
# every case: it still has a date, casualties and usually coordinates.
#
# Coordinates are checked against `NYC_BOUNDS`. A pair at exactly (0, 0) is what a failed geocode
# (an automatic address lookup) leaves behind, and a point outside the box cannot be a New York crash. Both are set to missing, not
# clipped to the box edge, since clipping would pile crashes onto streets at the boundary. The
# flag `is_located` marks rows with usable coordinates.
#
# Missing ZIP codes are not filled at this stage. Most rows without a ZIP still have coordinates,
# so the ZIP can be recovered from the surrounding crashes, but that is an integration problem and
# is solved in Section 6, where its accuracy is measured on held-out rows.

# %%
def clean_crash_geography(raw, frame):
    """Add borough, zip, lat, lon and is_located to a copy of the crash frame; return (frame, report)."""
    frame = frame.copy()
    borough_text = raw["borough"].str.strip().str.title()
    frame["borough"] = pd.Categorical(borough_text, categories=BOROUGHS)

    zip_text = raw["zip_code"].str.strip()
    clean_zip, implied_borough = standardise_zip(raw["zip_code"])
    frame["zip"] = clean_zip
    rejected = (zip_text.notna() & clean_zip.isna()).to_numpy()

    lat, lon, n_zero, n_outside = validate_coordinates(raw["latitude"], raw["longitude"])
    frame["lat"], frame["lon"] = lat.to_numpy(), lon.to_numpy()
    frame["is_located"] = frame["lat"].notna().to_numpy()

    stated = frame["borough"].astype(object)
    disagree = (implied_borough.notna().to_numpy() & stated.notna().to_numpy()
                & (implied_borough.to_numpy() != stated.to_numpy()))
    report = {
        "borough_raw_present": int(raw["borough"].notna().sum()),
        "borough_unrecognised": int(raw["borough"].notna().sum() - frame["borough"].notna().sum()),
        "zip_raw_missing": int(raw["zip_code"].isna().sum()),
        "zip_blank": int((zip_text == "").sum()),
        "zip_placeholder": int((zip_text == ZIP_PLACEHOLDER).sum()),
        "zip_rejected": int(rejected.sum()),
        "zip_rejected_values": zip_text[rejected].value_counts().head(6),
        "coordinates_raw_missing": int(raw["latitude"].isna().sum()),
        "coordinates_zero": n_zero, "coordinates_outside": n_outside,
        "zip_borough_disagree": int(disagree.sum()),
        "disagree_zips": frame.loc[disagree, "zip"].value_counts().head(5),
        "border_zip_boroughs": frame.loc[frame["zip"].isin(["11001", "11040"]), "borough"]
                               .value_counts().loc[lambda counts: counts > 0],
    }
    return frame, report


raw_coordinate_sample = crash_source[["latitude", "longitude"]].apply(to_number)
crashes, crash_geo_report = clean_crash_geography(crash_source, crashes)

log_rule("crashes", "borough to title case, category with the BOROUGHS spelling", "standardise",
         len(crashes), len(crashes), crash_geo_report["borough_raw_present"],
         "Upper case here, title case in housing; one spelling is needed to compare boroughs.")
log_rule("crashes", "zip_code blank, placeholder 10000 or outside NYC ranges set to missing",
         "standardise", len(crashes), len(crashes), crash_geo_report["zip_rejected"],
         "Junk codes would form fake neighbourhoods in the ZIP panel; the crash row is kept.")
log_rule("crashes", "coordinates at (0,0) set to missing", "standardise", len(crashes), len(crashes),
         crash_geo_report["coordinates_zero"], "A failed geocode leaves this placeholder; it marks no place.")
log_rule("crashes", "coordinates outside NYC_BOUNDS set to missing", "standardise", len(crashes),
         len(crashes), crash_geo_report["coordinates_outside"],
         "Domain rule: a crash in this table cannot lie outside the city; clipping would misplace it.")
log_rule("crashes", "rows without usable coordinates flagged (is_located = False)", "flag",
         len(crashes), len(crashes), int((~crashes["is_located"]).sum()),
         "Kept for borough and ZIP totals; excluded from distance-based analyses.")
log_rule("crashes", "missing zip left missing at this stage", "flag", len(crashes), len(crashes),
         int(crashes["zip"].isna().sum()),
         "Recovered from coordinates after integration (Section 6), where accuracy can be tested.")
log_rule("crashes", "ZIP range borough differs from stated borough (stated borough kept)", "flag",
         len(crashes), len(crashes), crash_geo_report["zip_borough_disagree"],
         "Checked only; ZIPs on the Brooklyn / Queens line cross the borough boundary.")

assert crashes.loc[crashes["is_located"], "lat"].between(*NYC_BOUNDS["lat"]).all()
assert crashes.loc[crashes["is_located"], "lon"].between(*NYC_BOUNDS["lon"]).all()
assert crashes["zip"].dropna().str.len().eq(5).all()
assert ZIP_PLACEHOLDER not in set(crashes["zip"].dropna().unique())

borough_before_after = pd.DataFrame({
    "raw spelling": crash_source["borough"].value_counts(dropna=False).rename(lambda v: v if pd.notna(v) else "(missing)"),
})
borough_before_after["clean spelling"] = [value.title() if value != "(missing)" else "(missing)"
                                          for value in borough_before_after.index]
borough_before_after["rows after"] = [
    int(crashes["borough"].isna().sum()) if value == "(missing)" else int((crashes["borough"] == value).sum())
    for value in borough_before_after["clean spelling"]]
display(borough_before_after.rename(columns={"raw spelling": "rows before"}))
display(crash_geo_report["zip_rejected_values"].rename("rows").to_frame().rename_axis("rejected zip_code"))

say(f"ZIP: {crash_geo_report['zip_raw_missing']:,} rows had none to begin with. Of the rest, "
    f"{crash_geo_report['zip_rejected']:,} were rejected ({crash_geo_report['zip_placeholder']} of them the "
    f"10000 placeholder, {crash_geo_report['zip_blank']} blank strings). After cleaning "
    f"{int(crashes['zip'].isna().sum()):,} rows ({pct(crashes['zip'].isna().sum(), len(crashes)):.1f}%) have no "
    f"ZIP and {crashes['zip'].nunique()} distinct ZIPs remain.")
say(f"Coordinates: {crash_geo_report['coordinates_raw_missing']:,} rows had none, "
    f"{crash_geo_report['coordinates_zero']:,} were at (0,0) and {crash_geo_report['coordinates_outside']} lay "
    f"outside the city box. After cleaning {int(crashes['is_located'].sum()):,} rows "
    f"({pct(crashes['is_located'].sum(), len(crashes)):.1f}%) are located.")
say(f"Borough is missing in {int(crashes['borough'].isna().sum()):,} rows "
    f"({pct(crashes['borough'].isna().sum(), len(crashes)):.1f}%); "
    f"{int((crashes['borough'].notna() & crashes['zip'].isna()).sum()):,} rows have a borough but no valid ZIP "
    f"and {int((crashes['borough'].isna() & crashes['zip'].notna()).sum())} the reverse, so the two fields are "
    f"missing together. The ZIP range and the stated borough disagree in "
    f"{crash_geo_report['zip_borough_disagree']} rows, mostly in ZIP "
    f"{join_and(crash_geo_report['disagree_zips'].index[:3])}. The two ZIPs shared with Nassau County carry "
    f"{int(crash_geo_report['border_zip_boroughs'].sum())} crashes, "
    f"{'all ' if len(crash_geo_report['border_zip_boroughs']) == 1 else ''}recorded in "
    f"{join_and(crash_geo_report['border_zip_boroughs'].index.astype(str))}.")

# %% [markdown]
# Borough and ZIP are missing in the same rows, which fits a record where the officer gave a
# location on a highway or bridge without a postal address. The disagreements between ZIP range
# and stated borough sit in ZIPs that lie across the Brooklyn and Queens boundary, so they are
# real places and the stated borough is left as it is. The two ZIPs shared with Nassau County are
# kept because every crash that carries them is recorded in Queens.
#
# The first figure validates the coordinate rule. The left panel counts what happened to every raw
# coordinate pair. The right panel draws a sample of the cleaned points, coloured and labelled by
# borough, with the points that have no borough drawn as hollow black rings.

# %%
raw_lat, raw_lon = raw_coordinate_sample["latitude"], raw_coordinate_sample["longitude"]
coordinate_outcomes = pd.Series({
    "kept (inside the city box)": int(crashes["is_located"].sum()),
    "no coordinates": crash_geo_report["coordinates_raw_missing"],
    "outside the city box": crash_geo_report["coordinates_outside"],
    "at (0, 0): failed geocode": crash_geo_report["coordinates_zero"],
})
assert coordinate_outcomes.sum() == len(crashes), "every raw coordinate pair has exactly one outcome"
map_sample = crashes[crashes["is_located"]].sample(MAP_SAMPLE_ROWS, random_state=SEED)


def plot_coordinate_cleaning(outcomes, sample):
    """Left: what happened to every raw coordinate pair. Right: a sample of the cleaned points.

    Boroughs carry a name on the map as well as a colour. Points with coordinates but no borough
    are hollow black rings, so they cannot be mistaken for the grey of Staten Island.
    """
    fig, (ax_outcome, ax_clean) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    outcome_colors = [PALETTE["housing"], PALETTE["before"], PALETTE["accent"], PALETTE["crash"]]
    outcome_bars = ax_outcome.barh(outcomes.index[::-1], outcomes.to_numpy()[::-1],
                                   color=outcome_colors[::-1], height=0.65)
    ax_outcome.bar_label(outcome_bars, labels=[f"{value:,}" for value in outcomes.to_numpy()[::-1]],
                         padding=4, fontsize=9, color=PALETTE["neutral"])
    ax_outcome.set_xlim(0, outcomes.max() * 1.18)
    thousands(ax_outcome, axis="x")
    ax_outcome.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax_outcome.grid(axis="y", visible=False)
    finish(ax_outcome, "Before: what happens to each raw coordinate pair", "Crashes", "")

    legend_handles = []
    for borough_name in BOROUGHS:
        points = sample.loc[sample["borough"] == borough_name]
        ax_clean.scatter(points["lon"], points["lat"], s=1, color=BOROUGH_COLORS[borough_name],
                         rasterized=True, zorder=2)
        ax_clean.text(points["lon"].median(), points["lat"].median(), borough_name, ha="center",
                      fontsize=9, fontweight="bold", color=PALETTE["neutral"], zorder=5,
                      path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
        legend_handles.append(mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=6,
                                               color=BOROUGH_COLORS[borough_name], label=borough_name))
    no_borough = sample.loc[sample["borough"].isna()]
    ax_clean.scatter(no_borough["lon"], no_borough["lat"], s=4, facecolors="none", edgecolors="black",
                     linewidths=0.25, alpha=0.5, rasterized=True, zorder=1)
    legend_handles.append(mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=6,
                                           markerfacecolor="none", markeredgecolor="black",
                                           label="No borough recorded"))
    map_axes(ax_clean)
    ax_clean.legend(handles=legend_handles, loc="upper left", fontsize=8.5)
    finish(ax_clean, f"After: a sample of {MAP_SAMPLE_ROWS:,} located crashes", "Longitude (degrees)",
           "Latitude (degrees)")
    plt.tight_layout()
    plt.show()


plot_coordinate_cleaning(coordinate_outcomes, map_sample)

# %%
raw_rejected_count = coordinate_outcomes["outside the city box"] + coordinate_outcomes["at (0, 0): failed geocode"]
say(f"{raw_rejected_count:,} coordinate pairs are rejected. Raw longitudes run from {raw_lon.min():.1f} to {raw_lon.max():.1f} degrees and raw latitudes from "
    f"{raw_lat.min():.1f} to {raw_lat.max():.1f}; after cleaning they run from {crashes['lon'].min():.3f} to "
    f"{crashes['lon'].max():.3f} and from {crashes['lat'].min():.3f} to {crashes['lat'].max():.3f}.")

# %% [markdown]
# The few rejected pairs would have stretched any map or bounding box across half the globe, as
# the raw ranges above show, and a distance calculation on them would have been meaningless. After
# cleaning, the sample draws the outline
# of the five boroughs and their street grid, and the borough colours sit where the boroughs are,
# which checks the borough labels and the coordinates against each other. The hollow rings have
# coordinates but no borough or ZIP, and they are the group whose ZIP Section 6 recovers.
#
# The second figure shows which location fields each crash has left, by year.

# %%
LOCATION_STATES = ["ZIP and coordinates", "Coordinates only", "ZIP only", "Neither"]
location_state = np.select(
    [crashes["zip"].notna() & crashes["is_located"], crashes["zip"].isna() & crashes["is_located"],
     crashes["zip"].notna() & ~crashes["is_located"]], LOCATION_STATES[:3], default=LOCATION_STATES[3])
location_by_year = (pd.crosstab(crashes["year"], location_state, normalize="index")
                    .reindex(columns=LOCATION_STATES, fill_value=0) * 100)
location_colors = [PALETTE["housing"], PALETTE["placebo"], PALETTE["accent"], PALETTE["before"]]


def plot_location_states(shares):
    """Stacked bars of the share of each year's crashes by which location fields survive cleaning."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bar_bottom = np.zeros(len(shares))
    for state, color in zip(LOCATION_STATES, location_colors):
        ax.bar(shares.index, shares[state], bottom=bar_bottom, color=color,
               edgecolor="white", linewidth=0.8, width=0.8, label=state)
        bar_bottom += shares[state].to_numpy()
    for year, share in shares["Coordinates only"].items():
        label_height = shares.loc[year, "ZIP and coordinates"] + share / 2
        ax.text(year, label_height, f"{share:.0f}", ha="center", va="center", fontsize=8.5, color="black")
    ax.set_xticks(shares.index)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    finish(ax, "Location fields available per crash after cleaning, by year", "Crash year",
           "Share of the year's crashes (%)")
    plt.show()


plot_location_states(location_by_year)

# %%
location_counts = pd.Series(location_state).value_counts().reindex(LOCATION_STATES)
coordinates_only = location_by_year["Coordinates only"]
say(f"Across all years {pct(location_counts['ZIP and coordinates'], len(crashes)):.1f}% of crashes have both "
    f"fields, {pct(location_counts['Coordinates only'], len(crashes)):.1f}% have coordinates only "
    f"({location_counts['Coordinates only']:,} rows), {pct(location_counts['ZIP only'], len(crashes)):.1f}% "
    f"have a ZIP only and {pct(location_counts['Neither'], len(crashes)):.1f}% have neither. The "
    f"coordinates-only share ranges from {coordinates_only.min():.0f}% in {int(coordinates_only.idxmin())} to "
    f"{coordinates_only.max():.0f}% in {int(coordinates_only.idxmax())}.")

# %% [markdown]
# The orange band is the reason ZIP recovery is worth doing: a large block of crashes has no ZIP
# but does have a point, and its size changes from year to year. A ZIP-level count built from the
# blue band alone would rise and fall with reporting practice and not only with crashes. The
# decision taken here is to leave these ZIPs missing and to keep the rows, so that Section 6 can
# fill them from coordinates and test the fill. Dropping them now would remove a share of crashes
# that differs by year, which is worse than either alternative. Only the grey band is lost to
# every geographic analysis below the city level.

# %% [markdown]
# ### 4.9 Crashes: casualty counts, and totals that do not match their parts
#
# Each crash has a total number of persons injured and killed, and the same two numbers broken
# out for pedestrians, cyclists and motorists. The columns are first checked for the basic
# properties of a count (whole, not negative) and converted to small integers.
#
# A few totals are blank while the parts are present. The total is then imputed as the sum of
# the parts, which is the definition of the field and not an estimate.
#
# The total and the parts also disagree in a number of rows, in both directions.
#
# Where the total is below the sum of the parts, the total is raised to that sum. The parts name
# specific people, so they are the more detailed record, and a total cannot be smaller than what
# it contains.
#
# Where the total is above the sum of the parts, nothing is overwritten. The three roles do not
# cover everyone who can be hurt: a rider of an e-scooter or e-bike is none of pedestrian, cyclist
# or motorist in this table. Lowering the total to the sum of the parts would delete injured
# people from the data, and adding the surplus to one of the three roles would be a guess. The
# surplus is kept in its own columns, `other_injured` and `other_killed`, so that the eight
# published counts stay as reported and the identity "total = pedestrians + cyclists + motorists
# + other" holds in every row. `casualty_mismatch` flags the rows involved.

# %%
def clean_casualties(raw):
    """Typed casualty counts with imputed totals, repaired deficits and the unclassified surplus.

    Returns (counts frame, report). The total is never lowered; see the text for the reasoning.
    """
    counts = pd.DataFrame({short: to_number(raw[column]) for column, short in CASUALTY_COLUMNS.items()})
    present = counts.stack()
    report = {"negative": int((present < 0).sum()), "fractional": int((present % 1 != 0).sum()),
              "above_fence": {}}
    assert report["negative"] == 0 and report["fractional"] == 0, "casualty counts must be whole and >= 0"

    for outcome in ["injured", "killed"]:
        part_columns = [f"{prefix}_{outcome}" for prefix in ROAD_USER_PREFIXES]
        assert counts[part_columns].notna().all().all(), "role counts are expected to be complete"
        parts_sum = counts[part_columns].sum(axis=1)
        reported = counts[outcome]
        report[f"{outcome}_missing"] = int(reported.isna().sum())
        report[f"{outcome}_deficit"] = int((reported < parts_sum).sum())
        report[f"{outcome}_surplus"] = int((reported > parts_sum).sum())
        report[f"{outcome}_surplus_people"] = int((reported - parts_sum).clip(lower=0).sum())
        # Missing total -> sum of parts (comparison with NaN is False, so fillna comes first);
        # total below its parts -> raised to the parts; total above its parts -> left as reported.
        counts[outcome] = np.maximum(reported.fillna(parts_sum), parts_sum)
        counts[f"other_{outcome}"] = counts[outcome] - parts_sum
        report[f"{outcome}_mismatch_mask"] = ((reported != parts_sum) & reported.notna()).to_numpy()
        report[f"{outcome}_surplus_mask"] = (reported > parts_sum).to_numpy()
        report[f"{outcome}_deficit_mask"] = (reported < parts_sum).to_numpy()

    # The narrow integer type is only safe while every count fits in it; a wrap would be silent.
    assert counts.max().max() <= np.iinfo(np.int8).max, "a casualty count no longer fits in int8"
    counts = counts.astype("int8")
    counts["casualty_mismatch"] = report["injured_mismatch_mask"] | report["killed_mismatch_mask"]
    return counts, report


casualties, casualty_report = clean_casualties(crash_source)
casualty_columns_clean = list(casualties.columns)
for column in casualty_columns_clean:
    crashes[column] = casualties[column].to_numpy()
crashes["is_injury_crash"] = ((crashes["injured"].astype("int16") + crashes["killed"]) > 0).to_numpy()
crashes["ped_cyc_harm"] = ((crashes[["ped_injured", "ped_killed", "cyc_injured", "cyc_killed"]]
                            .sum(axis=1)) > 0).to_numpy()

log_rule("crashes", "eight casualty columns to small integers; checked whole and non-negative",
         "standardise", len(crashes), len(crashes), casualty_report["negative"] + casualty_report["fractional"],
         "Counts were text; affected counts negative or fractional values.")
log_rule("crashes", "missing persons injured / killed total = sum of pedestrians, cyclists, motorists",
         "impute", len(crashes), len(crashes),
         casualty_report["injured_missing"] + casualty_report["killed_missing"],
         "The total is defined as the sum of its parts, so the fill is exact.")
log_rule("crashes", "persons total below the sum of its parts raised to the sum", "impute",
         len(crashes), len(crashes), casualty_report["injured_deficit"] + casualty_report["killed_deficit"],
         "A total cannot be smaller than the people it contains.")
log_rule("crashes", "persons total above the sum of its parts kept; surplus stored in other_injured / "
         "other_killed and flagged (casualty_mismatch)", "flag", len(crashes), len(crashes),
         casualty_report["injured_surplus"] + casualty_report["killed_surplus"],
         "The surplus is people outside the three roles (for example e-scooter riders); lowering "
         "the total would delete them.")

# Checks against the raw columns, so they can fail: the people counted after cleaning must equal the
# reported totals plus the imputed totals plus the deficits that were raised, and a row that was
# not flagged as a mismatch must have no unclassified people.
for outcome, raw_column in [("injured", "number_of_persons_injured"), ("killed", "number_of_persons_killed")]:
    reported_total = to_number(crash_source[raw_column]).to_numpy()
    parts_total = crashes[[f"{prefix}_{outcome}" for prefix in ROAD_USER_PREFIXES]].sum(axis=1).to_numpy()
    is_reported = ~np.isnan(reported_total)
    imputed_people = int(parts_total[~is_reported].sum())
    raised_people = int(np.clip(parts_total[is_reported] - reported_total[is_reported], 0, None).sum())
    assert int(crashes[outcome].sum()) == int(reported_total[is_reported].sum()) + imputed_people + raised_people
    assert (crashes.loc[~crashes["casualty_mismatch"], f"other_{outcome}"] == 0).all()
assert (crashes[[c for c in casualty_columns_clean if c != "casualty_mismatch"]] >= 0).all().all()
# The cleaned totals must line up row by row with the raw reported totals: a total is never lowered.
assert (crashes["injured"].to_numpy() >= to_number(crash_source["number_of_persons_injured"]).fillna(0).to_numpy()).all()

casualty_table = pd.DataFrame({
    "total missing (imputed)": [casualty_report["injured_missing"], casualty_report["killed_missing"]],
    "total below parts (raised)": [casualty_report["injured_deficit"], casualty_report["killed_deficit"]],
    "total above parts (kept)": [casualty_report["injured_surplus"], casualty_report["killed_surplus"]],
    "people in the surplus": [casualty_report["injured_surplus_people"], casualty_report["killed_surplus_people"]],
    "people after cleaning": [int(crashes["injured"].sum()), int(crashes["killed"].sum())],
}, index=["injured", "killed"])
display(casualty_table)

# %% [markdown]
# The explanation given for the surplus can be checked against the data. If the extra injured
# people are riders of vehicles outside the three roles, then the first vehicle type in those rows
# should look different from the first vehicle type in crashes generally, and the rows should be
# concentrated in the years after such vehicles became common.

# %%
surplus_mask = casualty_report["injured_surplus_mask"]
vehicle_type = crash_source["vehicle_type_code1"].str.strip().str.title().fillna("(missing)")
vehicle_comparison = pd.DataFrame({
    "share of surplus rows (%)": 100 * vehicle_type[surplus_mask].value_counts(normalize=True),
    "share of all crashes (%)": 100 * vehicle_type.value_counts(normalize=True),
}).dropna().sort_values("share of surplus rows (%)", ascending=False).head(6).round(2)
display(vehicle_comparison)

# %%
mismatch_by_year = pd.DataFrame({
    "Total above parts (kept, surplus stored)": pd.Series(surplus_mask).groupby(crashes["year"]).sum(),
    "Total below parts (raised to the parts)": pd.Series(casualty_report["injured_deficit_mask"])
                                               .groupby(crashes["year"]).sum(),
})


def plot_casualty_mismatch(by_year):
    """Paired bars per year: totals above their parts (kept) and totals below them (raised)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bar_width = 0.4
    ax.bar(by_year.index - bar_width / 2, by_year.iloc[:, 0], width=bar_width,
           color=PALETTE["crash"], label=by_year.columns[0])
    ax.bar(by_year.index + bar_width / 2, by_year.iloc[:, 1], width=bar_width,
           color=PALETTE["control"], label=by_year.columns[1])
    ax.set_xticks(by_year.index)
    thousands(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    finish(ax, "Injured total differs from the sum of its parts, by year", "Crash year",
           "Crashes")
    plt.show()


plot_casualty_mismatch(mismatch_by_year)

# %%
is_scooter_or_ebike = vehicle_type.isin(["E-Scooter", "E-Bike"]).to_numpy()
first_surplus_year = BREAK_YEAR + 1      # the surplus appears from here on, as the table above shows
last_error_year = BREAK_YEAR - 2         # the deficits cluster up to here
surplus_from_first_year = int(mismatch_by_year.loc[first_surplus_year:, mismatch_by_year.columns[0]].sum())
deficit_before_cutoff = int(mismatch_by_year.loc[:last_error_year, mismatch_by_year.columns[1]].sum())
say(f"The injured total exceeds its parts in {casualty_report['injured_surplus']:,} crashes "
    f"({pct(casualty_report['injured_surplus'], len(crashes)):.2f}% of all crashes), and "
    f"{pct(surplus_from_first_year, casualty_report['injured_surplus']):.0f}% of them fall in "
    f"{first_surplus_year} or later. An "
    f"e-scooter or e-bike is the first vehicle in {pct((is_scooter_or_ebike & surplus_mask).sum(), surplus_mask.sum()):.1f}% "
    f"of these rows against {pct(is_scooter_or_ebike.sum(), len(crashes)):.2f}% of all crashes. The opposite "
    f"case, a total below its parts, occurs in {casualty_report['injured_deficit']} crashes, "
    + ("all of them" if deficit_before_cutoff == casualty_report["injured_deficit"]
       else f"{deficit_before_cutoff} of them")
    + f" before {last_error_year + 1}. Keeping the "
    f"surplus preserves {casualty_report['injured_surplus_people']:,} injured people that a strict "
    f"\"parts only\" rule would have removed.")
say(f"After cleaning: {int(crashes['is_injury_crash'].sum()):,} injury crashes "
    f"({pct(crashes['is_injury_crash'].sum(), len(crashes)):.1f}% of crashes), "
    f"{int(crashes['ped_cyc_harm'].sum()):,} crashes in which a pedestrian or cyclist was injured or killed, "
    f"{int(crashes['injured'].sum()):,} people injured and {int(crashes['killed'].sum()):,} killed.")

# %% [markdown]
# The two kinds of mismatch have different histories. Totals below their parts are few and belong
# to the first years of the window, which points to occasional entry errors that later stopped.
# Totals above their parts barely exist before 2021 and then appear every year, and e-scooters and
# e-bikes are far more common in those rows than in crashes overall. That pattern fits a change in
# who is counted as a person involved, and it does not fit random error, so the surplus is treated
# as real people. Cars are still the most frequent first vehicle in these rows, so the surplus is
# not only riders of e-scooters and e-bikes, and the columns are named `other_` to avoid claiming more than
# the data shows.
#
# One consequence for later sections: `ped_cyc_harm` uses the pedestrian and cyclist columns only.
# Injured e-scooter and e-bike riders sit in `other_injured` and are not part of it.

# %% [markdown]
# ### 4.10 Crashes: extreme injury counts
#
# A crash with dozens of injured people could be a typing error or a bus crash. The IQR rule is
# tried first. The decision is again to flag and not to cap. Capping at a fixed number would turn
# a real mass-casualty event into an ordinary crash, and it would also break the agreement between
# the total and its parts that 4.9 established. A domain rule is used for the flag: more than
# `EXTREME_INJURED` people injured in one crash. The distribution itself was drawn in 3.2.8, with the
# same threshold marked.

# %%
def flag_extreme_injuries(frame):
    """Add is_extreme_injury (domain rule) to a copy of the frame and return it with the IQR fences."""
    frame = frame.copy()
    all_fence = tukey_upper_fence(frame["injured"])
    injury_fence = tukey_upper_fence(frame.loc[frame["injured"] > 0, "injured"])
    frame["is_extreme_injury"] = (frame["injured"] > EXTREME_INJURED).to_numpy()
    return frame, {"all": all_fence, "injury_only": injury_fence,
                   "above_all": int((frame["injured"] > all_fence).sum()),
                   "above_injury_only": int((frame["injured"] > injury_fence).sum())}


crashes, injury_fences = flag_extreme_injuries(crashes)
log_rule("crashes", f"more than {EXTREME_INJURED} injured in one crash flagged (is_extreme_injury), not capped",
         "flag", len(crashes), len(crashes), int(crashes["is_extreme_injury"].sum()),
         "Bus-scale events are real; capping would erase them and break total = sum of parts.")

extreme_rows = crashes["is_extreme_injury"].to_numpy()
extreme_table = crashes.loc[extreme_rows, ["crash_date", "borough", "injured", "ped_injured", "mot_injured",
                                           "killed"]].copy()
extreme_table["first_vehicle"] = vehicle_type[extreme_rows].to_numpy()
display(extreme_table.nlargest(8, "injured"))

# %%
bus_share = pct(vehicle_type[extreme_rows].eq("Bus").sum(), extreme_rows.sum())
injured_q1, injured_q3 = np.percentile(crashes["injured"], [25, 75])
say(f"The quartiles of injured persons per crash are {injured_q1:.0f} and {injured_q3:.0f}, so the "
    f"Tukey fence is {injury_fences['all']:.1f} and would call {injury_fences['above_all']:,} crashes outliers "
    f"({pct(injury_fences['above_all'], len(crashes)):.1f}% of the table). Restricted to injury crashes the "
    f"fence is {injury_fences['injury_only']:.1f} and would flag {injury_fences['above_injury_only']:,} rows, "
    f"that is every crash with more than one injured person. The domain rule flags "
    f"{int(crashes['is_extreme_injury'].sum())} crashes with more than {EXTREME_INJURED} injured, holding "
    f"{int(crashes.loc[extreme_rows, 'injured'].sum()):,} of the {int(crashes['injured'].sum()):,} injured "
    f"people; the maximum is {int(crashes['injured'].max())}. A bus is the first vehicle in {bus_share:.0f}% of "
    f"the flagged crashes against {pct(vehicle_type.eq('Bus').sum(), len(crashes)):.1f}% of all crashes.")

# %% [markdown]
# The IQR rule is not informative for this variable. Most crashes injure nobody, so both quartiles
# sit at the bottom of the scale and the fence lands between two and three people. What it flags
# is ordinary multi-injury crashes, which says the variable is a skewed count and nothing about
# errors. The distribution drawn in 3.2.8 shows the counts falling steadily with no gap before the
# largest values, and the table above shows that the largest events involve buses and other large
# vehicles far more often than crashes in general. They are kept at their reported values. The
# main outcome of this notebook is whether a crash injured anyone, which is not affected by how
# many people a single crash injured, and the flag lets the analyses that sum injured people check
# their sensitivity to these few rows.

# %% [markdown]
# ### 4.11 Crashes: contributing factors grouped into categories
#
# The first contributing factor is a free-form label with many distinct values. Some
# differ only in spelling or capitalisation, a few are bare numbers left over from an older coding
# scheme, and many are too rare to analyse on their own. For RQ10 they are reduced to ten groups.
#
# The mapping is an ordered list of keyword patterns (`FACTOR_RULES` at the top of this section).
# Each value is trimmed and lower-cased, which merges the case variants, and receives the group of
# the first pattern it matches. Keywords were chosen over a hand-written dictionary of every label
# because a keyword also catches a spelling variant that appears in a future download, and the
# order of the rules is commented where it matters. Missing values, the label "Unspecified" and
# numeric codes all mean that no factor was recorded, so they share one group. They are not
# dropped: the crash still happened, and the size of this group is itself a data-quality finding.
# A label that no pattern recognises goes to "Other" and is listed, so the mapping can be audited.

# %%
def group_factors(factor_text):
    """Map raw contributing-factor labels to FACTOR_GROUPS; return (category Series, mapping table).

    The patterns run on the distinct labels only (a few dozen), and the result is mapped back to
    the 1.5 million rows, which keeps the step fast.
    """
    label_counts = factor_text.value_counts(dropna=True)
    labels = pd.Series(label_counts.index.astype(str), index=label_counts.index)
    normalised = labels.str.strip().str.lower()
    group = pd.Series(np.nan, index=labels.index, dtype=object)
    for group_name, pattern in FACTOR_RULES:
        # only labels that no earlier rule claimed can be assigned: this is "first match wins"
        is_match = normalised.str.contains(pattern, regex=True) & group.isna()
        group[is_match] = group_name
    is_unrecognised = group.isna()
    group = group.fillna(FACTOR_FALLBACK)
    mapping = pd.DataFrame({"raw_label": labels.to_numpy(), "normalised": normalised.to_numpy(),
                            "factor_group": group.to_numpy(), "rows": label_counts.to_numpy(),
                            "unrecognised": is_unrecognised.to_numpy()})
    row_groups = factor_text.astype(object).map(dict(zip(mapping["raw_label"], mapping["factor_group"])))
    row_groups = row_groups.fillna("Unspecified")          # a missing factor is an unrecorded factor
    return pd.Categorical(row_groups, categories=FACTOR_GROUPS), mapping


crashes["factor_group"], factor_mapping = group_factors(crash_source["contributing_factor_vehicle_1"])
factor_missing = int(crash_source["contributing_factor_vehicle_1"].isna().sum())
numeric_codes = factor_mapping[factor_mapping["normalised"].str.fullmatch(r"\d+")]
log_rule("crashes", "contributing_factor_vehicle_1 mapped to ten factor groups by ordered keywords",
         "standardise", len(crashes), len(crashes), len(crashes) - factor_missing,
         "Dozens of labels with spelling variants and junk codes are too fragmented to compare.")
log_rule("crashes", "missing factor and numeric junk codes assigned to 'Unspecified'", "standardise",
         len(crashes), len(crashes), factor_missing + int(numeric_codes["rows"].sum()),
         "No factor was recorded; the crash is kept and the gap stays visible as its own group.")

assert crashes["factor_group"].notna().all()
assert set(crashes["factor_group"].unique()) == set(FACTOR_GROUPS), "a factor group is empty"

with pd.option_context("display.max_rows", 80):
    display(factor_mapping.groupby("factor_group")
            .agg(labels=("raw_label", "size"), rows=("rows", "sum"),
                 raw_labels=("raw_label", lambda values: "; ".join(values)))
            .reindex(FACTOR_GROUPS))

# %%
def plot_factor_groups(group_counts, raw_label_count):
    """Horizontal bars of crashes per factor group; groups without a usable cause are grey."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bar_colors = [PALETTE["before"] if group in ("Unspecified", "Other") else PALETTE["crash"]
                  for group in group_counts.index]
    ax.barh(group_counts.index.astype(str), group_counts.to_numpy(), color=bar_colors, height=0.72)
    for position, (group, count) in enumerate(group_counts.items()):
        ax.text(count + group_counts.max() * 0.01, position,
                f"{count:,}  ({pct(count, len(crashes)):.1f}%)", va="center", fontsize=9,
                color=PALETTE["neutral"])
    ax.set_xlim(0, group_counts.max() * 1.22)
    thousands(ax, axis="x")
    ax.grid(axis="y", visible=False)
    finish(ax, f"{raw_label_count} raw factor labels reduced to {len(FACTOR_GROUPS)} groups",
           f"Crashes, {YEAR_START} to {YEAR_END}", "Factor group (first contributing factor)",
           note="Grey bars carry no usable cause: the factor was not recorded or is a catch-all label.")
    plt.show()


factor_group_counts = crashes["factor_group"].value_counts().sort_values()
factor_injury_share = (crashes.groupby("factor_group", observed=True)["is_injury_crash"].mean() * 100)
plot_factor_groups(factor_group_counts, len(factor_mapping))

# %%
unrecognised_labels = factor_mapping.loc[factor_mapping["unrecognised"], "raw_label"].tolist()
informative = ~crashes["factor_group"].isin(["Unspecified", "Other"])
top_group = factor_group_counts.drop(["Unspecified", "Other"]).idxmax()
say(f"{len(factor_mapping)} distinct labels ({factor_mapping['normalised'].nunique()} after trimming and "
    f"lower-casing) plus {factor_missing:,} missing values become {len(FACTOR_GROUPS)} groups. Numeric codes "
    f"({', '.join(numeric_codes['raw_label'])}) account for {int(numeric_codes['rows'].sum())} rows. Labels "
    f"that no pattern recognised: {len(unrecognised_labels)}.")
say(f"'Unspecified' holds {pct(factor_group_counts['Unspecified'], len(crashes)):.1f}% of crashes, so "
    f"{pct(informative.sum(), len(crashes)):.1f}% have an informative factor group. The largest informative "
    f"group is {top_group} ({pct(factor_group_counts[top_group], len(crashes)):.1f}%). The share of crashes with an "
    f"injury or death ranges from {factor_injury_share.min():.0f}% ({factor_injury_share.idxmin()}) to "
    f"{factor_injury_share.max():.0f}% ({factor_injury_share.idxmax()}) across groups.")

# %% [markdown]
# Every raw label is claimed by a pattern, so the fallback is unused in this download. A large
# share of crashes has no recorded factor, which limits RQ10 to the crashes that do and has to
# be stated with its results. The groups differ widely in how often they involve an injury, so
# the grouping keeps real information.

# %% [markdown]
# ### 4.12 Final shape of the two frames
#
# The last step puts the columns in a fixed order, leaves out the raw columns that no later
# section uses, and checks the invariants of each frame once more. For housing, the finer
# geographic identifiers (tax lot, building number, census tract, neighbourhood area, council
# district) have no counterpart in the crash table, and the second coordinate pair is missing in
# the same rows as the first, so none of them can help integration. For crashes, the street name,
# the second contributing factor and the vehicle type were used above for checks and are not
# carried forward; the raw factor label is replaced by its group.

# %%
HOUSING_COLUMN_ORDER = [
    "housing_id", "project_id", "building_id", "project_name", "address",
    "borough", "zip", "lat", "lon", "community_board", "is_confidential", "is_located",
    "project_start_date", "project_completion_date", "building_completion_date", "completion_date",
    "start_year", "completion_year", "is_completed", "completion_before_start", "same_day_completion",
    "construction_type", "extended_affordability", "prevailing_wage",
    *INCOME_UNIT_COLUMNS, *BEDROOM_UNIT_COLUMNS, *TENURE_UNIT_COLUMNS, *TOTAL_UNIT_COLUMNS,
    "deep_affordable_units", "deep_affordable_share", "is_size_outlier", "size_band",
]
CRASH_COLUMN_ORDER = [
    "collision_id", "crash_date", "year", "month", "hour", "hour_band",
    "borough", "zip", "lat", "lon", "is_located",
    "injured", "killed", "ped_injured", "ped_killed", "cyc_injured", "cyc_killed",
    "mot_injured", "mot_killed", "other_injured", "other_killed",
    "is_injury_crash", "ped_cyc_harm", "factor_group",
    "casualty_mismatch", "is_extreme_injury", "is_possible_duplicate",
]
assert set(HOUSING_COLUMN_ORDER) == set(housing.columns), set(HOUSING_COLUMN_ORDER) ^ set(housing.columns)
assert set(CRASH_COLUMN_ORDER) == set(crashes.columns), set(CRASH_COLUMN_ORDER) ^ set(crashes.columns)
housing = housing[HOUSING_COLUMN_ORDER]
crashes = crashes[CRASH_COLUMN_ORDER]

# Raw columns that reach the clean frame under another name. Whatever is neither kept by name nor
# renamed is not carried forward, so the dropped set is derived and then compared with the lists.
HOUSING_RENAMED = {"postcode": "zip", "latitude": "lat", "longitude": "lon", "house_number": "address",
                   "street_name": "address", "reporting_construction_type": "construction_type",
                   "extended_affordability_status": "extended_affordability",
                   "prevailing_wage_status": "prevailing_wage"}
CRASH_RENAMED = {"zip_code": "zip", "latitude": "lat", "longitude": "lon", **CASUALTY_COLUMNS}
housing_dropped = sorted(set(raw_housing.columns) - set(housing.columns) - set(HOUSING_RENAMED))
crash_columns_not_kept = sorted(set(raw_crashes.columns) - set(crashes.columns) - set(CRASH_RENAMED))
assert housing_dropped == sorted(HOUSING_COLUMNS_NOT_KEPT), housing_dropped
assert set(HOUSING_RENAMED.values()) <= set(housing.columns)
assert set(CRASH_RENAMED.values()) <= set(crashes.columns)

log_rule("housing", f"{len(housing_dropped)} raw columns not carried forward "
         "(finer geographic ids, second coordinate pair)", "drop", len(housing), len(housing),
         len(housing_dropped), "No counterpart in the crash table. Affected counts columns, not rows.")
log_rule("crashes", f"{len(crash_columns_not_kept)} raw text columns not carried forward "
         "(replaced by typed columns or used only for checks)", "drop", len(crashes), len(crashes),
         len(crash_columns_not_kept), "Keeps the frame compact. Affected counts columns, not rows.")

# Invariants of the two frames that later sections rely on.
assert len(housing) == len(raw_housing), "housing rows are conserved (no rule dropped any)"
assert not repeats_key(housing).any()
assert len(crashes) == crash_rows_raw, "crash rows are conserved (no rule dropped any)"
assert crashes["collision_id"].is_unique
assert list(housing["borough"].cat.categories) == BOROUGHS == list(crashes["borough"].cat.categories)
assert housing["zip"].dtype == crashes["zip"].dtype

display(pd.DataFrame({"dtype": housing.dtypes.astype(str), "missing": housing.isna().sum()}).T)
display(pd.DataFrame({"dtype": crashes.dtypes.astype(str), "missing": crashes.isna().sum()}).T)

# %% [markdown]
# Missing values now remain only where the information does not exist: location fields of
# withheld or unaddressed housing records, completion dates of unfinished buildings, the share
# with a zero denominator, and the borough, ZIP and coordinates that the crash reports never had.

# %% [markdown]
# ### 4.13 Cleaning log and before/after summary
#
# The table lists every rule in the order it ran. `affected` is a number of rows, except where
# the rule says cells or columns.

# %%
cleaning_log_table = pd.DataFrame(CLEANING_LOG)
with pd.option_context("display.max_colwidth", None, "display.max_rows", 100):
    display(cleaning_log_table)
assert (cleaning_log_table["rows_after"] <= cleaning_log_table["rows_before"]).all()
cleaning_log_table["rows_dropped"] = cleaning_log_table["rows_before"] - cleaning_log_table["rows_after"]
display(cleaning_log_table.groupby(["dataset", "action"])
        .agg(rules=("rule", "size"), rows_dropped=("rows_dropped", "sum")))
assert set(cleaning_log_table["action"]) <= {"drop", "impute", "flag", "standardise"}

# %%
before_after = pd.DataFrame({
    "rows before": [len(raw_housing), crash_rows_raw],
    "rows after": [len(housing), len(crashes)],
    "columns before": [raw_housing.shape[1], crash_columns_raw],
    "columns after": [housing.shape[1], crashes.shape[1]],
    "memory before (MB)": [memory_mb(raw_housing), crash_memory_raw],
    "memory after (MB)": [memory_mb(housing), memory_mb(crashes)],
}, index=["housing", "crashes"]).round(1)
display(before_after)
say(f"{len(cleaning_log_table)} rules were logged: "
    f"{int((cleaning_log_table['dataset'] == 'housing').sum())} for housing and "
    f"{int((cleaning_log_table['dataset'] == 'crashes').sum())} for crashes. No rule removed a row, so both "
    f"tables keep their full length ({len(housing):,} and {len(crashes):,} rows). The typed crash frame takes "
    f"{before_after.loc['crashes', 'memory after (MB)']:.0f} MB against "
    f"{before_after.loc['crashes', 'memory before (MB)']:.0f} MB for the raw text.")

# %% [markdown]
# Neither table lost a row. That is a result of the policy and of the data: the keys were sound,
# and every other problem could be handled by standardising a format or by adding a flag. The
# crash frame is several times smaller than the raw text because counts are stored as one-byte
# integers and labels as categories.
#
# The raw crash table and the working copies made in this section are no longer needed and are
# released. The raw housing table is small and is kept for reference. Section 4 consumes
# `raw_crashes`, so running it a second time needs Section 2.1 to be run again first.

# %%
RELEASED_NAMES = [
    "raw_crashes", "crash_source", "crash_ids", "crash_dates", "crash_hour", "crash_minute", "casualties",
    "raw_coordinate_sample", "raw_lat", "raw_lon", "map_sample", "vehicle_type",
    "location_state", "surplus_mask", "is_scooter_or_ebike", "extreme_rows", "casualty_report",
]
release(*RELEASED_NAMES)
