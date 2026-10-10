# %% [markdown]
# ## 2. The data
#
# ### 2.1 Loading the two datasets
#
# Both tables come from the NYC Open Data API (Socrata), following the pattern in the assignment:
# a dataset identifier, a `$where` filter, a `$limit` and a stable `$order`.
#
# Housing is small, so it is downloaded whole in one request. Crashes is large, so the
# notebook downloads a filtered copy: the ten full years 2016 to 2025 and a subset of the
# columns. Section 2.5 explains that choice and prints the exact numbers. The download is paged
# on `collision_id`, which is unique and never missing, so pages cannot overlap or skip rows
# even if the table is updated mid-download. Each page is appended to a temporary file as it
# arrives, so an interrupted download continues from the last page it saved.
#
# The first run saves each table in the folder the notebook runs from (the working directory).
# Later runs read those two files, which takes seconds and works offline. Delete a file to
# download a fresh copy. A first run needs an internet connection and takes several minutes. On
# Colab the storage is lost when the runtime recycles, so the download then repeats. The notebook
# needs Python 3.12 or newer, which Section 1 checks.

# %%
# --- Constants shared by Sections 3 and 4 (defined once so the two sections cannot drift apart) --
# The break year comes from Section 1.2, where it is defined for the whole notebook.
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"   # the single timestamp layout the API returns for both tables
BYTES_PER_MB = 1_000_000
CONFIDENTIAL_NAME = "CONFIDENTIAL"     # project_name the publisher uses for withheld records
WITHHELD_ADDRESS = "----"              # house number and street of a withheld record
ZIP_PLACEHOLDER = "10000"              # not a real ZIP: used in the crash table when the officer had no code
IQR_MULTIPLIER = 1.5                   # Tukey's fence: beyond 1.5 interquartile ranges from the quartiles
EXTREME_INJURED = 10                   # persons; more than this cannot fit in a few cars, so it marks bus-scale events
INTEGER_BIN_COUNT = 30                 # log-spaced bin edges for histograms of unit counts
TOO_MANY_REQUESTS = 429                # HTTP status that means "slow down", which is worth a retry

# Housing unit columns. A blank in any of them means zero units (shown in 3.1.3, applied in 4.1).
INCOME_UNIT_COLUMNS = ["extremely_low_income_units", "very_low_income_units", "low_income_units",
                       "moderate_income_units", "middle_income_units", "other_income_units"]
BEDROOM_UNIT_COLUMNS = ["studio_units", "_1_br_units", "_2_br_units", "_3_br_units", "_4_br_units",
                        "_5_br_units", "_6_br_units", "unknown_br_units"]
TENURE_UNIT_COLUMNS = ["counted_rental_units", "counted_homeownership_units"]
TOTAL_UNIT_COLUMNS = ["all_counted_units", "total_units"]
UNIT_COLUMNS = INCOME_UNIT_COLUMNS + BEDROOM_UNIT_COLUMNS + TENURE_UNIT_COLUMNS + TOTAL_UNIT_COLUMNS


def join_and(items):
    """Join words into a readable list: 'a', 'a and b', 'a, b and c'."""
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def integer_log_bins(maximum):
    """Histogram edges for a whole-number variable on a log axis.

    Counts are integers, so edges sit halfway between integers. Plain log-spaced edges would leave
    empty bins between 1, 2 and 3 units and draw gaps that are not in the data.
    """
    integer_steps = np.unique(np.round(np.logspace(0, np.log10(maximum), INTEGER_BIN_COUNT)))
    return np.append(integer_steps - 0.5, maximum + 0.5)


def fetch_csv(dataset_id, params):
    """Request one page from the Socrata API and return it as an all-text DataFrame.

    Retries with exponential back-off, because the public endpoint sometimes answers with a
    429 (rate limit) or a 5xx. Any other 4xx answer means the request itself is wrong, so it stops
    at once. Every column is read as text so that nothing is silently converted before the
    cleaning section decides how to treat it.
    """
    url = API_URL.format(dataset_id) + "?" + urlencode(params)
    last_problem = "no attempt made"
    for attempt in range(REQUEST_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as error:
            last_problem = f"network error: {error}"      # a network hiccup: wait and try again
        else:
            if response.status_code == 200:
                text = response.text
                return pd.read_csv(io.StringIO(text), dtype=str) if text.strip() else pd.DataFrame()
            last_problem = f"HTTP {response.status_code}"
            is_permanent = 400 <= response.status_code < 500 and response.status_code != TOO_MANY_REQUESTS
            if is_permanent:
                raise RuntimeError(f"NYC Open Data rejected the request ({last_problem}): {url}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"NYC Open Data did not answer after {REQUEST_RETRIES} attempts "
                       f"(last problem: {last_problem}): {url}")


def download_housing(path):
    """Download the whole housing table in one request and save it as CSV."""
    table = fetch_csv(HOUSING_ID, {"$limit": HOUSING_ROW_LIMIT, "$order": "building_id, project_id"})
    # If the table ever outgrows the limit the request would be cut short without an error.
    assert len(table) < HOUSING_ROW_LIMIT, "housing table reached the row limit: raise HOUSING_ROW_LIMIT"
    table.to_csv(path, index=False)


def last_saved_collision_id(partial_path):
    """Largest collision_id already written to an interrupted download (0 when there is none)."""
    if not partial_path.exists():
        return 0
    saved = pd.read_csv(partial_path, usecols=["collision_id"], compression="gzip")
    return int(saved["collision_id"].max()) if len(saved) else 0


def download_crashes(path):
    """Download the filtered crash table page by page (keyset paging) and save it compressed.

    Keyset paging asks for 'the next PAGE_SIZE rows whose collision_id is above the last one
    seen'. Unlike $offset paging it stays correct while the live table receives new rows.
    Each page is appended to a temporary file that is renamed only when the last page is in, so
    memory stays at one page and an interrupted run continues where it stopped.
    """
    partial_path = path.with_name(path.name + ".part")
    last_id = last_saved_collision_id(partial_path)
    rows_saved = 0
    while True:
        page = fetch_csv(CRASH_ID, {
            "$select": ",".join(CRASH_COLUMNS),
            "$where": f"{CRASH_WHERE} AND collision_id > {last_id}",
            "$order": "collision_id",
            "$limit": PAGE_SIZE,
        })
        if page.empty:
            break
        # The API leaves out a column when it is empty on a page, so restore the full layout.
        page = page.reindex(columns=CRASH_COLUMNS)
        is_first_page = not partial_path.exists()
        page.to_csv(partial_path, mode="a", header=is_first_page, index=False, compression="gzip")
        last_id = int(pd.to_numeric(page["collision_id"]).max())
        rows_saved += len(page)
        print(f"  downloaded {rows_saved:>9,} crash rows", end="\r")
    partial_path.replace(path)


def load_raw(path, download):
    """Read a local copy if it exists, otherwise download it first. Returns all-text columns."""
    # Arrow-backed strings need the pyarrow package. Creating one empty array fails at once with a
    # clear message if it is missing, instead of after a long download.
    pd.array([], dtype="string[pyarrow]")
    if not path.exists():
        print(f"{path.resolve()} not found: downloading from NYC Open Data ...")
        download(path)
    # Arrow-backed strings hold 1.5 million text rows in a fraction of the memory of Python objects.
    return pd.read_csv(path, dtype="string[pyarrow]")


# %%
raw_housing = load_raw(HOUSING_FILE, download_housing)
raw_crashes = load_raw(CRASH_FILE, download_crashes)

print("housing columns:", list(raw_housing.columns))
print("crash columns:  ", list(raw_crashes.columns))

# %% [markdown]
# Four small helpers are shared by Sections 3 and 4, so each rule exists once: a number parser
# that can be strict or lenient, the upper outlier fence, the clock-time parser and the memory
# measure. The map helper gives longitude/latitude plots the proportions of the ground.

# %%
CLOCK_PATTERN = r"^(\d{1,2}):(\d{2})$"   # crash_time is written H:MM or HH:MM
MID_LATITUDE_DEG = float(np.mean(NYC_BOUNDS["lat"]))   # latitude at the middle of the city box, for map proportions


def to_number(text, strict=True):
    """Convert a text column to float64.

    With `strict` a non-blank value that is not a number stops the notebook, which is what the
    cleaning section needs. Without it the value becomes NaN, which lets the exploration count
    the failures instead of stopping at the first one.
    """
    numbers = pd.to_numeric(text, errors="coerce")
    number = pd.Series(numbers.to_numpy(dtype="float64", na_value=np.nan), index=text.index, name=text.name)
    if strict:
        unparseable = int((text.notna() & number.isna()).sum())
        assert unparseable == 0, f"{unparseable} non-numeric values in column {text.name}"
    return number


def tukey_upper_fence(values):
    """Upper Tukey fence: third quartile plus IQR_MULTIPLIER interquartile ranges."""
    first_quartile, third_quartile = np.percentile(values, [25, 75])
    return third_quartile + IQR_MULTIPLIER * (third_quartile - first_quartile)


def parse_hour(time_text):
    """Hour of day (0-23) and minute from 'H:MM' text; anything else becomes missing.

    Returns the hour as nullable Int8 and the minute as float64.
    """
    parts = time_text.str.strip().str.extract(CLOCK_PATTERN).astype("float64")
    is_valid = parts[0].between(0, 23) & parts[1].between(0, 59)
    return parts[0].where(is_valid).astype("Int8"), parts[1].where(is_valid)


def memory_mb(frame):
    """Memory held by a frame, in megabytes, counting the text inside string columns."""
    return frame.memory_usage(deep=True).sum() / BYTES_PER_MB


def map_axes(ax):
    """Give a longitude/latitude axes the proportions of the ground and remove the grid."""
    ax.set_aspect(1 / math.cos(math.radians(MID_LATITUDE_DEG)))
    ax.grid(False)
    return ax


# %% [markdown]
# ### 2.2 Dataset overview
#
# The table below describes the copy of the data this notebook analyses.

# %%
def dataset_card(name, dataset_id, domain, frame, grain, time_columns, place_columns):
    """One overview row per dataset: size, grain and the attributes usable for integration."""
    return {
        "dataset": name, "domain": domain, "id": dataset_id,
        "page": DATASET_PAGE.format(dataset_id), "rows": f"{len(frame):,}", "columns": frame.shape[1],
        "one row is": grain,
        "time attributes": ", ".join(c for c in time_columns if c in frame.columns),
        "place attributes": ", ".join(c for c in place_columns if c in frame.columns),
    }


overview = pd.DataFrame([
    dataset_card("Affordable Housing Production by Building", HOUSING_ID, "Housing & Development",
                 raw_housing, "a building inside a city-financed affordable housing project",
                 ["project_start_date", "project_completion_date", "building_completion_date"],
                 ["borough", "postcode", "community_board", "census_tract", "latitude", "longitude"]),
    dataset_card("Motor Vehicle Collisions - Crashes", CRASH_ID, "Transportation",
                 raw_crashes, "a police-reported motor vehicle collision",
                 ["crash_date", "crash_time"],
                 ["borough", "zip_code", "on_street_name", "latitude", "longitude"]),
]).set_index("dataset").T
display(overview)

# The assignment requires at least 9 columns per dataset and at least one shared integration attribute.
assert raw_housing.shape[1] >= 9 and raw_crashes.shape[1] >= 9
assert {"borough", "latitude", "longitude"} <= set(raw_housing.columns) & set(raw_crashes.columns)

# %% [markdown]
# ### 2.3 Why these two datasets
#
# The question is whether the streets around new affordable housing differ in their injury-crash
# patterns from other streets, and whether those patterns move when a building is completed.
# Answering it needs two things from each side: a place precise enough to talk about "the
# streets around" something, and a date precise enough to talk about "before" and "after".
#
# The housing table supplies the events. Each row is one building in a city-financed project,
# with a start date, a completion date, the number of units by income band, whether the work was
# new construction or preservation of existing homes, and a point location. Completion is the
# moment a new building can be occupied, so it gives a natural before and after. Preservation
# rows add no residents, which later makes them a useful comparison group.
#
# The crash table supplies the outcome. Each row is one police-reported collision with its date,
# time, point location, the number of people injured and killed split into pedestrians, cyclists
# and motorists, and the contributing factor the officer recorded. The split matters here
# because a residential building adds people on foot to a street before it adds anything else.
#
# Other tables in the two catalogs were considered and set aside.
#
# | Catalog | Alternative | Why it was weaker for this question |
# |---|---|---|
# | Housing | Department of Buildings (DOB) violations, Housing Preservation and Development (HPD) complaints | They describe the condition of buildings that already exist. They do not say when new homes open or how many households they hold. |
# | Housing | DOB permits and job filings | They record the intention to build. A filing carries no count of affordable units and no dependable completion date, so there is no clear "after". |
# | Transportation | Parking meters | An inventory of equipment with no time dimension and no outcome to measure. |
# | Transportation | Traffic counts, bike counts | Measured at a limited set of counting locations, so most housing sites would have no observation nearby. |
# | Transportation | Street permits | They describe work on the street, which is an activity and not a safety outcome. |
# | Transportation | Bus delays | Recorded by route and operator, without a coordinate that could be tied to the surroundings of a building. |
# | Transportation | Taxi trips | Located by taxi zone only, which is far coarser than a building, and too large to process in a notebook. |
#
# The deciding point is the overlap of integration attributes. Both chosen tables carry
# latitude and longitude, a ZIP code, a borough and an exact calendar date. That allows two
# independent ways of joining them (by ZIP code and year, and by distance) and lets one check
# the other. No other pair that was looked at offered all four on both sides.

# %% [markdown]
# ### 2.4 Feasibility check
#
# The course catalog lists the crash table under the label "SIRTA Data" (the catalog's name for
# the collisions table) and not under its NYC Open Data title, so a reader could take it for a
# dataset from outside the catalog. The assignment asks for a short feasibility check in that
# case, so one is included here.

# %%
def non_missing_share(frame, columns):
    """Percentage of rows in which every one of `columns` holds a non-blank value."""
    filled = pd.Series(True, index=frame.index)
    for column in columns:
        filled &= (frame[column].notna() & frame[column].str.strip().ne("")).fillna(False).astype(bool)
    return pct(int(filled.sum()), len(frame))


def date_span(frame, column):
    """First and last calendar date in a raw ISO-formatted text column."""
    days = frame[column].dropna().str.slice(0, 10)
    return f"{days.min()} to {days.max()}"


COORDINATE_COLUMNS = ["latitude", "longitude"]
BOROUGH_COLUMN = ["borough"]


def feasibility_card(frame, date_column, zip_column):
    """Size and integration attributes of one raw table."""
    return {
        "rows": f"{len(frame):,}",
        "columns": frame.shape[1],
        "date range": f"{date_span(frame, date_column)} ({date_column})",
        "rows with coordinates (%)": f"{non_missing_share(frame, COORDINATE_COLUMNS):.1f}",
        "rows with a ZIP code (%)": f"{non_missing_share(frame, [zip_column]):.1f}",
        "rows with a borough (%)": f"{non_missing_share(frame, BOROUGH_COLUMN):.1f}",
        "distinct ZIP code values": frame[zip_column].nunique(),
    }


def feasibility_table(housing, crashes):
    """Feasibility cards of both tables side by side, plus the column names they share."""
    table = pd.DataFrame({
        "Affordable Housing Production by Building": feasibility_card(housing, "project_start_date", "postcode"),
        "Motor Vehicle Collisions - Crashes": feasibility_card(crashes, "crash_date", "zip_code"),
    })
    table.loc["column names shared by both"] = ", ".join(sorted(set(housing.columns) & set(crashes.columns)))
    return table


display(feasibility_table(raw_housing, raw_crashes))

# %% [markdown]
# Reading the check against the four points the assignment lists:
#
# * Size. Both tables clear the nine-column minimum, and the crash table is large enough for
#   the surroundings of a single building to contain many collisions over ten years.
# * Geographic and temporal attributes. Both have a borough, a ZIP code, point coordinates and
#   a calendar date. The crash table also has the time of day.
# * Expected level of integration. There is no shared row key, because a crash does not belong
#   to a building. The tables meet at two levels: ZIP code by year, which is coarse but uses
#   only published fields, and distance, where a crash counts as near a building when it lies
#   within 250 m of it. Section 5 builds both.
# * Why the pair supports the question. One side gives dated, located changes in where people
#   live. The other gives dated, located harm on the street. Most rows on both sides have
#   usable coordinates, which is what the distance join needs, and the crash dates cover years
#   before and after most completions.
#
# Coverage is incomplete on both sides, and Section 3 shows that the gaps are not random.

# %% [markdown]
# ### 2.5 Why the crash table is a filtered copy
#
# The assignment allows a filtered sample of a large table as long as the choice is justified.
# The cell below prints the exact request the notebook sends, built from the constants in
# Section 1.2, together with the columns the request leaves out.

# %%
CRASH_COLUMNS_NOT_LOADED = [
    "location",                       # latitude and longitude repeated as one text field
    "off_street_name", "cross_street_name",
    "contributing_factor_vehicle_3", "contributing_factor_vehicle_4", "contributing_factor_vehicle_5",
    "vehicle_type_code2", "vehicle_type_code_3", "vehicle_type_code_4", "vehicle_type_code_5",
]


def describe_crash_request():
    """Show the Socrata query parameters used for the crash download, one per row."""
    request = pd.DataFrame({
        "value": {
            "$select": ", ".join(CRASH_COLUMNS),
            "$where": CRASH_WHERE + " AND collision_id > (last collision_id of the previous page)",
            "$order": "collision_id",
            "$limit": f"{PAGE_SIZE:,} rows per request, repeated until a request returns no rows",
            "columns not requested": ", ".join(CRASH_COLUMNS_NOT_LOADED),
        }
    })
    with pd.option_context("display.max_colwidth", None):
        display(request)


def check_crash_copy(crashes):
    """Assert that the copy matches the request, then describe it in one sentence."""
    crash_years = crashes["crash_date"].str.slice(0, 4).astype(int)
    assert crash_years.between(YEAR_START, YEAR_END).all(), "a crash lies outside the study window"
    assert crashes["collision_id"].is_unique, "paging returned a collision twice"
    assert not set(CRASH_COLUMNS_NOT_LOADED) & set(crashes.columns)
    cohort_years = full_window_years()
    say(f"The copy holds **{len(crashes):,} crashes**, every one dated between {YEAR_START} and "
        f"{YEAR_END}, and no `collision_id` appears twice. As text it takes "
        f"**{memory_mb(crashes):,.0f} MB** of memory. With "
        f"{EVENT_WINDOW} whole years on each side, buildings completed from {cohort_years[0]} to "
        f"{cohort_years[-1]} have a complete crash history before and after.")


describe_crash_request()
check_crash_copy(raw_crashes)
say(f"The request keeps {len(CRASH_COLUMNS)} of the {len(CRASH_COLUMNS) + len(CRASH_COLUMNS_NOT_LOADED)} "
    f"columns of the source table and the {len(STUDY_YEARS)} calendar years {YEAR_START} to {YEAR_END}.")

# %% [markdown]
# The filter restricts rows by date and columns by name. It does not sample. Inside the
# 2016 to 2025 window every row of the source table is loaded, so no count in this notebook is
# an estimate from a subset. Paging stops only when a request comes back empty.
#
# The reasons for each restriction:
#
# * Size and memory. The full table is much larger than the filtered copy. Read as text with
#   every column it leaves little room in a free Colab session for the intermediate tables the
#   spatial join needs.
# * 2026 is left out because the year is still in progress. A partial year would look like a
#   fall in crashes in every yearly chart.
# * Years before 2016 are left out because no research question reads them. The comparison of
#   before and after needs whole years on both sides of a completion, and the earliest window
#   the analysis uses starts in 2016. Loading the earlier years would add rows that nothing
#   reads.
# * The columns not requested are the contributing factors of the third to fifth vehicle, the
#   vehicle types of the second to fifth vehicle, and street and location text that repeats
#   what `on_street_name`, `latitude` and `longitude` already hold. The first vehicle's type
#   and the first two contributing factors are kept.
#
# The alternative was a random sample across all years. It was rejected because sampling thins
# out exactly the quantity being measured: the crashes within 250 m of one building in one year
# are already a small count, and a sample would turn many of those counts into zeros.

# %% [markdown]
# ### 2.6 Known issues at a glance
#
# These are the problems the next section demonstrates on the raw tables, with the place in the
# notebook where each one is handled.
#
# | Dataset | Issue | Handled in |
# |---|---|---|
# | Housing | Unit columns are blank where the count is zero | Section 4 |
# | Housing | Confidential rows have no address, ZIP code or coordinates | Section 4 (flagged, kept for borough totals) |
# | Housing | Some rows have no completion date | Section 4 |
# | Housing | The same building can appear under more than one project | Section 4 |
# | Housing | Building size is strongly right-skewed | Section 4 (flagged, not removed) |
# | Housing | Identifiers, dates and ZIP codes arrive as text | Section 4 |
# | Crashes | ZIP code and borough are missing together for a large share of rows, and the share changes by year | Section 6 (ZIP code recovered from coordinates) |
# | Crashes | Some coordinates are missing, sit at (0, 0) or fall outside the city | Section 4 |
# | Crashes | Reports of crashes without injuries fall sharply from 2020 | Section 3, after which every analysis uses injury crashes |
# | Crashes | Placeholder and malformed ZIP codes | Section 4 |
# | Crashes | Contributing factors have spelling and case variants and a large "Unspecified" group | Section 4 |
# | Crashes | Persons injured does not always equal pedestrians plus cyclists plus motorists | Section 4 |
# | Crashes | A few crashes report an unusually high number of injuries | Section 4 (flagged) |
# | Both | Borough names use different letter case in the two tables | Section 4 |
# | Both | Rows of one table that find no partner in the other once the tables meet | Section 6 |
