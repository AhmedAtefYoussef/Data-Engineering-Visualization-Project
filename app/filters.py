"""The filter model shared by the controls, the search parser, the data layer and the charts.

A `Filters` value is immutable and hashable, so it can key a cache and travel through a
`dcc.Store` as a plain dict. Empty tuples mean "no restriction".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

YEAR_START, YEAR_END = 2016, 2025
YEARS = list(range(YEAR_START, YEAR_END + 1))

BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
CONSTRUCTION_TYPES = ["New Construction", "Preservation"]
SIZE_BANDS = ["1-4", "5-19", "20-49", "50-149", "150+"]
FACTOR_GROUPS = [
    "Unspecified", "Pedestrian or cyclist error", "Road or environment", "Vehicle defect",
    "Impairment or medical", "Distraction", "Speed or aggressive driving",
    "Failure to yield or obey", "Improper maneuver", "Other",
]
HOUR_BANDS = ["Night", "Morning", "Afternoon", "Evening"]
HOUR_BAND_RANGES = {"Night": "00:00 to 05:59", "Morning": "06:00 to 11:59",
                    "Afternoon": "12:00 to 17:59", "Evening": "18:00 to 23:59"}

# Who was harmed. Each key names one count column of the crash tables (see tools/build_app_data.py).
# `noun` completes "N ..." in chart titles; `label` is the dropdown text.
HARM = {
    "injury": {"label": "Anyone injured or killed", "noun": "injury crashes",
               "column": "n_injury"},
    "pedcyc": {"label": "Pedestrians or cyclists", "noun": "crashes that hurt a pedestrian or cyclist",
               "column": "n_pedcyc"},
    "ped": {"label": "Pedestrians", "noun": "crashes that hurt a pedestrian", "column": "n_ped"},
    "cyc": {"label": "Cyclists", "noun": "crashes that hurt a cyclist", "column": "n_cyc"},
    "mot": {"label": "Motorists", "noun": "crashes that hurt a motorist", "column": "n_mot"},
    "fatal": {"label": "Someone killed", "noun": "fatal crashes", "column": "n_fatal"},
    "all": {"label": "All reported crashes (see note)", "noun": "reported crashes",
            "column": "n_all"},
}
DEFAULT_HARM = "injury"

# Where the crash happened relative to the housing sites (the spatial join of the two datasets).
SCOPE = {
    "city": "Anywhere in the city",
    "near": "Within 250 m of the selected housing",
}
DEFAULT_SCOPE = "city"
BUFFER_M = 250


@dataclass(frozen=True)
class Filters:
    """One selection of the eight controls."""

    boroughs: tuple[str, ...] = ()
    year_from: int = YEAR_START
    year_to: int = YEAR_END
    construction: tuple[str, ...] = ()
    sizes: tuple[str, ...] = ()
    factors: tuple[str, ...] = ()
    hours: tuple[str, ...] = ()
    harm: str = DEFAULT_HARM
    scope: str = DEFAULT_SCOPE
    # not a control: kept so `describe` can say where a selection came from
    source: str = field(default="controls", compare=False)

    @property
    def years(self) -> list[int]:
        """The selected calendar years, in order."""
        return list(range(self.year_from, self.year_to + 1))

    @property
    def n_years(self) -> int:
        return self.year_to - self.year_from + 1

    @property
    def harm_noun(self) -> str:
        return HARM[self.harm]["noun"]

    @property
    def harm_column(self) -> str:
        return HARM[self.harm]["column"]

    @property
    def has_housing_filter(self) -> bool:
        return bool(self.construction or self.sizes)

    def to_dict(self) -> dict:
        """JSON-safe dict for a dcc.Store."""
        return {key: (list(value) if isinstance(value, tuple) else value) for key, value in asdict(self).items()}


def _keep_known(values, allowed: list[str]) -> tuple[str, ...]:
    """Selected values in canonical order; anything unknown is dropped, and 'all selected' means no filter."""
    chosen = set(values or [])
    kept = tuple(value for value in allowed if value in chosen)
    return () if len(kept) == len(allowed) else kept


def _year(value, fallback: int) -> int:
    """A year clamped to the study window; `fallback` when the control is empty or malformed."""
    try:
        return max(YEAR_START, min(YEAR_END, int(value)))
    except (TypeError, ValueError):
        return fallback


def make_filters(boroughs=None, year_from=None, year_to=None, construction=None, sizes=None,
                 factors=None, hours=None, harm=None, scope=None, source: str = "controls") -> Filters:
    """Build a valid `Filters` from raw control values. Never raises: bad input falls back to the default."""
    first, last = _year(year_from, YEAR_START), _year(year_to, YEAR_END)
    if first > last:                       # a reversed range is read as the same range
        first, last = last, first
    return Filters(
        boroughs=_keep_known(boroughs, BOROUGHS),
        year_from=first, year_to=last,
        construction=_keep_known(construction, CONSTRUCTION_TYPES),
        sizes=_keep_known(sizes, SIZE_BANDS),
        factors=_keep_known(factors, FACTOR_GROUPS),
        hours=_keep_known(hours, HOUR_BANDS),
        harm=harm if harm in HARM else DEFAULT_HARM,
        scope=scope if scope in SCOPE else DEFAULT_SCOPE,
        source=source,
    )


def from_dict(stored: dict | None) -> Filters:
    """Inverse of `Filters.to_dict`, tolerant of missing keys."""
    names = {item.name for item in fields(Filters)}
    return make_filters(**{key: value for key, value in (stored or {}).items() if key in names})


def year_text(filters: Filters) -> str:
    """'2022' or '2019 to 2021'."""
    if filters.year_from == filters.year_to:
        return str(filters.year_from)
    return f"{filters.year_from} to {filters.year_to}"


def _joined(values: tuple[str, ...], everything: str) -> str:
    return ", ".join(values) if values else everything


def place_text(filters: Filters) -> str:
    """'Brooklyn' or 'all five boroughs'."""
    return _joined(filters.boroughs, "all five boroughs")


def crash_scope_text(filters: Filters) -> str:
    """One line naming every crash-side filter in force, for chart subtitles."""
    parts = [place_text(filters), year_text(filters), HARM[filters.harm]["label"].replace(" (see note)", "").lower()]
    if filters.factors:
        parts.append("factor: " + ", ".join(filters.factors).lower())
    if filters.hours:
        parts.append("time: " + ", ".join(filters.hours).lower())
    if filters.scope == "near":
        parts.append(f"within {BUFFER_M} m of " + housing_kind_text(filters))
    return " · ".join(parts)


def housing_kind_text(filters: Filters) -> str:
    """'new construction sites with 150+ units' or 'any affordable-housing site'."""
    if not filters.has_housing_filter:
        return "any affordable-housing site"
    kind = " or ".join(filters.construction).lower() if filters.construction else "affordable-housing"
    text = f"{kind} sites"
    if filters.sizes:
        text += " with " + ", ".join(filters.sizes) + " units"
    return text


def housing_scope_text(filters: Filters) -> str:
    """One line naming every housing-side filter in force."""
    parts = [place_text(filters), f"completed {year_text(filters)}",
             _joined(filters.construction, "both construction types").lower()]
    if filters.sizes:
        parts.append("size: " + ", ".join(filters.sizes) + " units")
    return " · ".join(parts)


def chips(filters: Filters) -> list[tuple[str, str]]:
    """(label, value) pairs for every control, used by the report header."""
    return [
        ("Borough", _joined(filters.boroughs, "All")),
        ("Years", year_text(filters)),
        ("Construction", _joined(filters.construction, "Both types")),
        ("Building size", _joined(filters.sizes, "Any") + (" units" if filters.sizes else "")),
        ("Contributing factor", _joined(filters.factors, "Any")),
        ("Time of day", _joined(filters.hours, "Any")),
        ("Who was harmed", HARM[filters.harm]["label"].replace(" (see note)", "")),
        ("Crash location", SCOPE[filters.scope]),
    ]
