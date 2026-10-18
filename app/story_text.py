"""Sentences for the story sections: number formatting and the text built from results.json.

Kept apart from app/story.py, which lays the sections out, so wording and layout change separately.
"""
from __future__ import annotations

import re

from app.data import Store


MINUS = "−"


def signed(value: float, digits: int = 1) -> str:
    """'+3.3' or '−4.6' with a real minus sign."""
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return text
    return ("+" if value > 0 else MINUS) + text


def interval_text(item: dict, digits: int = 1, suffix: str = "", prefix: str = "") -> str:
    """'−11.9% to +3.3%' for a dict with `low` and `high` (or `percent_low` and `percent_high`)."""
    return f"{signed(item[prefix + 'low'], digits)}{suffix} to {signed(item[prefix + 'high'], digits)}{suffix}"


def unit_digits(unit: str) -> int:
    return 3 if "correlation" in unit else 2


def number_word(count: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
            9: "nine", 10: "ten"}.get(count, str(count))


def plural(count: int, one: str, many: str) -> str:
    return one if count == 1 else many


SAME_AS_ZERO = re.compile(r"; the interval includes zero, so the data are compatible with no difference")


def reading_label(item: dict) -> tuple[str, str]:
    """(text, CSS class) of the tag under a result: includes zero, excludes zero, or not kept."""
    if not item["excludes_zero"]:
        return "Compatible with no change", "tag"
    if item["kept"]:
        return "Clear of zero", "tag tag-strong"
    return "Clear of zero, but not counted after adjusting for several tests", "tag"


def finding_extras(store: Store) -> dict[str, str]:
    """A second sentence for the questions whose headline needs a check or a caveat beside it."""
    detail = store.results["detail"]
    pretrend, rq7, reported = detail["rq3_pretrend"], detail["rq7"], detail["reported_zips"]
    by_type, threshold = detail["rq5_by_type"], detail["false_discovery_rate"]
    same_year, all_zips = reported["rq2_same_year"], reported["rq2_same_year_all_zips"]
    kept_on_reported = same_year["adjusted_p"] < threshold
    kept_on_all = all_zips["adjusted_p"] < threshold
    rq1 = reported["rq1"]
    includes_zero = rq1["low"] < 0 < rq1["high"]
    new_type, preservation = by_type["New Construction"], by_type["Preservation"]
    return {
        "RQ1": f"Counting only crashes whose ZIP code was reported, the change in forecast error is "
               f"{signed(rq1['estimate'], 2)} injury crashes per ZIP per year ({interval_text(rq1, 2)}), which "
               f"{'includes' if includes_zero else 'excludes'} zero. With either version of the data, adding "
               "housing information did not make the forecast better.",
        "RQ2": f"Of five timing slopes, the one for the completion year is "
               f"{signed(all_zips['estimate'], 2)}% per 100 units (the change in a ZIP's injury crashes in the "
               "year new units open, compared with that ZIP's other years). With all ZIP codes it is "
               f"{'kept' if kept_on_all else 'not kept'} after adjustment (p = {all_zips['adjusted_p']:.2f}). On "
               f"reported ZIP codes alone the adjusted p is {same_year['adjusted_p']:.2f}, "
               f"{'below' if kept_on_reported else 'above'} the {threshold} false-discovery limit, so there it is "
               f"{'kept' if kept_on_reported else 'reported and not counted as a finding'}. It is an association, "
               "not evidence that housing lowers crashes.",
        "RQ3": f"Check on the comparison: in the three years before completion the two groups already differ by "
               f"{signed(pretrend['percent_change'])}% ({signed(pretrend['percent_low'])}% to "
               f"{signed(pretrend['percent_high'])}%). The interval includes zero but runs as far as "
               f"{signed(pretrend['percent_high'], 0)}%, so this check cannot rule out a pre-existing gap of that size.",
        "RQ5": f"Against the same comparison sites, new construction changes by {signed(new_type['percent'])}% "
               f"({interval_text(new_type, suffix='%')}) and preservation by "
               f"{signed(preservation['percent'])}% ({interval_text(preservation, suffix='%')}). "
               "Preservation renovates buildings that already have residents, so a change there is not expected "
               "to come from new residents. We use it as a check.",
        "RQ7": f"The benchmark is a grid of street points across each borough, so this shows where the buildings "
               f"stand. It cannot show that affordable housing is placed on busier streets than market-rate "
               f"housing would be. Part of the gap is a difference between neighbourhoods: ranked inside their "
               f"own ZIP code, new-construction sites are {rq7['within_zip']['estimate']:.1f} percentile points "
               f"above the middle ({interval_text(rq7['within_zip'])}). All intervals resample whole clusters of "
               f"linked sites.",
        "RQ9": f"On reported ZIP codes alone the interval is {interval_text(reported['rq9'], 3)}. We treat this as a "
               "hint and do not count it as a finding.",
    }


def interval_level_text(questions: dict) -> str:
    """Which interval level the bars show, read from each stored result."""
    levels = sorted({item["level"] for item in questions.values()})
    if len(levels) == 1:
        return f"The bar shows the {levels[0]:.0%} interval."
    return ("Each bar shows the interval at the level printed beside it ("
            + ", ".join(f"{level:.0%}" for level in levels) + ", depending on the question).")


def zip_level_text(store: Store) -> str:
    """'Three of the ZIP-level results (RQ1, RQ2 and RQ9)', from the questions stored under reported ZIP codes."""
    keys = sorted({key.split("_")[0].upper() for key in store.results["detail"]["reported_zips"]},
                  key=lambda key: int(key[2:]))
    names = keys[0] if len(keys) == 1 else ", ".join(keys[:-1]) + " and " + keys[-1]
    return f"{number_word(len(keys)).capitalize()} of the ZIP-level results ({names})"


def findings_lead(questions: dict) -> str:
    """How many intervals include zero, which exclude it, and which of those are not kept."""
    excluding = [key for key, item in questions.items() if item["excludes_zero"]]
    not_kept = [key for key in excluding if not questions[key]["kept"]]
    text = (f"Each of the {number_word(len({item['owner'] for item in questions.values()}))} team members asked two questions that need both datasets; RQ1 to RQ{len(questions)} "
            f"are the question numbers used throughout this page. {len(questions) - len(excluding)} of the "
            f"{len(questions)} intervals include zero and {len(excluding)} exclude it ({', '.join(excluding)}).")
    if not_kept:
        text += (f" {', '.join(not_kept)} {plural(len(not_kept), 'is', 'are')} not kept after adjustment for "
                 f"the tests made with {plural(len(not_kept), 'it', 'them')}.")
    if "RQ1" in excluding and "RQ7" in excluding:
        rq7 = questions["RQ7"]
        text += (" RQ1 is a forecast result: adding housing data made the forecast less accurate. RQ7 is the "
                 f"largest effect: it describes the streets before anyone moves in, and its interval "
                 f"({interval_text(rq7)} percentile points) is far from zero.")
    return text


def robustness_text(store: Store) -> str:
    """How many of the one-change versions include zero, how the sign moves, and what the exception is."""
    table = store.robustness[store.robustness["estimate"].notna()]
    checks = table[table["kind"] == "check"]
    unscreened = set(store.results["detail"]["unscreened_variants"])
    screened = checks[~checks["variant"].isin(unscreened)]
    excluding = checks[checks["interval_excludes_zero"]]
    text = (f"We changed one choice at a time: the radius, the outcome, the screen for nearby completions, "
            f"the cohorts, one building per project and the comparison inside boroughs. "
            f"{len(checks) - len(excluding)} of the {len(checks)} intervals include zero. ")
    text += (f"Across the {len(screened)} versions that keep the screen, the intervals run from "
             f"{signed(screened['percent_low'].min())}% to {signed(screened['percent_high'].max())}%. ")
    rises = screened[screened["percent_change"] > 0]
    if len(rises) and (screened["percent_change"] < 0).any():
        top = rises.loc[rises["percent_change"].idxmax()]
        text += (f"The estimate itself changes sign with the choice of sites (for example "
                 f"{signed(top['percent_change'])}% in the version \u201c{top['variant']}\u201d), so the data "
                 "support no direction. ")
    if len(excluding) and set(excluding["variant"]) <= unscreened:
        text += ("The exceptions all switch the screen off, which admits sites with other completions "
                 "close by. ")
        without = table[table["variant"].str.contains("without the most influential cluster", case=False)]
        if len(without):
            row = without.iloc[0]
            text += (f"One large development carries them: leaving its cluster out gives "
                     f"{signed(row['percent_change'])}% ({signed(row['percent_low'])}% to "
                     f"{signed(row['percent_high'])}%).")
    elif len(excluding):
        text += "The exceptions: " + "; ".join(excluding["variant"]) + "."
    return text.strip()
