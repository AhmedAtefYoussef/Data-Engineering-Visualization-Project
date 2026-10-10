"""Research-question registry: 10 (figure, statistic, headline) triples, run_all() and summary helpers.

Public API (SPEC s6): Artifacts, apply_filters, RQS, rqN_fig / rqN_stat, run_all, write_results.
All findings text is generated from numbers; wording is associational and hedged by the CI sign."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from analysis.data import (FILTER_LABELS, Artifacts, FKey, apply_filters, cube_agg, default_filters, fkey, housing_mask,  # noqa: F401
                           scope_text, zip_borough, zip_year)
from analysis.rq_event import rq3_fig, rq3_stat, rq4_fig, rq4_stat, rq5_fig, rq5_stat
from analysis.rq_expo import rq6_fig, rq6_stat, rq7_fig, rq7_stat, rq10_fig, rq10_stat
from analysis.rq_zip import rq1_fig, rq1_stat, rq2_fig, rq2_stat, rq8_fig, rq8_stat, rq9_fig, rq9_stat

END = " This is an association, not proof of cause."
MEMBERS = {"M1": "ZIP-level relationships", "M2": "Before/after at housing sites", "M3": "Housing type & vulnerable road users",
           "M4": "Maps & borough patterns", "M5": "Income mix, severity & causes"}


def _sign(lo: float, hi: float) -> int:
    """+1 / -1 when the 95% CI excludes zero, else 0."""
    return 1 if lo > 0 else -1 if hi < 0 else 0


def _pick(sign: int, up: str, down: str, flat: str) -> str:
    return {1: up, -1: down, 0: flat}[sign]


def _ci(lo: float, hi: float, d: int = 2) -> str:
    return f"95% CI {lo:+.{d}f} to {hi:+.{d}f}"


# Headline generators: one per RQ, numbers in -> hedged one-sentence text out.
def h1(s: dict) -> str:
    return (f"Across {s['n']} ZIPs, " + _pick(_sign(s["ci_lo"], s["ci_hi"]), "ZIPs with more completed affordable units tended to show a larger crash growth ratio",
            "ZIPs with more completed affordable units tended to show a smaller crash growth ratio", "completed units and crash growth ratio showed no clear monotone link")
            + f" (Spearman ρ = {s['rho']:+.2f}, {_ci(s['ci_lo'], s['ci_hi'])}).")


def h2(s: dict) -> str:
    return ("Within the same ZIP, " + _pick(_sign(s["ci_lo"], s["ci_hi"]), "years with more completions had higher crash counts than that ZIP and year normally show",
            "years with more completions had lower crash counts than that ZIP and year normally show", "years with more completions showed no clear difference from that ZIP's and year's normal crash count")
            + f" ({s['pct_per_100_units']:+.1f}% per +100 units; {s['n_zips']} ZIPs).")


def h3(s: dict) -> str:
    if s.get("did_unavailable"):
        return (f"The difference-in-differences estimate is unavailable: {s['reason']}. Looking only at the {s['n_treated']} completed sites, adjusted crashes within 250 m "
                f"went from {s['treated_pre']:.2f} before to {s['treated_post']:.2f} after completion per building-year "
                f"(change {s['treated_change']:+.2f}, {_ci(s['t_ci_lo'], s['t_ci_hi'])}), with no comparison group to separate this from citywide trends.")
    text = (_pick(_sign(s["ci_lo"], s["ci_hi"]), "Adjusted crashes within 250 m rose more (or fell less) around newly completed sites than around comparison sites",
                  "Adjusted crashes within 250 m fell more (or rose less) around newly completed sites than around comparison sites", "Adjusted crashes within 250 m changed no differently around newly completed sites than around comparison sites")
            + f" (DiD {s['did']:+.2f} per building-year, {_ci(s['ci_lo'], s['ci_hi'])}; {s['n_treated']} treated sites).")
    if s.get("cohorts_without_control"):
        text += f" Cohorts {', '.join(map(str, s['cohorts_without_control']))} are left out because they have too few comparison sites."
    return text


def h4(s: dict) -> str:
    ok = [size_bin for size_bin in s["bins"] if "did" in size_bin]
    pos, neg = sum(_sign(r["ci_lo"], r["ci_hi"]) == 1 for r in ok), sum(_sign(r["ci_lo"], r["ci_hi"]) == -1 for r in ok)
    return (f"In {pos + neg} of {len(ok)} size bins the post-minus-pre change differed clearly from comparison sites "
            f"({pos} higher, {neg} lower); across sites, building size and change had Spearman ρ = {s['trend_rho']:+.2f}.")


def h5(s: dict) -> str:
    ok = {row["level"]: row for row in s["types"] if "did" in row}
    parts = [f"{k}: DiD {v['did']:+.2f} ({_ci(v['ci_lo'], v['ci_hi'])}, {v['n']} sites)" for k, v in ok.items()]
    return "Changes relative to comparison sites by construction type were " + "; ".join(parts) + " (Preservation adds no new structure, so it acts as a quasi-placebo)."


def h6(s: dict) -> str:
    return (_pick(_sign(s["ci_lo"], s["ci_hi"]), "A higher share of crashes harmed pedestrians or cyclists near completed housing",
                  "A lower share of crashes harmed pedestrians or cyclists near completed housing", "The share of crashes harming pedestrians or cyclists was not clearly different near completed housing")
            + f" than away from any housing ({s['rd'] * 100:+.2f} percentage points, {_ci(s['ci_lo'] * 100, s['ci_hi'] * 100)}; n = {s['n']:,} crashes).")


def h7(s: dict) -> str:
    if not np.isfinite(s["rho"]):
        return (f"Across {s['n']:,} map cells there was not enough variation in housing units to relate them to crash counts; "
                "this map is context, not evidence of an effect.")
    return (f"Across {s['n']:,} map cells, crash counts and completed housing units were {_pick(_sign(s['rho'] - .05, s['rho'] + .05), 'positively', 'negatively', 'only weakly')} "
            f"related (ρ = {s['rho']:+.2f}); both concentrate in dense areas, so this map is context, not evidence of an effect.")


def h8(s: dict) -> str:
    r = f"was ρ = {s['rho']:+.2f}" if np.isfinite(s["rho"]) else "could not be computed (not enough variation)"
    return (f"Most units were completed in {s['top_borough']} in {s['top_year']} ({s['top_units']:,}); across {s['n']} borough-years the descriptive rank "
            f"correlation between units and year-over-year crash change {r}.")


def h9(s: dict) -> str:
    return (f"Across {s['n']} ZIPs, " + _pick(_sign(s["ci_lo"], s["ci_hi"]), "a larger low-income share of completions went with more injuries per crash",
            "a larger low-income share of completions went with fewer injuries per crash", "the low-income share of completions showed no clear link with injuries per crash")
            + f" (ρ = {s['rho']:+.2f}, {_ci(s['ci_lo'], s['ci_hi'])}).")


def h10(s: dict) -> str:
    ci = f"95% CI {s['top_lo']:.2f} to {s['top_hi']:.2f}"
    if not s["n_significant"]:
        return (f"No clear difference by contributing factor near completed housing ({s['n']:,} crashes). The largest point estimate was {s['top_factor']} "
                f"at {s['top_lift']:.2f}x its usual share ({ci}), which cannot be told apart from 1.")
    return (f"{s['n_significant']} factor group(s) differed clearly in share near completed housing; the largest clear gap was {s['top_factor']} at "
            f"{s['top_lift']:.2f}x its usual share ({ci}).")


def _rq(i: int, member: str, title: str, question: str, ignores: list[str], head, fig, stat) -> dict:
    return dict(id=i, key=f"RQ{i}", member=member, member_title=MEMBERS[member], title=title, question=question, ignores=ignores,
                headline_fn=head, chart=fig.__name__, fig=fig, stat=stat)


L = FILTER_LABELS
_EV = [L["factor"], L["severity"], L["hours"], L["ped_cyc"]]
RQS: list[dict] = [
    _rq(1, "M1", "Completions vs crash growth", "Do ZIPs where more affordable units were completed show faster crash growth (last 3 vs first 3 years in range)?", [], h1, rq1_fig, rq1_stat),
    _rq(2, "M1", "Within-ZIP year-to-year co-movement", "Within a ZIP, do years with more completions coincide with higher or lower crash counts than that ZIP and year normally have?", [], h2, rq2_fig, rq2_stat),
    _rq(3, "M2", "Event study: sites vs comparison", "Do crashes within 250 m change differently after a completion than around buildings that are completed only later?", _EV, h3, rq3_fig, rq3_stat),
    _rq(4, "M2", "Dose-response by building size", "Is the post-minus-pre change relative to comparison sites larger for bigger buildings?", _EV, h4, rq4_fig, rq4_stat),
    _rq(5, "M3", "New construction vs preservation", "Does the change differ between new construction and preservation (which adds no new structure)?", _EV, h5, rq5_fig, rq5_stat),
    _rq(6, "M3", "Pedestrian/cyclist share of crashes", "Are crashes within 250 m of completed housing more likely to harm pedestrians or cyclists than crashes away from housing?", [L["construction"], L["ped_cyc"]], h6, rq6_fig, rq6_stat),
    _rq(7, "M4", "Relationship map", "Where do completed housing and crash concentrations overlap?", [L["factor"], L["hours"]], h7, rq7_fig, rq7_stat),
    _rq(8, "M4", "Borough-year completions and crash change", "How do completions per borough and year line up with year-over-year changes in crashes?", [], h8, rq8_fig, rq8_stat),
    _rq(9, "M5", "Low-income share vs severity", "Do ZIPs with a larger extremely/very-low-income share of completions have more injuries per crash?", [L["construction"]], h9, rq9_fig, rq9_stat),
    _rq(10, "M5", "Contributing-factor lift", "Which contributing factors are over- or under-represented in crashes near completed housing?", [L["construction"], L["factor"]], h10, rq10_fig, rq10_stat),
]
BY_KEY = {r["key"]: r for r in RQS}


def headline(rq: dict, stat: dict) -> str:
    """One hedged, associational sentence generated from a stat dict (never hard-coded findings)."""
    if stat.get("insufficient"):
        return f"Not enough data under these filters (n = {stat.get('n', 0)}; at least {config.MIN_N} units of analysis needed)." + END
    return rq["headline_fn"](stat) + END


def _clean(o):
    """Recursively convert numpy / NaN values into JSON-safe Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_all(art: Artifacts | str | Path, filters: dict | None = None, B: int = config.BOOT_PIPELINE) -> dict:
    """Compute every RQ's stat (default filters, pipeline bootstrap size) -> rq_results.json structure."""
    art = art if isinstance(art, Artifacts) else Artifacts.load(art)
    out = {}
    for rq in RQS:
        st = rq["stat"](art, filters, B)
        out[rq["key"]] = dict(id=rq["key"], member=rq["member"], title=rq["title"], question=rq["question"],
                              ignores=rq["ignores"], stat=_clean(st), headline=headline(rq, st), scope=scope_text(fkey(filters)))
    return out


def write_results(art: Artifacts | str | Path, B: int = config.BOOT_PIPELINE) -> Path:
    """Run all RQs and write rq_results.json next to the artifacts (called by the pipeline)."""
    art = art if isinstance(art, Artifacts) else Artifacts.load(art)
    path = art.dir / "rq_results.json"
    path.write_text(json.dumps(run_all(art, None, B), indent=1))
    art.__dict__.pop("rq_results", None)
    return path


def kpis(art: Artifacts, filters: dict | None) -> dict:
    """Headline numbers under the filters: crashes, injured, killed, ped+cyc injured, buildings, units, ZIPs."""
    fk = fkey(filters)
    t = cube_agg(art, fk, ("year",))[["crashes", "injured", "killed", "ped_injured", "cyc_injured"]].sum()
    h = art.housing[housing_mask(art.housing, fk)]
    return dict(crashes=int(t.crashes), injured=int(t.injured), killed=int(t.killed), ped_cyc_injured=int(t.ped_injured + t.cyc_injured),
                buildings=len(h), units=int(h["total_units"].sum()), zips=int(zip_year(art, fk)[0].shape[0]))


def zip_summary(art: Artifacts, filters: dict | None, top: int = 15) -> pd.DataFrame:
    """Top-`top` ZIPs by units completed under filters, with crash and injury context."""
    fk = fkey(filters)
    U, C, I = zip_year(art, fk)
    first, last = C.iloc[:, :3].sum(axis=1), C.iloc[:, -3:].sum(axis=1)
    df = pd.DataFrame({"ZIP": U.index, "Borough": zip_borough(art).reindex(U.index).values, "Units completed": U.sum(axis=1).astype(int).values,
                       "Crashes": C.sum(axis=1).astype(int).values, "Injured per 100 crashes": (100 * I.sum(axis=1) / C.sum(axis=1).replace(0, np.nan)).round(1).values,
                       "Growth ratio": ((last + 1) / (first + 1)).round(2).where(first >= 30).values if C.shape[1] >= 6 else np.nan})
    return df.sort_values(["Units completed", "Crashes"], ascending=False).head(top).reset_index(drop=True)
