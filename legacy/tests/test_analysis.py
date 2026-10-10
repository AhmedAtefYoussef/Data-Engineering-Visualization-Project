"""Analysis tests: stats helpers, every RQ fig/stat on the artifacts under test, empty-filter safety, run_all contract."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tests._art import get_art  # noqa: F401  (sets NYC_PROCESSED_DIR before config import)
import config
from analysis import stats
from analysis.rq import RQS, Artifacts, apply_filters, headline, kpis, run_all, write_results, zip_summary

FILTERS = [
    {},
    dict(boroughs=["Brooklyn", "Queens"], years=(2018, 2023)),
    dict(construction=["New Construction"], severity=["Injury"], factor=["Speed/Aggressive"], hours=["Night"]),
    dict(boroughs=["Staten Island"], years=(2016, 2016), severity=["Fatal"], ped_cyc=True),     # sparse -> insufficient
    dict(boroughs=["Manhattan"], years=(2024, 2025), construction=["Other"], factor=["Vehicle/Road Defect"]),
]


@pytest.fixture(scope="module")
def art() -> Artifacts:
    return get_art()


def test_spearman_boot_known():
    x = np.arange(50.0)
    r = stats.spearman_boot(x, x ** 2, 100, 1)
    assert r["rho"] == pytest.approx(1.0) and r["ci_lo"] > 0.99


def test_within_slope_recovers_effect():
    rng = np.random.default_rng(0)
    X = rng.poisson(1, (40, 8)).astype(float)
    Y = 0.3 * X + rng.normal(0, .1, X.shape) + rng.normal(size=(40, 1)) + rng.normal(size=(1, 8))
    r = stats.within_slope(Y, X, 100, 0)
    assert 0.25 < r["slope"] < 0.35 and r["ci_lo"] < r["slope"] < r["ci_hi"]


def test_did_boot_known_effect():
    rng = np.random.default_rng(1)
    dt, dc = rng.normal(1.0, .3, (60, 1)), rng.normal(0.0, .3, (80, 1))
    r = stats.did_boot(dt, np.zeros(60, int), dc, np.zeros(80, int), 1, 200, 0)
    assert r["diff"]["lo"][0] > 0.7 and abs(r["diff"]["est"][0] - 1.0) < .15


def test_risk_diff_cramers_loglift():
    rd = stats.risk_diff(30, 100, 20, 100)
    assert rd["rd"] == pytest.approx(.1) and rd["ci_lo"] < .1 < rd["ci_hi"]
    assert stats.cramers_v(np.array([[30, 70], [20, 80]], float)) == pytest.approx(.1, abs=.02)
    ll = stats.log_lift(np.array([20.0]), 100, np.array([10.0]), 100)
    assert ll["lift"][0] == pytest.approx(2) and ll["lo"][0] < 2 < ll["hi"][0]


def test_registry_shape():
    assert [r["id"] for r in RQS] == list(range(1, 11))
    assert {r["member"] for r in RQS} == {f"M{i}" for i in range(1, 6)}
    assert all(callable(r["fig"]) and callable(r["stat"]) and r["question"] for r in RQS)


@pytest.mark.parametrize("filters", FILTERS)
def test_every_rq_fig_and_stat(art, filters):
    for rq in RQS:
        fig, st = rq["fig"](art, filters, 30), rq["stat"](art, filters, 30)
        assert isinstance(fig, go.Figure) and isinstance(st, dict) and "n" in st
        fig.to_plotly_json()
        assert headline(rq, st).endswith("This is an association, not proof of cause.")
        if st.get("insufficient"):
            assert fig.layout.annotations and "Insufficient" in fig.layout.annotations[0].text


def test_empty_filter_safety(art):
    f = dict(boroughs=[], years=(2030, 2031))      # no overlap with any year
    for rq in RQS:
        assert isinstance(rq["fig"](art, f, 20), go.Figure)
    k = kpis(art, f)
    assert k["crashes"] == 0 and len(zip_summary(art, f)) == 0


def test_apply_filters_and_kpis(art):
    v = apply_filters(art, dict(boroughs=["Bronx"], years=(2017, 2019)))
    assert set(v["cube"]["year"].unique()) <= {2017, 2018, 2019}
    assert kpis(art, dict(boroughs=["Bronx"], years=(2017, 2019)))["crashes"] == int(v["cube"]["crashes"].sum())
    assert kpis(art, None)["crashes"] >= kpis(art, dict(boroughs=["Bronx"]))["crashes"]


def test_cube_memory_dtypes(art):
    c = art.cube
    assert str(c["zip"].dtype) == "category" and c["year"].dtype == np.int16 and c["crashes"].dtype == np.int32
    assert c.memory_usage(deep=True).sum() < 400e6


def test_run_all_and_write(art, tmp_path):
    res = run_all(art, B=30)
    assert list(res) == [f"RQ{i}" for i in range(1, 11)]
    for v in res.values():
        assert {"title", "question", "member", "stat", "headline"} <= set(v)
    json.dumps(res)                                  # JSON-safe
    # write into a scratch copy so the artifact dir is never modified by tests
    import shutil
    for p in art.dir.glob("*"):
        shutil.copy(p, tmp_path / p.name)
    out = write_results(tmp_path, B=30)
    assert out.exists() and json.loads(out.read_text())["RQ1"]["headline"]


def test_did_boot_clusters_by_site():
    """Rows of one site move together: 40 rows from 4 sites give a wider interval than 40 independent sites."""
    rng = np.random.default_rng(3)
    dt = rng.normal(1, .3, (40, 1))
    dc = np.repeat(rng.normal(0, 1, 4), 10)[:, None]
    z = lambda n: np.zeros(n, int)  # noqa: E731
    iid = stats.did_boot(dt, z(40), dc, z(40), 1, 300, 0)
    clu = stats.did_boot(dt, z(40), dc, z(40), 1, 300, 0, np.arange(40), np.repeat(np.arange(4), 10))
    w = lambda r: r["control"]["hi"][0] - r["control"]["lo"][0]  # noqa: E731
    assert w(clu) > 1.5 * w(iid)
    assert stats._clusters(np.array([5, 5, 2]), 3)[0].tolist() == [2]


def test_unknown_borough_has_own_stratum():
    from analysis import rq_event
    assert rq_event.stratum(np.array([2020, 2020]), pd.Series(["Bronx", "Unknown"])).tolist() == [0, 5]


def test_rq7_headline_without_variation():
    from analysis.rq import h7
    txt = h7(dict(n=30, rho=float("nan")))
    assert "nan" not in txt and "not enough variation" in txt and " - " not in txt


def test_rq7_housing_cells_match_crash_grid(art):
    """Housing and crash grids use the same floor-centre cells, so some cells carry housing and rho is defined."""
    from analysis.data import default_filters
    from analysis.rq_expo import _centre
    s = art.__class__ and __import__("analysis.rq_expo", fromlist=["x"]).rq7_stat(art, default_filters())
    assert s["share_cells_with_housing"] > 0 and np.isfinite(s["rho"])
    gd = config.GRID_DEG
    assert _centre(pd.Series([40.7012]), gd).iloc[0] == pytest.approx(40.7025)


def test_log_ticks_clean():
    from app.charts import log_ticks
    assert log_ticks(1200, 6000) == [2000.0, 5000.0]
    assert log_ticks(0.4, 25) == [1.0, 10.0]
    assert len(log_ticks(0.94, 1.14)) >= 3


def test_clean_log_axes_skips_categorical_axes():
    from app.charts import clean_log_axes
    fig = go.Figure(go.Bar(x=["1-10", "11-50"], y=[3, 30]))
    fig.update_yaxes(type="log")
    out = clean_log_axes(fig)                       # a text x axis must not raise
    assert list(out.layout.yaxis.tickvals) == [10.0] or len(out.layout.yaxis.tickvals) >= 1
    assert out.layout.xaxis.tickvals is None


# ------------------------------------------------------------------ audit regressions
def test_rq10_headline_uses_significant_factor():
    from analysis.rq import h10
    base = dict(n=5000, top_factor="Speed", top_lift=1.8, top_lo=0.7, top_hi=3.0)
    none = h10({**base, "n_significant": 0, "top_is_significant": False})
    assert none.startswith("No clear difference by contributing factor") and "cannot be told apart from 1" in none and "Speed" in none
    some = h10({**base, "n_significant": 2, "top_factor": "Distraction", "top_lift": 1.4, "top_lo": 1.1, "top_hi": 1.8, "top_is_significant": True})
    assert "2 factor group(s)" in some and "Distraction" in some and "largest clear gap" in some


def test_rq10_top_factor_comes_from_significant_set(art):
    from analysis.rq_expo import rq10_stat
    s = rq10_stat(art, {}, 50)
    if s.get("insufficient"):
        pytest.skip("not enough data")
    clear = [f for f in s["factors"] if f["lo"] > 1 or f["hi"] < 1]
    assert s["n_significant"] == len(clear)
    top = next(f for f in s["factors"] if f["factor"] == s["top_factor"])
    if clear:
        assert top in clear and s["top_is_significant"]
        assert abs(np.log(top["lift"])) == max(abs(np.log(f["lift"])) for f in clear)
    else:
        assert not s["top_is_significant"]


class _EventArt:
    """Minimal stand-in for Artifacts holding a hand-built event_building table (hashable for the lru_cache)."""
    def __init__(self, event: pd.DataFrame):
        self.event = event


def _event_frame(treated_by_cohort: dict, control_by_cohort: dict, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, site = [], 0
    for group, spec in (("treated", treated_by_cohort), ("control", control_by_cohort)):
        for cohort, n_sites in spec.items():
            for _ in range(n_sites):
                site += 1
                for rel in range(-3, 4):
                    rows.append(dict(building_id=site, site_id=site, group=group, cohort=cohort, contaminated=False, comp_year=cohort,
                                     rel_year=rel, construction_type="New Construction", borough="Bronx", size_bin="11-50", total_units=30,
                                     crashes=1, adj=float(rng.normal(2.0 + (0.5 if group == "treated" and rel > 0 else 0), .2)),
                                     ped_cyc_crashes=0, injured=0))
    return pd.DataFrame(rows)


def test_event_study_without_enough_controls_is_treated_only():
    from analysis.rq import BY_KEY
    from analysis.rq_event import rq3_fig, rq3_stat
    art_ = _EventArt(_event_frame({2021: 12, 2022: 10}, {2021: 3, 2022: 2}))
    s = rq3_stat(art_, {}, 50)
    assert s["did_unavailable"] and "did" not in s and "ci_lo" not in s
    assert s["n_treated"] == 22 and s["treated_post"] > s["treated_pre"] and s["t_ci_lo"] < s["treated_change"] < s["t_ci_hi"]
    text = headline(BY_KEY["RQ3"], s)
    assert "unavailable" in text and "no comparison group" in text and "DiD +" not in text
    fig = rq3_fig(art_, {}, 50)
    assert isinstance(fig, go.Figure) and "DiD unavailable" in fig.layout.annotations[-1].text
    fig.to_plotly_json()


def test_event_study_drops_only_cohorts_lacking_controls():
    from analysis.rq import BY_KEY
    from analysis.rq_event import rq3_stat
    art_ = _EventArt(_event_frame({2019: 12, 2021: 12}, {2019: 15, 2021: 2}, seed=1))
    s = rq3_stat(art_, {}, 50)
    assert not s.get("did_unavailable") and s["cohorts_without_control"] == [2021] and s["n_treated"] == 12
    assert "2021" in headline(BY_KEY["RQ3"], s) and np.isfinite(s["did"]) and s["ci_lo"] <= s["did"] <= s["ci_hi"]


def test_event_study_rq4_rq5_survive_no_controls():
    from analysis.rq_event import rq4_fig, rq4_stat, rq5_fig, rq5_stat
    art_ = _EventArt(_event_frame({2022: 12}, {2022: 1}))
    assert rq4_stat(art_, {}, 50)["insufficient"] and rq5_stat(art_, {}, 50)["insufficient"]
    assert isinstance(rq4_fig(art_, {}, 50), go.Figure) and isinstance(rq5_fig(art_, {}, 50), go.Figure)
