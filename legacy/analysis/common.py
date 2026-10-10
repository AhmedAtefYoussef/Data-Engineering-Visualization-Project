"""Shared plumbing for the RQ modules: insufficient-data handling and fig/stat factories."""
from __future__ import annotations

from typing import Callable

import plotly.graph_objects as go

import config
from analysis.data import Artifacts, FKey, fkey, scope_text
from app.charts import empty_fig

SEED = config.SEED


def fmt_rho(rho: float) -> str:
    """'ρ = 0.12' or a plain-words fallback when the correlation is undefined (no variation)."""
    return f"ρ = {rho:.2f}" if rho == rho else "ρ undefined (not enough variation)"


def insufficient(n: int, why: str = "") -> dict:
    """Stat dict flagging that fewer than MIN_N analysis units are available (F8)."""
    return dict(n=int(n), insufficient=True, reason=why or f"fewer than {config.MIN_N} units of analysis")


def is_insufficient(stat: dict) -> bool:
    """True when a stat dict is the insufficient-data placeholder."""
    return bool(stat.get("insufficient"))


def make_api(core: Callable, plot: Callable, title: str, name: str) -> tuple[Callable, Callable]:
    """Build the public (fig, stat) pair for one RQ from a cached core(art, fk, B) and a plot(data, title) function."""
    def stat(art: Artifacts, filters: dict | None = None, B: int = config.BOOT_APP) -> dict:
        return core(art, fkey(filters), B)["stat"]

    def fig(art: Artifacts, filters: dict | None = None, B: int = config.BOOT_APP) -> go.Figure:
        fk: FKey = fkey(filters)
        ttl = f"{title}<br><sup>{scope_text(fk)}</sup>"
        try:
            d = core(art, fk, B)
        except FileNotFoundError:
            return empty_fig(ttl, "Processed data not found. Run the pipeline first.")
        if is_insufficient(d["stat"]):
            return empty_fig(ttl, f"Insufficient data for these filters (n = {d['stat']['n']}, at least {config.MIN_N} needed). Widen or reset the filters.")
        return plot(d, ttl)
    stat.__doc__, fig.__doc__ = f"Statistics for {name}.", f"Plotly figure for {name}."
    stat.__name__, fig.__name__ = f"{name}_stat", f"{name}_fig"
    return fig, stat
