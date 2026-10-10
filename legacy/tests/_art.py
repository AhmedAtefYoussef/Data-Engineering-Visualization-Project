"""Helper for my tests: pick the artifact dir (mock by default, real fixture output when NYC_FIXTURE=1)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not os.environ.get("NYC_FIXTURE") and not os.environ.get("NYC_PROCESSED_DIR"):
    os.environ["NYC_PROCESSED_DIR"] = str(ROOT / "data" / "mock_processed")


def get_art():
    """Open (building the mock set if needed) the artifacts under test."""
    import config
    from analysis.rq import Artifacts
    from tests.make_mock_artifacts import make
    d = Path(config.PROCESSED_DIR)
    if not (d / "meta.json").exists() and not os.environ.get("NYC_FIXTURE"):
        make(d)
    return Artifacts.load(d)
