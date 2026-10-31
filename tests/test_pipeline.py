"""The data pipeline's output is a checked contract: the schema is verified at start-up, the export
is atomic, and meta.json says what the data was built from."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from app import data

ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "tools" / "build_app_data.py"
REBUILD = "python tools/build_app_data.py"


def _load_build_module():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("build_app_data_under_test", BUILD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


@pytest.fixture
def data_copy(tmp_path):
    """A private copy of app/data that a test may damage."""
    target = tmp_path / "data"
    shutil.copytree(data.DATA_DIR, target)
    yield target
    data.load.__wrapped__(data.DATA_DIR)          # load() sets REL_YEARS and the filter window: put them back


def _load(folder: Path):
    return data.load.__wrapped__(folder)           # the undecorated function: no cache between cases


def _edit_table(folder: Path, name: str, edit) -> None:
    path = folder / f"{name}.parquet"
    frame = pd.read_parquet(path)
    edit(frame).to_parquet(path, index=False)


def _digests(folder: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(folder.iterdir())}


# --------------------------------------------------------------------------------------------
# The schema is checked when the site starts
# --------------------------------------------------------------------------------------------
def test_committed_data_loads():
    assert len(_load(data.DATA_DIR).sites) > 0


@pytest.mark.parametrize("table,column", [("sites", "borough_percentile"), ("crash_cube", "n_pedcyc"),
                                          ("event_sites", "injury_-3"), ("zip_year", "units_completed")])
def test_removed_column_fails_load(data_copy, table, column):
    _edit_table(data_copy, table, lambda frame: frame.drop(columns=column))
    with pytest.raises(data.DataContractError) as error:
        _load(data_copy)
    message = str(error.value)
    assert f"'{table}'" in message and f"'{column}'" in message and REBUILD in message


def test_renamed_column_fails_load(data_copy):
    _edit_table(data_copy, "crash_cube", lambda frame: frame.rename(columns={"n_pedcyc": "n_pedcycle"}))
    with pytest.raises(data.DataContractError, match="n_pedcyc"):
        _load(data_copy)


def test_unexpected_column_fails_load(data_copy):
    # An extra n_* column would silently become a count, and any other column a dimension.
    _edit_table(data_copy, "hour_cube", lambda frame: frame.assign(n_extra=1))
    with pytest.raises(data.DataContractError, match="n_extra"):
        _load(data_copy)


def test_wrong_column_type_fails_load(data_copy):
    _edit_table(data_copy, "sites", lambda frame: frame.assign(lat=frame["lat"].astype(str)))
    with pytest.raises(data.DataContractError, match="'sites', column 'lat'"):
        _load(data_copy)


def test_table_that_disagrees_with_meta_fails_load(data_copy):
    # Same kind (unsigned integer) but a different width from the one meta.json recorded.
    _edit_table(data_copy, "crash_cube", lambda frame: frame.assign(month=frame["month"].astype("uint32")))
    with pytest.raises(data.DataContractError, match="different builds"):
        _load(data_copy)


def test_event_window_in_meta_drives_the_event_columns(data_copy):
    meta = json.loads((data_copy / "meta.json").read_text(encoding="utf-8"))
    meta["constants"]["event_window"] = 2
    (data_copy / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(data.DataContractError, match="event_sites"):
        _load(data_copy)


def test_rel_years_follow_the_event_window():
    meta = json.loads((data.DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    window = meta["constants"]["event_window"]
    _load(data.DATA_DIR)
    assert data.REL_YEARS == list(range(-window, window + 1))


def test_expected_schema_covers_every_exported_table():
    assert set(data.EXPECTED_SCHEMA) == set(build.OUTPUT_TABLES)
    meta = json.loads((data.DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["schema"]) == set(build.OUTPUT_TABLES)
    window = meta["constants"]["event_window"]
    for name, recorded in meta["schema"].items():
        assert set(data.expected_schema(name, window)) == set(recorded), name


# --------------------------------------------------------------------------------------------
# The export is atomic
# --------------------------------------------------------------------------------------------
def _old_output(folder: Path) -> Path:
    folder.mkdir()
    (folder / "meta.json").write_text('{"old": true}', encoding="utf-8")
    (folder / "sites.parquet").write_bytes(b"old parquet bytes")
    return folder


def _no_staging_left(parent: Path) -> bool:
    return [path for path in parent.iterdir() if path.name.startswith(".")] == []


def test_failed_write_leaves_old_output_untouched(tmp_path, monkeypatch):
    out = _old_output(tmp_path / "out")
    before = _digests(out)

    def broken(ns, folder, seconds, previous):
        (folder / "sites.parquet").write_bytes(b"half written")
        raise build.DataCheckError("conservation failed")

    monkeypatch.setattr(build, "write_all", broken)
    with pytest.raises(build.DataCheckError, match="conservation"):
        build.export({}, out, 1.0)
    assert _digests(out) == before
    assert _no_staging_left(tmp_path)


def test_failed_website_check_leaves_old_output_untouched(tmp_path, monkeypatch):
    out = _old_output(tmp_path / "out")
    before = _digests(out)

    def fine(ns, folder, seconds, previous):
        (folder / "meta.json").write_text('{"new": true}', encoding="utf-8")

    def failing_check(ns, folder):
        raise build.DataCheckError("website number differs from the notebook")

    monkeypatch.setattr(build, "write_all", fine)
    monkeypatch.setattr(build, "check_against_notebook", failing_check)
    with pytest.raises(build.DataCheckError, match="website number"):
        build.export({}, out, 1.0)
    assert _digests(out) == before
    assert _no_staging_left(tmp_path)


def test_successful_export_replaces_the_old_output(tmp_path, monkeypatch):
    out = _old_output(tmp_path / "out")

    def fine(ns, folder, seconds, previous):
        assert previous == {"old": True}              # the old meta.json is read before the swap
        (folder / "meta.json").write_text('{"new": true}', encoding="utf-8")

    monkeypatch.setattr(build, "write_all", fine)
    monkeypatch.setattr(build, "check_against_notebook", lambda ns, folder: None)
    build.export({}, out, 1.0)
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == {"new": True}
    assert not (out / "sites.parquet").exists()
    assert _no_staging_left(tmp_path)


def test_output_folder_with_foreign_files_is_refused(tmp_path, monkeypatch):
    out = _old_output(tmp_path / "out")
    (out / "vercel.json").write_text("{}", encoding="utf-8")
    before = _digests(out)
    monkeypatch.setattr(build, "write_all", lambda *args: pytest.fail("must refuse before writing"))
    with pytest.raises(build.DataCheckError, match="vercel.json"):
        build.export({}, out, 1.0)
    assert _digests(out) == before


def test_relative_out_is_resolved_before_comparing(tmp_path, monkeypatch):
    out = _old_output(tmp_path / "out")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build, "write_all", lambda ns, folder, seconds, previous: (folder / "meta.json").write_text("{}"))
    monkeypatch.setattr(build, "check_against_notebook", lambda ns, folder: None)
    build.export({}, Path("out"), 1.0)                 # a relative path reaches the same folder
    assert (out / "meta.json").read_text() == "{}"


def test_checks_are_not_assert_statements():
    """`python -O` removes assert, so no data check may rely on it."""
    tree = ast.parse(BUILD_PATH.read_text(encoding="utf-8"))
    assert [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)] == []
    with pytest.raises(build.DataCheckError):
        build.require(False, "a failed check raises")


# --------------------------------------------------------------------------------------------
# Provenance comes from content
# --------------------------------------------------------------------------------------------
def test_meta_records_what_the_data_was_built_from():
    meta = json.loads((data.DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    for key in ["schema", "raw_sha256", "source_sha256", "packages"]:
        assert meta.get(key), f"meta.json has no {key}; {REBUILD}"
    assert all(len(digest) == 64 for digest in meta["raw_sha256"].values())
    assert {"pandas", "numpy", "pyarrow"} <= set(meta["packages"])
    assert meta["data_as_of"]["latest_crash"] == "2025-12-31"


def test_sources_unchanged_since_the_last_build():
    meta = json.loads((data.DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    assert build.source_hash() == meta["source_sha256"], (
        "tools/notebook_src/* or app/stats.py changed since app/data was built, so the exported numbers may be "
        f"stale. Rebuild with: {REBUILD}")


def test_raw_files_unchanged_since_the_last_build():
    meta = json.loads((data.DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    for name, recorded in meta["raw_sha256"].items():
        assert build.file_sha256(ROOT / "notebooks" / name) == recorded, (
            f"notebooks/{name} differs from the file app/data was built from. Rebuild with: {REBUILD}")


def _tree(root: Path, text: str) -> None:
    (root / "tools" / "notebook_src").mkdir(parents=True)
    (root / "app").mkdir()
    (root / "tools" / "notebook_src" / "01_a.py").write_bytes(text.encode())
    (root / "app" / "stats.py").write_bytes(b"x = 1\n")


def test_source_hash_follows_content_not_line_endings_or_dates(tmp_path):
    _tree(tmp_path / "lf", "a = 1\nb = 2\n")
    _tree(tmp_path / "crlf", "a = 1\r\nb = 2\r\n")
    _tree(tmp_path / "changed", "a = 1\nb = 3\n")
    assert build.source_hash(tmp_path / "lf") == build.source_hash(tmp_path / "crlf")
    assert build.source_hash(tmp_path / "lf") != build.source_hash(tmp_path / "changed")
    (tmp_path / "lf" / "app" / "stats.py").write_bytes(b"x = 2\n")
    assert build.source_hash(tmp_path / "lf") != build.source_hash(tmp_path / "crlf")
