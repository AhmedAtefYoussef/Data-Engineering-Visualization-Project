"""Assemble the notebook from the section sources in tools/notebook_src/ and optionally execute it.

Each section is a Python file in jupytext "percent" format (`# %%` code cells, `# %% [markdown]`
text cells). Sections are concatenated in file-name order into one self-contained notebook, so the
notebook itself never imports anything from this repository.

Examples
--------
    python tools/assemble_notebook.py --execute                      # full build, executed, to notebooks/
    python tools/assemble_notebook.py --sections 01 02 04 --execute \
        --out /tmp/run/dev.ipynb --show 04                           # partial dev run, print section 04 outputs
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "tools" / "notebook_src"
NOTEBOOK_DIR = ROOT / "notebooks"          # execution directory: the two data files live here
DEFAULT_OUT = NOTEBOOK_DIR / "NYC_Housing_Safety.ipynb"
MAX_TEXT = 1800                            # characters of text output printed per cell with --show


def section_files(prefixes: list[str] | None) -> list[Path]:
    """Return the section files in order, optionally restricted to the given name prefixes."""
    files = sorted(SRC_DIR.glob("*.py"))
    if prefixes:
        files = [f for f in files if any(f.name.startswith(p) for p in prefixes)]
    if not files:
        raise SystemExit(f"no section files matched {prefixes} in {SRC_DIR}")
    return files


def build(files: list[Path]) -> nbformat.NotebookNode:
    """Parse every section and concatenate the cells, remembering which section each came from."""
    cells = []
    for path in files:
        part = jupytext.reads(path.read_text(encoding="utf-8"), fmt="py:percent")
        for cell in part.cells:
            cell.metadata["section"] = path.stem
            cells.append(cell)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


def execute(notebook: nbformat.NotebookNode) -> bool:
    """Run all cells in notebooks/ (where the data files are). Return False if a cell raised."""
    client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
    try:
        client.execute()
        return True
    except CellExecutionError as error:
        failed = next((c for c in notebook.cells if c.cell_type == "code" and any(
            o.get("output_type") == "error" for o in c.get("outputs", []))), None)
        if failed is not None:
            print(f"\n=== FAILED CELL (section {failed.metadata.get('section')}) ===")
            print(failed.source[:1500])
        print("\n=== ERROR ===")
        print(str(error)[-3500:])
        return False


def show_outputs(notebook: nbformat.NotebookNode, prefixes: list[str], image_dir: Path) -> None:
    """Print the text outputs of the chosen sections and save their figures as PNG files."""
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(notebook.cells):
        section = cell.metadata.get("section", "")
        if cell.cell_type != "code" or not any(section.startswith(p) for p in prefixes):
            continue
        print(f"\n--- cell {index} [{section}] {cell.source.splitlines()[0][:90] if cell.source else ''}")
        for number, output in enumerate(cell.get("outputs", [])):
            kind = output.get("output_type")
            if kind == "stream":
                print(output.text[:MAX_TEXT])
            elif kind in ("display_data", "execute_result"):
                data = output.get("data", {})
                if "image/png" in data:
                    target = image_dir / f"cell{index:03d}_{number}.png"
                    target.write_bytes(base64.b64decode(data["image/png"]))
                    print(f"[figure saved: {target}]")
                elif "text/markdown" in data:
                    print(data["text/markdown"][:MAX_TEXT])
                elif "text/plain" in data:
                    print(data["text/plain"][:MAX_TEXT])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="*", help="file-name prefixes to include (default: all)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .ipynb path")
    parser.add_argument("--execute", action="store_true", help="run the notebook and store outputs")
    parser.add_argument("--show", nargs="*", default=[], help="print outputs of these section prefixes")
    args = parser.parse_args()

    notebook = build(section_files(args.sections))
    ok = True
    if args.execute:
        started = time.time()
        ok = execute(notebook)
        print(f"executed in {time.time() - started:.0f}s: {'OK' if ok else 'FAILED'}")
    if args.show:
        show_outputs(notebook, args.show, args.out.with_suffix("").parent / (args.out.stem + "_img"))
    for cell in notebook.cells:
        cell.metadata.pop("section", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, str(args.out))
    code_cells = sum(c.cell_type == "code" for c in notebook.cells)
    print(f"wrote {args.out} ({len(notebook.cells)} cells, {code_cells} code)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
