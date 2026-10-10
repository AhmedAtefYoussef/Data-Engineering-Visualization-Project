# Legacy code

This folder holds the first version of the project: a file-based pipeline (`pipeline/`), an analysis
package (`analysis/`), their tests and the old notebook generator. It was only ever run on synthetic
fixtures and is kept for reference. Nothing in the current notebook, website or tests imports it.

The current project is the self-contained notebook in `notebooks/` (built from `tools/notebook_src/`)
and the website in `app/`.

`legacy/docs/` holds the documents that described that version: `SPEC.md`, `DESIGN.md` and `PRODUCT.md`.
They refer to `pipeline/`, `config.py` and a tabbed site that no longer exist in the current project.
