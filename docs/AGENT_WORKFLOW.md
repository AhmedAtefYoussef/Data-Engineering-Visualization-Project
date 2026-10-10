# How the project was built with AI agents

The assignment asks how AI was used, so this file records the process. The prompts are in `docs/PROMPTS.md`.

## Two versions

The first version was a file-based pipeline (`pipeline/`), an analysis package (`analysis/`), a Dash site and a notebook generator. An orchestrating session wrote a specification, an adversarial reviewer attacked it, implementers wrote the code against it, and an auditor checked the result against the project description. The sandbox it ran in could not reach NYC Open Data, so it was tested only on synthetic data. That version was removed from the repository once the rebuild replaced it. It remains in the git history, in the first commit.

The second version, which is the current one, was built after the datasets could be downloaded. It was designed around one rule: the notebook is the source of truth, and everything else is derived from the same code.

## How the second version was built

1. One assistant per notebook section. The notebook was split into section files in `tools/notebook_src/` (title, setup, data, exploration, cleaning, integration, post-integration cleaning, toolkit, research questions 1 to 5, research questions 6 to 10, robustness, findings and reproducibility). Each file had one author, who could edit only that file. All authors received the same brief (quoted in part in `docs/PROMPTS.md`), which fixed the datasets, the outcome, the two integration levels, the coding rules and the writing rules.
2. Run on the real data. Each author executed their section, together with the sections it depends on, through `tools/assemble_notebook.py` and looked at the figures before handing it in. Every author also reported the objects and numbers that later sections could rely on, and the points where they had to deviate from the brief. The authors of later sections used those reports.
3. Review from several points of view. Further assistants read the executed notebook, each for one concern: the requirements of the project description, the code, the statistics, the prose and the charts. They returned findings with a location, a severity, the problem and a fix. They did not edit the notebook.
4. Fix passes. The section authors applied the findings. In the largest pass, 54 of 57 findings on one group of sections were applied. Some changed results: the RQ7 headline was restricted to new construction, two comparisons inside cohort and borough and on reported ZIP codes only were added to the robustness section, and the RQ8 and RQ10 intervals were widened for the number of tests. Findings that belonged to another author were passed on to that author.
5. The website. A separate assistant rebuilt the site so that it reads only `app/data/`, which `tools/build_app_data.py` produces by running the notebook sections. The script asserts that the site's code reproduces the notebook's RQ3 and RQ7 estimates. A second assistant did a design and motion pass over the style files without touching the layout or the callbacks. The assistant that rebuilt the site also wrote the tests in `tests/test_site.py`.
6. Documents. The README and the files in `docs/` were brought in line with the final state, with every figure checked against `app/data/` or the section sources. The documents of the first version that no longer matched (`SPEC.md`, `DESIGN.md`, `PRODUCT.md`) were removed with the first version.

## Model choice

In the first version, Opus 5.5 reviewed the specification and audited the result against the requirements, and Sonnet 5.5 wrote the pipeline, the site, the notebook builder and the documents (see the table in `docs/PROMPTS.md`). A later prompt asked for the strongest model on complex tasks and a cheaper one for syntax and routine code, to cut token use. The records kept in the repository do not show which model handled each task of the second version, so this file does not claim a per-task assignment.

## Prompt patterns

- A shared brief for parallel authors, with rules that can be checked: no data-derived number typed into text, an assertion for every invariant, one figure per cell, no claim of cause.
- Ownership boundaries: each assistant edits only its own file, so parallel work does not collide and responsibility is traceable.
- Reviewers who read but cannot edit what they judge, and a fixed shape for their findings.
- Execution before acceptance: nothing counts as finished until it has run on the real data and its figures have been looked at.
- Reports from authors that state the interface they provide and anything left unresolved.

## What this process does not cover

- The deployed site. It has not been deployed, and gunicorn has not been run on Windows. Memory on Render's free plan has not been measured.
- The stored outputs of the committed notebook are a snapshot. Rebuild with `python tools/assemble_notebook.py --execute` after any change to a section file, and rebuild the website data with `python tools/build_app_data.py`.
- Whether each student has read and can explain their parts. The code and text were drafted by AI assistants under the team's direction. The checks each student should do are in `docs/CONTRIBUTIONS.md`.
- Names and student IDs of members 2 to 5, which are not in the repository.
