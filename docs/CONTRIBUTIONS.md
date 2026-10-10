# Contributions

The team has five members, so the project has ten research questions and ten charts, two questions per member. The same table appears in Section 14 of the notebook.

The split below is the planned one. The team has not confirmed it, and it is not a record of who did what. Member 1 is Ahmed Atef Youssef. The names and student IDs of members 2 to 5 are not in the repository and are marked TO FILL IN. Replace the split with what each person actually did before submission, because the assignment asks for an honest per-member record.

## Research questions

| Member | Name | Student ID | Research questions |
|---|---|---|---|
| M1 | Ahmed Atef Youssef | TO FILL IN | RQ1 forecast of a ZIP's injury crashes with housing added; RQ2 within-ZIP change as the unit stock grows |
| M2 | TO FILL IN | TO FILL IN | RQ3 injury crashes near new buildings after completion; RQ4 change by building size |
| M3 | TO FILL IN | TO FILL IN | RQ5 new construction against preservation; RQ6 pedestrian and cyclist share by borough |
| M4 | TO FILL IN | TO FILL IN | RQ7 where the sites stand before the project starts; RQ8 hour of day |
| M5 | TO FILL IN | TO FILL IN | RQ9 deep-affordable share across ZIP codes; RQ10 contributing factors |

## Other work

Section numbers refer to the notebook.

| Member | Data cleaning | Integration | Website | Other |
|---|---|---|---|---|
| M1 | Housing: blank unit counts, duplicates, geographic identifiers, dates, unit consistency, size outliers (4.1 to 4.6) | ZIP by year panel (5.2) | Dropdown filters and the Generate report button | Setup and data loading (Sections 1 and 2), housing exploration (3.1), README setup and run steps |
| M2 | Crashes: key, duplicates, dates, time of day, borough and ZIP (4.7, 4.8) | Distance join with the ball tree and its brute-force check, the 250 m buffer (5.3) | Event-study cards and the bootstrap callback | Crash exploration (3.2), bootstrap and difference-in-differences functions (Section 7) |
| M3 | Crashes: coordinates, casualty totals, extreme injury counts, contributing-factor groups (4.8 to 4.11) | Sites, clusters and the screen for nearby completions (5.4) | Search box and query parser | Robustness checks (Section 9), data dictionary |
| M4 | ZIP recovery from coordinates and its validation (6.1) | Tagging crashes with the nearest site (5.5), conservation and uniqueness checks (6.4) | Map and heat-map cards, page layout, accessibility | Deployment configuration, architecture notes |
| M5 | Post-integration cleaning: borough from two sources, redundant columns, data types (6.2, 6.3) | Cumulative unit stock and citywide index (6.2, 6.5), final table schema | Findings and data sections of the site | Findings, limits, impact, prompts and reproducibility (Sections 10 to 15), deployment on the hosting platform |

## To verify before submission

Each student should check their own row and record the result in their own words. Nothing in this repository records that any student has done these checks.

| Member | To verify |
|---|---|
| M1 | Run the notebook from the top and read Sections 2 and 4.1 to 4.6. Read the RQ1 and RQ2 code and be able to explain why RQ1's interval leaves out the uncertainty of fitting. Open the deployed site in a private window. |
| M2 | Read the ball tree code and its brute-force check, and the RQ3 code in Section 8.3 with the difference-in-differences function in Section 7. Check the event-study card on the site against the notebook. |
| M3 | Read the cluster and screen logic (5.4) and explain how the screen changes the sample. Try ten phrases in the search box and confirm that search only sets filters. Read Section 9. |
| M4 | Read the ZIP recovery and its holdout validation (6.1), and the RQ7 code in Section 8.7 and why its benchmark is street points and not other housing. Check the map and heat map at phone width. Confirm `render.yaml` and `requirements.txt`. |
| M5 | Run `python tools/build_app_data.py` and `python -m pytest -q tests`. Read the cleaning log and Sections 10 to 15 for numbers that do not match the executed outputs. Read `docs/AGENT_WORKFLOW.md` and `docs/PROMPTS.md` for accuracy. Deploy on Render and test the address. |

## How to record individual contributions

Each member keeps a short list of the commits or pull requests they wrote (with links), the research questions they own and can explain, and anything they ran or checked. Questions in the evaluation can target any part of the notebook, so every member should be able to explain the cleaning and integration decisions, including those they did not write.

## AI assistance

AI coding assistants drafted the code and text under the team's direction, as described in `docs/AGENT_WORKFLOW.md` and `docs/PROMPTS.md`. Members remain responsible for understanding, running and checking what they submit.
