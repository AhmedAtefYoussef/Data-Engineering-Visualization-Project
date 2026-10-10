# Methodology

This file summarizes the methods of the notebook `notebooks/NYC_Housing_Safety.ipynb`. The notebook has the code, the reasoning for each choice and the executed outputs. The website in `app/` uses the same code through `tools/build_app_data.py`, so both show the same numbers. Where this file and the notebook differ, the notebook is right.

## 1. Question and design

The question is whether the streets around city-financed affordable housing have different police-reported crash records from comparable streets, before and after the buildings are completed. The design is observational. Nobody assigned buildings to streets at random, so every result is worded as an association.

The team has five members, so there are ten research questions and ten charts, two per member.

The study window is the ten full calendar years 2016 to 2025, because 2026 is incomplete. The main outcome is the injury crash, a crash in which at least one person was injured or killed. Reports of crashes with property damage only collapsed in 2020, so counts of all crashes cannot be compared across that year. With 2019 set to 100, all reported crashes stand at 40 in 2025 and injury crashes at 82. All-crash counts appear only in the exploration and as one robustness check.

## 2. The two datasets

The housing table (`hg8x-zxpr`) has one row per building of a city-financed project. It gives the project's start date, the building's completion date, its size, its construction type (new construction or preservation) and, for most rows, its location. The crash table (`h9gi-nx95`) has one row per police-reported collision, with date, time, location, persons hurt by road-user type and the contributing factors.

The two tables share no key. The notebook says so in Section 5.1 and does not invent a row-level join. It combines them at two levels instead (section 4 below).

Section 2 of the notebook loads the housing table in one request and downloads a filtered copy of the crash table: crashes dated 2016-01-01 to 2025-12-31 and 19 of the 29 columns. The download pages on `collision_id`, appends each page to a temporary file and resumes from it after an interruption.

## 3. Cleaning

Section 4 of the notebook follows one policy. A row is dropped only if it cannot be used at all: no key, a repeated key, or a date outside the window. A value is imputed only if it follows from other fields of the same row without guessing. A value or row is flagged when it is unusual but plausible, or wrong in a way that cannot be repaired. A value is standardised when only its form changes. Every rule writes an entry to a cleaning log (rows before, rows after, rows affected, reason), and rules that find nothing are logged too. The log has 51 entries (21 for housing, 25 for crashes, 5 for the ZIP panel) and the website shows it.

No cleaning rule removes a row from the housing or crash table. The decisions that matter most for the results are these.

- Blank unit counts in the housing table mean zero units, so 106,309 cells were set to 0. The true key is project and building together. Deduplicating on `building_id` alone would have dropped 124 rows.
- Confidential housing rows (1,949 rows, 20.4% of rows but 0.63% of units) have no location. They are kept and flagged, and they take part only in analyses that need no location.
- Completion date is the building's date, else the project's date. Rows without either are not completed. 21 rows complete before they start and 2,824 complete on their start date; both are flagged and kept.
- Building size is right-skewed. A Tukey fence on units flags 1,171 buildings, which are kept. Size bands (1-4, 5-19, 20-49, 50-149, 150 or more units) limit the influence of very large buildings.
- Crash coordinates at (0, 0) or outside a box around the five boroughs are set to missing. 8.8% of crashes have no usable location, and 16.4% of the crashes of 2016.
- A ZIP code is missing for 508,922 crashes. Placeholder and out-of-range codes are set to missing first.
- Pedestrian, cyclist and motorist counts should add up to the persons total. In 10,178 crashes the total is higher, 96% of them from 2021 on. The surplus is stored separately and flagged, because it fits e-scooter and e-bike riders being counted as injured without belonging to the three categories (the notebook says the table does not confirm this). Where the total is lower than its parts, it is raised to the sum.
- 136 crashes with more than 10 injured are flagged and not capped.
- Free-text contributing factors are mapped by ordered keywords into ten groups. A blank factor or a numeric code becomes "Unspecified".

Section 6 repairs the gaps that the joins create. ZIP-years with crashes and no completion get zero for the housing measures. ZIP codes with fewer than 500 crashes in the ten years are left out of the panel (60 ZIP codes). The crashes' missing boroughs are filled from the main borough of their ZIP.

### ZIP recovery

About a third of the crashes arrive without a ZIP code, and the share differs by year, so dropping them would bias the ZIP panel. Most of them have coordinates. The notebook learns ZIP areas from the crash table itself, without a boundary file. The city is cut into square cells of 0.001 degree, each cell takes the ZIP reported most often by the crashes inside it, and a crash without a ZIP takes its cell's ZIP. If the cell holds no labelled crash, a coarser cell of 0.004 degree is tried, and otherwise the ZIP stays missing. A random tenth of the crashes with a known ZIP was hidden to measure the method, and the chosen setting got 97.7% of them right. It filled 391,799 ZIP codes, and 117,123 crashes still have none. Before the lookup is learned, 441 reports whose borough is contradicted by at least 99% of the reports in their area are set aside as evidence, because one wrong report on a highway where little else is labelled would otherwise be copied to the crashes around it. A located crash that still has no ZIP takes its borough, and only its borough, from a vote of the 25 nearest reported crashes within 5 km (27,599 crashes). Recovered ZIPs are flagged, and results at ZIP level are repeated on reported ZIPs only (section 7).

## 4. Integration

Level A is a ZIP by year panel. Housing is summed by completion year and crashes by crash year, each table is aggregated first, and the two aggregates are merged on ZIP and year. The panel is complete: 171 ZIP codes by 10 years, 1,710 rows, with zeros where a ZIP had no completion. It also carries the cumulative stock of units completed since 2014, so a ZIP's earlier buildings count. The panel covers 91.9% of the crashes.

Level B is spatial. A crash is near a building if it lies within 250 m, roughly three short Manhattan blocks. Distances are great-circle distances (haversine), and a ball tree over the crashes finds all building and crash pairs within the radius. A brute-force check confirms the tree in the notebook.

A site is one located housing row. Buildings of this programme stand close together, and two sites less than 250 m apart count many of the same crashes, so their counts are not independent. Sites linked directly or through a chain of neighbours form a cluster (562 clusters; the three largest hold 59% of the 7,593 sites). Every bootstrap in the spatial analysis resamples whole clusters, with one exception in RQ7. Each site also has a block id, the one-kilometre square it stands in.

For the before-and-after comparison, calendar years are re-expressed relative to the completion year, from -3 to +3. A cohort is the set of sites completed in one year, and only cohorts whose whole window lies inside 2016 to 2025 are used (2019 to 2022). Each cohort gets comparison sites observed over the same calendar years: located sites that are not completed by the end of the cohort's window. A comparison site is also dropped from the main analysis if another site within 250 m was completed in its window or earlier, and a treated site is dropped if a neighbour completed in a different year of its window. Neighbours completed in the same year count as one event. This screen keeps 306 of the 2,086 treated candidates, and the rows stay in the table with their flags so that Section 9 can relax it. Because the programme is densely packed, the screened sites are the less dense part of it.

Every crash is also tagged with its nearest site within 250 m (or none), the site's completion year and the crash's year relative to it.

## 5. Statistics

All intervals are 95% percentile intervals from 1,000 bootstrap resamples with seed 42. Samples smaller than 30 get no statistic.

- Before and after at the same places is measured as the log of after over before, with 0.5 added to both counts so that sites with a zero count stay in. A difference of logs treats a fall from 4 to 2 crashes like a fall from 40 to 20, and a few busy intersections cannot decide the answer. Results are reported as percent changes.
- Crash reporting changed during the window, so a site's own before and after difference mixes the building with the citywide change. The difference-in-differences subtracts the mean log ratio at comparison sites over the same years from the mean at treated sites. Comparisons are made inside strata (cohort), and the strata are averaged with the number of treated sites as weights. The estimate is an association under the assumption that both groups would otherwise have changed alike.
- Composition questions (RQ6, RQ8, RQ10) compare the change in a category's share of injury crashes at completed sites with the change at comparison sites, in percentage points, weighted by cohort.
- Questions that report several comparisons side by side have Benjamini-Hochberg adjusted p-values at a false discovery rate of 5%. For RQ8 and RQ10 the headline is the largest of 4 or 9 gaps, so its interval is widened by a Bonferroni correction (99% and 99.4% intervals).
- The forecast comparison in RQ1 splits by time, not at random, and scores models on years they have not seen. Its interval resamples ZIP codes and holds the fitted models fixed, so it leaves out the uncertainty of fitting.

An interval that includes zero is read as "the data cannot tell a difference from no difference". That is weaker than showing there is none.

## 6. The ten questions

| RQ | Method | Headline in the executed data |
|---|---|---|
| 1 | Linear regression forecasts of a ZIP's injury crashes, trained on 2018 to 2022 and scored on 2023 to 2025. Crash history alone against history plus completions or stock. | Adding housing raises the error by 0.60 injury crashes per ZIP-year (0.17 to 1.10, 171 ZIPs) |
| 2 | Two-way fixed effects (ZIP and year) of log injury crashes on the cumulative unit stock, plus a timing regression with leads (placebo) and lags of completions. | +0.07% per 100 units (-0.06 to +0.41, 171 ZIPs) |
| 3 | Difference-in-differences of the log ratio at new-construction sites against comparison sites, cohort strata. | -4.6% (-11.9 to +3.3, 128 sites) |
| 4 | Slope of the before and after change on the log of building units, with a Spearman check. | -1.1% per doubling of units (-4.7 to +2.2, 128 sites) |
| 5 | The RQ3 contrast for new construction against preservation, as a ratio of ratios. | -8.1% (-19.6 to +4.9, 306 sites) |
| 6 | Change in the pedestrian and cyclist share of injury crashes, completed against comparison sites, plus a table by borough. | +1.2 points (-1.1 to +3.4, 306 sites) |
| 7 | Rank of each site's 2016 to 2018 injury crashes within 250 m among a grid of street reference points of its borough, for sites whose project started after 2018. Block resampling. | New construction +19.90 percentile points above the middle (16.56 to 22.99, 2,789 sites) |
| 8 | The RQ6 design for four parts of the day (and 24 hours as a detail). | Largest gap +1.4 points, afternoon (-2.1 to +4.2) |
| 9 | Spearman correlation across ZIP codes, within borough, between the share of deeply affordable units and the pedestrian and cyclist share of injury crashes. ZIPs with at least 50 affordable units. | -0.22 (-0.39 to 0.00, 123 ZIPs), adjusted p 0.10, not kept |
| 10 | The RQ6 design for nine contributing-factor groups. | Largest gap -0.8 points, impairment or medical (-2.8 to +0.7) |

Three of the ten intervals exclude zero: RQ1, RQ7 and RQ9. The RQ9 interval ends at zero and the result does not survive the adjustment for its two outcomes, so the notebook calls it a hint.

RQ7 asks where the buildings are, not what they do. Its benchmark is a grid of street points and not other housing, and apartment buildings of any kind stand on dense blocks, so the result does not show that the programme picks more dangerous streets than other housing would.

## 7. Robustness

Section 9 repeats the central comparison (RQ3) in 16 variants, each with its own interval, which with the central row make 17: buffer radius of 150 m and 400 m (with the screen rebuilt and with the same sites), a 2-year window, all crashes instead of injury crashes, the screen switched off and a stricter screen, 2020 left out as a year and as a cohort, blocks resampled instead of clusters, comparison inside cohort and borough, and two placebos that date the completion three years early. The notebook also repeats the ZIP-level questions on reported ZIP codes only.

What the checks show:

- One of the 17 intervals excludes zero: the estimate with the screen switched off (+19.9%, +0.2 to +79.8). The notebook traces that result to one development of small houses on streets that had almost no injury crashes before. The central estimate and the other 15 intervals include zero, including the two placebos.
- The sign of the estimate depends on which sites are admitted: it turns positive at 150 m with the screen rebuilt and with the screen off. It does not depend on the counting radius, the window, the outcome, the handling of 2020 or the unit that is resampled.
- The raw before and after change in all crashes is about -47% at both treated and comparison sites, against about -14% and -10% for injury crashes. The comparison group cancels the reporting break.
- On reported ZIP codes only, the RQ1 loss disappears, the completion-year slope of RQ2 is no longer kept after adjustment, and the interval of RQ9 includes zero.

## 8. Limits that apply to every result

- Confounding. Neighbourhood change, transit, street redesign and traffic safety programmes can move crashes and housing together.
- The before and after comparison rests on 128 new-construction sites (306 of both kinds), the less densely built edge of the programme, and the estimates are wide.
- Completion date is not move-in date.
- The 250 m radius is a judgement, and a crash near several sites is counted for each of them.
- ZIP-level results are ecological. They say nothing about individual streets or people.
- Only police-reported crashes are counted, and reporting practice changed during the window. Recovered ZIP codes depend on the cell vote described above.
- Confidential housing rows (20.4% of rows) have no location.
- Ten questions with several estimates each are examined, and an interval that barely excludes zero is weak evidence.
