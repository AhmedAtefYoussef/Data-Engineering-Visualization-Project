# Methodology and answers

This file summarizes the methods of the notebook `notebooks/NYC_Housing_Safety.ipynb` and answers the ten research questions in plain words. The notebook has the code, the reasoning for each choice and the executed outputs. The website in `app/` uses the same code through `tools/build_app_data.py`, so both show the same numbers. Where this file and the notebook differ, the notebook is right. Every estimate below is read from `app/data/results.json`, `app/data/robustness.parquet` or the executed notebook.

## 1. Question and design

The question is whether the streets around city-financed affordable housing have different police-reported crash records from comparable streets, before and after the buildings are completed. The design is observational. Nobody assigned buildings to streets at random, so every result is worded as an association.

The team has five members, so there are ten research questions and ten charts, two per member.

The study window is the ten full calendar years 2016 to 2025, because 2026 is incomplete. The main outcome is the injury crash, a crash in which at least one person was injured or killed. Reports of crashes with property damage only collapsed in 2020, so counts of all crashes cannot be compared across that year. With 2019 set to 100, all reported crashes stand at 40 in 2025 and injury crashes at 82. All-crash counts appear only in the exploration and as one robustness check.

"Near a building" means within 250 m of it.

## 2. The two datasets

The housing table (`hg8x-zxpr`) has one row per building of a city-financed project. It gives the project's start date, the building's completion date, its size, its construction type (new construction or preservation) and, for most rows, its location. The crash table (`h9gi-nx95`) has one row per police-reported collision, with date, time, location, persons hurt by road-user type and the contributing factors.

The two tables share no key. The notebook says so in Section 5.1 and does not invent a row-level join. It combines them at two levels instead (section 4 below).

Section 2 of the notebook loads the housing table in one request and downloads a filtered copy of the crash table: crashes dated 2016-01-01 to 2025-12-31 and 19 of the 29 columns. The download pages on `collision_id`, appends each page to a temporary file and resumes from it after an interruption. The local copies were saved on 10 October 2026 (`meta.json`, `raw_files_saved`).

## 3. Cleaning

Section 4 of the notebook follows one policy. A row is dropped only if it cannot be used at all: no key, a repeated key, or a date outside the window. A value is imputed only if it follows from other fields of the same row without guessing. A value or row is flagged when it is unusual but plausible, or wrong in a way that cannot be repaired. A value is standardised when only its form changes. Every rule writes an entry to a cleaning log (rows before, rows after, rows affected, reason), and rules that find nothing are logged too. The log has 49 entries (17 for housing, 27 for crashes, 5 for the ZIP panel) and the website shows it.

No cleaning rule removes a row from the housing or crash table. The decisions that matter most for the results are these.

- Blank unit counts in the housing table mean zero units, so 106,309 cells were set to 0. The true key is project and building together. Deduplicating on `building_id` alone would have dropped 124 rows.
- Confidential housing rows (1,949 rows, 20.4% of rows but 0.63% of units) have no location. They are kept and flagged, and they take part only in analyses that need no location.
- Completion date is the building's date, else the project's date. Rows without either are not completed. 21 rows complete before they start and 2,824 complete on their start date; both are flagged and kept.
- Building size is right-skewed. A Tukey fence on units flags 1,171 buildings, which are kept. Size bands (1-4, 5-19, 20-49, 50-149, 150 or more units) limit the influence of very large buildings.
- Crash coordinates at (0, 0) or outside a box around the five boroughs are set to missing. 8.8% of crashes have no usable location, and 16.4% of the crashes of 2016.
- Crashes logged at exactly 0:00 (23,717, 1.6%) are flagged `time_unknown`. 0:00 is the default for a missing time, and 4,547 crashes are logged at 1:00. They are excluded from hour-of-day work (RQ8) and stay in every other count.
- A ZIP code is missing for 508,922 crashes. Placeholder and out-of-range codes (638) are set to missing first.
- Pedestrian, cyclist and motorist counts should add up to the persons total. In 10,178 crashes the total is higher, 96% of them from 2021 on. The surplus is stored separately and flagged, because it fits e-scooter and e-bike riders being counted as injured without belonging to the three categories (the notebook says the table does not confirm this). Where the total is lower than its parts, it is raised to the sum.
- 136 crashes with more than 10 injured are flagged and not capped.
- Free-text contributing factors are mapped by ordered keywords into ten groups. A blank factor or a numeric code becomes "Unspecified".

Section 6 repairs the gaps that the joins create. ZIP-years with crashes and no completion get zero for the housing measures (870 of 1,720). 59 ZIP codes with fewer than 500 crashes in the ten years are left out of the panel. The crashes' missing boroughs are filled from the main borough of their ZIP.

### ZIP recovery

About a third of the crashes arrive without a ZIP code, and the share differs by year, so dropping them would bias the ZIP panel. Most of them have coordinates. The notebook learns ZIP areas from the crash table itself, without a boundary file. The city is cut into square cells, each cell takes the ZIP reported most often by the crashes inside it, and a crash without a ZIP takes its cell's ZIP. If the cell holds no labelled crash, a coarser cell of 0.004 degree is tried, and otherwise the ZIP stays missing.

The accuracy test hides whole locations, not single rows, because 81.7% of labelled crashes share their exact coordinates with another one and a row-level holdout would let a crash be scored against its own twin. The labelled crashes are split by exact location into a selection part and a separate test part. The selection part chooses the fine cell size (0.0005, 0.001 or 0.002 degree; 0.0005, about 56 m, scored best) and the test part reports accuracy. On the test locations the recovered ZIP equals the reported one for 96.9% of covered crashes. Highways, bridges and ramps are 2.5% of labelled crashes but 36.9% of the crashes to fill, and accuracy there is 90.7%; reweighted to that mix the expected accuracy of the filled values is about 94.7%. Both figures are upper bounds, because the rows to fill lie by definition where fewer neighbours report a ZIP.

The method filled 391,799 ZIP codes, and 117,123 crashes still have none. Before the lookup is learned, 441 reports whose borough is contradicted by at least 99% of the reports in their area are set aside as evidence, because one wrong report on a highway where little else is labelled would otherwise be copied to the crashes around it. A located crash that still has no ZIP takes its borough, and only its borough, from a vote of reported boroughs in the cells around it (27,597 crashes, 99.7% on the same kind of holdout). Recovered ZIPs are flagged, and the ZIP-level results are repeated on reported ZIPs only (section 7).

## 4. Integration

Level A is a ZIP by year panel. Housing is summed by completion year and crashes by crash year, each table is aggregated first, and the two aggregates are merged on ZIP and year. A building that was completed again under a later project is counted once: 78 completed rows repeat an earlier building and their 5,955 units are not counted a second time. The unit columns by kind (new construction, preservation) count affordable units, the units the programme counts, so they add up to the affordable total. The panel is complete: 172 ZIP codes by 10 years, 1,720 rows, with zeros where a ZIP had no completion. It also carries the stock of affordable units completed since 2014, the first year in the housing table, so a ZIP's earlier buildings count (12,363 units when 2016 starts, 220,701 at the end of 2025). The panel covers 92.0% of the crashes.

Level B is spatial. A crash is near a building if it lies within 250 m, roughly three short Manhattan blocks. Distances are great-circle distances (haversine), and a ball tree over the crashes finds all building and crash pairs within the radius. A brute-force check confirms the tree in the notebook.

A site is one located housing row (7,593). Buildings of this programme stand close together, and two sites less than 250 m apart count many of the same crashes, so their counts are not independent. Sites linked directly or through a chain of neighbours form a cluster (562 clusters; the three largest hold 59% of the sites). Every bootstrap in the spatial analysis, RQ7 included, resamples whole clusters.

For the before-and-after comparison, calendar years are re-expressed relative to the completion year, from -3 to +3. A cohort is the set of sites completed in one year, and only cohorts whose whole window lies inside 2016 to 2025 are used (2019 to 2022). Each cohort gets comparison sites observed over the same calendar years: located sites that are not completed by the end of the cohort's window. A comparison site is also dropped from the main analysis if another site within 250 m was completed in its window or earlier, and a treated site is dropped if a neighbour completed in a different year of its window. Neighbours completed in the same year count as one event. This screen keeps 306 of the 2,086 treated candidates, and the rows stay in the table with their flags so that Section 9 can relax it. Because the programme is densely packed, the screened sites are the less dense part of it.

Every crash is also tagged with its nearest site within 250 m (or none). The website uses that tag.

## 5. Statistics

All intervals are 95% percentile intervals from 1,000 bootstrap resamples with seed 42. Samples smaller than 30 get no statistic. The bootstrap, the difference-in-differences and the event-study profile live once, in `app/stats.py`. The notebook pastes that file into Section 7 and the website imports it, so both draw the same resamples.

- Before and after at the same places is measured as the log of after over before, with 0.5 added to both counts so that sites with a zero count stay in. A difference of logs treats a fall from 4 to 2 crashes like a fall from 40 to 20, and a few busy intersections cannot decide the answer. Results are reported as percent changes.
- Crash reporting changed during the window, so a site's own before and after difference mixes the building with the citywide change. The difference-in-differences subtracts the mean log ratio at comparison sites over the same years from the mean at treated sites. Comparisons are made inside strata (cohort), and the strata are averaged with the number of treated sites as weights. The estimate is an association under the assumption that both groups would otherwise have changed alike.
- Composition questions (RQ6, RQ8, RQ10) compare the change in a category's share of injury crashes at completed sites with the change at comparison sites, in percentage points, weighted by cohort.
- Questions that report several comparisons side by side have Benjamini-Hochberg adjusted p-values at a false discovery rate of 5%. For RQ8 and RQ10 the headline is the largest of 4 or 9 gaps, so its interval is widened by a Bonferroni correction (98.75% and 99.4% intervals). In RQ2 the adjustment covers the three lags only; the two leads are placebo diagnostics.
- The forecast comparison in RQ1 splits by time, not at random, and scores models on years they have not seen. Its interval resamples ZIP codes and holds the fitted models fixed, so it leaves out the uncertainty of fitting.

An interval that includes zero is read as "the data cannot tell a difference from no difference". That is weaker than showing there is none.

## 6. The ten questions

The table gives the method and the estimate with its interval, from `results.json`. The answers that follow say what each result means and what it does not.

| RQ | Method | Estimate (interval, n) |
|---|---|---|
| 1 | Linear regression forecasts of a ZIP's injury crashes, trained on 2018 to 2022 and scored on 2023 to 2025. Crash history alone against history plus completions or stock. | Adding housing raises the error by +0.59 injury crashes per ZIP-year (+0.17 to +1.11, 172 ZIPs) |
| 2 | Two-way fixed effects (ZIP and year) of log injury crashes on the cumulative unit stock, plus a timing regression with leads (placebo) and lags of completions. | +0.06% per 100 units (-0.07 to +0.39, 172 ZIPs) |
| 3 | Difference-in-differences of the log ratio at new-construction sites against comparison sites, cohort strata. | -4.6% (-11.9 to +3.3, 128 sites) |
| 4 | Slope of the before and after change on the log of the units completed nearby in that year. | -2.2% per doubling of units (-5.8 to +1.1, 128 sites) |
| 5 | The RQ3 contrast for new construction against preservation (a negative control), as a ratio of ratios. | -8.1% (-19.6 to +4.9, 306 sites) |
| 6 | Change in the pedestrian and cyclist share of injury crashes, completed against comparison sites, plus a table by borough. | +1.22 points (-1.08 to +3.39, 306 sites) |
| 7 | Rank of each site's 2016 to 2018 injury crashes within 250 m among a grid of street reference points of its borough, for sites whose project started after 2018. Linked site clusters resampled. | New construction +19.99 percentile points above the middle (+10.39 to +23.06, 2,789 sites in 350 clusters) |
| 8 | The RQ6 design for four parts of the day, without the crashes of unknown time. | Largest gap +1.38 points, afternoon (98.75% interval -2.16 to +4.29, 306 sites) |
| 9 | Spearman correlation across ZIP codes, within borough, between the share of deeply affordable units and the pedestrian and cyclist share of injury crashes. ZIPs with at least 50 affordable units. | -0.202 (-0.376 to +0.012, 123 ZIPs), adjusted p 0.136, not kept |
| 10 | The RQ6 design for nine contributing-factor groups. | Largest gap -0.83 points, impairment or medical (99.4% interval -2.76 to +0.67, 306 sites) |

Two of the ten intervals exclude zero: RQ1 and RQ7. The RQ9 interval ends just above zero (+0.012), and its adjusted p-value (0.136) is above the 5% threshold, so the notebook calls it a hint.

### Member 1: Ahmed Atef (16007032)

**RQ1. Does knowing how much affordable housing a ZIP code completed help forecast its injury crashes?** No. A forecast built only on the ZIP code's own crash history was already as good as we could get, and adding housing made it slightly worse. For 172 ZIP codes a model was trained on 2018 to 2022 and tested on 2023 to 2025. One version used the last two years of injury crashes. The other also used completed units and the stock of units built so far. The ZIP-years scored average about 212 injury crashes, and the history-only model misses by about 20 (19.82). The housing version missed by 0.59 more per ZIP-year (95% interval 0.17 to 1.11). On crashes whose ZIP the police reported, the difference disappears (-0.16, interval -0.58 to +0.35), so the notebook reads it as "housing did not improve the forecast" and not as "housing made it worse". Last year's crashes are the best guide to next year's; adding completions does not help a planner forecast where injuries will happen.

**RQ2. Inside one ZIP code, do injury crashes move as its stock of affordable units grows?** We found no clear movement. The same 172 ZIP codes were followed for ten years and each was compared with itself, after removing what was common to all ZIP codes in a year, so a citywide change such as 2020 does not count as a housing effect. For every 100 units added, injury crashes changed by +0.06% (interval -0.07% to +0.39%). The two years before completion, used as a false test, showed nothing. The completion year itself shows a dip of -0.39% (adjusted p 0.012). Split by kind it is -0.41% for preservation, which brings no new residents, and +0.38% (interval -0.46% to +1.28%) for new construction, so the notebook reports it and does not count it as a finding. It stays below the 5% threshold on reported ZIPs (adjusted p 0.030). A ZIP is large, so this says little about the street in front of a new building; RQ3 looks at that.

### Member 2: Nour Mostafa Aly (16001618)

**RQ3. Do injury crashes near a new building change after it opens?** We could not detect a change. For buildings completed in 2019 to 2022 we counted injury crashes within 250 m in the three years before and after completion, and did the same over the same calendar years for sites not yet completed. Only sites with no other completion nearby in the window were kept, which left 128 of the 1,142 new-construction sites. Injury crashes fell by 13.8% at new buildings and 9.7% at comparison sites; relative to the comparison sites the change is -4.6% (interval -11.9% to +3.3%). The interval holds a moderate fall as well as a small rise, so this is not evidence of no change. The pre-period placebo (years -3 to -1) is +6.6% (-6.9% to +21.0%). The answer covers isolated buildings only; sites in dense clusters of projects had to be left out.

**RQ4. Do bigger buildings come with a bigger change?** No. On the same 128 sites, each doubling of the units completed within 250 m in that year went with a change of -2.2% in injury crashes (interval -5.8% to +1.1%). If new residents were driving crashes, large buildings should show it most. They do not, which fits RQ3.

### Member 3: Yahia Mohamed (16004934)

**RQ5. Is the change different for new construction than for preservation?** Not in a way we can measure. Preservation renovates buildings that already have residents, so nobody new arrives, and it serves as a negative control: a change that comes with new residents should appear around new construction and not around preservation. Around new construction the change was -4.6% (interval -12.1% to +3.3%, 128 sites). Around preservation it was +3.8% (interval -7.0% to +16.2%, 178 sites). The difference is -8.1% (interval -19.6% to +4.9%). The control is weak: it is imprecise, 70% of its sites are in the 2019 cohort, and preserved buildings stand on older streets than new ones.

**RQ6. Does the share of injury crashes that hurt pedestrians and cyclists change after completion?** No clear change. Near completed and near comparison sites we computed the share of injury crashes in which a pedestrian or cyclist was hurt, before and after, and compared the two changes. The gap is +1.22 percentage points (interval -1.08 to +3.39), from 306 completed and 696 comparison sites. Borough gaps run from -0.7 to +1.9 points, none survives the adjustment, and Staten Island has too few sites. New construction alone gives +1.16 points. The interval is wide enough to hide a shift of a few points.

### Member 4: Moaz Ahmed (16003110)

**RQ7. Before construction starts, do these sites already stand on streets with more injury crashes?** Yes, and this is the clearest result of the project. For sites whose project started after 2018 we counted injury crashes within 250 m in 2016 to 2018 and ranked each site against 9,985 street reference points of the same borough. A typical street point sits at the 50th percentile. New-construction sites sit near the 70th, +19.99 points higher (interval +10.39 to +23.06, 2,789 sites in 350 clusters). 80.3% of them are above their borough's middle, with a median of 62 nearby injury crashes against 27 at a typical street point. Ranked inside their own ZIP code they are still 8.4 points above the middle (4.8 to 12.1). Preservation sites also rank above the middle (mean percentile 59.5 against 70.0 for new construction). This counts injury crashes, not risk per traveller, so it says "more injury crashes" and not "more dangerous". It describes where the buildings stand. It does not show that the programme picks worse streets than market-rate housing would, because we have no market-rate data. For the transportation department, planned housing sites are a reasonable list of places to check for safety work before people move in.

**RQ8. Does the time of day of crashes near sites shift after completion?** No. Injury crashes near sites were split into night, morning, afternoon and evening, leaving out the 23,717 crashes of unknown time, and the change in each share at completed sites was compared with the change at comparison sites. The largest gap is the afternoon, +1.38 points, with an interval of -2.16 to +4.29 once widened for four bands. None of the four passes the adjustment (adjusted p for the afternoon 0.53). With new construction alone the gap is +0.34 points. Weekdays and weekends are pooled, so opposite changes would cancel.

### Member 5: Yassin Fady (16005869)

**RQ9. Do ZIP codes with more deeply affordable units have a different pedestrian and cyclist share of injuries?** There is a hint, and we do not count it as a finding. For 123 ZIP codes with at least 50 affordable units we compared the share of units reserved for the lowest incomes with the share of injury crashes that hurt a pedestrian or cyclist, within each borough, so that Manhattan against Staten Island does not drive the result. The rank correlation is -0.202 (interval -0.376 to +0.012). The interval includes zero by a small margin, the bootstrap p-value is 0.068 and the adjusted p-value is 0.136 (two outcomes). On reported ZIP codes alone the interval is -0.392 to +0.012. Deaths per 1,000 injury crashes show no association (+0.125, interval -0.086 to +0.299). We cannot say that the lowest-income housing sits in places with more or less pedestrian harm; the question is worth asking again with counts of people walking.

**RQ10. Do the recorded causes of crashes near sites change after completion?** No factor group changed in a way we can separate from noise. Police record a contributing factor for each crash. The labels were grouped into nine families, and the change in each family's share (among crashes with a recorded factor) at completed sites was compared with the change at comparison sites. The largest gap is impairment or medical, -0.83 points, with an interval of -2.76 to +0.67 after widening for nine tests. None of the nine survives the adjustment (smallest adjusted p 0.93). One side result: crashes with no recorded factor became less common near completed sites (-3.5 points, interval -6.4 to -0.9), but that came from preservation sites (-5.7 points) and not new construction (-0.8), so it is not about new residents. About a quarter of crashes (24.5%) have no factor recorded, which limits what anyone can learn from this field.

### What the ten answers add up to

After a building opens, we see no measurable change in how many people are hurt nearby, who is hurt, when, or why. Before it opens, the site is already on a street with more injury crashes than most. For New York that puts the attention on the streets where housing is planned, since that is where the records show a difference.

## 7. Robustness

Section 9 repeats the central comparison (RQ3) as 11 rows, the central estimate and 10 variants, each with its own interval: all crashes instead of injury crashes, the screen switched off and the screen off without its most influential cluster, two placebos that date the completion three years early (with a 3-year and a 2-year window), comparison inside cohort and borough, without the 2019 cohort, one building per project, and buffer radii of 150 m and 400 m with the screen and clusters rebuilt at each. The notebook also repeats the ZIP-level questions on reported ZIP codes only.

What the checks show:

- One of the 11 intervals excludes zero: the estimate with the screen switched off (+19.9%, +0.2 to +79.8). The notebook traces that result to one development of small houses on streets that had almost no injury crashes before; without that cluster the estimate is +3.9% (-1.2 to +11.4). The central estimate and the other nine intervals include zero, including the two placebos (+2.1% and -0.4%).
- The sign of the estimate depends on which sites are admitted: it turns positive at 150 m (+5.4%, -4.1 to +14.7) and with the screen off. It does not depend on the outcome (all crashes: -2.2%, -9.9 to +5.2), the cohort (without 2019: -7.0%, -17.1 to +2.6), the strata (inside cohort and borough: -5.9%, -13.1 to +1.7), repeated buildings in one project (-5.6%, -11.4 to +0.6) or the unit that is resampled.
- The raw before and after change in all crashes is about -47% at both treated and comparison sites, against -13.8% and -9.7% for injury crashes. The comparison group cancels the reporting break.
- On reported ZIP codes only, the RQ1 loss disappears, the RQ2 completion-year slope remains below the 5% threshold (adjusted p 0.030, against 0.012 on all codes), and the RQ9 interval still ends at +0.012.

## 8. Limits that apply to every result

- Confounding. Neighbourhood change, transit, street redesign and traffic safety programmes can move crashes and housing together.
- The before and after comparison rests on 128 new-construction sites (306 of both kinds), the less densely built edge of the programme, and the estimates are wide.
- Completion date is not move-in date, and 36% of completed rows have a completion date equal to the start date.
- The 250 m radius is a judgement, and a crash near several sites is counted for each of them.
- ZIP-level results are ecological. They say nothing about individual streets or people.
- Only police-reported crashes are counted, and reporting practice changed during the window. About 28% of the crashes in the ZIP panel carry a recovered ZIP code, and the recovery accuracy is an upper bound.
- Confidential housing rows (20.4% of rows) have no location.
- Ten questions with several estimates each are examined, and an interval that barely excludes zero is weak evidence.
