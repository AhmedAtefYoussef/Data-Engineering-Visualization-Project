# %% [markdown]
# # Affordable housing and street safety in New York City
#
# What the crash records show on the streets around city-financed affordable buildings, before
# and after they are completed.
#
# ICS 503 Data Engineering and Visualization, Winter Semester 2026, Project Milestone 1
#
# ## Abstract
#
# When an affordable building is proposed, neighbours often worry that it will make the streets
# around it less safe. We wanted to know whether the crash records support that worry. We
# combined two NYC Open Data tables, one from each domain of the project: Affordable Housing
# Production by Building, which lists every building the city has financed since 2014 with its
# completion date, size and location, and Motor Vehicle Collisions - Crashes, which lists every
# police-reported collision, here for the ten years 2016 to 2025. The tables share no key, so
# we joined them in two ways: by ZIP code and year, after reducing both tables to that grain,
# and by distance, counting the crashes within 250 m of each building. A large share of the
# crashes arrive without a ZIP code, and most of those gaps were filled from the coordinates once
# the tables met. The outcome throughout is the injury crash, because reports of crashes without
# injuries fell sharply from 2020 for reasons that have nothing to do with housing. Our main
# finding has two parts. New-construction sites already stood, before their project started, on
# streets with more injury crashes than the typical street location of their borough. The
# benchmark is a grid of street points and not other housing, so this shows where the buildings
# are and not that the programme picks more dangerous streets than other housing would. After
# completion we could not detect a change in injury crashes near the new buildings we could
# isolate, once the same years at not-yet-completed sites are taken into account. Around
# completed sites of either kind, new or preserved, we could also not detect a change in who was
# hurt, at what hour or with which recorded contributing factor. The main limit is that sites are not placed
# at random and only a small, less densely built part of the programme could be isolated for the
# before and after comparison, so every result is an association and none can be read as an
# effect of housing.
#
# ## Team
#
# The team has five members, so the notebook has ten research questions and ten charts, two per
# member. The label in the first column is the one used under each question heading in Section 8.
#
# | Member | Name | Student ID | Research questions |
# |---|---|---|---|
# | M1 | Ahmed Atef Youssef | TO FILL IN | RQ1 Do housing completions improve a forecast of a ZIP code's injury crashes? RQ2 Do injury crashes in a ZIP code move as its stock of affordable units grows? |
# | M2 | TO FILL IN | TO FILL IN | RQ3 Do injury crashes within 250 m of a new building change after completion? RQ4 Does that change grow with the size of the building? |
# | M3 | TO FILL IN | TO FILL IN | RQ5 Is the change different for new construction than for preservation? RQ6 Does the pedestrian and cyclist share of injury crashes change after completion, by borough? |
# | M4 | TO FILL IN | TO FILL IN | RQ7 Before the project starts, are the sites already on streets with more injury crashes than typical street locations of the borough? RQ8 Does the hour-of-day mix of injury crashes shift after completion? |
# | M5 | TO FILL IN | TO FILL IN | RQ9 Is the share of deeply affordable units in a ZIP code associated with crash severity or the pedestrian and cyclist share of harm? RQ10 Which contributing-factor groups are over- or under-represented near completed housing? |
#
# ## The two datasets
#
# | Domain | Dataset | Page | API |
# |---|---|---|---|
# | Housing & Development | Affordable Housing Production by Building (`hg8x-zxpr`) | https://data.cityofnewyork.us/d/hg8x-zxpr | https://data.cityofnewyork.us/resource/hg8x-zxpr.csv |
# | Transportation | Motor Vehicle Collisions - Crashes (`h9gi-nx95`) | https://data.cityofnewyork.us/d/h9gi-nx95 | https://data.cityofnewyork.us/resource/h9gi-nx95.csv |
#
# ## Where each requirement of the milestone is met
#
# | Requirement | Section |
# |---|---|
# | Loading through the Open Data API, with the column names checked | 2.1 |
# | Dataset overview: rows, columns, geographic and temporal attributes | 2.2 |
# | Reasoning behind the selection, with the alternatives | 2.3 |
# | Feasibility check | 2.4 |
# | Justification of the filtered crash copy | 2.5 |
# | Known issues | 2.6, and 3.3 with their measured size |
# | Exploratory data analysis with statistics and plots | 3.1 (housing), 3.2 (crashes) |
# | Pre-integration cleaning: missing values, drop against impute | 4.1, 4.3, 4.4, 4.7, 4.8, 4.9 |
# | Pre-integration cleaning: outliers (IQR and domain rules) | 4.6, 4.8, 4.10 |
# | Pre-integration cleaning: inconsistent data | 4.4, 4.5, 4.8, 4.9 |
# | Pre-integration cleaning: formats (dates, strings, categories, geographic identifiers) | 4.1, 4.3, 4.4, 4.7, 4.8, 4.11 |
# | Pre-integration cleaning: duplicates | 4.2, 4.7 |
# | Cleaning log and before/after summary | 4.13 |
# | Integration: join key, granularity and aggregation, with the reasons | 5.1 to 5.6 |
# | Post-integration cleaning: new missing values from the joins | 6.1 |
# | Post-integration cleaning: inconsistent or redundant columns | 6.2 |
# | Post-integration cleaning: data type mismatches | 6.3 |
# | Conservation checks on the integrated tables | 6.4 |
# | Statistics functions used by the questions, with a self-test | 7 |
# | Ten research questions, one chart each, with the owner named | 8.1 to 8.10 |
# | Robustness of the main comparison and of the ZIP-level results | 9 |
# | The discovery: what we found and what surprised us | 10 |
# | The limits | 11 |
# | The impact: who can use this | 12 |
# | Prompts used in the project | 13 |
# | Individual contributions | 14 |
# | How to run the notebook, versions and data-as-of stamp | 15 |
