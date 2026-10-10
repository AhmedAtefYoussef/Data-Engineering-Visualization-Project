# Answers to the ten research questions

Each team member owns two questions. For every question this page gives the short answer, how we measured it, and what it means for New York. All numbers come from the executed notebook (`notebooks/NYC_Housing_Safety.ipynb`, Sections 8 and 9), where the code and the charts are.

Three things apply to every answer.

We count injury crashes, meaning crashes where at least one person was hurt or killed. Total reported crashes fell by more than half after 2019 while injuries stayed close to their old level, so the total mostly tracks how crashes were reported.

"Near a building" means within 250 metres of it.

Every result is an association. Housing sites are not chosen at random and neighbourhoods change for many reasons, so none of these numbers shows that housing caused anything. A 95% interval that includes zero means the data fit no change as well as a small rise or a small fall.

## Member 1: Ahmed Atef Youssef

### RQ1. Does knowing how much affordable housing a ZIP code completed help forecast its injury crashes?

No. A forecast built only on the ZIP code's own crash history was already as good as we could get, and adding housing made it slightly worse.

How we measured it: for 171 ZIP codes we trained a model on the earlier years and tested it on 2023 to 2025, which it had never seen. One version used the last two years of injury crashes. The other also used completed units and the stock of units built so far. We compared the average miss of each version. The housing version missed by 0.59 more injury crashes per ZIP code per year (95% interval 0.16 to 1.10), on a base of about 20. When we kept only crashes whose ZIP code was reported by the police, the difference disappeared (−0.17, interval −0.63 to +0.32).

What it means for the city: last year's crashes are the best guide to next year's. A planner who wants to know where injuries will happen does not gain anything by adding affordable-housing completions to the forecast.

### RQ2. Inside one ZIP code, do injury crashes move as its stock of affordable units grows?

We found no clear movement. ZIP codes that added more units did not see their injury crashes rise or fall in step.

How we measured it: we followed the same 171 ZIP codes over ten years and compared each one with itself, after removing what was common to all ZIP codes in a given year. That way a busy ZIP code is not compared with a quiet one, and a citywide change such as 2020 does not count as a housing effect. For every 100 units added, injury crashes changed by +0.07% (interval −0.06% to +0.41%). The years before completion, which we used as a false test, showed nothing either. There was a small dip in the completion year itself, but it did not hold up on reported ZIP codes and it came from renovated buildings, where no new residents arrive.

What it means for the city: at the scale of a ZIP code, building affordable housing did not come with a visible change in injury crashes. A ZIP code is large, so this says little about the street in front of a new building. RQ3 looks at that.

## Member 2: name to fill in

### RQ3. Do injury crashes near a new building change after it opens?

We could not detect a change. The estimate is a small fall, and the interval includes zero.

How we measured it: for buildings completed in 2019 to 2022 we counted injury crashes within 250 metres in the three years before and the three years after completion. We did the same over the same calendar years for sites that were not yet completed, and compared the two changes. This is a difference-in-differences: the comparison sites absorb what happened to the whole city, including the pandemic. We only kept sites with no other completion nearby during the window, which left 128 of 1,142 new buildings. Relative to the comparison sites, injury crashes changed by −4.6% (interval −11.9% to +3.3%). The intervals come from resampling groups of neighbouring sites, because buildings close to each other share the same crashes.

What it means for the city: the common worry that a new affordable building brings more injuries to the block is not supported by these records. They do not prove safety either. The answer covers isolated buildings only, since sites in dense clusters of projects had to be left out.

### RQ4. Do bigger buildings come with a bigger change?

No. The change around a site did not grow or shrink with the number of units.

How we measured it: on the same 128 sites we related the before-to-after change to the size of the building, on a doubling scale because a few buildings are very large. Each doubling of units went with a change of −2.2% in injury crashes (interval −5.8% to +1.1%).

What it means for the city: if new residents were driving crashes, large buildings should show it most. They do not, which fits the answer to RQ3.

## Member 3: name to fill in

### RQ5. Is the change different for new construction than for preservation?

Not in a way we can measure. The two kinds of project look alike.

How we measured it: preservation renovates buildings that already have residents, so nobody new arrives. We used it as a check. If something other than new residents were moving crashes around housing sites, preservation would show it too. Around new construction the change was −4.6% (interval −12.1% to +3.3%, 128 sites). Around preservation it was +3.8% (interval −7.0% to +16.2%, 178 sites). The difference between them is −8.1% with an interval of −19.6% to +4.9%.

What it means for the city: buildings that add residents and buildings that do not behave the same in the crash records. That is one more reason to read RQ3 as no detectable change.

### RQ6. Does the share of injury crashes that hurt pedestrians and cyclists change after completion?

No clear change.

How we measured it: near completed sites and near comparison sites we computed the share of injury crashes in which a pedestrian or cyclist was hurt, before and after, and compared the two changes. The gap is +1.2 percentage points (interval −1.1 to +3.4), from 306 completed and 696 comparison sites. No single borough stood out after correcting for testing several of them.

What it means for the city: more residents usually means more people on foot. The records do not show that turning into a larger share of pedestrian and cyclist injuries, though the interval is wide enough to hide a shift of a few points.

## Member 4: name to fill in

### RQ7. Before construction starts, do these sites already stand on streets with more injury crashes?

Yes, and this is the clearest result of the project.

How we measured it: we took sites whose project started after 2018 and counted injury crashes within 250 metres in 2016 to 2018, before anything was built. We ranked each site against a grid of street points in the same borough. A typical street point sits at the 50th percentile. New-construction sites sit near the 70th, which is 19.7 points higher (interval 16.3 to 22.8, 2,789 sites). About 79% of them are above their borough's middle, with a median of 62 nearby injury crashes against 29 at a typical street point. Ranked inside their own ZIP code they are still 8.6 points above the middle.

What it means for the city: apartment buildings go up on busy corridors, where crashes are also concentrated. Residents of new affordable housing move onto streets that were already more dangerous than average. This describes where the buildings stand. It does not show that the programme picks worse streets than market-rate housing would, because renovated buildings also rank above the middle and we have no market-rate data. For the transportation department, planned housing sites are a reasonable list of places to check for safety work before people move in.

### RQ8. Does the time of day of crashes near sites shift after completion?

No.

How we measured it: we split injury crashes near sites into night, morning, afternoon and evening and compared how each share changed at completed sites against comparison sites. The largest gap was the afternoon, at +1.4 percentage points, with an interval of −2.1 to +4.2 once we widen it for having looked at four parts of the day. None of the four passed the test, and neither did any single hour.

What it means for the city: there is no sign that completions move crashes toward school hours, the commute or the night. Time-based measures near these sites can follow the same pattern as the rest of the neighbourhood.

## Member 5: name to fill in

### RQ9. Do ZIP codes with more deeply affordable units have a different pedestrian and cyclist share of injuries?

There is a hint, and we do not count it as a finding.

How we measured it: for 123 ZIP codes with at least 50 affordable units we computed the share of units reserved for the lowest incomes and the share of injury crashes that hurt a pedestrian or cyclist. We compared ZIP codes only within the same borough, so that Manhattan against Staten Island does not drive the result. The rank correlation is −0.22 (interval −0.39 to −0.001). The interval misses zero by a hair, the p-value is 0.10 after correcting for the two outcomes we looked at, and on reported ZIP codes alone the interval includes zero.

What it means for the city: we cannot say that the lowest-income housing sits in places with more or less pedestrian harm. The question is worth asking again with better exposure data, such as counts of people walking.

### RQ10. Do the recorded causes of crashes near sites change after completion?

No factor group changed in a way we can separate from noise.

How we measured it: police record a contributing factor for each crash. We grouped the labels into nine families, such as distraction, failure to yield and speed, and compared how each family's share changed at completed sites against comparison sites. The largest gap was impairment or medical causes, at −0.8 percentage points, with an interval of −2.8 to +0.7 after widening it for nine tests. One side result: crashes with no recorded factor became less common near completed sites, but that came mainly from preservation sites, so it is not about new residents.

What it means for the city: completions do not come with a new kind of crash. About a quarter of crashes have no factor recorded at all, which limits what anyone can learn from this field.

## What the ten answers add up to

After a building opens, we see no measurable change in how many people are hurt nearby, who is hurt, when, or why. Before it opens, the site is already on a street with more injury crashes than most. For New York that puts the attention on the streets where housing is planned, since that is where the records show a difference.
