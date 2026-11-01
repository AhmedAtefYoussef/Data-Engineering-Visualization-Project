# LinkedIn post

Images, in this order: screens/01_hero_before_after.png, 05_story_before_after.png, 02_explore_before_after.png, 03_report_before_after.png, 04_findings_before_after.png, 06_phone_before_after.png

Post text (paste everything between the lines):
--------------------------------------------------------------------
I kept working on our ICS 503 project after the first version went live. It still asks whether injury crashes rise around New York's affordable housing once a building opens, joining 9,551 housing records to 1.5 million crash reports. Before and after screenshots are below.

The dashboard
The first version was a long white page of charts. The new one opens on a map of 7,593 buildings drawn as points of light, which regroup into each chart as you scroll through five chapters. I added three tools that didn't exist before. Your block looks up any ZIP code or project and shows how crash-heavy its streets were before construction. Test it lets you change who is compared and recomputes the main result instantly, so you can try to break it yourself. Explore turns a question typed in plain words into a report with a shareable link and a CSV.

Phones got their own pass. On an emulated iPhone 13 with a slowed processor, first content went from 18 s to 3.5 s, blocked main-thread time from 21.6 s to 2.5 s, and the download from 7.6 MB to 2.4 MB. axe-core now finds no WCAG 2.2 AA violations.

The pipeline
The notebook is now the single source of truth. A build script runs the same notebook code to export the site's data, and the site won't start if the schema doesn't match. No number on the page is typed by hand. The test suite grew from 208 to 816 tests at 98% coverage, and separate code recomputes what the site shows from the raw files under 40 random filter combinations. That check caught a real bug: one result depended on how a library broke distance ties, so it could come out differently on another machine. I fixed it and logged every number that moved. I also traced an intermittent production failure to Vercel's 500 MB function limit and fixed it.

How I built it
I orchestrated a team of Claude Code subagents, each with one bounded brief: its role, the files it could read, the only files it could edit, the quality bar and the format of its report. They ran in parallel waves, each in its own git worktree and branch, and I merged a branch only after its checks passed. Auditors reviewed the statistics, code, prose, charts and design but couldn't edit what they judged, and writers applied their findings. Nothing counted as done until it had run on the real data, passed the tests and been checked in a headless browser at phone and desktop widths. I measured every performance change before and after.

The headline result is still modest: -4.6% after opening (95% interval -11.9% to +3.3%), which the data can't tell from no change. New buildings did go up on streets already busier than most of their borough.

Thanks to my teammates Nour Mostafa Aly, Yahia Mohamed, Moaz Ahmed and Yassin Fady.

Live dashboard: https://data-engineering-visualization-proj-three.vercel.app/
Code: https://github.com/AhmedAtefYoussef/Data-Engineering-Visualization-Project

#DataEngineering #DataVisualization #Python #ClaudeCode #NYCOpenData
--------------------------------------------------------------------
