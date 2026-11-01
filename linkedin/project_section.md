# LinkedIn > Projects > edit this project

Name: Affordable Housing and Street Safety in New York City

Project URL: https://data-engineering-visualization-proj-three.vercel.app/

Associated with: ICS 503 Data Engineering and Visualization

Skills: Data Engineering · Data Visualization · Python · pandas · Plotly Dash · Statistics · Claude Code · GitHub Actions · Vercel

Media: attach screens/new_01_story_hero.png, new_04_your_block.png, new_05_test_it.png, new_06_explore_report.png, new_07_findings.png, new_09_phone.png

Description (paste everything between the lines):
--------------------------------------------------------------------
Do injury crashes rise around city-financed affordable housing once a building opens? We joined two NYC Open Data tables that share no key, 9,551 housing records and 1,504,722 police crash reports from 2016 to 2025, by ZIP code and year and by distance (crashes within 250 m of a building).

The live dashboard is one page. A scroll story draws the 7,593 located buildings as points of light that regroup into each chapter's chart. Your block looks up any ZIP code, project or borough and shows how crash-heavy its streets were before construction. Test it lets you change who is compared and recomputes the main result on the spot. Explore turns a question typed in plain words, or a set of filters, into a report with a shareable link and a CSV. Findings and Method hold the ten research questions, the robustness checks and the full cleaning log.

For the 128 new buildings that could be compared fairly, injury crashes within 250 m changed by -4.6% after completion (95% interval -11.9% to +3.3%), which the data cannot tell from no change. New-construction sites stood on streets already busier than most of their borough, at the 69th percentile before construction.

The notebook is the single source of truth. The site's data is exported from the same code, checked against a schema at start-up, and verified by 816 tests (98% coverage) on every push. On a phone the page downloads 2.4 MB and shows content in about 3.5 s, and axe-core reports no WCAG 2.2 AA violations.

I led the project and built it by orchestrating Claude Code subagents in parallel, with separate auditors and automated checks verifying every change.
--------------------------------------------------------------------
