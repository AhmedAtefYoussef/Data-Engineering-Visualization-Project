# Prompts used

AI coding assistants (Claude models from Anthropic, used through Claude Code) drafted the code and text of this project. The notebook and the website send nothing to a language model.

Prompts whose text the team has are quoted in full. The others are described, and the description says so.

## 1. The first requests

The first prompt came with the project description and the data catalogue. It asked the assistant to analyse both, to compare the available Claude models, to split the work among several assistants with one responsibility each, and to audit the result against the requirements of the project description. Its text is not available word for word, so it is described here and not quoted.

The follow-up in the same conversation was:

> apply these skills as necessarily as possible: 21st-ui, design-motion-principles, design-taste-frontend, humanizer, impeccable, ui-ux-pro-max

A team of five members and a light, editorial look for the website were decided in the same conversation.

## 2. The briefs given to the assistants

The first version of the project (removed from the repository, kept in the first commit of the git history) was built in steps, each handled by one assistant with its own brief. Every brief had the same parts: a role, the files to read, the files the assistant was allowed to edit, the task, the quality bar and the form of the report. The briefs are summarized below, and their full text is in the session transcript, not in this repository.

| Step | Model | Role in the brief | Task | Output asked for |
|---|---|---|---|---|
| Spec review | Opus 5.5 | Adversarial architect | Find defects in the written specification: event-study design, meaning of the exposure flag, imputation leakage, paging, memory, gaps in the contract | Numbered findings with severity and exact fix text, under 600 words |
| Analysis and app | Sonnet 5.5 | Dash engineer and designer | Ten research-question functions, the Dash website, the search parser, the design brief | Files, tests, screenshot findings |
| Notebook and docs | Sonnet 5.5 | Technical writer | Notebook builder, README, methodology, data dictionary | Notebook execution result |
| Final audit | Opus 5.5 | Independent auditor | Check every requirement in the project description against the repository | Requirement-by-requirement table |

The prompt patterns were sections marked with XML-style tags (role, context, ownership, task, output format), one responsibility and one set of editable files per assistant, a written specification that all of them had to follow, reviewers that could read but not edit what they judged, and a cap on the length of each report.

## 3. The request for the current notebook

The second version of the project, with the self-contained notebook and the rebuilt website, was requested with this prompt. The project description and the data catalogue were attached.

> Reverse engineer the entirety of the codebase in terms of architecture, technical code implementation, documentation and the finalized notebook deliverable. Apply a structural dynamic workflow incorporating multiple sub agents with their respective tasks concurrently, allowing the each subagent to effectively and productively focus on one specific task at a time. For example, a subagent is responsible for code literacy and syntax, another is responsible for logical reasoning and developing the best possible strategies to counteract potential problems in all aspects, etc. The structural workflow should allow each subagent to complete their respective tasks concurrently with maximum level of precision, accuracy and efficiency with the least possible token consumption. Finally, apply all of the necessary plugins, skills, etc. You may research through the Claude chat regarding this project to maximize your knowledge, intel, etc. Finally, the finalized version of the notebook should be perfect in all the aspects such as modularization, code organization, documentation, architecture in addition to coordinating a sub agent that checks whether the finalized notebook satisfies all of the requirements from the project description. Also, use the best datasets from the data catalog folder and I need the notebook to run independently without needing external folders or files, just both of the datasets used. Apply the best skills I have here such as humanizer, impeccable, design-motion-principle, etc. Go all out. Make no mistake.

The same conversation contained three further prompts, and the two-word prompt "Try again" was sent twice.

> Build high quality top notch interactive web application according to the project requirements and specification by incorporating these skills as well; /humanizer

> and these skills as well /anthropic-skills:impeccable /anthropic-skills:design-motion-principles /anthropic-skills:design-taste-frontend /anthropic-skills:ui-ux-pro-max

> While you are continuing building this project, I need you to cut down the token usage by incorporating cheaper models for simpler tasks. For example, allow Claude opus 5.5 to execute and perform the complex tasks while sonnet 5.5 is responsible for code syntax and literacy. Basically apply the structural workflow, maintaining the quality and productivity of the work productivity and effectiveness overall while cutting down token usage in a strategic approach.

After a usage limit interrupted the work, the request was "I hit my usage limit while you were working, but it has reset now. Please continue from where you left off." The documentation update (README and `docs/`) was done in that continued session.

## 4. The shared brief for the section authors

Each notebook section was drafted by a separate assistant from one shared brief. The passages that set the content of the notebook are quoted here. Passages that only name working folders and build commands are left out.

> Datasets: H2 Affordable Housing Production by Building (hg8x-zxpr) and T4 Motor Vehicle Collisions - Crashes (h9gi-nx95). Crashes are a filtered copy: 2016-01-01 to 2025-12-31 and 19 of 29 columns.

> Crash reporting broke in 2020: total crashes halve (211k in 2019, 113k in 2020, 86k in 2025) while injuries stay near 50-60k a year, because property-damage-only reports collapsed. So the PRIMARY OUTCOME everywhere is the **injury crash** (a crash with at least one person injured or killed). All-crash counts appear only in EDA and as a robustness check. Say this plainly wherever a time comparison is made.

> Two integration levels, both justified in the text: (a) ZIP x year panel; (b) spatial proximity, a crash is "near" a building if within `BUFFER_M` = 250 m (haversine BallTree). No row-level key join exists or is faked.

> Charts: matplotlib/seaborn only, static, no plotly, no map tiles, no network at view time. Use `PALETTE`, `BOROUGH_COLORS`, `finish()`. Every chart has a title that states what it shows, axis labels with units, a legend when there is more than one series, and a labelled log axis when a log axis is used. One figure per code cell, followed by a markdown or `say()` reading of it. Pick the chart type for the question; no pie charts, no 3D, no dual axes.

> Small named functions with docstrings; descriptive names (`housing_clean`, not `h2` or `df1`); no one-letter frames; no semicolon-joined statements; no magic numbers (name them); no dead code; no try/except that hides errors.

> Comment the non-obvious logic (outlier rule, spatial index, imputation, bootstrap, difference-in-differences), explaining why, not restating the line.

> Add `assert` checks for every invariant you claim (row conservation, uniqueness, no negative counts).

> A markdown cell before every code block says what is about to be done and why. After the output, the reading of it. For every cleaning or design decision: the problem as observed in THIS data, the decision (drop, impute, flag, standardise), why, and the alternative considered and why it was rejected.

> Never type a data-derived number into markdown. Numbers that come from the data are printed by code, using `say(f"...")` for sentences. Markdown carries the reasoning; it may state design constants and qualitative conclusions that you have verified in the executed output.

> No claim of causation anywhere. This is observational data: say "is associated with", "coincides with", "differs from". State n and uncertainty with each result.

The brief also listed the ten research questions with their owners, in the wording of the team table at the top of the notebook.

## 5. Review and fix passes

Each draft was executed on the real data before it was accepted. Further assistants then read the executed notebook, each from one point of view: the requirements of the project description, the code, the statistics, the prose and the charts. Each was asked for a list of findings with a location, a severity, the problem and a proposed fix. The briefs of these reviewers are not available word for word. The authors of each section then applied the findings, rebuilt the notebook and opened the figures they changed. A separate assistant rebuilt the website data and the website from the same section sources, and another did a design and motion pass on it.

## 6. Prompts inside the product

None. The notebook and the website make no model calls.
