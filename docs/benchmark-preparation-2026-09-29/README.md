# Benchmark preparation — 29 September 2026

Status: **prepared, not executed**. No paid generation request has been sent. Gemini 3.6 Flash is accessible with the configured account, verified by a read-only model-list request. Application history contains 15 distinct document hashes; 49 extractable source PDFs have no matching run hash. There are currently no approved source coverage catalogs.

## Candidate selection

Candidates were selected before looking at condition outputs. The 49 unused-by-history files were sorted by extracted text length, divided into strata of 16, 17 and 16, then two were sampled per stratum with seed 20260929. This reduces discretionary selection. It does **not** establish that the files were never used outside this application's recorded history.

| ID | PDF | Pages | Extracted characters | Stratum |
|---|---|---:|---:|---|
| D001 | 2009 - model manager.pdf | 13 | 21,358 | short |
| D002 | 2010 - home 1.3.pdf | 15 | 24,965 | short |
| D003 | 2004 - sprat.pdf | 33 | 45,541 | medium |
| D004 | 2006 - stewards.pdf | 45 | 66,942 | medium |
| D005 | 2007 - mdot.pdf | 56 | 102,830 | long |
| D006 | 2001 - elsfork.pdf | 69 | 114,173 | long |

## Concrete execution choices

All conditions use the same generation model and thinking level, a proposed 100,000-token generation ceiling and a 100,000-token Judge ceiling. The conditions are single prompt, staged single agent, corrected multi-agent control, and efficient multi-agent.

| Scope | Trials | Maximum scheduling reservation |
|---|---:|---:|
| Small operational pilot: D001 and D002, one repeat per condition | 8 | 1,600,000 tokens |
| Full feasibility study: all six candidates, three repeats per condition | 72 | 14,400,000 tokens |

These are sums of per-trial ceilings, not forecasts of actual usage or prices. The small pilot can check execution and obtain preliminary costs; two documents cannot establish general quality preservation. The six-document study is also a feasibility design, not a statistical power calculation. A smaller user-approved budget stops scheduling earlier and may leave an incomplete comparison. Do not silently substitute an incomplete comparison for the planned study.

Neither scope is approved merely by this document. The actual total budget remains unset. Source catalog preparation by external model calls would need its own allocation; manual human annotation has no model API token charge. Unexpected usage reported after a request may exceed a reservation; the scheduler stops further work once accounted limits are reached.

## Review materials

- `candidate-selection.json`: selection method, file paths and hashes, sample sizes and unresolved decisions.
- `D001-source.json` through `D006-source.json`: canonical source chunks and citation locations.
- `D001-pages.txt` through `D006-pages.txt`: extracted page text for navigation. Inspect original PDFs for tables, diagrams and reading order; text extraction alone is insufficient.
- `catalog-schema.json`: required source-coverage-unit format. Prepare one JSON object with a nonempty `units` array per document. Use unique unit IDs and exact 5–25-word references from the source chunks. The schema validates structure; a human must review support and completeness.
- `source-review.csv`: independent source-review checklist. Blank review and approval fields are intentional; no review has been fabricated.
- `readiness.json`: frozen preparation status, implementation hashes and proposed execution sizes. This is not an approved experiment manifest.

D001 already illustrates an important review issue: the Model Manager source contains explicit TBDs, question marks and conditional features. Unspecified defaults, job priorities, model options and unclear restart/post-processing settings must remain ambiguous. Catalog construction must not turn these into invented exact expected values. The source also distinguishes ordinary users' own jobs from a super user's ability to control any job; those actor boundaries must survive generation and review.

## Remaining order of work

1. Confirm total token/spending limit and which candidate files truly have not been used for tuning. Freeze the selected scope and configuration.
2. Independently annotate and review the selected source catalogs before inspecting generated suites. Human approval remains separate from model-generated draft annotations.
3. Run the frozen randomized schedule with the benchmark CLI; preserve failures and all attempt costs.
4. Give two independent human raters the blinded artifacts and source documents. Never label model-produced ratings as human ratings.
5. Analyze paired document-level cost and quality, completion, missing evaluations, critical errors and inter-rater agreement. Leave quality preservation inconclusive if evidence is insufficient.

The implementation and CLI instructions are in [the rollout and research protocol](../token-efficiency-rollout-2026-09-29.md). No production setting, historical result or application code was changed during this preparation.
