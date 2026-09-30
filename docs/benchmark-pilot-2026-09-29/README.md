# Pilot: comparisons recorded; human ratings pending

The two PDFs and the **1,600,000-token total budget are approved**. On 2026-09-30 the user approved both source catalogs and explicitly authorized sending these PDFs, excerpts and derived content to Google’s Gemini API. Both catalogs are now approved in the local database. All eight comparison trials are recorded: four completed generation runs and four failures. Both multi-agent variants failed on both documents. Generated-suite human ratings remain pending. See [catalog approval](catalog-approval.json), [external-transfer authorization](external-transfer-authorization.json) and [current status](pilot-status.json).

## Results and remaining review

- [Cost, outcomes, research discussion and recommended next experiment](cost-and-quality-report.md).
- [Blinded review package: start here](results/blind/START-HERE.md).
- Use two actual, independent raters and return `ratings-R1.csv` and `ratings-R2.csv`. Prefer people who have not viewed this task’s condition outcomes. Five saved artifacts are available, including one partial requirements-only artifact; four are completed generation runs. Three failures produced no reviewable bundle.
- **522,328 tokens accounted; 1,077,672 unspent.** No further model calls are scheduled. Quality preservation is not established.

## Approved source catalogs

1. [Model Manager catalog: 91 proposed units](D001-catalog-review.md).
2. [DigitalHome catalog: 84 proposed units](D002-catalog-review.md).
3. [Source ambiguities and critical checks](source-review-notes.md), especially exact limits, actor permissions, manual overrides and unresolved TBDs.

Check whether the obligations are supported, sufficiently atomic and complete, and whether repeated statements describe the same behavior. The source PDF paths are listed in `authorization.json`. Page numbers in the catalogs refer to actual PDF pages. The quoted excerpts preserve extraction artifacts; they have not been silently rewritten to look cleaner.

The drafts include testable process and inspection constraints as well as runtime behavior, consistently with the current source-coverage protocol. Document-purpose text, descriptive glossary definitions, named project members and bibliography entries are treated as context, not independent product obligations. Explicitly unresolved features remain in the ambiguity notes rather than receiving invented exact test oracles. In particular, DigitalHome's humidistat “manual temperature” wording requires reviewer clarification.

The user’s **“Approve both source catalogs”** approval applies to the exact draft hashes recorded in `catalog-approval.json`. It does not certify generated test suites or supply the later blinded human ratings. The `.draft.json` filenames are retained to preserve the frozen manifest references.

These catalogs were drafted by the assistant and approved by the user. Exact citation validation succeeded for all **175** units; it does not prove semantic completeness. No generated-suite human ratings have been supplied.

## Budget and the live preparation result

| Item | Tokens / count |
|---|---:|
| User-authorized total budget | 1,600,000 |
| Catalog preparation already accounted | 23,613 |
| Original ceiling for all generation/evaluation trials | 1,576,387 |
| Comparison generation/evaluation tokens accounted | 498,715 |
| Total including preparation | 522,328 |
| Unspent authorization | 1,077,672 |
| Paid catalog call attempts | 3 |
| Successful calls / failed calls | 1 / 2 |
| Benchmark trials planned / recorded | 8 / 8 |
| Completed generation runs / failures | 4 / 4 |

The automated catalog attempt hit an 8,000-token output cap, split once, saved a successful child task, and then stopped when the other child also exceeded the cap. Its raw calls and checkpoint are retained in `D001-calls.jsonl` and `D001-checkpoints.jsonl`. The failed calls remain included in the 23,613 total. The original failed preparation record is preserved as `preparation-status.json`.

The first failed call reported 1,464 input tokens, 309 visible output tokens and 7,676 reasoning tokens: 9,449 tokens total. The reasoning plus visible output consumed nearly the entire response allowance. This is direct evidence of an output-budget failure for that call; it is not evidence that the input context window was full. It also shows why passing scripted tests does not prove that the real model will finish under the same cap.

After that failure, no more Gemini catalog calls were made. The review drafts were prepared directly from the two PDFs in this task and built with `build_review.py`; source references were verified against the application's canonical extracted chunks. The partial Gemini glossary-oriented output was preserved as evidence, not automatically adopted as ground truth.

## Prepared comparison

The source documents, code/dependency hashes, model, thinking level, condition definitions, seed and randomized order are recorded in `pilot-manifest.json` and `planned-trials.json`. Each PDF will receive one fresh run of each condition: single prompt, staged single agent, corrected multi-agent and efficient multi-agent. All share the chosen generation settings and fixed independent Judge protocol.

The benchmark runner's `max_total_budget_tokens` is **1,576,387**, so the spent preparation tokens cannot be spent again. The manifest also records the original total and preparation cost. Per-run generation and Judge ceilings remain 100,000 each. The runner checks remaining funds before each trial and can stop short of eight trials if actual spending leaves insufficient capacity; it must never increase the approved budget to force completion.

No generation or evaluation implementation was modified in response to these held-out source-catalog observations. The live output-limit failure is a recorded limitation, not hidden by changing the treatment halfway through a study. If a later engineering change is needed, record a new version and use a separate experiment rather than quietly replacing results.

The approved catalog hashes were checked and the same source catalog is reused by every condition; it is never fed into generation. Before any condition run, runtime preflight detected local/deployed differences in `google-genai` (2.17.0 → 2.19.0) and `pypdf` (6.15.0 → 6.16.2). The manifest was frozen against the deployed runtime after verifying all 175 exact references and that no old or new trial IDs existed. Source code, catalogs, model settings and trial order are unchanged. The original local manifest and [preflight audit](runtime-preflight.json) are preserved.

The final report must add preparation spend to all attempted generation/evaluation spend, report failures and unavailable scores, and include blinded human ratings and critical-error checks. With two documents and one repetition per condition this remains a feasibility pilot; it cannot establish broad quality equivalence.

## Verification

- Both source PDF hashes match the approved selection.
- Both draft files pass the existing coverage-unit schema.
- All 175 unit references match exact source spans.
- Three recorded paid attempts reconcile to 23,613 tokens.
- The remaining comparison budget plus preparation spending equals exactly 1,600,000.
- The manifest validates against current source and implementation/dependency hashes and contains eight planned trials.

Machine-readable state: [pilot-status.json](pilot-status.json). Application code remains frozen; approval records and new benchmark runs are stored separately from historical runs.
