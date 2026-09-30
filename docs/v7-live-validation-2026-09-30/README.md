# Live v7 development validation — 30 September 2026

**Both attempted runs failed to complete. Quality-preserving token savings are not established.** These are development regressions on the two previously used and approved PDFs, not fresh held-out evidence. All original eight pilot results are preserved.

| Document | Outcome | Accounted tokens | Saved artifacts | Automated F1 |
|---|---|---:|---|---|
| D001 — Model Manager | Invalid evidence reference after one correction | 18,406 | Two validated extraction/audit checkpoints; no bundle | Unavailable |
| D002 — DigitalHome | Generation budget guard stopped the next call | 191,485 | 104 requirements, 141 scenarios, 40 tests; incomplete | Unavailable |

The runs made 33 charged model calls and one blocked reservation. All 209,891 tokens came from generation; neither run reached evaluation. No estimated charges were needed. Earlier preparation and comparisons used 522,328 tokens. **Total: 732,219 / 1,600,000; remaining: 867,781.** The validation used 209,891 of its separate 600,000-token ceiling.

The budget is cumulative across calls. Neither failure indicates a Gemini context-window limit: D001 rejected a source reference; D002 stopped locally because its next reservation needed 10,838 tokens but only 8,515 remained. Aggregate tokens must not be compared to a single request's context capacity.

## Where DigitalHome spent its budget

| Stage | Tokens |
|---|---:|
| Extraction | 29,588 |
| Source-gap audits | 24,165 |
| Curation | 37,254 |
| Scenario generation | 74,983 |
| Test writing, incomplete | 25,495 |
| **Total** | **191,485** |

Extraction, audit, curation and scenario generation consumed 165,990 tokens before test writing began. Scenario generation was the largest stage. The saved artifacts demonstrate expansion of intermediate content, but their count alone does not prove redundancy: distinct boundary and state cases can be necessary. Their cost does show that smaller handoff payloads alone did not produce a completed suite within this budget.

The earlier offline reductions measured request characters on Telescope with fixed replayed decisions. They were not measured end-to-end token savings. This live run used different model-generated artifacts and cannot be used to claim those offline percentages transferred to production.

## Failure handling and verification

D001 exposed a reporting defect: after the second invalid response, a raw `ValueError` escaped the runner and left the database status as running. The process had already exited. Its four immutable call records and two validated checkpoints were exported, then the existing repository finalization method recorded a semantic failure. No call was repeated; aggregate metrics remain unavailable rather than reconstructed as if the runner had saved them. See `D001-finalization-provenance.json` and `D001-before-finalization.json`.

After both frozen trials ended, the shared task boundary was patched to translate an exhausted validation error into `PipelineOutputError`, which the runner already records as a terminal semantic failure. The patch changes failure reporting, not evidence acceptance, prompts, retry count or generation quality. A regression reproduces the exception against the frozen image; all 127 focused pipeline/runner tests pass with the patch. The deployed image and source hashes are recorded separately in `reporting-fix-deployment.json`.

`verification.json` reconciles exported results with the database, checks the unchanged original eight trials, verifies both approved catalogs and PDF hashes, verifies all 14 archived source files, and totals every call against both budgets. The trial implementation is preserved in `frozen-implementation.tar.gz` and Docker image `citd-v7-validation:20260930`. The post-trial reporting patch is not retroactively part of the experiment. Do not rerun the frozen manifest against changed application code.

## Recommended next design

1. **Use the staged single-agent baseline as the practical reference.** In the original pilot it completed both documents; the efficient multi-agent variant still has no completed suite. This does not establish that the baseline meets an acceptable quality threshold. Keep its human review pending and the original controls intact.
2. **Ablate the separate scenario-authoring stage first.** Prototype generating a concise scenario and its executable tests together for each assigned requirement group, retaining the same traceability and output schemas. Avoid composing and then resending a complete intermediate scenario corpus. Verify coverage of actors, boundaries, negation and state transitions; do not solve cost by imposing arbitrary artifact counts. This is a proposed experiment, not an implemented or measured saving.
3. **Make evidence-reference correction specific.** Report which returned ID failed and the allowed local scope, or evaluate short task-local IDs mapped deterministically to stable evidence IDs. Keep exact-source validation. The present records cannot distinguish a nonexistent ID from a known but out-of-scope one, and invalid raw responses were not retained, so the exact model mistake cannot be reconstructed.
4. **Plan the budget across the whole pipeline.** Estimate remaining test-writing and review cost after extraction, reserve their budget, and stop with an explicit incomplete outcome if the plan cannot fit. Stage allocations alone do not guarantee completion; calibrate them on development data before a new study. Keep the existing per-call guard and checkpoint recovery.
5. **Run a new comparison only after local calibration.** Freeze the same generation budget, model settings and evaluator across conditions, use genuinely unused approved documents, predeclare an acceptable quality margin, and obtain two independent human reviews. Report completion rate and cost per human-accepted suite, not cheap early failures or test counts as success.

These proposals apply the communication-redundancy motivation in Zhang et al.'s [*Cut the Crap / AgentPrune*](https://arxiv.org/abs/2410.02506), but do not implement its pruning algorithm or inherit its measured savings. Liu et al.'s [*Lost in the Middle*](https://aclanthology.org/2024.tacl-1.9/) motivates focused evidence instead of assuming larger context solves retrieval; its experiments were on different tasks and models. Smit et al.'s [ICML 2024 comparison of multi-agent debate](https://proceedings.mlr.press/v235/smit24a.html) motivates strong simple baselines; debate is different from this deterministic pipeline, so it does not establish that this application's multi-agent design is universally inferior.

## What remains

The live v7 validation and failure-reporting correction are finished. The broader quality-preserving efficiency goal is not finished: invalid evidence generation and end-to-end budget completion remain unresolved. Neither run supplies an automated quality score, and both original human-rating forms remain blank. Reusing development documents, raising the generation ceiling from 100,000 to 200,000, and changing the evaluator prevent a like-for-like improvement claim.

The original [human review instructions](../benchmark-pilot-2026-09-29/results/blind/START-HERE.md) and [blinded review ZIP](../benchmark-pilot-2026-09-29/human-review-package.zip) remain the place for two actual reviewers to submit independent ratings. The new DigitalHome partial bundle is separately exported in `results/blind`; it must not be treated as a completed comparison suite. No human scores have been generated by an LLM.
