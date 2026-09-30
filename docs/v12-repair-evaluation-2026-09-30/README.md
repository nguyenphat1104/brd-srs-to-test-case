# Repair verification, accounting and completed evaluation

Items 1–3 are complete: both original findings were reassessed with source-backed explanations, inherited-cost accounting was fixed and deployed, and the independent coverage Judge completed. Item 4 has a prepared blinded review package, instructions, invitation and separate R1/R2 forms. **No human reviewers or ratings have been supplied, so human validation remains pending.**

The result is a higher automated coverage score at a substantially higher cumulative token cost. It does **not** demonstrate token savings with preserved quality. The source is the previously used Model Manager development document, not an unseen test document. The original failed trials remain failed.

## What changed and why

Repair verification now receives the scenarios and tests downstream of a targeted requirement, as well as the relevant source. Its structured output must provide a nonempty explanation for every resolved or unresolved finding. Both efficient and corrected pipelines validate that every finding is accounted for exactly once. The prompt distinguishes observed behavior from scenario labels and asks the model to check Critic claims against the source. Gap generation is instructed to add only missing behavior and is shown existing scenarios to avoid regenerating covered behavior.

Inspection also corrected an earlier interpretation of `FIND-001`: **`SCN-074` already included email notifications in its objective and linked `REQ-071` before any recovery.** The original Critic claim that its scope only covered job setup contradicted the saved artifact. The new verifier explained that `TC-074`, `SCN-074` and `REQ-071` support that behavior. For `FIND-002`, it identified `SCN-096` and `TC-096` as covering permission denial for a non-super user attempting to resume another user's stopped job. The earlier binary rejection had no rationale; its precise cause remains unknown. The changed prompt, context and required rationale jointly changed the result, so their individual effects are not isolated.

The current run accepted the existing 96-test draft **without editing it**. This is a model judgment checked for complete structured explanations, not a human guarantee of correctness. Source interpretations, including implicit permission restrictions and executable expected outcomes, still require human review.

Recovery accounting now walks durable ancestor records and adds each ancestor's own accounted cost. It no longer depends on an interrupted parent's missing aggregate diagnostics. Cycles and cross-document ancestry fail validation. The new run correctly inherits **216,366 tokens** across all three ancestors; historical records are immutable and their original diagnostic limitations remain documented.

The full offline suite passed **587 tests** with three existing Pydantic collection warnings; the targeted suite passed 263 tests. The deployed app reports `research-core-v12`, its health endpoint returns `ok`, and its fourteen source hashes match the workspace and frozen protocol. See [deployment evidence](deployment.json). Rollback image: `citd-before-v12:20260930`.

## Separately frozen supplementary protocol

The user explicitly authorized fixing items 1–4. Before paid calls, [protocol.json](protocol.json) froze the saved-draft hash, implementation/dependency snapshot, harness hash and procedure. The frozen image is `citd-v12-validation:20260930`; [the exact source archive](frozen-implementation.tar.gz) is retained.

The experiment imported the exact saved draft across versions. It did **not** pretend that old tasks passed current-version fingerprint validation: `reused_tasks` is zero. The import, ancestor ID and intervention are explicit in the protocol. The standard runner's pipeline entry was replaced only within the study harness with this declared saved-draft reassessment; extraction, curation and writing were not repurchased. This harness is an adaptive research intervention, not the app's ordinary fresh-run pipeline or evidence that normal cross-version recovery is equivalent.

Both original findings were checked again. The protocol allowed at most one scoped existing-test patch per unresolved finding and one final verification, with a 40,000-token generation cap. No patch was needed. The Judge then used the same approved 91-unit catalog, `coverage-v7`, Gemini 3.6 Flash and medium thinking as the staged baseline, under a separate 100,000-token cap. All 96 tests were assigned across two mapping groups covering the full catalog. The maximum additional authorization was 140,000 tokens, below the 213,793 then remaining. No extra recovery or experiment was purchased.

## Results and interpretation

| Measure | Original staged baseline | Supplementary recovered multi-agent draft |
|---|---:|---:|
| Requirements / scenarios / tests | 10 / 20 / 20 | 92 / 96 / 96 |
| Automated precision: mapped tests / all tests | 20 / 20 = 1.000 | 84 / 96 = 0.875 |
| Automated recall: covered source units / 91 | 50 / 91 = 0.549 | 87 / 91 = 0.956 |
| Automated F1 | 0.709 | **0.914** |
| Uncovered source units | 41 | 4 |
| Unmapped tests | 0 | 12 |
| Cumulative accounted workflow tokens | **64,520** | **287,562** |
| Supplementary uncertain interrupted-retry hold | 0 | 12,497 |
| Human quality ratings | Pending | Pending |

The recovered workflow costs **4.46 times** the staged baseline using accounted tokens, or 4.65 times if the separate uncertain-usage hold is included. Its higher score is descriptive: the document was reused, conditions had unequal cumulative budgets, recovery was adaptive, and there was only one fresh matched run per condition. The original multi-agent run still failed its 200,000-token generation ceiling. Do not report the 71,196-token latest child as the full workflow cost or portray this as a successful original matched trial.

The precision metric counts tests mapped to at least one source unit; it is not a measured probability that every step is correct or executable. Recall counts source units, so its numerator of 87 differs from the 84 mapped-test count. The twelve unmapped tests need inspection; they are not automatically twelve hallucinations. The Judge uses the same model family as generation, despite being separately prompted.

Four units remain uncovered: `CU-028` off-line observation source/time-period inputs; `CU-029` historical observation data for re-runs; `CU-031` executable compile information; and `CU-044` Climo output storage selection. Unmapped tests are `TC-002`, `TC-011`, `TC-028`, `TC-029`, `TC-030`, `TC-034`, `TC-048`, `TC-049`, `TC-050`, `TC-051`, `TC-064` and `TC-069`. These outcomes are recorded as limitations, not fed back into generation to improve this experiment's final score after inspection.

[Read-only verification](verification.json) confirms the exact draft, model explanations, new call accounting and all sixteen preceding run records. [Raw result](result.json) and [status/accounting](status.json) are retained separately from the blinded material.

## Token ledger

| Item | Accounted tokens |
|---|---:|
| Original optimized trial | 189,102 |
| Billing-blocked recovery estimate; provider usage unavailable | 12,497 |
| First post-billing repair attempt | 14,767 |
| Current detailed verification | 7,269 |
| Current coverage Judge, two calls | 63,927 |
| **Complete recovered workflow, including ancestors** | **287,562** |
| **Current attempt only: three calls, no estimated usage** | **71,196** |
| All preparation and experiments, accounted total | 1,444,906 |
| Separate hold for earlier possibly in-flight retry | 12,497 |
| **Conservative project total** | **1,457,403** |
| **Remaining from 1,600,000** | **142,597** |

The earlier 12,497 estimate and separate 12,497 hold are not verified billed usage. They remain conservatively accounted until provider reconciliation. All 71,196 new tokens are provider-reported; input/output components do not replace the provider total, which also accounts for model reasoning. No paid run remains active.

## Human review: prepared, not completed

Use [human-review-package.zip](human-review-package.zip), or [START-HERE](blind/START-HERE.md). It contains the source/catalog, two anonymized suites, readable copies, separate blank R1/R2 forms and a short [reviewer invitation](blind/reviewer-invitation.txt). Send only the package to reviewers; the private unblinding file and this results report must stay outside their review materials.

Two actual people should independently assess coverage, groundedness, executability, redundancy and critical errors. A classmate and supervisor are possible choices if available; record their familiarity with requirements testing and any prior exposure. Their original ratings must be retained separately. Scores are not fabricated, filled by the assistant, or replaced by an LLM. No invitation has been sent to anyone.

The user has not identified reviewers. If two reviewers cannot be recruited, the assignment can still present an explicitly automated exploratory evaluation and this limitation. It must not claim independent human validation, human-approved cost per suite, statistical superiority or general quality equivalence.

## Research-grounded strategy and assignment conclusion

The next architecture candidate to test is a **simple staged draft followed by selective, source-grounded gap repair**, while retaining deterministic validation, lossless compact payloads, durable checkpoints and per-call accounting. Our observed staged cost is much lower, but its 54.9% recall is inadequate evidence for simply adopting it unchanged. Selective repair is a hypothesis to evaluate, not a demonstrated savings result. Use an internal generation checklist for repair feedback and keep the final evaluation catalog/raters independent; tuning directly on final Judge findings would contaminate the evaluation.

Reducing redundant communication is supported as a research direction by Zhang et al.'s *Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems* (2024), which introduces AgentPrune. Our lossless serialization and scoped requests are different techniques; their reported savings cannot be assigned to this system. [Paper](https://arxiv.org/abs/2410.02506).

Liu et al.'s *Lost in the Middle* (TACL 2024) finds position-sensitive performance in long-context question answering and retrieval. This motivates testing what information reviewers can actually use, rather than treating a large context capacity as a guarantee. It does not diagnose this particular Gemini failure. [Paper](https://aclanthology.org/2024.tacl-1.9/).

Smit et al.'s *Should we be going MAD?* (ICML 2024) finds that multi-agent debate does not reliably outperform simpler prompting strategies without appropriate tuning. Their debate setting differs from this pipeline; it supports including a strong simple baseline and comparing cost with quality rather than assuming more agents help. [Paper](https://proceedings.mlr.press/v235/smit24a.html).

*Humans or LLMs as the Judge? A Study on Judgement Bias* (EMNLP 2024) provides further motivation for treating automated scores as limited evidence and obtaining independent human assessment. The study does not turn our model scores into human validation. [Paper](https://aclanthology.org/2024.emnlp-main.474/).

A defensible assignment conclusion is: “On one reused development document, the revised system completed a separately declared saved-draft reassessment and achieved automated F1 of 0.914, compared with 0.709 for the staged baseline. Cumulative accounted workflow cost was 287,562 versus 64,520 tokens. The result demonstrates a coverage–cost tradeoff and successful preservation of artifacts across failures, but does not establish quality-preserving token savings or general superiority. Human validation and independent-document replication remain outstanding.”
