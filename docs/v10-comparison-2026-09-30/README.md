# Historical v10 comparison and recovery results

**Current follow-up:** the [separately frozen v12 reassessment](../v12-repair-evaluation-2026-09-30/README.md) resolved the verification gate with explanations and completed coverage evaluation at F1 0.914. The existing 96-test draft was unchanged. Automatic ancestry accounting is fixed; 142,597 tokens remain. Human reviews are still pending. The sections below preserve the outcomes and budget at the end of the earlier v10 billing resumption; they are not the current project status. Its statement about `SCN-074` requiring further scope inspection is superseded by inspection showing that email notifications and `REQ-071` were already included.

The payload reductions are implemented and verified offline. The matched experiment completed, but **the optimized pipeline still failed its 200,000-token generation cap**. After a billing-blocked recovery, the user increased the provider cap and authorized another attempt. Gemini then completed five new calls, but semantic verification still rejected one finding. The retained draft has 96 tests; independent evaluation has not run. Quality preservation is not established, and no human ratings have been supplied.

## Matched result: one development document

Both conditions used the same approved Model Manager PDF and 91-unit catalog, Gemini 3.6 Flash, minimal generation thinking, a 200,000-token generation ceiling and the same coverage-v7 Judge with medium thinking and a 100,000-token ceiling. The document had already been used in development. See the [frozen methods and limitations](methods.md), [manifest](manifest.json) and [database/accounting verification](verification.json).

| Measure | Staged baseline | Optimized multi-agent |
|---|---:|---:|
| Generation status | Completed | Failed at first repair reservation |
| Requirements / scenarios / tests saved | 10 / 20 / 20 | 92 / 93 / 93 |
| Generation tokens | 39,512 | 189,102 |
| Judge tokens | 25,008 | 0 |
| Total accounted tokens | **64,520** | **189,102** |
| Source units covered by the Judge | 50 / 91 | Unavailable |
| Precision / recall / F1 | 1.000 / 0.549 / **0.709** | Unavailable |
| Human ratings | Pending | Pending; retained draft |

The optimized run completed all extraction, source audits, curation, test writing and six initial Critic batches. One Critic batch needed one correction. The Critic retained two findings: a test/scenario mapping issue in `TC-074`, and a missing permission-denial scenario for `REQ-092`. The first repair needed a 12,497-token reservation but only 10,898 remained. The guard blocked that call without charging it. The resulting 93 tests are a draft, not a semantically approved suite.

This experiment favors the staged baseline for **completion and cost at the tested cap**. Its 54.9% automated source-unit recall is also a material limitation. Neither condition is demonstrated to meet an acceptable human-quality threshold. The optimized run cannot be described as cheaper, higher quality, or quality-equivalent to staged from these results; its F1 is missing, not zero. Do not report a savings percentage against the earlier Telescope incident, which used a different document and pipeline.

## Implemented reductions

The Critic receives tables with field names written once. A test shares a scenario field only when the values are exactly equal; different preconditions, titles, requirement links and citations remain explicit. Every scoped artifact can be reconstructed exactly. The full requirement index and relevant source evidence remain available. Review groups can contain up to 24 requirements under the unchanged 48,000-character request bound.

The Judge uses the same table representation and larger bounded groups. Every test still meets every approved coverage unit; no top-k retrieval or omitted catalog subset is introduced. Its separate request bound is 80,000 characters, with up to 100 units and 48 tests per group. The cumulative Judge ceiling remains 100,000 tokens. This changes prompt size and grouping, so model-quality effects still require evaluation.

On the **fixed previous v9 draft**, the [offline replay](offline-measurement.json) measured:

| Initial planned requests | Before | After | Character reduction |
|---|---:|---:|---:|
| Critic | 10 calls; 381,676 characters | 6 calls; 233,983 characters | 38.7% |
| Judge | 18 calls; 565,338 characters | 3 calls; 177,048 characters | 68.7% |

All 9,555 test–catalog-unit pairs remain in the Judge replay. These quantities include serialized requests and response schemas, but exclude generated output, reasoning, corrections and repairs. **They are not measured token savings.** The new live draft differs from the replay draft, so its stage cost is not a controlled before/after estimate.

The motivation and boundaries are grounded in [AgentPrune](https://arxiv.org/abs/2410.02506), [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) and [Smit et al.'s multi-agent baseline study](https://proceedings.mlr.press/v235/smit24a.html), as explained in [methods.md](methods.md). Their benchmark gains are not attributed to this application.

## Recovery and provider billing block

A separately declared [post-hoc recovery](recovery/protocol.json) reused all 31 validated checkpoints without regenerating earlier work. Its additional generation ledger was capped at 100,000 tokens, with a separate 100,000-token Judge ceiling. This was supplementary recovery, not a rerun or replacement of the matched failure.

Gemini rejected the first repair with: **“Your project has exceeded its monthly spending cap.”** The recovery container was stopped to prevent repeated attempts, and its durable record was [sealed as a provider failure](recovery/status.json). It produced no new validated repair and no evaluation. The original matched records and retained draft are unchanged. Billing settings were not changed and no alternate project was used.

The incident also exposed an application retry bug: a hard spending-cap 429 was treated like a temporary rate limit. The subsequent v11 patch stops application retries for explicit monthly/project spending-cap messages, while preserving bounded retries for temporary 429 responses. This patch is separate from the frozen v10 experiment and makes no new model calls. **582 tests passed**, and the [deployed v11 app is healthy](v11-deployment.json); frozen v10 and rollback images remain available.

## Budget reconciliation

| Item | Tokens |
|---|---:|
| Preparation and trials before this comparison | 1,092,824 |
| Matched staged run | 64,520 |
| Matched optimized run | 189,102 |
| **Accounted before the billing-blocked recovery** | **1,346,446** |
| Persisted recovery failure estimate; provider usage unavailable | 12,497 |
| **Recorded accounted total** | **1,358,943** |
| Separate conservative hold for a possibly in-flight retry at termination | 12,497 |
| **Conservative total including that hold** | **1,371,440** |
| Available before the billing resumption, after the hold | 228,560 |
| New provider-reported tokens after billing was restored | 14,767 |
| **Current recorded accounted total** | **1,373,710** |
| **Current conservative total including the hold** | **1,386,207** |
| **Current remaining from 1,600,000 after the hold** | **213,793** |

The two recovery amounts are **not verified billed tokens**. One is the failed call's persisted reservation estimate; the other is a separate hold for uncertain interrupted usage, not a fabricated call record. Reconcile them against provider usage before releasing the hold. A remaining experiment token allocation does not override the provider's monetary spending cap. No further paid work is running.

## Retry after the provider cap increase

The user authorized retrying after increasing the provider budget. The [new protocol](billing-resumption/protocol.json) kept the original 1,600,000-token experiment authorization and the prior uncertain-usage hold. It reserved at most 100,000 additional generation tokens and 100,000 Judge tokens. The frozen v10 backend reused all 31 checkpoints. An explicitly recorded external guard stops application retries on a renewed hard billing-cap error; successful generation prompts, schemas and checkpoint fingerprints were unchanged.

The [verified continuation](billing-resumption/status.json) used **14,767 provider-reported tokens**, with no estimated new usage or retries:

| New task | Tokens | Result |
|---|---:|---|
| Patch `TC-074` | 5,426 | Saved |
| Verify `FIND-001` | 4,118 | Model marked resolved |
| Add scenarios for `REQ-092` | 1,091 | Three scenarios saved |
| Write their tests | 1,731 | Three tests saved |
| Verify `FIND-002` | 2,401 | Model marked unresolved |

The retained bundle contains 92 requirements, 96 scenarios and 96 tests, and passes deterministic reference/structure validation. Its semantic status remains unresolved, so the runner correctly retains a failed result and does not purchase Judge evaluation. `FIND-002` requests a state-transition scenario for a non-super user denied permission to resume another user's stopped job. New `TC-096` tests permission denial, but its linked scenario is classified `negative`. The verifier records only resolved/unresolved IDs, without a rationale; the cause of rejection cannot be established from that output. Do not assume the type label is the only problem or relabel the result as accepted.

The first repair also illustrates why a model acceptance flag is insufficient evidence: `TC-074` still combines job-queue observation and email notifications in one expected result, although the verifier marked the original finding resolved. A source-based human review must determine whether its scenario and requirement links now support the behavior. More cases and accepted model flags do not establish quality preservation.

The harness's final accounting assertion exposed another limitation: an interrupted parent has no aggregate diagnostics, so the frozen runner reported only that direct parent's 12,497 estimated tokens as inherited cost, omitting the original 189,102-token trial. The terminal result had already been saved. Read-only verification traversed both ancestor records and reconciled **201,599 inherited accounted tokens**, **216,366 accounted workflow tokens including this retry**, and **228,863 including the uncertain hold**. The original raw result and all 15 preceding records remain unchanged; the corrected accounting is explicit in the status report. The frozen harness is retained as executed and must not be rerun to obtain a different result.

No second recovery or evaluation was purchased after this semantic failure. The next technical work is to make failed repair verification actionable, constrain gap additions to the missing behavior, and make inherited-cost reporting follow durable ancestor records when aggregate diagnostics are absent. Any changed generation protocol needs a separately recorded experiment; it cannot retroactively convert these failures into successful trials.

## Human review and what remains

The [original blinded review package](human-review-package.zip) preserves the two matched suites. The [latest supplementary package](billing-resumption/human-review-package.zip) pairs the same staged baseline with the 96-test continuation draft. Each includes the approved source/catalog, readable artifacts, instructions and separate blank R1/R2 forms, excluding costs, model identities and private mappings. The optimized suite remains a draft. Use [the latest review instructions](billing-resumption/blind/START-HERE.md) for the recovered artifacts; do not combine these ratings with the original matched comparison without identifying which artifact version was reviewed.

Two actual people must rate independently and disclose prior exposure. Check source coverage, supported expected outcomes, executability, duplication and critical errors; retain their original scores separately. Do not replace these ratings with model judgments. No human-approved cost or quality-preservation conclusion can be calculated yet.

Provider access is restored. Remaining work is to resolve the failed semantic verification with source-backed evidence, perform independent evaluation if generation passes, and obtain the two human reviews. Any further completion must keep the original matched budget failure in the report and include all inherited costs. A claim about unseen documents would need a separately frozen study on unused approved sources. The current assignment can already report a reproducible exploratory result: fewer serialized characters improved task progress, but total-budget completion and quality preservation were not demonstrated.
