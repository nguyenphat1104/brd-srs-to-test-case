# v9 development validation: budget failure during review

**Later work:** the [v10 comparison](../v10-comparison-2026-09-30/README.md) reduced review/evaluation payloads, recorded a completed staged baseline and a new optimized budget failure, and prepared blinded reviews. A subsequent recovery encountered the provider spending cap. The v9 result and budget figures below remain historical.

The v9 Model Manager run saved 99 requirements, 105 scenarios and 105 tests, then stopped during Critic review. **It did not complete generation or reach independent evaluation. Quality-preserving token savings are still unproven.** The predeclared gate prevented a DigitalHome run. No further paid run was started after this failure.

The run used the previously approved Model Manager PDF and catalog, `research-core-v9`, Gemini `gemini-3.6-flash`, minimal generation thinking, a 200,000-token generation ceiling and a separate 100,000-token Judge ceiling. This is a development regression on a reused document, not a new held-out result. Its immutable run ID is `228f8d258152-b992e6e7f67752d1980370f383c70209`; the frozen manifest hash is `44c18e67f818b6f6ca0f0816fe4b5b586fb12fa320a3b38221e15f2032d294a9`.

| Stage | Accounted tokens |
|---|---:|
| Extraction | 24,618 |
| Source-gap audits | 17,318 |
| Curation, including one semantic correction | 43,280 |
| Combined scenario and test writing | 56,206 |
| Critic, five completed batches | 45,534 |
| **Generation total** | **186,956** |
| Independent evaluation | 0 |

There were 32 charged provider calls and one zero-cost blocked reservation, with no estimated usage. The next request required a reservation of 17,619 tokens, while 13,044 remained. This is an application budget rejection, not a Gemini context-window error. The largest reported input for any individual call was 10,020 tokens; cumulative usage was 125,792 input plus 61,164 output tokens across calls. These sums must not be interpreted as a single context window.

Five Critic batches returned no findings and passed validation. The v8 repair-role error did not recur in those batches, but they produced no findings and therefore did not exercise routing of a real finding. Local regression checks cover that routing change; this live result does not establish its behavior across a completed review/repair cycle. The retained draft has valid structural links and citations, but those checks do not establish source completeness or correct expected outcomes. F1 is **unavailable**, not zero.

## Remaining budget

| Work | Accounted tokens |
|---|---:|
| Catalog preparation | 23,613 |
| Original eight v6 comparisons | 498,715 |
| Two v7 development trials | 209,891 |
| One v8 development trial | 173,649 |
| One v9 development trial | 186,956 |
| **Total** | **1,092,824** |
| **Remaining from 1,600,000** | **507,176** |

The 600,000-token maximum in this validation's protocol was a reservation limit, not the amount spent. DigitalHome was not purchased, and no Judge call was made. Earlier experiments and this failure remain unchanged.

## Offline diagnosis

[The reproducible payload replay](measure_review.py) used the saved draft and exact frozen source/dependencies, with networking disabled and no API credentials. It calls no provider. It enumerates initial review groups using in-memory control-flow placeholders; those placeholders are not review judgments or experimental results. [Measured payload details](review-payload.json) show:

| Measure | Value |
|---|---:|
| Extracted source characters | 21,358 |
| Planned initial Critic batches | 10 |
| Completed / still pending | 5 / 5 |
| Serialized messages across all planned batches | 368,226 characters |
| Serialized response schemas across those batches | 13,450 characters |
| Artifact JSON across batches | 167,529 characters |
| Full requirement index, repeated ten times | 110,400 characters |
| Evidence text across batches | 48,738 characters |
| Test-case appearances / unique test cases | 106 / 105 |

The component character counts are measured before embedding/escaping into messages; they are diagnostic quantities, not an additive token ledger. Five actual Critic calls consumed **45,454 input tokens for 80 output tokens**. Input repetition dominates this stage. Simply reducing its output allowance would not remove that input cost, and raising the ceiling would not improve efficiency.

The next optimization should address the repeated whole-document index and duplicated scenario/test fields. A smaller source file can still produce large cumulative usage when many stages repeatedly serialize expanded artifacts. This run supports that mechanism; it does not establish a percentage saving against the original Telescope incident, which used a different document and pipeline.

## Next strategy and acceptance gates

1. **Keep the staged pipeline as the practical baseline.** It completed both original pilot documents, with higher automated F1 than the single-call condition in that pilot. Do not promote the efficient multi-agent profile as superior based on incomplete drafts. The original evaluator and ceilings differ from v9, so a direct cross-version quality comparison would be invalid.
2. **Measure lossless payload reductions before another paid trial.** Share identical scenario/test fields and source references once, with explicit references and an exact reconstruction check. Preserve distinct objectives, preconditions, data, steps, expected results and artifact IDs. First measure on this saved bundle; then validate model behavior separately. Fewer serialized characters alone do not prove equal quality.
3. **Separate local review from global checks.** Local review should carry assigned artifacts, required dependencies and supporting source. Cross-group duplication and contradictions need a deliberate global pass or tested retrieval, rather than silently removing the full index. Full extraction and source audits must still cover every owned source section. Compare this change separately so omissions and cross-group errors remain observable.
4. **Plan the remaining work before spending.** Estimate generation, all review groups, possible corrections/repairs and independent evaluation against the remaining budget. Stop early with a retained draft when the planned work cannot fit. Keep the existing hard reservation guard. Do not label an unreviewed draft completed or automatically raise its cap.
5. **Freeze a feasible matched comparison, then obtain actual human ratings.** Hold model, evaluator, source catalogs and budgets constant across conditions; include failed attempts in cost/completion results. Treat Model Manager and DigitalHome as development data after tuning. A claim about unseen documents needs newly selected, approved sources. If the remaining budget cannot support that study, submit the current work as an exploratory engineering study with explicit negative results and limitations. A successful optimization is not required to report a valid experiment honestly.

These are the next acceptance gates, not changes already deployed. The v9 implementation and its experimental records remain frozen. Further paid development repetitions should follow a measured payload change and a new explicit protocol, rather than repeat this failure unchanged.

## Research basis and limits

Zhang et al.'s [AgentPrune (2024)](https://arxiv.org/abs/2410.02506) studies redundant communication in multi-agent systems and prunes communication graphs. It motivates measuring repeated handoffs here. This application has not implemented that algorithm, and cannot claim its reported benchmark savings.

Liu et al.'s [Lost in the Middle (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/) shows position-sensitive performance in long-context question answering and key-value retrieval. It motivates focused evidence with explicit source coverage, not a claim that shorter prompts automatically preserve quality on requirements documents.

Smit et al.'s [Should we be going MAD? (ICML 2024)](https://proceedings.mlr.press/v235/smit24a.html) finds that multi-agent debate does not reliably outperform simpler alternatives without suitable configurations. Our pipeline is not a debate system; the relevant experimental lesson is to retain strong simple baselines and test whether additional model work justifies its cost.

## Verification and handoff

[Verification](verification.json) confirms that the saved result equals PostgreSQL, all eleven earlier trial records are unchanged, source/catalog/archive hashes match, per-call costs reconcile, the total stays within authorization, and the skipped second trial does not exist. The measurement script also verifies the deployed implementation/dependency snapshot before replay. No production source was changed by this validation/reporting step.

The new partial draft is under `results/blind`, separate from the original pilot. The original [human-review package](../benchmark-pilot-2026-09-29/human-review-package.zip) and [review instructions](../benchmark-pilot-2026-09-29/results/blind/START-HERE.md) remain available. R1 and R2 ratings have not been supplied; no scores or human approval have been inferred. The assignment's quality-preservation gate remains open.
