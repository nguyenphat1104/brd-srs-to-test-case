# Matched development comparison

The later user-authorized [v12 saved-draft reassessment](../v12-repair-evaluation-2026-09-30/README.md) is supplementary. Its modified verification context/schema, explicit artifact import and additional budget are frozen in a separate protocol; it does not change the matched methods or historical failures below.

This study asks whether the revised multi-agent workflow completes within a bounded budget and offers a useful quality/cost tradeoff against the simpler staged workflow. It does not assume that extra agents improve quality, or that a smaller serialized prompt preserves model behavior.

## Design frozen before paid calls

- One document: the previously approved Model Manager PDF, with its unchanged 91-unit source catalog. This document has already been used in development; it is not held out.
- Two conditions: staged and efficient multi-agent, one fresh run each, staged first. The order is fixed for a budget/feasibility gate, not a counterbalanced experiment.
- Both use Gemini `gemini-3.6-flash`, temperature 0, minimal generation thinking, the same input PDF, 200,000-token generation ceilings and `research-core-v10`.
- Both use the same `coverage-v7` Judge, medium thinking and a separate 100,000-token ceiling. The approved catalog contents are copied exactly into the new evaluator-version namespace.
- The original total authorization is 1,600,000 tokens. Prior work consumed 1,092,824. This comparison can spend at most the remaining 507,176; each next run requires enough remaining authorization to reserve its whole generation plus Judge ceiling. Failures remain in the denominator and cost ledger.
- If staged generation or evaluation fails, the optimized trial is not purchased. There are no automatic paid recovery runs. Source hashes, dependencies, implementation files and trial identities are frozen in `manifest.json` and `frozen-implementation.tar.gz`.

## Revised representation

Critic tables store column names once. When a test and its referenced scenario have exactly equal titles, preconditions, requirement IDs or source references, those fields are shared explicitly. Differences remain explicit. Offline checks reconstruct the original scoped artifacts, retain every assigned requirement and retain the complete global index in each review request. Full relevant source evidence remains visible. Review grouping allows up to 24 requirements but still applies the 48,000-character request bound and existing response/split/validation rules.

The Judge also uses tables. It groups the approved catalog using the actual mapping prompt and the largest test, and checks every test against every catalog group; there is no top-k filter. Up to 100 coverage units and 48 tests can share a request, subject to a separate 80,000-character request bound. The existing 16,000-token maximum response and 100,000-token cumulative Judge budget remain. Larger requests are an explicit changed condition: exact input reconstruction is not proof that the model attends to every field equally well.

The fixed v9 draft replay contains 99 requirements and 105 tests. Critic request characters fall from 381,676 to 233,983 (38.7%), with ten initial calls reduced to six. Judge request characters fall from 565,338 to 177,048 (68.7%), with 18 initial calls reduced to three. All 9,555 test–coverage-unit pairs remain assigned. These are **offline serialization measurements**, excluding live output, reasoning, corrections and repairs; they are not token savings or quality scores. The old and new paths are checked against the same saved artifacts.

## Outcomes and interpretation

Report generation completion, evaluation availability, per-phase accounted tokens, generation cost per completed suite, precision/recall/F1, and retained artifact counts. Model-reported usage and estimated reservations are distinguished. A failed or unevaluated run has unavailable F1, not zero. More generated cases and perfect internal traceability are not evidence of semantic coverage.

The Judge is independently prompted, but uses the same model family as generation. Its scores may share model biases; they do not replace source-based human review. R1 and R2 must be different people who rate blinded artifacts independently, using the existing four-point rubric for coverage, groundedness, executability and redundancy, plus explicit critical-error counts. Their original ratings must remain separate; later adjudication must not overwrite them. Prior exposure to the document or earlier results must be disclosed.

One reused document and one run per condition do not support significance, general superiority, non-inferiority or a quality-equivalence claim. No F1 tolerance was supplied or retroactively chosen. Human-approved cost remains unavailable until actual ratings and critical-error assessment exist. If a generalization claim is needed, a later study must select unused approved documents and freeze its conditions before examining their results.

## Research rationale

[AgentPrune, Zhang et al. (2024)](https://arxiv.org/abs/2410.02506) motivates investigating redundant inter-agent communication. This implementation uses deterministic serialization and grouping, not AgentPrune's learned graph pruning; none of its benchmark gains are claimed here.

[Lost in the Middle, Liu et al. (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/) demonstrates position effects in long-context retrieval tasks. It supports measuring the quality effects of our changed prompt sizes, rather than assuming that a large context window guarantees complete use of the supplied evidence.

[Should we be going MAD?, Smit et al. (ICML 2024)](https://proceedings.mlr.press/v235/smit24a.html) motivates a strong simple baseline and measured cost/quality tradeoffs. Their debate systems differ from this requirements-to-tests workflow; this is methodological motivation, not direct evidence of effectiveness here.

## Separately declared post-hoc recovery

The matched experiment ended with staged completion and an optimized budget failure at 189,102 generation tokens. The optimized run had completed its initial reviews, which retained two findings, but could not reserve the first repair request. Its result stays failed and is never replaced by a recovery result.

After inspecting this failure, a supplementary checkpoint-recovery protocol was saved under `recovery/protocol.json`, before new paid calls. A dry execution proved that all 31 validated tasks were reused and the first new provider request was the unfinished repair. The original settings and code remain identical for checkpoint validation; an external study restriction lowers the new generation ledger to 100,000 additional tokens, with the existing 100,000-token Judge limit. This reserves at most 200,000 additional tokens against the 253,554 then remaining.

This is an adaptive development extension, not a fresh run or a success at the original 200,000-token cumulative cap. Its complete workflow cost must include the failed parent's 189,102 tokens plus all new repair, verification and Judge calls. Only the child's new calls are added to overall project spending, so inherited cost is not counted twice. Any supplementary quality comparison with the staged suite is descriptive and must identify the different cumulative generation budgets. No further recovery is planned if this extension fails.

## Subsequent user-authorized billing resumption

The first recovery was blocked by a hard provider spending cap and stopped. After the user reported increasing that cap and explicitly requested another attempt, `billing-resumption/protocol.json` was frozen before new calls. This is a disclosed amendment to the earlier no-further-recovery decision, prompted by changed provider availability. It does not enlarge the original 1,600,000-token experiment authorization.

The new limit was 100,000 additional generation plus 100,000 Judge tokens, within the conservatively remaining 228,560. The prior 12,497-token interrupted-retry hold was retained. The frozen v10 runtime and settings reused all 31 completed tasks. An external harness guard makes explicit monthly/project spending-cap errors nonretryable; this failure-only change and the exact harness/image hashes are recorded in the protocol. The continuation links to the billing-blocked parent and preserves the entire failed ancestry.

Five generation calls used 14,767 provider-reported tokens. One repair finding remained unresolved; the continuation was retained as a semantic failure, without Judge calls or a subsequent paid recovery. The retained 96-test draft is separately available for human review. A post-run reporting assertion identified incomplete inherited costs in the frozen runner when the direct parent lacks aggregate diagnostics. Read-only verification traverses the recorded ancestry to include both parents; it does not rewrite their diagnostics or trial outcomes. See `billing-resumption/status.json` for the verified ledger and limitations.
