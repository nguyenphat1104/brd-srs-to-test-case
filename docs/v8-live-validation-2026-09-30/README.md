# v8 validation and v9 routing correction

**Later result:** [v9 live validation](../v9-live-validation-2026-09-30/README.md) stopped during Critic review at its budget guard after spending 186,956 tokens. Cumulative spending is now 1,092,824 of 1,600,000. The v8 and pre-validation v9 account below is preserved as historical context.

The combined generation step produced a complete structural draft for Model Manager, but the v8 run failed during Critic validation. It did not reach independent evaluation. The newly discovered routing defect is corrected in v9; that correction has local regression coverage but no new live validation. **Quality-preserving token savings remain unproven.**

## Implemented changes

The default efficient pipeline now generates executable tests and their scenarios in one request per requirement group. The model authors each design's title, objective, type, preconditions, data, steps and supporting evidence. Code assigns IDs and derives the linked scenario and test, sharing preconditions and citations exactly. There is no fixed total case count. Separate scenario and test functions remain for targeted gap repairs and the corrected baseline remains available.

Every request presents short labels such as `E1`, mapped back to stable evidence IDs before validation and checkpoint storage. Only labels supplied in that request are accepted. Literal source text is unchanged. An invalid label receives one correction attempt naming the invalid value and allowed range. Exact citation, source ownership and artifact-scope checks remain in place. This reduces identifier-copying complexity; it does not guarantee that a valid citation logically supports an authored claim.

The live run exposed another avoidable model choice: Critic returned a valid requirement ID with an incompatible repair role. In v9 the Critic no longer generates `responsible_role`. Code derives it from the target artifact type, then applies the existing target validation. Mixed artifact kinds require separate findings; unknown IDs remain errors. The finding, requested correction and evidence are not discarded or marked resolved by this routing step. Gap additions and semantic repairs still require verification.

## Frozen v8 development result

| Measure | Model Manager |
|---|---:|
| Canonical requirements saved | 94 |
| Scenarios saved | 99 |
| Tests saved | 99 |
| Generation tokens | 173,649 |
| Evaluation tokens | 0 |
| Charged calls | 31 |
| Automated F1 | Unavailable |

All extraction, source audits, curation and combined test-writing tasks finished. Four Critic batches were saved before a later batch failed, including its one correction attempt. The reported error was: `Finding FIND-001 assigns REQ-057 to the wrong responsible role scenario_architect.` This is an application validation failure, not a context-window failure or a budget rejection. The 99 tests remain a draft: structural completeness does not establish semantic completeness or quality.

| Stage | Accounted tokens |
|---|---:|
| Extraction | 24,266 |
| Source-gap audits | 16,861 |
| Curation | 29,399 |
| Combined scenario and test writing | 49,595 |
| Critic, incomplete | 53,528 |
| **Total** | **173,649** |

The predeclared gate required Model Manager to complete generation and evaluation before purchasing DigitalHome. It failed that gate, so **DigitalHome was not run**. Do not report this as two completed trials. The script reserves up to 600,000 tokens for both documents, but spent only 173,649 before stopping; it did not automatically retry the failed run.

Prior preparation and experiments used 732,219 tokens. **Total now: 905,868 of 1,600,000; remaining authorization: 694,132 tokens.** The subsequent v9 routing patch made no paid model calls.

## Interpretation for the assignment

The v8 run passed the extraction point that failed in v7 and produced a draft through the combined writer. This is a development observation on one reused document, not proof of a general reliability improvement. Short evidence labels and joint generation changed together, so this run cannot isolate their effects.

Model Manager's 49,595-token combined-writing cost must not be subtracted from DigitalHome's earlier 74,983-token scenario-stage cost to claim a percentage saving: these are different documents, different generated artifacts and incomplete pipelines. A defensible savings claim still requires matched completed runs with the same budgets, evaluator and quality criteria, plus human reviews.

The design follows the communication-redundancy motivation of Zhang et al.'s [AgentPrune paper](https://arxiv.org/abs/2410.02506), without implementing its pruning algorithm or inheriting its benchmark gains. Focused source evidence is motivated by Liu et al.'s [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/); those experiments do not establish performance on these requirements PDFs. Keep the simple staged baseline in the next comparison, consistent with the need to test whether extra agents justify their cost discussed by [Smit et al.](https://proceedings.mlr.press/v235/smit24a.html). Their work concerns debate, which differs from this pipeline.

## Verification and remaining work

The v8 suite passed 574 tests before the live run. The final v9 verification and deployment are recorded separately in `v9-deployment.json`. New checks cover request-local evidence labels, unchanged literal source text, invalid-label correction, split-task scenario/test links, partial checkpoint retention, and deterministic repair ownership without accepting unknown targets.

`verification.json` checks the frozen v8 manifest and source archive, the approved PDFs/catalogs, immutable database results, preservation of all ten previous trials, every charged call, both budgets and the absence of a second-document run. The post-trial v9 changes are not retroactively included in the v8 experiment. `results/blind` contains the draft separately from the original pilot's review package.

Remaining work is end-to-end validation of the final routing correction, demonstrated completion within the generation budget, a controlled comparison on genuinely unused approved documents, and two independent human reviews. The original [blinded human-review package](../benchmark-pilot-2026-09-29/human-review-package.zip) remains available; no human ratings have been invented. Further paid runs should target the remaining validation gap rather than silently repeat the failed experiment under its original manifest.
