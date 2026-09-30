# Token-efficiency feasibility pilot — 30 September 2026

The eight planned trials are recorded: **4 completed generation runs**, **4 available automated coverage scores**. Quality preservation is **not established**. Two independent human ratings remain pending.

## Cost and outcomes

Total accounted Gemini tokens: **522,328 / 1,600,000**. This includes **23,613** catalog-preparation tokens and **498,715** comparison tokens, including unsuccessful calls. Unspent authorization: **1,077,672** tokens. The budget is a ceiling, not a target to exhaust.

| Document | Condition | Generation result | Generation tokens | Evaluation tokens | Saved test cases | Available F1 |
|---|---|---|---:|---:|---:|---:|
| D001 | multi_corrected | failed | 22,735 | 0 | 0 | Unavailable |
| D001 | multi_efficient | failed | 19,567 | 0 | 0 | Unavailable |
| D001 | single | completed | 14,357 | 53,679 | 11 | 0.600 |
| D001 | staged | completed | 37,860 | 79,698 | 18 | 0.704 |
| D002 | multi_corrected | failed | 23,500 | 0 | 0 | Unavailable |
| D002 | multi_efficient | failed | 86,017 | 0 | 0 | Unavailable |
| D002 | single | completed | 28,819 | 24,407 | 9 | 0.303 |
| D002 | staged | completed | 62,972 | 45,104 | 16 | 0.539 |

D001 = Model Manager (91 approved coverage units); D002 = DigitalHome 1.3 (84 units). All four conditions reuse the same approved catalog for each document. The catalogs are withheld from generation. Missing evaluations are **not F1 = 0**. Counts of saved tests do not certify correctness. Per-condition totals and exploratory paired statistics are in `results/summary.json`; per-stage costs and trial-level precision/recall are in `results/analysis.json`.

Comparison accounting: paid calls = 56; locally blocked calls = 1; incomplete responses = 5. Reported components: 300,426 input, 123,704 visible output and 74,585 thinking tokens. Estimated charges: 0. These are application/provider token-accounting records, not a currency invoice. Cumulative run tokens are not the size of one context window.

## Recorded failures

- **D002 / multi_efficient — budget_exhaustion:** Need 17861 tokens; 13983 remain.
- **D001 / multi_efficient — semantic_validation:** Gap candidate has no owned evidence.
- **D001 / multi_corrected — schema_failure:** Provider stopped at the output token limit before completing structured data.
- **D002 / multi_corrected — schema_failure:** Provider stopped at the output token limit before completing structured data.

The efficient DigitalHome run used 28,730 tokens for extraction, 23,430 for source audits and 33,857 for curation before the next reservation was blocked. It retained 62 requirements but no scenarios or tests. This is an incomplete pipeline, not a low-cost success. The efficient Model Manager run stopped when a gap candidate lacked evidence owned by its assigned extraction task; successful structured responses did not guarantee semantic validity.

On Model Manager, the staged condition used 37,860 generation tokens and 79,698 evaluator tokens. Evaluator spending therefore exceeded generation spending. Evaluator reasoning and retries are part of the measured cost, not free infrastructure. The earlier catalog attempt also exhausted an 8,000-token output allowance while reasoning consumed most of that allowance. Google's [thinking documentation](https://ai.google.dev/gemini-api/docs/thinking#token-limits-and-max_output_tokens) explains that the response cap includes thinking tokens and can truncate an answer without reducing reasoning effort.

## Interpretation and next engineering experiment

These are observations from a feasibility pilot. They do not validate the label “efficient,” prove that every simpler architecture is better, or establish savings against the earlier 973,822-token Telescope incident: the documents and outcomes differ. A method that fails early cannot win on token count alone. Report completion rate and cost per usable, human-reviewed suite alongside coverage.

The next version should simplify the mandatory pipeline before adding more agents:

1. **Reserve enough budget to finish.** Estimate extraction, writing and verification costs before generation. Allocate stage allowances and report whether the document can fit. A global ceiling currently prevents continued spending but can stop after expensive upstream work with no tests.
2. **Reduce repeated payloads and mandatory passes.** Pass cited spans with necessary surrounding context; retain full documents locally. Investigate a staged generator with targeted audits for ambiguous, uncovered or conflicting obligations. Compare it with the current mandatory per-group audit/curation path. Do not assume that smaller batches mean fewer total tokens.
3. **Repair the failing unit.** For evidence-ownership violations, use one bounded local correction referencing the actual validation error and allowed evidence IDs. Keep the semantic check; do not relax it merely to make runs pass. Preserve valid checkpoints and make incomplete output visible.
4. **Tune the evaluator separately.** Test lower thinking effort and adequate response headroom on development data; compare its mappings with human judgments before freezing a new evaluator. Reduce repeated catalog serialization. Any candidate-retrieval shortcut must measure missed matches; skipping difficult coverage pairs would inflate apparent efficiency.
5. **Run a new versioned study.** Keep these eight outcomes unchanged. Use development documents for fixes, then freeze code/settings and use new held-out documents, repeated runs and two actual human raters. These two documents have now been inspected and must not be described as untouched held-out data for later tuning.

These are proposed engineering changes, not changes applied during this pilot. Quality and cost effects need measurement.

## Research basis

- Zhang et al., [*Cut the Crap / AgentPrune*](https://arxiv.org/abs/2410.02506), studies redundant communication in multi-agent systems. It motivates testing fewer repeated messages and unnecessary interactions. This application does not implement AgentPrune and cannot inherit the paper's savings.
- Liu et al., [*Lost in the Middle*](https://aclanthology.org/2024.tacl-1.9/), finds that relevant-information position affects performance on long-context tasks. This motivates scoped evidence and explicit coverage checks; a large advertised context window does not guarantee effective use of all evidence.
- Smit et al., [*Should we be going MAD?*](https://proceedings.mlr.press/v235/smit24a.html), reports that multi-agent debate does not reliably beat simpler alternatives under its tested protocols. That supports retaining competitive simple baselines; its debate findings are not direct evidence about this requirements pipeline.
- Chen et al., [*Humans or LLMs as the Judge?*](https://aclanthology.org/2024.emnlp-main.474/), documents biases in human and LLM judgment. Automatic F1 should therefore remain one measure alongside independent human review and critical-error checks.
- Dror et al., [*The Hitchhiker’s Guide to Testing Statistical Significance in NLP*](https://aclanthology.org/P18-1128/), motivates choosing analysis suited to the study design. Two documents with one repetition and no predeclared noninferiority margin cannot establish general quality equivalence; paired intervals here are exploratory.

## Controls, provenance and remaining work

Generation uses Gemini 3.6 Flash with minimal thinking; the shared Judge uses the frozen medium-thinking evaluator. Each trial has a 100,000-token generation allowance and a separate 100,000-token Judge allowance. Execution order was randomized with seed 20260929 before results. The source catalogs were assistant-drafted, exact-citation checked and explicitly approved by the user; approval is not evidence that they are a perfect independent gold standard.

Before any trial, runtime preflight found two dependency differences from the local manifest: google-genai 2.17.0 → 2.19.0 and pypdf 6.15.0 → 6.16.2. The deployed runtime was frozen after confirming no old/new trial records existed and revalidating all 175 source references. Source code, catalogs, model settings and trial order were unchanged. Both manifests and the preflight audit are retained. An initial launch was blocked by automatic approval review; the user then explicitly authorized the Gemini data transfer before execution.

The blinded package is `results/blind/START-HERE.md`. Keep condition mapping, costs, logs and this report away from raters until their original ratings are frozen. Anyone exposed to trial outcomes in this task should disclose that exposure; preferably select two raters who have not viewed it. Collect separate coverage, groundedness, executability and redundancy scores, plus critical errors. Calculate agreement and record adjudication separately without replacing initial ratings. No human scores have been fabricated. Item 5 remains incomplete until these ratings and the final analysis are finished.
