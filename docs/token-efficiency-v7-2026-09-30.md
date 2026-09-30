# Post-pilot fixes — 30 September 2026

The next implementation is `research-core-v7`, with evaluator `coverage-v6:coverage-catalog-v1:gemini:gemini-3.6-flash:medium`. This version addresses observed failure mechanisms; it has not established quality-preserving token savings.

**Later live result:** both development trials failed (invalid evidence and budget exhaustion). They spent 209,891 additional tokens; cumulative spending is 732,219 of 1,600,000. See the [live validation report](v7-live-validation-2026-09-30/README.md). The implementation-stage measurements below are preserved as recorded before those trials.

## Changes

- The source-gap audit explicitly lists owned chunks and requires every added candidate to cite owned evidence. Neighboring source remains read-only. Invalid task output gets one scoped correction attempt; a second invalid result fails and preserves validated checkpoints. Transport/configuration failures do not receive this semantic retry.
- Curator, scenario and writer handoffs contain exact cited spans plus surrounding source windows. Extraction, source-gap audits and Critic checks still read their full relevant source chunks. Numeric operators and negation are preserved verbatim. Unrelated dynamically added citation spans are omitted from these scoped handoffs.
- Curator batches grow only while the complete request fits, up to 48 candidates. The global curator index was removed because only the supplied detailed peers are legal merge targets. Writer batches allow up to eight scenarios and Critic batches up to twelve requirements; output artifacts are not capped to arbitrary counts.
- Planning and execution share the same size calculation, including JSON escaping, schemas, role instructions and bounded correction overhead. The bound remains an engineering character limit, not a model-context guarantee. Oversized or truncated tasks can split once; an indivisible oversized task fails before a paid call.
- Evaluation now includes preconditions and test data, which were previously omitted. Repeated source references are serialized once per evaluation request. Judge response headroom rises from 8,000 to 16,000 tokens; medium thinking and the 100,000-token evaluation ceiling remain unchanged. Catalogs/scores from the old evaluator retain their original version.

## Verification and measurements

The full existing suite passed **570 tests**, including dedicated-database persistence checks; three existing pytest collection warnings remain. Regressions cover neighbor-only citations, one successful or exhausted correction, exact evidence windows, compact curator batching, escaped request overhead and evaluator test conditions.

The offline replay uses the previously inspected Telescope development artifacts: 283 candidates, 230 requirements, 62 scenarios and 63 tests. It makes no model calls. Curator decisions retain all candidates identically in both versions solely to compare serialized requests; they are not new model decisions or reference annotations.

| Stage | v6 planned tasks | v7 planned tasks | v6 request characters | v7 request characters | Reduction |
|---|---:|---:|---:|---:|---:|
| Curator | 16 | 9 | 929,883 | 370,333 | 60.17% |
| Test writer | 21 | 9 | 416,249 | 299,354 | 28.08% |

All measured v7 curator/writer requests are below 48,000 characters before runtime splitting. These counts exclude generated outputs, retries and actual tokenizer behavior. They cannot be reported as measured token savings. The fixed-batch evaluator payload changed only from 1,106,241 to 1,103,725 characters while adding previously omitted semantic inputs; no material evaluator cost reduction is established.

Reproduce with `PYTHONPATH=src .venv/bin/python docs/measure-efficiency-v7.py /tmp/citd-phase01-telescope-trace.json`. The replay verifies archived source hashes; the trace hash and measurements are in [the JSON result](token-efficiency-v7-offline-2026-09-30.json).

## Preserved experiment and remaining work

The eight v6 trial records and their results were not rerun or rewritten. The exact implementation files were archived and verified in `benchmark-pilot-2026-09-29/frozen-implementation.tar.gz`; the deployed pilot image is retained as `citd-pilot-v6:20260930`. The original manifest intentionally does not validate against changed v7 code.

**Before the subsequent live validation, additional Gemini spending was zero.** At that point, total accounted authorization was 522,328 of 1,600,000 tokens, leaving 1,077,672. Human ratings are still pending in [the blinded review package](benchmark-pilot-2026-09-29/results/blind/START-HERE.md).

A full pipeline may still exceed a chosen budget. No calibrated stage-budget allocation or selective-audit policy has been introduced. Source windows and lexical selection of eight detailed peers can miss distant dependencies/duplicates; full-source audits reduce but do not prove absence of that risk. Larger batches may need truncation retries. Live cost, completion and quality validation is still required before claiming the pilot failures are resolved in production. Use a new versioned experiment and new held-out documents after development/calibration; preserve these failed pilot outcomes.

The changes test the communication-redundancy principle studied in [AgentPrune](https://arxiv.org/abs/2410.02506), without implementing its graph-pruning algorithm or inheriting its results. Scoped evidence is motivated by [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/), whose tasks differ from this application. Retain simple baselines and human review as discussed in [the pilot research report](benchmark-pilot-2026-09-29/cost-and-quality-report.md).
