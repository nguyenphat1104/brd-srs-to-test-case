# Token efficiency: implemented changes and experiment protocol

Updated 29 September 2026. This note accompanies `research-core-v6`, schema v3 and the coverage-v5 evaluator. It supersedes the implementation status, not the historical measurements, in the earlier assessment.

**Latest empirical status, 30 September:** [v12 repair reassessment and evaluation](v12-repair-evaluation-2026-09-30/README.md) completed with 0.914 automated F1 for the saved 96-test suite. The staged baseline scored 0.709, but cumulative accounted cost is 287,562 versus 64,520 tokens. This is an adaptive supplementary result, not a replacement for the original matched budget failure. Both findings now have source-backed model explanations; ancestry accounting is fixed and deployed. All 587 tests passed. Conservative remaining authorization is 142,597 tokens. The two human reviews remain pending; the report and review package are prepared. Quality-preserving savings remain unproven. The rollout details below are historical.

## Delivery status

| User item | Delivered | Still requires evidence or input |
|---|---|---|
| 1. Deploy the corrected release | The first release was built and deployed locally, with an image rollback tag and database backup. Historical Telescope usage remains 973,822. | Final deployment verification is recorded below. |
| 2. Compact evidence and handoffs | Stable source-span IDs, exact citation reconstruction, one extraction owner per chunk, boundary context, source gap audits, retain/merge/reject curation, compact requirement index. | Held-out cost and omission measurements. |
| 3. Bound generation and review | Scoped scenario/writer/Critic tasks, one split for oversized/truncated tasks, targeted repairs, gap additions and verification, bounded independent catalog/mapping tasks. | Live provider performance and quality comparison. |
| 4. Durable recovery | Immediate PostgreSQL checkpoints; matching source/configuration/code/upstream fingerprints; linked recovery runs; process-level database leases; idempotent request IDs; preserved original records and drafts. UI recovery action is tested. | A paid recovery benchmark, if desired, must be separate from fresh repetitions. |
| 5. Research comparison | Frozen manifests, randomized four-condition schedule, total budget gate, independent catalog import/approval, blinded exports and rating template, separate phase accounting, paired document-level bootstrap. | User-selected held-out PDFs, total budget, supervisor-approved quality/error criteria, actual runs, two independent human ratings and analysis. |

No model API call was made during this implementation. Offline tests use scripted providers and the dedicated `brd_srs_test` database. Passing these tests establishes software behavior, not preserved generation quality.

## Architecture and bounds

The default centralized pipeline is:

`source chunks → owned extraction → source gap audit → curator decisions → scoped scenarios → scoped tests → deterministic validation → scoped Critic → targeted correction/gap addition → verification`

Each validated task is persisted immediately. Independent coverage evaluation follows generation and cannot erase a completed suite on an expected evaluator failure. Public artifacts still contain full source references. Models exchange short evidence IDs where the new wire schemas apply; Python resolves them to exact source spans before validation and export.

The efficient profile uses 6,000-character extraction groups, up to 18 candidates per curation group, up to 12 requirements per scenario group, up to three scenarios per writer group, and up to six requirements per Critic group. Actual prompt size can make groups smaller. The input policy is a 48,000-character engineering bound including schema, setup and a reserve for short schema-repair instructions. It is **not** a provider tokenizer count or a Gemini context-window guarantee. Bounded outputs are capped at 8,000 tokens or a lower configured role maximum. An oversized or truncated multi-item task may split once; an indivisible or still-oversized child fails explicitly with saved work intact. Transport retry and schema-repair attempts remain visible in the call ledger.

These are per-task size controls, not limits on how many supported requirements or tests a document may contain. Source content is not truncated to meet the limits. Missing or ambiguous source obligations remain explicit unresolved failures.

The evaluator uses a Cartesian set of bounded test/catalog groups: every test is assessed against every catalog unit, and mapping IDs are unioned deterministically before scoring. A test may appear in multiple calls when the catalog is large. This avoids top-k retrieval omissions at the cost of extra evaluation calls. The same evaluator protocol applies to all experimental conditions.

Checkpoint reuse requires a matching complete input/configuration fingerprint and revalidation. Recovery creates a new run and reports current-attempt tokens separately from inherited attempt costs. Recovering a completed generation after a Judge failure requires no generation calls. A new independent repetition receives a new request ID and does not inherit checkpoints.

## Measured offline effects

The previous targeted-repair replay reduced the three-finding repair payload from 436,699 characters to 39,353 characters including verification, approximately 91%. This was a scripted replay of the historical failure, not a newly generated suite or token measurement.

The new [serialization measurement](token-efficiency-phase234-offline-2026-09-29.json) uses the saved Telescope draft: 230 requirements, 62 scenarios, 63 tests and 99 chunks.

| Measurement | Characters |
|---|---:|
| Full artifact JSON | 275,927 |
| Artifact JSON using evidence IDs | 211,392 |
| Artifact-only reduction | 23.39% |
| Extracted source text | 152,543 |
| Rendered registry including IDs, headings and referenced spans | 206,152 |

All citations reconstructed exactly. The registry is larger than raw source text, and source gap audits add calls. Consequently the 23.39% figure must **not** be reported as overall token/cost savings. Scope reduction and avoiding full-bundle rewrites are expected to be the larger effects; actual end-to-end savings remain unmeasured.

## Deliberate limits

- Original source text is preserved. There is no new heuristic header/footer removal, avoiding accidental deletion of repeated requirements or table headings.
- Scenario tasks receive a compact global index, relevant evidence and known transitive dependencies. There is no extra global planning model pass. Dependencies discovered during curation can reference already retained requirements; forward dependency discovery and a comprehensive cross-batch contradiction pass remain limitations to evaluate on dependency-heavy documents.
- Detailed curator duplicate candidates are selected by simple lexical overlap, with at most eight prior detailed requirements. The full index remains available; unsupported merges are rejected. This can leave cross-batch duplicates and must be assessed under redundancy scoring.
- Boundary context consists of neighboring short source spans, not an automatic reconstruction of arbitrary multi-page tables or remote definitions. Such documents belong in the held-out challenge set.
- The efficient orchestration is sequential and bounded. No new worker service, vector database, learned compressor or model router was added. No new provider-side cancellation facility is claimed; a request already sent can still incur cost.
- The existing single-prompt architecture remains one generation prompt. The efficient staged path has durable stage checkpoints, but the research staged control uses the corrected previous staged protocol. Experimental conditions keep their defining architecture differences.
- Aggregate F1 and a passing citation validator cannot establish correct expected outcomes. The benchmark leaves human-approved cost and the quality-preservation conclusion unavailable until ratings and critical-error review are complete.

## Reproducible benchmark

Run commands from the repository root with `PYTHONPATH=src` and the existing virtual environment. Inventory, freeze, source-pack, catalog import and summarize make no model requests. Only `run` performs paid generation/evaluation.

1. Inspect [the document inventory](benchmark-document-inventory-2026-09-29.json): 62 PDFs were found, 61 could be extracted and one failed parsing. This is an inventory, not a declared held-out split. Telescope is development data and the freeze command rejects its hash.
2. Copy [the draft specification](benchmark-spec.draft.json). Select genuinely unseen documents, declare the total budget and per-run ceiling, model/thinking settings, repeats and any supervisor-approved F1 tolerance. Set `held_out_approved` to true only after that selection. A suggested six-document feasibility pilot with four conditions and three repeats is **72 runs**; it is not a statistical power calculation.
3. Freeze source hashes, effective protocol, code/dependency hashes and random seed:

   ```sh
   PYTHONPATH=src .venv/bin/python -m brd_srs_testgen.benchmark freeze /path/to/spec.json /path/to/manifest.json
   PYTHONPATH=src .venv/bin/python -m brd_srs_testgen.benchmark source-pack /path/to/manifest.json /path/to/annotations
   ```

4. Independently annotate source coverage units using the exported chunks and `catalog-schema.json`, before looking at generated condition outputs. Each file is a JSON object with a `units` array; each unit has a unique `CU-001`-style ID, title, description, unit_type and full verbatim source references. Review boundaries, negation, tables and exact expected behavior. Import without approval for a draft, or use `--approve` after human review:

   ```sh
   PYTHONPATH=src .venv/bin/python -m brd_srs_testgen.benchmark import-catalog /path/to/manifest.json D001 /path/to/D001-catalog.json --approve
   ```

   `DATABASE_URL` must point to the intended application database. Existing catalog contents are immutable. Manual annotation adds no model tokens, but report annotation time separately. If a model was used externally to draft annotations, report that one-time preparation cost separately too.

5. Run the frozen schedule with `DATABASE_URL` and `GEMINI_API_KEY` supplied through the existing environment:

   ```sh
   PYTHONPATH=src .venv/bin/python -m brd_srs_testgen.benchmark run /path/to/manifest.json /path/to/results
   ```

   Conditions are `single`, `staged`, `multi_corrected` and `multi_efficient`. They share generation model/thinking settings, generation budget and the fixed evaluator. No approved catalog is passed into generation. Each document/condition/repeat gets a deterministic unique request ID. Restarting the CLI reuses an already recorded trial rather than buying it again; an interrupted trial requires inspection. A fresh experimental repetition must have its own manifest/repeat identity.

   Before a new trial, the scheduler requires remaining accounted budget for its entire generation ceiling plus the 100,000-token Judge ceiling. Provider-reported and conservatively reserved unknown usage are distinguished in saved attempt records. A provider may report unexpected usage after a request returns; the accounting budget is not a guaranteed currency cap. No further trial starts after overspend.

6. Give raters the source materials, `blind/*.json`, rating instructions and `ratings.csv`. Keep `private/unblinding.json`, raw results and condition costs away from raters. Use two independent stable pseudonymous rater IDs. Copy the blank template once per rater before entering ratings; the exporter does not overwrite an existing ratings file. Rate coverage, groundedness, executability and redundancy on the existing 1–4 rubric; record critical expected-result, boundary and negation errors separately. Freeze ratings before unblinding and adjudication.
7. `summary.json` reports attempted generation/catalog/evaluation tokens, completion and unavailable-evaluation rates, latency, available F1 and exploratory paired document-level bootstrap intervals. Missing evaluations are not assigned zero. The automatic F1 interval statement is not a quality-equivalence conclusion. Human approval, inter-rater agreement, critical errors and final cost per approved suite still require the existing rating/adjudication workflow and analysis of the collected data. Use **all attempt spend**, including failed attempts, in the numerator.

Recovery and provider caching experiments are separate from the primary fresh-run comparison. Joint changes are a package treatment: do not attribute their combined effect to evidence IDs alone without an ablation. This delivery supplies the four-condition main comparison; per-component ablations are not yet implemented.

## Research basis

These are adaptations and experimental motivations, not replications or claims that the cited papers prove this implementation preserves quality.

- [AgentPrune](https://arxiv.org/abs/2410.02506) motivates reducing redundant agent communication. This implementation scopes handoffs; it does not learn a communication graph.
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) motivates testing focused evidence and omission safeguards. It does not establish a Gemini-specific context threshold.
- [Smit et al., ICML 2024](https://proceedings.mlr.press/v235/smit24a.html) motivates retaining strong simpler baselines rather than assuming more agents improve quality.
- [Chen et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.474/) motivates explicit judge-bias controls and independent human review.
- [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/) motivates paired analysis suited to the experimental unit. Repeats on one document do not replace independent documents; limited sample sizes must remain visible.

## Verification and deployment record

Final verification: **564 tests passed**, with three existing Pytest collection warnings about model class names. Tests ran under the application Docker runtime against the dedicated test database; no model API calls were made. `git diff --check` passed.

The new image `2fcf413c88e0` was deployed to the existing local service on 29 September 2026 at approximately 22:37 Singapore time. The container reported healthy and `/_stcore/health` returned `ok`. Ten relevant application/source/schema file hashes matched between the verified working tree and running container. The home page and historical Telescope result rendered in the browser; its 973,822 accounted tokens and 56 blackboard records remain readable.

The migration adds checkpoint columns and permits additional stage names; original runs are not rewritten. The new recovery feature requires fingerprinted checkpoints from this release. Legacy Telescope handoffs have no such fingerprints and are not automatically reused as if they had been validated by the new protocol.

Rollback artifacts: `citd-phase01:20260929` (previous app image), `citd-before-efficiency:20260929` (original app image), and `/tmp/citd-before-phase2345-20260929.dump` (database backup). These backups were retained, not restored. Detailed verification hashes are in [the deployment record](token-efficiency-deployment-2026-09-29.json).

The only pending execution is the research experiment and its human evaluation. The draft specification intentionally has no held-out documents and a zero total budget, so it cannot accidentally start paid trials.
