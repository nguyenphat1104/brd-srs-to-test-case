Implementation plan: reduce token consumption while preserving test-suite quality

Prepared 29 September 2026. Updated 30 September: Phases 0–4 have an implemented, offline-tested delivery. Phase 5 now includes a completed supplementary reassessment of the saved 96-test suite: automated F1 0.914 versus the staged baseline's 0.709, at cumulative accounted cost 287,562 versus 64,520 tokens. The original matched budget failure remains unchanged. Repair explanations and inherited-cost accounting are fixed and deployed; 587 tests passed. Two actual human reviews remain pending, with a prepared blinded package. Quality-preserving savings remain unproven. Conservative remaining authorization is 142,597 tokens. See the [latest verified results and remaining human gate](../../v12-repair-evaluation-2026-09-30/README.md), [implementation and protocol](../../token-efficiency-rollout-2026-09-29.md), and [first release results](../../token-efficiency-implementation-2026-09-29.md). The sections below preserve the original plan.

The goal is a reliable, measurable BRD/SRS-to-test pipeline whose cost comes from useful reasoning and output, with failures recoverable at the task that failed. Preserve the three assignment conditions: single-prompt, staged single-agent and centralized multi-agent. Begin with the current Python/Pydantic/PostgreSQL implementation and its existing tests. A new agent framework, vector database or orchestration service is unnecessary for this scope.

The [assessment](/Users/jun/projects/citd-final/brd-srs-to-test-case/docs/token-efficiency-assessment-2026-09-29.md) and [saved measurements](/Users/jun/projects/citd-final/brd-srs-to-test-case/docs/token-efficiency-audit-2026-09-29.json) establish the incident baseline: 973,822 recorded tokens, 230 requirements, 62 scenarios and 63 tests before final repair failed. The original run is an observed failure case, not evidence of baseline quality.

**Additional findings change the implementation priorities.**

| Finding | Evidence | Required improvement |
|---|---|---|
| Citation matching loses meaningful symbols | `_evidence_key` removes punctuation. An offline probe confirmed that a source containing `voltage < 5 V` accepts a cited quote containing `voltage > 5 V`. | Preserve comparison operators, signs, decimal values and units during validation; do not confuse a matching word sequence with an exact quote. |
| Judge failures can destroy completed generation | Offline probes injected `BudgetExceeded` and incomplete `StructuredOutputError` during catalog extraction. Both caused the runner to return a failed run with `bundle=None`. | Separate generation outcome from evaluator outcome; preserve generated artifacts and evaluator costs. |
| Current repair forbids requested citation-only fixes | The preceding audit reproduced rejection of the Critic's requested `SCN-013` correction. | Explicitly distinguish citation, relationship and semantic repairs. |
| Source coverage and trace links are different | Existing validation counts requirement IDs linked to scenarios/tests. A linked requirement can still lack an adequate test action or expected outcome. | Keep traceability checks, but evaluate behavior and expected-result correctness separately. |
| Checkpoints are currently finalized together | `record_stage` appends in memory; `RunRepository.finalize` inserts stage outputs at the end. | Persist completed tasks immediately so process termination does not lose all unfinalized work. |
| Development examples risk contaminating the experiment | Telescope has already been inspected and used to diagnose the implementation. | Use Telescope for development and recovery validation; use other documents for held-out claims. |

The first two new findings were reproduced without a database or model request. Relevant code: [citation matching](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/documents.py:28), [Judge exception boundary](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/runner.py:811), [stage persistence](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/storage.py:408).

**Deliver in six phases, each with an observable acceptance gate.**

| Phase | Outcome | Dependency |
|---|---|---|
| 0. Freeze and measure | Trustworthy per-call accounting and reproducible versions | First, before any new paid benchmark |
| 1. Correctness and local repair | A small correction cannot erase a completed suite | Phase 0 for measured pilot results |
| 2. Evidence and compact handoffs | Reduce repeated input and citation regeneration | Phase 1's stricter evidence checks |
| 3. Bounded generation and review | Large documents do not require one oversized JSON response | Phase 2's evidence references |
| 4. Durable recovery | Retry only invalidated or incomplete tasks | Stable task contracts from Phases 1–3 |
| 5. Research evaluation | Defensible cost/quality comparison | Frozen implementation and approved evaluation data |

Implement and check one phase at a time. The first useful release is Phases 0–1. Do not bundle optional model-routing or agent-removal experiments into that release.

**Phase 0: establish what each request costs.**

Work in [providers.py](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/providers.py), [pipelines.py](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/pipelines.py), [models.py](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/models.py), [storage.py](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/storage.py), [schema.sql](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/schema.sql), [runner.py](/Users/jun/projects/citd-final/brd-srs-to-test-case/src/brd_srs_testgen/runner.py) and [app.py](/Users/jun/projects/citd-final/brd-srs-to-test-case/app.py).

- Add one append-only call-attempt record keyed by run, stage, task and attempt. Record model, effective output cap, request/schema hashes, observed usage metadata, latency, outcome and available provider request/finish identifiers. Use nullable fields for unavailable usage; unknown is not zero. Exclude credentials and connection details.
- Preserve the provider's reported total as authoritative when available. Store raw usage fields so thought/cache counts can be interpreted according to the provider API without double counting. Keep conservative reservation charges as estimates in a separate field.
- Separate generation, catalog construction and evaluation mapping totals. Report failed-call spend and maximum observed request input alongside total tokens. Retain legacy metrics with explicit legacy labels; do not invent a historical stage breakdown.
- Store a digest of actual source/prompt/schema content, effective configuration and dependency versions. A Git commit alone is insufficient while the working tree contains edits. Cache/recovery fingerprints must also include upstream artifacts.
- Add a preflight view: extracted pages/chunks, scheduled initial tasks, per-call output caps and an explicitly labeled estimate. Explain the run budget separately from model context and output limits. Do not invent downstream task counts before requirements/scenarios exist.

Acceptance: scripted successful, truncated and unknown-usage calls reconcile correctly; provider totals are not inflated by adding thought tokens twice; failure costs remain visible; no credential appears in snapshots. No paid call is needed for these checks.

**Phase 1: correct the known failure paths.**

Use the existing strict models and validators rather than a generic patch framework.

- Introduce a small repair response containing only requested replacement artifacts. Build prompts from findings, affected artifacts, required dependencies and supporting evidence. Merge by existing ID in Python. Route mixed findings to their responsible roles; limit each finding to one repair attempt followed by verification.
- Permit citation-only changes when the finding requests a citation correction and the replacement is grounded. Keep unrelated artifacts byte-for-byte equivalent after canonical serialization. Reject changed IDs, unknown targets, omitted targets and unsolicited changes. Semantic findings must resolve the described behavior, not merely alter trace links.
- Make citation verification whitespace-tolerant but preserve meaningful punctuation and values. Disable fuzzy quote rewriting as an automatic proof of support. Ambiguous matches need an explicit unresolved result or correction, not a silent choice of the first matching chunk.
- Catch expected Judge budget, transport and structured-output failures within evaluation. Preserve a successfully generated bundle and its validation state; record the failed evaluator attempt even if catalog construction failed before a catalog ID existed. Use Phase 0's attempt records for that case, preserving current catalog foreign-key integrity. Programming errors must remain visible rather than being converted into successful evaluations.
- Retain completed drafts when generation repair fails. Represent generation status, deterministic validation and semantic approval distinctly. A draft with unresolved findings must not be labeled an approved suite. Adjust UI/download counts to show saved work and outstanding issues.
- After repair, run deterministic checks and a small source-grounded verification of semantic findings; do not ask the model to rewrite the entire suite again. Apply any revised review protocol consistently and record its cost.

Acceptance: replay the three Telescope findings offline with scripted repair responses; only `SCN-013`, `TC-013`, and `TC-062` change. Citation-only repairs succeed; unrelated edits fail; the command result is corrected to the supported behavior. Both injected Judge failures preserve the completed generation. Add focused negative citation cases for `<`/`>`, `<=`/`<`, signed values, decimals and altered negation. These are correctness examples, not a new benchmark score.

Primary files: `models.py`, `prompts.py`, `pipelines.py`, `documents.py`, `runner.py`, `app.py`, and existing document/pipeline/runner tests. Add storage changes only where needed to persist the chosen draft representation. Preserve old run readability.

**Phase 2: store evidence once and shorten agent handoffs.**

- Add stable span IDs derived from document/chunk identity and offsets in a versioned canonical text representation. Resolve page, section and original quote from that evidence registry. Keep existing public exports with full citations; use compact IDs between agents.
- Models select span IDs and receive the corresponding verbatim text when reasoning requires it. An ID validates location, not entailment. Add definitions, table headers, relevant neighboring text and explicit cross-references to each evidence package.
- Assign source sections to Scouts exactly once, with narrow boundary context for continuity. Filter known repeated headers/footers deterministically, preserving a mapping to original text. Keep an exhaustive source pass; do not replace extraction with top-k retrieval.
- Reuse unchanged candidate fields during curation. Ask for retain/merge/reject decisions and edits only where necessary. Preserve actors, triggers, state, outcomes and numerical constraints when comparing candidates. Global deduplication must not collapse distinct rules because titles look similar.
- Maintain a compact global requirement/dependency index. Pass full detail only to the tasks that need it. Reuse `_relevant_chunks` and `_dependency_context` before adding new retrieval machinery.
- Run a small source-to-requirement gap check by section, reusing extraction evidence. Track missing or ambiguous source obligations explicitly. Generation-side coverage tracking stays separate from the independent evaluator catalog.

Acceptance: every included source section has an assigned owner or a recorded exclusion reason; cross-page clauses and table headers survive; resolved citations reconstruct unchanged evidence; scenarios depending on requirements from another section receive that dependency. Measure input and correction-call reductions against the corrected baseline. Failed or unsupported source content cannot be silently dropped to improve the cost metric.

Primary files: `documents.py`, `models.py`, `prompts.py`, `pipelines.py`; extend storage for evidence records only after the representation is fixed. Version extraction and schemas so old citations remain interpretable.

**Phase 3: bound both requests and outputs.**

- Separate the overall run budget, per-stage observed spending and per-request input/output capacity. Include schema/instruction overhead in preflight accounting. Output ceilings remain ceilings, not targets to fill.
- Keep a compact global Scenario Architect pass for dependency/coverage decisions, then generate details in groups sized by evidence and expected output. Reuse existing grouping helpers. Stable task-derived IDs and deterministic merge order should make output independent of thread completion order.
- Batch writer, Critic and evaluator work by size as well as item count. On truncation, split the unfinished task once; if an indivisible item still fails, persist an explicit partial failure. Bound split depth and total recovery calls. Do not truncate the source or silently cap the number of required tests.
- Run deterministic checks before semantic review. Use a global index to identify cross-batch omissions and contradictions, then review the affected evidence locally. Keep the Critic enabled in the primary optimized system.
- Add controlled gap repair that can create a missing scenario/test for an explicit source-supported obligation. Allocate new IDs in code, require supporting evidence, then verify links and behavior. Keep this separate from replacement-only repair so an ordinary correction cannot expand the suite arbitrarily.
- Generate positive, negative, boundary and transition cases only when source-supported. Surface ambiguity rather than inventing precise expected results or assuming every requirement needs all five categories.
- Batch independent catalog extraction and test mapping; preserve a single approved catalog and deterministic aggregation. Evaluate every test exactly once. An error in one evaluation batch must not become F1 zero or erase generation.

Acceptance: oversized scripted output triggers a bounded split; no identical oversized request is replayed indefinitely; all assigned obligations are represented or explicitly unresolved; no duplicate/missing IDs after merges; unavailable budget stops new work without destroying completed results. A tiny document should not trigger empty Scout model calls. Record output growth and source coverage, not just token reduction.

Primary files: `pipelines.py`, `prompts.py`, `providers.py`, `coverage.py`, `runner.py`, and relevant existing tests.

**Phase 4: recover completed work after interruption.**

- Persist each validated stage/task immediately using the existing blackboard table and a narrow repository callback. Make repeated insertion idempotent only when content hashes match; reject conflicting attempts. Finalization must not reinsert or overwrite different checkpoint outputs.
- A recovery action creates a new linked run/attempt. Reuse only completed tasks with matching source, code/prompt/schema, model/configuration and upstream fingerprints. Mark reused work explicitly. Downstream tasks are invalidated when their inputs change.
- Distinguish newly incurred tokens from inherited computation. Include all attempt costs in operational cost-per-completed-suite reporting. Do not charge the same inherited artifact twice, and do not claim its original computation was free.
- Show the failed stage, saved artifacts, unresolved findings and recovery scope before execution. A UI rerun or double-click must not duplicate generation calls. Use the existing database's transactions and uniqueness constraints; no additional queue service is required for this assignment.
- Stop scheduling new work on cancellation or budget exhaustion. Record already-running calls when they finish; do not imply that a Python thread cancellation cancels provider billing.

Acceptance: simulate interruption after a completed task and recover without reissuing that task; duplicate completion does not duplicate artifacts/costs; a changed prompt invalidates dependent reuse; historical runs remain immutable and readable. Check against the dedicated test database, never the application database.

Primary files: `storage.py`, `schema.sql`, `runner.py`, `pipelines.py`, `models.py`, `app.py`, and storage/runner tests. Use additive schema changes and test migration against existing-format records. Any scoring or evidence change receives a new version rather than modifying historical results.

**Phase 5: demonstrate the quality/cost tradeoff.**

- Treat Telescope as development data. Select other held-out documents across document length, requirement density and table/state-machine content. Freeze the split and all effective configurations before evaluation.
- Apply common correctness fixes and accounting to every condition. Keep model and thinking settings fixed in the primary experiment. Respect the defining differences between single-prompt, staged and multi-agent conditions; record any common retry policy so call counts remain transparent.
- Establish a corrected multi-agent baseline before evaluating context optimizations. The historical failed run remains an incident case. Use small ablations for compact evidence, bounded generation and optional combined scenario/test writing; do not attribute their joint gains to one component.
- Use one independently approved frozen coverage catalog per document/evaluator version. Generate it from source, with human inspection, before inspecting condition results. Never feed it into generation or tune prompts against held-out scores. Draft catalogs may support exploratory scoring but cannot enter official comparisons.
- Blind human raters to architecture, provider, token cost and condition identity. Retain the four current dimensions: coverage, groundedness, executability and redundancy. Add a checklist for expected-result correctness and critical boundary/negation mistakes within those dimensions. Source-backed links alone are insufficient.
- Report generation, evaluation and one-time catalog costs separately, along with completion, failure/recovery rate, latency and supported behavior coverage. Use total attempted spend divided by approved completed suites as an operational efficiency measure. Label unavailable scores as unavailable; also report their frequency.
- Begin with the proposed feasibility pilot of six held-out documents × four conditions (including the corrected multi-agent control) × three repeats = 72 runs, after offline and small development checks. This is not a statistical power justification. More repetitions on one document do not replace more independent documents.
- Analyze paired document-level differences with uncertainty intervals. Predeclare a quality tolerance and any critical-error rule with the supervisor before the held-out runs. The earlier suggested 0.02 F1 margin is illustrative, not an accepted standard. If uncertainty is too wide, report quality preservation as inconclusive. Do not equate a non-significant difference with equivalence.
- Fresh independent repetitions must regenerate artifacts. Checkpoint reuse belongs to a separately labeled recovery experiment. Report cold/warm provider caching conditions if later tested.

The main research question is: “How much can scoped evidence and bounded artifact repair reduce generation cost while maintaining supported behavior coverage and executable test quality?” A successful assignment may find that a simpler baseline offers the best tradeoff; the evaluation must not be tuned to guarantee multi-agent superiority.

**The research motivates the design; it does not predetermine the results.**

| Source | Design connection |
|---|---|
| [Zhang et al., AgentPrune](https://arxiv.org/abs/2410.02506) | Redundant communication is a concrete efficiency target. This plan adapts scoped handoffs rather than implementing the paper's graph-pruning algorithm. |
| [Liu et al., TACL 2024](https://aclanthology.org/2024.tacl-1.9/) | Evidence placement and context length can affect retrieval performance; evaluate focused evidence with omission safeguards. The paper does not establish results for this Gemini model. |
| [Smit et al., ICML 2024](https://proceedings.mlr.press/v235/smit24a.html) | Multi-agent debate does not uniformly dominate simpler methods; maintain strong baselines and control budgets. Its debate setting differs from this role pipeline. |
| [Chen et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.474/) | Human and LLM judges can exhibit bias. Use independent source inspection, blinded ratings and explicit evaluator versions; do not assume these eliminate bias. |
| [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/) | Statistical analysis must fit the experimental design. Use paired document-level analysis and disclose limited sample size. |

**Defer changes that would obscure the first result.**

Keep model routing, selective Critic skipping, lossy prompt compression, fine-tuning, a vector database and a new multi-agent framework outside the first implementation. Test combining scenario and test generation only as a later ablation. Add these only if measured bottlenecks remain after the simpler changes, with a separate quality/cost comparison.

Verification policy: use the smallest relevant existing offline tests for each phase, add regression cases for the observed failures, and run storage checks only against the dedicated test database when persistence changes. Inspect partial-result, repair-failure and recovery UI states after their implementation. Before official runs, freeze versions and run the repository's required final checks. The initial planning task authorized no deployment or paid generation. The later “continue 1–5” instruction authorizes implementing and deploying the improvements. Paid held-out evaluation remains pending the selected documents and total budget; historical results are not rewritten.
