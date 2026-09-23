# Coverage-First Multi-Agent Enhancement Design

**Date:** 2026-09-23
**Status:** Approved in conversation

## 1. Purpose

Redesign the centralized multi-agent BRD/SRS-to-test-case condition so that it
has a fair, research-grounded opportunity to outperform the staged condition on
unseen documents without changing the generation model or total token ceiling.

The reported benchmark result is:

- staged single agent: F1 `0.817`;
- centralized multi-agent: F1 `0.672`; and
- single prompt: F1 `0.673`.

The objective is not to tune until multi-agent wins. The objective is to remove
known architectural and measurement disadvantages, freeze a valid experiment,
and report the result even if the redesigned condition does not outperform the
staged baseline.

## 2. Confirmed decisions

- The exact provider, model, temperature, thinking level, source document, core
  schemas, and total generation-token ceiling are controlled across conditions.
- Different conditions may use different call graphs and internal token
  allocations, but actual usage is reported.
- Temperature remains fixed at `0.0`.
- The application retains flexible standalone-run configuration and adds a
  locked fair-comparison mode.
- Public research motivates architectural and evaluation choices.
- The final experiment uses held-out documents and does not tune against the
  final benchmark.
- Multi-agent superiority is a hypothesis, not an assumption.
- The fixed semantic Judge remains outside the generation graph and cannot
  provide coverage targets to generation agents.
- Human evaluation is blinded and includes inter-rater reliability measured
  with quadratic-weighted Cohen's kappa for ordinal scores.
- Existing Python orchestration, Pydantic models, provider adapters,
  PostgreSQL, and Streamlit remain the technical foundation.
- No agent framework, vector database, free-form supervisor, or unbounded
  debate loop is added.

## 3. Research basis

Multi-agent collaboration can improve reasoning and factuality when agents
produce independent proposals and then deliberate or aggregate them. It does
not reliably outperform simpler prompting merely because more model instances
are involved.

The design uses the following findings:

1. Du et al. showed that multi-agent proposal and debate can improve reasoning
   and factual validity across several tasks.
   <https://proceedings.mlr.press/v235/du24e.html>
2. Smit et al. found that multi-agent debate does not reliably outperform
   self-consistency and ensembling without careful protocol tuning. This rules
   out an assumption that agent count alone creates quality.
   <https://proceedings.mlr.press/v235/smit24a.html>
3. Mixture-of-Agents demonstrated a layered pattern in which later agents use
   multiple prior outputs as inputs. This supports explicit synthesis instead
   of deterministic concatenation.
   <https://arxiv.org/abs/2406.04692>
4. ReConcile found that diversity is important to multi-agent consensus. The
   experiment's same-model constraint prevents model-family diversity, so this
   design creates controlled diversity through independent evidence assignments
   and specialist responsibilities.
   <https://arxiv.org/abs/2309.13007>
5. Self-Refine showed that bounded feedback followed by refinement can improve
   an initial output without additional model training. This supports one
   critic-directed repair rather than an open-ended loop.
   <https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html>
6. Requirements-to-test research reports useful results from prompt chaining,
   while also identifying redundancy, applicability, domain nuance, and exact
   action sequences as areas requiring expert oversight.
   <https://arxiv.org/abs/2412.03693>
   <https://arxiv.org/abs/2404.12772>
7. LLM-as-a-Judge research documents position and other judgment biases. This
   supports condition blinding, a frozen evaluator, and human calibration.
   <https://aclanthology.org/2025.ijcnlp-long.18/>
   <https://aclanthology.org/2024.emnlp-main.474/>
8. Cohen's weighted kappa was designed to allow scaled disagreement or partial
   credit between ordinal categories. Quadratic weights are therefore used for
   the four-point human rubric.
   <https://pubmed.ncbi.nlm.nih.gov/19673146/>
9. Statistical-test selection must reflect the paired experimental design and
   metric distribution rather than use an arbitrary significance test.
   <https://aclanthology.org/P18-1128/>
10. Repeated evaluations are required because temperature zero does not ensure
    identical output from all hosted LLM systems.
    <https://arxiv.org/abs/2410.03492>

These sources motivate the design. They do not establish that this exact
pipeline will outperform the staged condition; the held-out experiment answers
that question.

## 4. Current-system diagnosis

### 4.1 Fragmented evidence

Remote multi-agent runs balance chunks by text length. Sorting chunks by weight
before assigning them can place disconnected pages in the same worker context
and remove surrounding document structure.

Local runs preserve order through bounded groups, so remote and local
multi-agent executions also test different evidence-partition strategies.

### 4.2 Order-dependent requirement loss

Worker requirements are grouped only when normalized titles and descriptions
are exactly equal. Semantically equivalent paraphrases remain separate. The
implementation then retains the first 20 groups in insertion order. Valid
requirements from later workers can be discarded without semantic review.

### 4.3 No global test strategy

After requirement merging, workers receive small requirement subsets and only
the chunks cited by those requirements or dependencies. Each worker generates
both scenarios and detailed test cases in one call. No role sees the full
canonical catalog to design integrated workflows, minimize duplication, or
balance positive and non-positive coverage globally.

### 4.4 Reviewer is normally inactive

The configured Reviewer does not review every multi-agent bundle. It normally
runs only when deterministic validation reports an issue. Deterministic checks
can confirm IDs and links but cannot determine that an important behavior was
missed or that an expected result is too weak.

### 4.5 Unequal output constraints

Staged generation exposes per-step output limits and defaults to 16,000 tokens
for requirements, 24,000 for scenarios, and 48,000 for test cases. Multi-agent
workers use hard-coded 8,000-token call limits and cannot configure them in the
run UI. Equal total ceilings therefore do not currently mean comparable stage
capacity.

### 4.6 Unstable evaluation denominator

The Judge extracts a new coverage-unit catalog for every run before mapping its
test cases. Persisted results show that identical document hashes received
different unit counts across conditions. Examples include:

- `c34ebce...`: 26, 64, and 70 units;
- `77ff45c...`: 23, 33, 41, and 45 units; and
- `fe7216c...`: 30, 40, and 48 units.

Because recall uses the run-specific unit count as its denominator, condition
scores are not guaranteed to represent the same target space. This does not
invalidate every observed direction, but it prevents a strong causal claim
from the current F1 values.

### 4.7 Incomplete audit trail

Only aggregate coverage scores and uncovered IDs are stored. Coverage-unit
definitions and test-to-unit mapping decisions are not persisted. Judge
failures are informational and may be swallowed, leaving an unavailable score
without a reconstructable evaluation failure.

## 5. Alternatives considered

### 5.1 Patch the existing split-and-merge pipeline

Fix chunk ordering, remove first-20 truncation, align worker output limits, and
always invoke the Reviewer.

This is the smallest implementation but preserves the main weakness: isolated
workers still generate local scenarios and detailed tests without a global
coverage strategy.

### 5.2 Hierarchical blackboard pipeline

Use ordered evidence scouts, global requirement synthesis, a global scenario
plan, parallel test writing, an independent critic, and one targeted repair.

This is the selected approach. It creates explicit information gain at each
handoff, retains bounded deterministic control, and can be evaluated through
component ablations.

### 5.3 Full multi-round debate

Ask multiple agents to generate complete suites and debate them over multiple
rounds.

This is rejected because it duplicates large source contexts, consumes more
tokens, is vulnerable to same-model conformity, and has not shown reliable
superiority to simpler inference-time strategies.

## 6. Proposed architecture

### 6.1 Deterministic policy orchestrator

Python owns:

- fixed stage order;
- evidence assignments;
- concurrency and shared budget reservations;
- typed input and output contracts;
- retries and cancellation;
- stage provenance;
- deterministic validation;
- one issue-routed repair; and
- terminal status.

An LLM never chooses arbitrary roles, changes budgets, or decides whether an
evaluation target should be exposed to generation.

### 6.2 Stage A: ordered evidence map

The existing canonical document chunks remain the evidence source. Python
builds an ordered map containing chunk ID, page, section, and content length.

Assignment rules are:

- preserve original page and chunk order;
- create contiguous section-aware batches;
- repeat one boundary chunk between adjacent batches when possible;
- assign every chunk at least once;
- record the assignment manifest; and
- give every worker the global section index but only its assigned full text.

The section index supplies global orientation without tripling the full source
in every prompt. Overlap reduces requirements lost at batch boundaries.

### 6.3 Stage B: Requirement Scouts

At most three Scout calls execute concurrently. The total task count may exceed
three for long documents; concurrency remains capped.

Each Scout extracts `CandidateRequirement` records containing:

- candidate ID local to the task;
- one atomic requirement statement;
- functional, non-functional, or business type;
- priority and module;
- conditions, exceptions, constraints, and ambiguity;
- candidate dependencies;
- exact source references;
- assignment ID; and
- originating Scout task.

Scouts inspect their assignments independently and do not see peer outputs.
They must return all supported candidates rather than target a shared global
count.

### 6.4 Stage C: Requirement Curator

The Curator receives all candidates, their cited excerpts, and the document
section index. It performs the semantic work missing from the current merge:

- cluster paraphrases and overlaps;
- preserve distinct conditional behavior;
- merge citations without losing provenance;
- resolve dependencies after canonical renumbering;
- identify source conflicts and ambiguities;
- record a decision for every candidate; and
- produce the canonical `RequirementBatch`.

The `RequirementDecision` record contains candidate IDs, action (`retain`,
`merge`, or `reject`), canonical requirement ID when applicable, and a concise
reason. Rejection requires a duplicate relationship or an explicit evidence
problem; list position is never a reason.

The shared final schemas impose no requirement, scenario, or test-case count
cap. The total token ceiling bounds output size. Per-task schemas may impose a
task-local maximum to keep worker responses complete, but the orchestrator must
create enough tasks to cover every accepted plan. All experimental conditions
use the same uncapped final schemas.

### 6.5 Stage D: Scenario Architect

The Scenario Architect receives the complete canonical requirement catalog and
its evidence. It creates a global `ScenarioPlanBatch` rather than detailed test
cases.

Each `ScenarioPlan` records:

- planned scenario ID;
- covered requirement IDs;
- positive, negative, boundary, edge, or state-transition type;
- coverage objective;
- required preconditions;
- required observable assertions;
- relevant source references;
- dependency or cross-requirement context; and
- risk or priority.

The Architect must cover every requirement while avoiding one-case-per-item
mechanical duplication. Cross-requirement workflows are planned before work is
distributed.

### 6.6 Stage E: Test Writers

Scenario plans are divided into balanced tasks. Each Test Writer receives:

- assigned plans;
- referenced canonical requirements;
- dependency context;
- relevant evidence chunks; and
- global ID rules.

Writers expand plans into executable manual cases with ordered actions and
observable expected results. They cannot alter requirement meaning or introduce
new expected behavior. The orchestrator namespaces IDs and verifies that each
output implements an assigned plan.

### 6.7 Stage F: independent Critic

The Critic always runs after deterministic validation. It receives the merged
bundle, canonical requirements, scenario plans, validation report, and relevant
evidence.

It returns typed `ReviewFinding` records containing:

- finding ID and severity;
- issue type;
- affected artifact IDs;
- supporting source references;
- responsible role;
- required correction; and
- whether the issue is repairable within the source.

Issue types are:

- missing source obligation;
- unsupported artifact;
- weak or unobservable expected result;
- missing negative, boundary, or transition behavior;
- duplicate intent;
- broken relationship or citation;
- cross-requirement inconsistency; and
- unresolved source ambiguity.

### 6.8 Stage G: one targeted repair

The orchestrator groups repairable findings by owning role and selects the
highest-severity group that fits the reserved budget. It invokes one repair
call and then reruns deterministic validation.

The repair must change substantive content. Adding a requirement ID to an
unrelated existing scenario or test case is forbidden. Unresolved findings
remain visible and reduce the final result; there is no second repair cycle.

### 6.9 Blackboard and provenance

The shared blackboard is persisted typed state, not a free-form conversation.
It contains:

- evidence assignment manifest;
- Scout candidates;
- Curator decisions and canonical requirements;
- scenario plans;
- Test Writer outputs;
- deterministic validation results;
- Critic findings;
- repair request and output; and
- per-stage usage, latency, retry, and status data.

Agents read only declared predecessor artifacts. This makes each handoff
reconstructable and supports ablation analysis.

## 7. Budget and fair-comparison policy

### 7.1 Locked variables

Fair-comparison mode freezes:

- source PDF hash and extracted chunks;
- provider and exact model ID;
- temperature `0.0`;
- thinking level;
- total generation-token ceiling;
- shared artifact schemas and limits;
- core safety, grounding, and citation rules;
- maximum semantic-repair count;
- prompt, workflow, and schema versions; and
- evaluator and catalog versions.

Standalone runs retain per-agent provider, model, prompt, thinking, and token
configuration.

### 7.2 Initial allocation

The initial multi-agent budget allocation is a preregistered engineering
hypothesis, not a claim derived directly from prior research:

- Requirement Scouts: 25%;
- Requirement Curator: 15%;
- Scenario Architect: 15%;
- Test Writers: 30%;
- Critic: 10%; and
- repair reserve: 5%.

The shared ledger counts input and output reservations. Unused allocations
return to the remaining balance. Stage output maxima are configurable for
standalone runs and frozen for comparison runs.

Equal ceiling, not forced equal consumption, is the fairness constraint. The UI
reports actual input, output, charged tokens, latency, retries, and repairs.

### 7.3 Concurrency

Remote providers allow at most three concurrent calls. Local providers remain
sequential when required by model-server capacity. The assignment content and
stage graph remain the same so concurrency changes latency, not task semantics.

## 8. Evaluation design

### 8.1 Frozen coverage catalog

Each unique document hash receives one immutable, versioned
`CoverageUnitCatalog`. It is created once per evaluator configuration and reused
for every condition and repetition.

The catalog records:

- catalog ID and document hash;
- evaluator provider, model, thinking level, prompt version, and schema version;
- atomic coverage units;
- unit type and canonical statement;
- source chunk IDs and excerpts;
- creation time and review status; and
- superseded catalog ID when revised.

Generation agents cannot read the catalog. Final-benchmark catalogs are human
reviewed without access to generated outputs or condition identities, then
frozen before final condition runs begin.

### 8.2 Mapping and scoring

The fixed Judge maps every test case to the same frozen catalog for its source
document. Condition names and architecture metadata are absent from Judge
inputs.

Coverage calculations remain deterministic after mapping:

- supported-case precision: generated cases mapped to at least one valid unit
  divided by all generated cases;
- unit recall: unique covered units divided by all frozen catalog units; and
- coverage F1: harmonic mean of supported-case precision and unit recall.

The report also displays:

- uncovered unit IDs;
- unmapped case IDs;
- duplicate-test rate;
- requirement and scenario traceability metrics; and
- catalog and mapping audit records.

The metric definition is explicitly described as semantic coverage F1. It is
not executed-code coverage or defect-detection effectiveness.

### 8.3 Judge failure

Judge schema, transport, or budget failure produces an explicit evaluation
failure record with sanitized diagnostics. It does not change generation
success, but the run is excluded from score aggregation until evaluation is
completed. Failures are never silently treated as missing-at-random scores.

### 8.4 Human rubric

Two independent reviewers evaluate blinded, randomized outputs using the same
four-point ordinal dimensions:

- coverage;
- groundedness;
- executability; and
- redundancy control.

Each rating stores a pseudonymous rater ID, run ID, dimension, score, reason,
rubric version, review round, and timestamp. Reviewers submit independently
before seeing peer or Judge ratings.

Disagreements are adjudicated only after the independent ratings are frozen.
The adjudicated score is stored as a separate record and never overwrites either
original rating.

### 8.5 Agreement statistics

For each rubric dimension, report:

1. quadratic-weighted Cohen's kappa between the two humans before adjudication;
2. quadratic-weighted kappa between the fixed Judge rating and adjudicated
   human rating;
3. exact agreement;
4. adjacent-category agreement; and
5. bootstrap 95% confidence intervals.

Kappa is computed over a set of runs, never interpreted per individual run.
The report presents the estimate and interval rather than relying only on
qualitative labels. The preregistered operational threshold is `0.70` for both
human-human and Judge-human quadratic-weighted kappa; this is a project quality
gate, not a universal interpretation label. If human-human kappa is below the
gate, the rubric is revised on development data and held-out ratings are
recollected. If Judge-human kappa is below the gate, Judge F1 is reported as
exploratory and adjudicated human evaluation becomes the primary semantic
result.

### 8.6 Dataset split and repetitions

Documents with completed source labels are divided before prompt tuning into:

- a development set for architecture, prompt, and allocation decisions; and
- an untouched held-out set for the final comparison.

The split uses 30% development and 70% held-out documents, stratified by source
length quartile with a stored random seed. The split manifest stores document
hashes and is committed before final runs. A confirmatory claim requires at
least 20 held-out documents; a smaller set is explicitly reported as an
exploratory study. No held-out score is used to change prompts or budgets.

Each condition runs exactly three times per held-out document. Condition order
is randomized within each document and repetition block using a stored seed.
All settings snapshots and provider model identifiers are stored.

### 8.7 Primary analysis

The primary outcome is document-level macro coverage F1:

1. calculate F1 for each run;
2. average repetitions within document and condition; and
3. average document means so large documents do not dominate.

The primary contrast is paired `multi-agent - staged` F1 for each held-out
document. Report:

- both condition means and medians;
- paired mean and median delta;
- document-level percentile bootstrap 95% confidence interval using 10,000
  document resamples and a stored seed;
- a two-sided paired randomization test as the primary significance test and a
  Wilcoxon signed-rank sensitivity analysis; and
- actual token and latency deltas.

Single prompt remains a secondary baseline.

### 8.8 Secondary analysis and ablation

Secondary outcomes are precision, recall, validation success, duplicate rate,
invalid-reference rate, token usage, latency, retries, repairs, and failure rate.

The required ablation compares:

- staged single agent;
- hierarchical multi-agent without Critic/repair; and
- complete hierarchical multi-agent.

An optional development-only diagnostic may compare the corrected ordered
split-and-merge pipeline. It is not another headline final condition.

## 9. Persistence changes

PostgreSQL remains the source of truth. The design adds normalized storage for:

- coverage catalogs and coverage units;
- test-case-to-unit mapping decisions;
- evidence assignment manifests;
- agent stage tasks and typed artifact snapshots;
- requirement merge decisions;
- scenario plans;
- Critic findings and repair requests;
- repeated-run comparison blocks;
- independent human ratings and adjudications; and
- computed agreement and comparison statistics or their reproducible inputs.

Existing run and artifact tables remain readable. Old coverage scores keep their
historical meaning but are marked as using per-run catalogs and excluded from
the new controlled aggregate unless they can be remapped to a frozen catalog.

Raw PDF bytes, credentials, and provider base URLs remain excluded from
PostgreSQL.

## 10. User experience

### 10.1 Standalone run

Standalone creation preserves current flexibility. Multi-agent settings add
per-stage output-token controls matching staged configurability. The immutable
snapshot shows stage roles, prompts, models, thinking levels, limits, and total
ceiling.

### 10.2 Fair comparison

The comparison workflow selects one provider/model configuration and applies it
to all generation roles and conditions. The UI shows which values are locked
and prevents condition-specific edits after the comparison starts.

The result view separates:

- generation results;
- deterministic quality and traceability;
- frozen-catalog semantic evaluation;
- human-review agreement;
- tokens and latency; and
- statistical comparison and ablations.

Condition identities remain hidden during human review and are revealed only
after independent ratings are locked.

## 11. Error handling

- A missing evidence assignment or dropped candidate is a pipeline error.
- A mandatory stage that cannot complete within its reserved and shared budget
  fails the condition with an explicit category.
- One bounded transport retry policy and existing schema-repair policy remain.
- Sibling tasks cancel after the first unrecoverable worker failure.
- A repair failure retains the pre-repair bundle and records unresolved findings
  rather than inventing success.
- Evaluation failure does not rewrite generation status.
- A coverage catalog cannot be used when its document hash or evaluator version
  differs from the comparison manifest.
- Partial comparison blocks remain inspectable but are excluded from paired
  aggregate inference until all required condition runs are evaluated.

## 12. Verification strategy

### 12.1 Unit checks

- ordered partitioning assigns every chunk and only permitted overlaps;
- section order is preserved;
- semantic merge decisions cannot silently drop candidates;
- canonical IDs and dependencies remain valid after synthesis;
- scenario plans cover every canonical requirement;
- Test Writers reference only assigned plans and known requirements;
- review findings route to valid roles;
- repair count cannot exceed one;
- fair-comparison settings cannot diverge across conditions;
- shared budget reservations are thread-safe; and
- the same document hash always resolves to the same frozen catalog version.

### 12.2 Integration checks

Scripted fake providers exercise the complete hierarchical graph, persistence,
cancellation, retry, repair, and result reopening without live model calls.

One evaluator invariant test scores different condition bundles for the same
document and asserts identical catalog ID and total coverage-unit count.

Database tests verify append-only human ratings, adjudication lineage, catalog
versioning, mapping persistence, and compatibility with old runs.

### 12.3 Statistical checks

Synthetic paired results with known deltas verify macro aggregation, bootstrap
intervals, the selected paired test, and quadratic-weighted kappa edge cases.
Degenerate rating distributions return an explicit undefined agreement result
rather than a misleading zero.

### 12.4 Live development smoke test

Two development documents exercise all three conditions and the Judge before
the held-out run. The smoke test checks contracts and resource sizing only; its
scores may tune development settings but are not reported as final evidence.

## 13. Success criteria

The feature is complete when:

1. every experimental condition for a document uses the same frozen coverage
   catalog;
2. multi-agent no longer uses noncontiguous length-balanced remote assignments,
   exact-string-only reconciliation, or insertion-order truncation;
3. the Scenario Architect creates a global plan before parallel test writing;
4. the Critic runs on every multi-agent bundle and at most one substantive
   repair occurs;
5. fair-comparison mode freezes every controlled variable and enforces one
   shared ceiling per condition;
6. intermediate tasks and evaluation decisions are reconstructable from
   PostgreSQL;
7. two independent human ratings and adjudication can be recorded without
   overwriting history;
8. quadratic-weighted kappa and confidence intervals are reproducible;
9. the offline, database, and manual comparison smoke gates pass; and
10. the held-out report states the result without changing the design in
    response to final scores.

The research hypothesis is supported only if held-out multi-agent macro-F1
exceeds staged and the paired uncertainty analysis supports a positive effect,
without a material precision, validity, or budget regression. If the interval
includes no improvement, the conclusion is that superiority was not
demonstrated under the controlled protocol.

## 14. Implementation order

1. Freeze and persist one evaluation catalog per document and make evaluation
   failures explicit.
2. Add ordered evidence assignments, Scout candidates, Curator decisions, and
   canonical requirement synthesis.
3. Add the global Scenario Architect and parallel plan-based Test Writers.
4. Add the always-on Critic and one targeted repair.
5. Add fair-comparison locking and per-stage multi-agent token controls.
6. Add repeated-run grouping, independent human ratings, weighted kappa,
   confidence intervals, and ablation reporting.
7. Freeze the benchmark split and execute the final held-out experiment.

This order repairs measurement validity before using the metric to optimize
generation behavior.
