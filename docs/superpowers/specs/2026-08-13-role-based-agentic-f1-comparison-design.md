# Role-Based Agentic Generation and F1 Comparison Design

**Date:** 2026-08-13  
**Status:** Approved in conversation

## 1. Purpose

Extend the BRD/SRS test-case research core with a reproducible role-based agent graph and compare it directly with the existing SRS-only single-prompt baseline.

The agentic condition uses a deterministic policy orchestrator to invoke a Requirement Analyst, conditional internet Researcher, Scenario Generator, Test Generator, and Validator. The comparison holds the source document, provider, model, temperature, and total model-token ceiling equal. It evaluates both conditions against versioned human-labelled requirements and test intents using separate requirement and test-coverage precision, recall, and F1 scores.

This work also replaces the current low-information result presentation with a professional comparison workspace inspired by the useful workflow and master-detail patterns in the reference application. It does not copy the reference application's unrelated CI/CD, infrastructure, incident simulation, or decorative dashboard features.

## 2. Confirmed decisions

- Human-labelled benchmark data is the evaluation ground truth.
- Internet research is clarification-only and cannot create or rewrite SRS requirements.
- A deterministic hybrid policy orchestrator owns routing; an LLM does not freely choose the graph.
- Baseline and agentic conditions use the same source, provider, exact model, temperature, and total model-token ceiling.
- Requirement F1 and test-coverage F1 are reported separately; there is no combined overall F1.
- The primary experiment measures end-to-end improvement: SRS-only single prompt versus the complete role-based agentic system with conditional research.
- Generation results are immutable. Human match decisions are appended as versioned evaluation revisions.

## 3. Success criteria

The feature is complete when:

1. one comparison request runs the single-prompt and role-based agentic conditions against the same parsed PDF;
2. the agentic condition visibly spawns only roles required by deterministic routing policy;
3. internet research is invoked only for externally researchable, test-blocking ambiguities and returns auditable citations;
4. research findings remain distinct from SRS evidence and cannot silently add product requirements;
5. uncovered requirements remain uncovered rather than being relinked to unrelated existing scenarios or test cases;
6. both conditions are evaluated through blinded, human-confirmed, one-to-one matching against versioned gold labels;
7. the UI reports requirement and test-coverage precision, recall, F1, score deltas, tokens, and latency;
8. agent decisions, artifacts, validation findings, research provenance, and evaluation revisions are reconstructable from PostgreSQL; and
9. the complete offline, database, and browser verification gates pass.

## 4. Scope

### 4.1 In scope

- A new `role_based_agentic` condition alongside `single_prompt`.
- A comparison aggregate that owns two condition runs under one immutable configuration.
- Five specialist roles and a deterministic Python policy orchestrator.
- Conditional, citation-backed internet research.
- A shared thread-safe agentic token ledger with bounded stage reservations.
- Deterministic validation plus one targeted agent repair cycle.
- A `completed_with_gaps` lifecycle state for structurally valid but incomplete output.
- Versioned gold requirements and gold test intents.
- Blinded human match confirmation and versioned evaluation revisions.
- Requirement and test-coverage micro/macro precision, recall, and F1.
- PostgreSQL persistence for comparisons, tasks, research, labels, decisions, and metrics.
- Comparison overview, agent activity, artifact, traceability, and evaluation views in Streamlit.
- Backward-compatible browsing of existing standalone runs.

### 4.2 Out of scope

- A free-form LLM supervisor that invents agents or arbitrary handoffs.
- Allowing web findings to become SRS requirements or unsupported expected behavior.
- Open-ended review or repair loops.
- OCR, non-PDF inputs, automated browser-test execution, CI/CD simulation, infrastructure control, incident alerting, or Excel export.
- Provider-versus-provider comparison as an experimental variable.
- Automatically using an LLM judge as ground truth.
- Multi-reviewer agreement statistics in the first slice.
- A 54-run scheduler, resumable batch runner, or statistical significance testing.
- New agent-framework, graph-framework, vector-database, or UI dependencies.

## 5. Architecture

The existing Python package, strict Pydantic models, provider boundary, document chunking, PostgreSQL repository, and Streamlit entry point remain the foundation. No agent framework is added.

### 5.1 Policy orchestrator

The orchestrator is ordinary Python control flow. It owns:

- fixed stage order;
- conditional role routing;
- task inputs and allowed outputs;
- at most three concurrent model calls;
- shared token reservations and cancellation;
- retries and one targeted repair cycle;
- provenance and lifecycle events; and
- terminal condition status.

It never asks a model to choose arbitrary tools, spawn unbounded roles, or alter budget and validation policy.

### 5.2 Specialist roles

#### Requirement Analyst

The Analyst always runs first. It extracts source-grounded functional, non-functional, and business requirements, dependencies, priorities, and structured ambiguities.

Each ambiguity contains:

- the affected requirement ID;
- category: `business_decision`, `source_conflict`, `external_standard`, or `external_fact`;
- why it blocks or weakens testability;
- whether it is test-blocking;
- a generic, privacy-safe search query when externally researchable; and
- the source references that exposed it.

#### Researcher

The Researcher is conditional. It may run only when an ambiguity is both test-blocking and categorized as `external_standard` or `external_fact`. Business decisions and conflicts inside the SRS remain unresolved for a human; internet search must not guess them.

The first live research adapter uses Gemini grounded Google Search with the same selected Gemini model and credential as the comparison. Research-enabled comparisons therefore require the Gemini provider. LM Studio and Ollama may still run baseline-versus-agentic comparisons with research disabled and a visible capability warning, but those results are excluded from research-enabled benchmark aggregates.

Each finding records the query, target ambiguity and requirement, concise claim, source title, URL, short supporting excerpt, retrieval time, and resolution status. Only primary or authoritative sources are retained. No webpage instruction is treated as executable direction.

#### Scenario Generator

The Scenario Generator receives validated requirements, unresolved ambiguities, allowed research clarifications, and cited SRS chunks. It produces positive, negative, boundary, edge, and state-transition scenarios where supported. Requirement batches may fan out across at most three isolated workers.

#### Test Generator

The Test Generator converts canonical scenarios into manual, observable test cases with preconditions, test data, ordered actions, expected results, priorities, and source links. It may batch work within the same three-call concurrency ceiling. It cannot convert a research recommendation into an SRS-backed expected result.

#### Validator

The Validator independently reviews groundedness, completeness, duplication, testability, unsupported assumptions, and relationship consistency. Its structured findings supplement, but never replace, deterministic validation.

### 5.3 Repair routing

After deterministic and agent validation, the orchestrator may issue one targeted repair task to the role responsible for a repairable issue:

- extraction or requirement issue → Requirement Analyst;
- missing or defective scenario → Scenario Generator;
- missing or defective test case → Test Generator.

The repair must create or correct evidence-supported content. It may not mark coverage complete by adding a requirement ID to an unrelated existing scenario or test case. There is no second repair cycle.

## 6. Routing and data flow

### 6.1 Comparison setup

1. The user uploads one text-extractable PDF and, for evaluation, selects a saved gold-label version or uploads a strict JSON label set matching the gold models.
2. The document is parsed once into canonical page-aware chunks.
3. A comparison manifest freezes the source hash, provider, exact model, temperature `0.0`, equal per-condition token ceiling, research capability, prompt/schema versions, gold-label version, and start time.
4. Gold labels remain inaccessible to all generation prompts, agents, validators, and search queries.

### 6.2 Baseline condition

The baseline executes the existing SRS-only single structured prompt. It receives the canonical SRS chunks and no research evidence, gold labels, agent output, or semantic repair. Existing schema-repair and transport-retry rules remain bounded.

### 6.3 Agentic condition

The orchestrator executes this graph:

1. Requirement Analyst.
2. Conditional Researcher tasks for eligible ambiguities, with no more than three research queries per comparison.
3. Scenario Generator workers.
4. Test Generator workers.
5. Deterministic validation and Validator review.
6. At most one targeted repair.
7. Final deterministic validation, RTM construction, and structural metrics.

Research findings use `research_references`; SRS-derived artifacts continue to use `source_references`. The two evidence types are never merged.

### 6.4 Evaluation

After both conditions terminate, the system creates blinded candidate-match work. The evaluation UI presents stable aliases `Condition A` and `Condition B` in an order derived from the comparison ID. It hides condition architecture, agent activity, tokens, and latency until match decisions are locked.

A reviewer confirms or rejects requirement and test-intent matches. Locking the review appends an immutable evaluation revision, computes metrics, and reveals the condition identities and deltas. A later correction creates a new evaluation revision rather than mutating the prior one.

## 7. Budget and concurrency policy

Each condition receives the same total model-token ceiling. Equal ceiling, not equal actual consumption, is the fairness constraint; actual input/output/charged tokens are reported.

The agentic condition shares one thread-safe ledger. Default stage reservation ceilings are:

- Requirement Analyst: 20%;
- Researcher: up to 10%;
- Scenario Generator: 20%;
- Test Generator: 30%;
- Validator: 10%; and
- targeted repair reserve: 10%.

These are ceilings within the shared ledger, not guaranteed spending. Released or unused reservations return to the shared balance for later stages. The orchestrator refuses a task before execution when its conservative input reservation plus maximum output would exceed the remaining ceiling.

Research prompts and grounded responses count against the agentic model-token ledger. Search retrieval calls, elapsed time, and any provider-reported search charges are recorded separately. Parallel workers use the same ledger and a maximum of three concurrent model calls.

## 8. Validation and lifecycle states

Deterministic code verifies the existing schema, ID, dependency, relationship, citation-containment, orphan, coverage, and duplicate rules for both conditions. It additionally verifies:

- research citations have a query, URL, retrieval time, and target ambiguity;
- research references never appear in `source_references`;
- gold labels never appear in generation task inputs or stored task prompts;
- each repaired artifact addresses an existing validator issue; and
- task outputs reference only declared task inputs and canonical IDs.

Terminal states are:

- `completed`: structurally valid, fully linked output with no unresolved test-blocking ambiguity;
- `completed_with_gaps`: structurally valid output with uncovered requirements or unresolved test-blocking ambiguity;
- `failed`: invalid schema, citation, relationship, unsupported claim, exhausted required stage, or unrecoverable provider failure.

The current `_repair_coverage` behavior is removed. Uncovered requirements stay visible and reduce evaluation recall.

## 9. Human-labelled benchmark and F1

### 9.1 Gold models

Each benchmark document has a versioned label set containing:

- `GoldRequirement`: stable ID, canonical statement, requirement type, priority, and source-page anchors;
- `GoldTestIntent`: stable ID, linked gold requirement IDs, scenario type, intended behavior, and essential assertion; and
- label-set metadata: document hash, version, author, creation time, and status.

The document hash must match the uploaded PDF. A label set must contain at least one gold requirement and one gold test intent. Labels are rejected when IDs are duplicated, anchors are outside the document, test intents reference unknown requirements, or required fields are blank.

### 9.2 Candidate proposals and blinded review

Candidate pairs are proposed without an LLM judge. Normalized token overlap, compatible artifact type, and source-page intersection rank up to five gold candidates for each prediction. The reviewer may confirm one candidate, select another gold item, or mark the prediction unmatched.

Confirmed edges pass through deterministic maximum-cardinality one-to-one matching. A predicted requirement or test case and a gold item can each contribute to at most one true positive. This prevents duplicates, merges, and splits from inflating results.

### 9.3 Metrics

For requirements:

- precision = matched predicted requirements / all predicted requirements;
- recall = matched gold requirements / all gold requirements; and
- F1 = `2 * precision * recall / (precision + recall)`, or `0` when both are zero.

For test coverage, each generated test case is one predicted test intent:

- precision = matched predicted test intents / all predicted test intents;
- recall = matched gold test intents / all gold test intents; and
- test-coverage F1 uses the same harmonic mean rule.

Unsupported or duplicate predictions remain in the precision denominator. Missed gold items remain in the recall denominator.

A metric whose denominator is zero is `0`. A failed condition receives zero precision, recall, and F1 rather than being omitted; its failure flag remains visible so zero quality is not confused with a reviewed successful output.

The primary multi-document outcome is macro-F1: calculate F1 per document, then average documents so large SRS files do not dominate. Micro precision, recall, and F1 aggregate raw counts and remain secondary. The UI shows baseline, agentic, and `agentic - baseline` deltas separately for requirement and test-coverage scores.

## 10. Persistence and compatibility

PostgreSQL remains the source of truth. The schema adds normalized records for:

- comparison manifests and their two condition run IDs;
- agent tasks, routing reasons, dependencies, usage, latency, retries, and status;
- ambiguities and research findings;
- gold label sets, requirements, and test intents;
- blinded candidate matches and reviewer decisions; and
- immutable evaluation revisions and metrics.

Existing run tables and artifact models are reused where possible. Existing `single_prompt`, `staged_single_agent`, and `centralized_multi_agent` records remain readable. The new comparison workflow creates `single_prompt` and `role_based_agentic` condition runs; it does not rewrite old records.

Generation results, task events, and research findings become immutable when a condition terminates. Evaluation decisions are append-only revisions. The raw PDF and provider credentials remain excluded from PostgreSQL. Search queries contain only generic ambiguity terminology or standard names; raw proprietary SRS text and source excerpts are never transmitted to internet search.

## 11. User experience

### 11.1 Runs home and creation

The runs-first home remains the entry point and adds **Create comparison** as the primary action. Existing standalone runs remain browsable.

Comparison creation contains:

- one PDF upload;
- one matching versioned gold-label selection or strict JSON upload;
- the saved provider/model/token-ceiling summary;
- explicit **Enable web research** consent and privacy guidance;
- a warning that research-enabled comparison currently requires Gemini; and
- one **Run comparison** primary action.

### 11.2 Comparison workspace

The comparison detail uses a restrained QA-workbench shell with these views:

- **Overview:** baseline versus agentic requirement F1, test-coverage F1, precision, recall, tokens, latency, and deltas;
- **Agent activity:** an auditable timeline showing spawned roles, routing reasons, research decisions, usage, and status;
- **Requirements:** searchable master-detail view with source citations, ambiguity state, allowed research clarifications, and evaluation state;
- **Scenarios and test cases:** filterable master-detail views rather than long stacks of collapsed expanders;
- **Traceability:** requirement-to-scenario-to-test mappings with uncovered items explicit;
- **Evaluation:** blinded candidate confirmation, unmatched predictions, missed gold items, score revision history, and identity reveal after lock; and
- **Configuration:** immutable provider, model, budget, versions, timings, capability flags, and safe diagnostics.

### 11.3 Visual system and accessibility

The interface uses white and slate surfaces, blue primary actions, semantic green/amber/red states with text labels, an 8px spacing rhythm, strong heading hierarchy, restrained shadows, and one primary action per view. Controls have at least 44px hit areas, visible labels, clear disabled/loading/error states, and 3px visible focus rings.

Desktop uses master-detail columns with one main page scroll. Mobile collapses to a single column at 375px without horizontal scrolling or nested scroll traps. Body text remains at least 16px on mobile. Motion is limited to meaningful 150-300ms state transitions and respects reduced-motion settings.

No emoji is used as a structural icon. Streamlit-native controls and the existing dependency set are preferred; no chart or component library is added.

## 12. Error handling and security

- A Researcher timeout, search failure, or lack of authoritative sources records the ambiguity as unresolved and continues the agentic condition toward `completed_with_gaps` when other artifacts are valid.
- A required Analyst, Generator, or Validator failure ends only the agentic condition; the baseline result remains inspectable.
- Failure in one condition never deletes or hides the other condition.
- Partial artifacts, token usage, routing events, and safe diagnostics remain persisted.
- Web content is untrusted quoted data. It cannot change orchestration, request uploads, execute instructions, or override the SRS.
- Only minimal generic search queries are sent externally. The application never sends the raw PDF, full chunks, credentials, gold labels, browsing data, or unrelated local data.
- Provider credentials remain browser-local under the existing disclosed boundary and are redacted from errors, events, downloads, and snapshots.
- Research citations are presentation-escaped and never rendered as trusted HTML.
- Evaluation blinding hides condition identity and operational metadata until decisions are locked.

## 13. Testing and verification

Automated tests use fake model and search providers and make no live external calls. They cover:

1. routing rules and conditional Researcher spawning;
2. business-decision/source-conflict ambiguities never triggering search;
3. search-query minimization, authoritative citation normalization, and research/SRS evidence separation;
4. agent isolation, three-call concurrency, shared-ledger accounting, cancellation, and stage ceilings;
5. one targeted repair cycle and prohibition of relationship-only false coverage repair;
6. `completed`, `completed_with_gaps`, and `failed` lifecycle rules;
7. hidden gold labels never entering generation inputs;
8. equal source, provider, model, temperature, and total ceilings across conditions;
9. candidate ranking, human decisions, maximum one-to-one matching, zero-denominator behavior, micro metrics, macro metrics, and score deltas;
10. comparison, task, research, label, decision, and evaluation-revision storage and reconstruction;
11. backward-compatible loading of existing runs;
12. Streamlit comparison creation, overview, activity, artifact, traceability, evaluation, empty, loading, and error states; and
13. credential and source-content redaction at every external and persisted boundary.

Fresh verification runs the full offline test suite, dedicated PostgreSQL tests with no skips, Python compilation, import checks, and `git diff --check`. Browser verification covers desktop and 375px widths, keyboard navigation, visible focus, readable contrast, research consent/capability states, blinded evaluation, identity reveal after lock, and absence of horizontal scrolling.

One optional manual Gemini smoke comparison verifies live grounded search provenance and confirms that only generic queries leave the application.

## 14. Acceptance criteria

The slice is accepted when:

1. a user can launch one immutable two-condition comparison from one PDF and one matching gold-label version;
2. the role-based graph executes with deterministic, visible routing and bounded concurrency;
3. Researcher tasks are conditional, authoritative, citation-backed, privacy-minimized, and clarification-only;
4. agentic and baseline conditions share the frozen experimental controls and equal token ceilings;
5. incomplete generation remains incomplete and cannot be converted to success by relinking IDs;
6. a blinded reviewer can lock one-to-one matches and create an immutable evaluation revision;
7. separate requirement and test-coverage precision, recall, F1, macro-F1, micro-F1, and condition deltas are correct;
8. comparison views expose the evidence, agents, traceability, gaps, evaluation, cost, and latency needed to explain the outcome;
9. existing standalone runs remain readable; and
10. all fresh automated, database, compilation, and browser checks pass.

## 15. Deferred follow-up slices

- Multi-reviewer assignment, adjudication, and inter-rater agreement.
- Frozen multi-document scheduling, resumability, and statistical significance analysis.
- Provider-stratified research adapters beyond Gemini grounded search.
- CSV or Excel benchmark import/export and formatted reporting.
- Automated test-script generation or execution.
- CI/CD, infrastructure, and incident integrations when a real operational requirement exists.
