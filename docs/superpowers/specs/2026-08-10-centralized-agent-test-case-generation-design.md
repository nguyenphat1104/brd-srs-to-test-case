# Centralized Agent Test-Case Generation: Design and Experiment Specification

**Status:** Approved design

**Date:** 2026-08-10

**Project:** `brd-srs-to-test-case`

## 1. Purpose

Build a standalone Streamlit application that turns a BRD/SRS PDF into four traceable artifacts:

1. a structured requirement list;
2. test scenarios;
3. detailed manual test cases; and
4. a requirement traceability matrix (RTM).

The application is also the experimental instrument for a university study of agent-system architecture. Product usefulness matters, but the primary contribution is a controlled comparison of test-case generation approaches.

The design follows the task-alignment principle in [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296): use centralized coordination where work is decomposable, preserve sequential dependencies where it is not, and measure quality, cost, and error propagation rather than assuming that more agents are better.

## 2. Research question

> Under comparable model and token budgets, does a centralized multi-agent pipeline generate higher-quality and more traceable manual test cases from BRD/SRS documents than a staged single-agent pipeline?

The study uses a neutral hypothesis. Centralized multi-agent coordination may improve coverage, groundedness, or validation, but it may also increase latency, token cost, failure rate, or coordination errors.

The primary causal comparison is staged single-agent versus centralized multi-agent. A single-prompt system is retained as a contextual reference, not as the sole baseline, because comparing only single-prompt with multi-agent would confound decomposition, validation, and agent count.

## 3. Scope

### In scope

- Text-extractable PDF input.
- Page-aware document parsing and exhaustive requirement extraction.
- Functional, non-functional, and business requirements.
- Positive, negative, boundary, edge, and state-transition scenarios.
- Detailed manual test cases with steps and expected results.
- Source citations from requirements through test cases.
- Deterministic RTM construction and coverage checks.
- Three controlled generation conditions.
- Automated metrics, blinded human evaluation, and experiment exports.
- English UI, prompts, code, schemas, logs, and generated artifacts.

### Out of scope

- OCR or scanned-PDF recovery.
- DOC, DOCX, RTF, HTML, or Markdown input in the first version.
- Automated Selenium, Playwright, API, unit, or integration test scripts.
- Execution of generated test cases against a real application.
- User accounts, collaboration, cloud deployment, or a database.
- Editing artifacts during controlled experimental runs.
- A vector database or embedding-based retrieval.
- Comparing model providers or model families.

## 4. Experimental conditions

All conditions use the same Gemini model, PDF parser, schemas, source text, prompt examples where roles overlap, temperature, and total-token ceiling. The pilot procedure freezes the exact Gemini API model identifier before evaluation; model choice is not an experimental variable.

### 4.1 Single-prompt reference

One model request receives the parsed document text and asks for requirements, scenarios, test cases, and traceability links in one structured response. It receives no semantic repair loop. Deterministic schema and RTM checks still run after generation.

This condition represents a simple application baseline. It is reported alongside the primary comparison but is not used alone to claim a multi-agent effect.

### 4.2 Staged single-agent

One agent context performs the semantic stages sequentially:

1. extract and consolidate requirements;
2. validate requirements;
3. generate scenarios;
4. generate detailed test cases; and
5. review groundedness and completeness.

The same agent retains the workflow state and uses source-linked document chunks. Python enforces stage order, schemas, IDs, revision limits, and budget accounting.

### 4.3 Centralized multi-agent

A centralized supervisor coordinates three isolated worker-agent contexts.

1. The document is divided into balanced page-aware chunk groups.
2. Workers extract candidate requirements from separate groups in parallel.
3. The supervisor reconciles overlap, dependencies, ambiguity, and duplicates.
4. Validated requirements are divided into three balanced groups.
5. Workers generate scenarios and test cases for their assigned groups in parallel.
6. The supervisor performs centralized semantic validation and requests bounded revisions.
7. Deterministic code constructs the RTM and metrics.

Workers do not communicate peer-to-peer. They receive only their assigned requirements, the document overview, relevant source chunks, shared schemas, and remaining budget. This limits coordination overhead and prevents unbounded context sharing.

Python owns the fixed control flow and budget ledger. The LLM supervisor performs only semantic tasks that deterministic code cannot perform reliably: reconciliation, relevance review, and evidence-grounded critique.

## 5. Shared document evidence service

Vector search is unnecessary for the selected experiment and would add a retrieval variable. The shared evidence service instead provides deterministic, exhaustive access:

1. extract text per PDF page;
2. detect section headings when possible;
3. divide text into bounded chunks without losing page identity;
4. assign stable chunk IDs;
5. scan every chunk during requirement extraction; and
6. retrieve later evidence directly through stored chunk IDs and page references.

Every requirement must cite at least one source chunk. Scenarios and test cases inherit and may extend those citations. A short document overview supplies global purpose, actors, modules, and business rules; source chunks supply local evidence.

This design separates model understanding from retrieval: the LLM understands text placed in its context, while the evidence service controls which verified source text is supplied at each stage.

## 6. Data model

JSON is the canonical internal representation. Excel is an export only.

### 6.1 DocumentChunk

- `chunk_id`
- `page_number`
- `section`
- `text`
- `content_hash`

### 6.2 SourceReference

- `chunk_id`
- `page_number`
- `section`
- `excerpt`

The excerpt must be a short, verbatim supporting passage from the referenced chunk. Code verifies that the excerpt occurs in the chunk after whitespace normalization.

### 6.3 Requirement

- `requirement_id`
- `title`
- `description`
- `requirement_type`: `functional`, `non_functional`, or `business`
- `module`
- `priority`: `high`, `medium`, or `low`
- `ambiguities`
- `dependency_ids`
- `source_references`

### 6.4 Scenario

- `scenario_id`
- `title`
- `objective`
- `scenario_type`: `positive`, `negative`, `boundary`, `edge`, or `state_transition`
- `preconditions`
- `requirement_ids`
- `source_references`

A scenario may cover more than one requirement so that end-to-end workflows are representable.

### 6.5 TestCase

- `test_case_id`
- `scenario_id`
- `requirement_ids`
- `title`
- `priority`: `P1`, `P2`, or `P3`
- `preconditions`
- `test_data`
- `steps`
- `source_references`

Each step contains `step_number`, `action`, and `expected_result`.

### 6.6 ExperimentRun

- `run_id`
- `document_id` and document hash
- `condition`
- exact model identifier
- temperature
- total-token budget and usage
- prompt and schema versions
- repetition number
- start/end timestamps and latency
- retry and revision counts
- status and failure category
- artifact paths

### 6.7 EvaluationScore

- randomized `blind_id`
- hidden run and test-case mapping
- evaluator ID
- correctness score
- completeness score
- executability/readability score
- groundedness/traceability score
- unsupported-or-hallucinated flag
- optional comment

## 7. Traceability and RTM

The LLM emits relationship IDs, but it does not build or calculate the RTM. Deterministic code derives:

`Requirement <-> Scenario <-> Test Case <-> SourceReference`

Validation rejects:

- duplicate artifact IDs;
- references to missing parents;
- missing or invalid source chunks;
- source excerpts absent from their chunks;
- orphan scenarios or test cases; and
- requirements with no scenario or test case.

Uncovered requirements remain in the output with an explicit coverage status. They are not silently removed.

## 8. Streamlit application

The UI contains four tabs.

### 8.1 Generate

- Accept a Gemini API key into session state only.
- Upload one PDF.
- Select one condition or all three.
- Show parsing status, run progress, budget use, validation results, and failures.

### 8.2 Artifacts

- Requirements table with source pages and excerpts.
- Scenarios grouped by requirement.
- Detailed manual test cases.
- RTM and coverage status.
- JSON and Excel downloads.

### 8.3 Experiment

- Select and validate the six evaluation PDFs.
- Show the frozen experiment manifest before execution.
- Execute or resume the 54 controlled runs.
- Preserve completed runs and execute only missing work.
- Show automated metrics and aggregate comparisons.

### 8.4 Human Evaluation

- Present randomized, blinded test-case samples.
- Collect the four rubric scores, hallucination flag, and comments.
- Prevent evaluators from seeing condition identities.
- Export anonymized scores and the private blind-ID mapping separately.

## 9. Persistence and reproducibility

Local files are sufficient; no database is required.

```text
runs/<run_id>/
  manifest.json
  chunks.json
  requirements.json
  scenarios.json
  test_cases.json
  metrics.json
  events.jsonl
```

Writes use a temporary file followed by atomic replacement. Completed outputs are immutable. A new semantic attempt receives a new run ID; transient transport retries remain events within the same run.

The API key is never persisted or logged. Logs contain prompts/response metadata, tool actions, validation results, and errors, but never hidden chain-of-thought.

The experiment manifest contains document hashes, selected condition order, exact model settings, token ceiling, prompt/schema hashes, repetition count, sampling seed, and software version. Any manifest change creates a new experiment identity.

## 10. Dataset and run protocol

Select six text-extractable PDFs from the corpus before running the experiment:

- two short documents of at most 15 pages;
- two medium documents of 16-60 pages; and
- two long documents over 60 pages.

The six documents should represent different application domains and must fit the selected model's input constraints. Record selection rules and file hashes before generation. Do not replace a document because its results are poor.

Use one additional PDF, excluded from evaluation, for pilot calibration. The pilot selects the smallest total-token ceiling under which all three conditions can complete the pilot. That ceiling is then frozen for all 54 evaluation runs.

For each of six documents, run each of three conditions three times:

`6 documents x 3 conditions x 3 repetitions = 54 runs`

Use temperature `0.0`. Repetitions remain necessary because hosted model execution can still vary. Condition order is randomized per document and repetition. A failed or budget-exhausted run remains a recorded outcome.

## 11. Budget and revision policy

A shared budget ledger counts total API input and output tokens per run. The orchestrator refuses a call whose declared maximum would exceed the remaining ceiling.

The staged single-agent and centralized multi-agent conditions receive identical limits:

- one schema-repair attempt per malformed artifact response;
- one verifier-directed semantic revision per artifact; and
- bounded transient transport retries with exponential backoff.

The single-prompt reference receives no semantic revision because a repair loop would change its definition. It may retry only transient transport failures.

Budget exhaustion, validation failure, provider rejection, timeout, and parsing failure are distinct failure categories.

## 12. Automated metrics

Compute metrics for every run, including failed runs where applicable:

- completion rate;
- schema-valid artifact rate;
- requirement source-citation coverage;
- requirement-to-scenario coverage;
- requirements with at least one positive scenario;
- requirements with at least one negative, boundary, edge, or state-transition scenario;
- RTM completeness;
- orphan artifact rate;
- invalid source-reference rate;
- total artifacts generated;
- token usage;
- latency;
- retry and revision counts; and
- budget-exhaustion rate.

Automated duplicate rate uses a preregistered deterministic heuristic: lowercase and normalize punctuation/whitespace, construct token trigrams from each test case's title and steps, and mark a pair as duplicate when Jaccard similarity is at least `0.85`. Report this as a heuristic duplicate rate, not semantic ground truth.

Automated coverage measures structural presence, not correctness. Human ratings determine whether generated content is useful and supported.

## 13. Blinded human evaluation

Three independent QA/test practitioners evaluate a stratified sample.

For each document-condition pair, select six test cases across the three repetitions, preferring two cases per repetition. Include at least one case from every available scenario type, then fill remaining positions with seeded random sampling. If fewer than six cases exist, evaluate all available cases and report the shortage.

This yields at most:

`6 documents x 3 conditions x 6 cases = 108 cases per evaluator`

Each evaluator scores every selected case from 1 to 5 for:

1. correctness;
2. completeness;
3. executability/readability; and
4. groundedness/traceability.

Evaluators also flag unsupported or hallucinated cases. Output order and condition labels are randomized. Evaluators receive the relevant source evidence but not the system identity or research hypothesis.

Report ordinal Krippendorff's alpha for inter-rater agreement per rubric dimension. Preserve individual ratings; do not force consensus after scoring.

## 14. Analysis and reporting

The primary comparison is centralized multi-agent minus staged single-agent, paired by document. The single-prompt condition is descriptive context.

For every metric, report:

- per-document results;
- condition median and interquartile range;
- paired document-level differences;
- a 95% percentile bootstrap confidence interval resampled by document;
- completion and failure counts; and
- quality versus token and latency trade-offs.

With only six documents, emphasize effect sizes, uncertainty, and observed patterns rather than treating a p-value as decisive. Repetitions estimate run variability, but they are not independent documents and must not be used as pseudoreplicates.

The final report must retain null and negative results. No metric, threshold, document, or repetition may be removed after outputs are observed unless a preregistered exclusion rule applies.

## 15. Error handling

- Reject encrypted PDFs.
- Flag empty pages and reject documents with insufficient extractable text.
- Stop a call before it can exceed the remaining token budget.
- Preserve partial artifacts when a run fails.
- Never substitute missing artifacts with another condition's output.
- Resume only missing steps in incomplete runs.
- Keep completed runs immutable.
- Display actionable errors in Streamlit while retaining machine-readable failure codes.

OCR remains out of scope; the UI instructs users to provide a text-extractable PDF.

## 16. Verification strategy

Automated checks cover:

- page-aware PDF chunking and stable chunk IDs;
- source-excerpt verification;
- schema validation;
- ID and parent-link integrity;
- deterministic RTM construction;
- coverage and duplicate metrics;
- token-budget enforcement;
- blinded sampling and deterministic randomization;
- immutable run persistence and resume behavior; and
- all three pipelines using mocked Gemini responses.

One optional live smoke test uses the pilot PDF and a real Gemini API key. It is excluded from routine automated checks because it costs money and can vary externally.

Before the 54 evaluation runs, experiment preflight verifies document hashes, model settings, token ceiling, prompt/schema hashes, repetition count, random seed, and output directory state.

## 17. Acceptance criteria

The implementation is ready for experimentation when:

1. one text-extractable PDF can produce validated requirements, scenarios, test cases, citations, and RTM under each condition;
2. the staged single-agent and centralized multi-agent runs share the same evidence, schemas, revision policy, and total-token ceiling;
3. every artifact can be traced to valid source chunks;
4. completed runs are reproducible from their manifests and cannot be overwritten;
5. experiment preflight can create and validate a 54-run manifest;
6. human samples are blinded, balanced, and repeatable from the stored seed;
7. automated metrics and Excel/JSON exports are generated without manual editing; and
8. focused automated checks pass without calling the live Gemini API.
