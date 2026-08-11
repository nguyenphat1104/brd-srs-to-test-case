# BRD/SRS Test-Case Research Core Design

**Status:** Approved in conversation

**Date:** 2026-08-11

**Project:** `brd-srs-to-test-case`

## 1. Purpose

Build the smallest complete research slice that can compare three agent-system conditions on one text-extractable BRD/SRS PDF. The application must turn the document into traceable requirements, scenarios, manual test cases, an RTM, and structural metrics using either Gemini or Ollama.

This slice establishes the research kernel and a thin English Streamlit interface. It does not yet run the full 54-run study or collect human evaluations.

## 2. Success criteria

A user can upload one text-extractable PDF, select Gemini or Ollama, configure the provider and model, and run all three conditions:

1. single-prompt reference;
2. staged single-agent; and
3. centralized multi-agent.

Every successful condition produces validated JSON requirements, scenarios, test cases, RTM rows, metrics, usage, and latency. Every run is persisted immutably and can also be downloaded from Streamlit. Failed conditions remain visible and preserve a categorized failure record.

## 3. Scope

### In scope

- One text-extractable PDF per comparison run.
- English UI, prompts, logs, and generated artifacts.
- Gemini and Ollama as operational provider alternatives.
- One provider and model fixed across all three conditions in a comparison run.
- Page-aware deterministic evidence chunks and source citations.
- Functional, non-functional, and business requirements.
- Positive, negative, boundary, edge, and state-transition scenarios.
- Detailed manual test cases with actions and expected results.
- Deterministic referential validation, RTM construction, and structural metrics.
- Equal temperature and token ceiling for the three conditions.
- Immutable local JSON persistence and JSON downloads.
- Mocked automated tests and documented manual provider smoke checks.

### Out of scope

- OCR or scanned-PDF recovery.
- DOCX, HTML, Markdown, or other input formats.
- Excel export.
- Automated test-script generation or execution.
- Comparing Gemini with Ollama as a research variable.
- Bulk datasets, randomized condition scheduling, resume across a 54-run study, or aggregate statistical analysis.
- Human-evaluation sampling, blinding, scoring, or agreement analysis.
- User accounts, collaboration, cloud deployment, a database, vector search, or artifact editing.

## 4. Architecture

The implementation is a focused Python package behind a thin Streamlit entry point.

### 4.1 Streamlit interface

The UI owns presentation only. It:

- accepts one PDF;
- selects `Gemini` or `Ollama`;
- collects the provider credential or local endpoint, explicit model identifier, and per-condition token ceiling;
- fixes temperature at `0.0` and worker count at `3`;
- launches the three conditions sequentially while allowing the centralized condition to make parallel worker calls;
- displays parsing, generation, budget, validation, completion, and failure status; and
- shows side-by-side summaries with individual artifact downloads and one complete JSON bundle per condition.

Provider credentials remain in Streamlit session state and are never passed to storage or event logging.

### 4.2 Document evidence

The evidence component hashes the uploaded PDF bytes with SHA-256, rejects encrypted PDFs, extracts text per page, normalizes whitespace, and creates chunks of at most 6,000 characters without crossing page boundaries or adding overlap. Empty pages are retained in page accounting but do not create chunks. Oversized paragraphs are split at sentence or word boundaries; text is never silently truncated.

Chunk IDs combine page number, page-local chunk number, and the first 12 hexadecimal characters of the normalized chunk SHA-256 hash. Every chunk stores its page, normalized text, and full content hash. Its section is the first non-empty page line of 3-100 characters that is either all uppercase or begins with a numeric heading such as `2.1`; otherwise the section is empty.

Every requirement cites at least one chunk. Scenarios and test cases inherit or add citations. A citation excerpt is valid only when its whitespace-normalized text occurs in the cited chunk.

### 4.3 Canonical models

Strict models reject unknown fields and represent:

- `DocumentChunk`;
- `SourceReference`;
- `Requirement`;
- `Scenario`;
- `TestStep`;
- `TestCase`;
- `ArtifactBundle`;
- `ValidationReport`;
- `RTMRow`;
- `RunManifest`; and
- `RunMetrics`.

Requirements, scenarios, and test cases use explicit unique IDs. Scenarios may reference multiple requirements. Every test case references exactly one scenario and one or more requirements. Each step has a positive integer order, action, and expected result.

The field definitions and enumerated values from the [broader approved design](2026-08-10-centralized-agent-test-case-generation-design.md) remain canonical unless this focused specification narrows them explicitly.

### 4.4 Provider boundary

Gemini and Ollama adapters accept the same logical request: message history, target JSON schema, temperature, and maximum output tokens. They return parsed structured content plus normalized input tokens, output tokens, latency, model identifier, and retry metadata.

Before a call, the adapter estimates input tokens and the ledger reserves that estimate plus maximum output tokens. Gemini uses its token-count endpoint. Ollama conservatively reserves the UTF-8 byte length of the serialized prompt and schema because its standard generation response reports prompt tokens only after execution. Reported usage settles the reservation. If actual usage pushes the ledger past its ceiling, the condition is marked budget-exhausted and makes no further calls.

The adapters contain provider-specific transport and usage parsing. Pipeline code does not branch on provider type. A direct `if` selection is sufficient; this slice does not add a registry, plugin system, or provider factory hierarchy.

Ollama defaults may be populated with the existing prototype values, but the base URL and model remain editable. Gemini requires an explicit API key and model identifier. Automated tests never make live provider calls.

### 4.5 Pipelines

All conditions consume the same chunks and schemas. Each receives an independent equal token ceiling so one condition cannot consume another condition's budget.

#### Single-prompt reference

One structured request receives the complete parsed evidence and produces requirements, scenarios, and test cases. It may retry transient transport failures and perform one schema-repair request for malformed output. It receives no semantic revision.

#### Staged single-agent

One continuing logical context performs these stages in order:

1. extract and consolidate requirements;
2. review requirements;
3. generate scenarios;
4. generate test cases; and
5. review groundedness and completeness.

Each request includes the prior state required to preserve the single-agent context. The pipeline permits one schema repair per malformed artifact and one verifier-directed semantic revision per artifact.

#### Centralized multi-agent

A coordinator controls three isolated worker contexts:

1. balance page-aware chunks across workers;
2. extract candidate requirements concurrently;
3. reconcile candidates centrally;
4. balance validated requirements across workers;
5. generate scenarios and test cases concurrently; and
6. review and reconcile outputs centrally.

Workers do not communicate with peers. They receive only their assignment, necessary global context, canonical schemas, and the remaining budget allocated to their work. A worker failure that remains after allowed retries fails the centralized condition; the pipeline never silently changes architecture.

### 4.6 Deterministic validation and metrics

Code, not the model, verifies:

- schema validity and unique IDs;
- requirement dependencies;
- scenario-to-requirement links;
- test-case-to-scenario and test-case-to-requirement links;
- chunk existence, page agreement, and excerpt containment;
- orphan scenarios and test cases; and
- requirement coverage by scenarios and test cases.

Code derives the RTM as `Requirement <-> Scenario <-> Test Case <-> SourceReference`. Uncovered requirements remain visible with an uncovered status.

Metrics include condition completion, schema validity, citation coverage, requirement-to-scenario coverage, requirement-to-test-case coverage, positive and non-positive scenario coverage, RTM completeness, orphan and invalid-reference rates, artifact counts, token usage, latency, retries, revisions, and budget exhaustion. Duplicate test-case rate uses normalized token trigrams and Jaccard similarity at the existing `0.85` threshold; it is labeled as a heuristic.

## 5. Data flow

1. Streamlit validates the provider configuration and uploaded PDF.
2. The evidence component hashes and parses the PDF once.
3. A comparison manifest freezes the document hash, provider, exact model, temperature, token ceiling, prompt/schema versions, condition order, and start time.
4. The three conditions run in the recorded order against the same evidence. Conditions are independent: failure in one does not prevent the others from running.
5. Each provider call reserves its declared budget before execution and reconciles the reservation against reported usage afterward.
6. Structured outputs pass through strict parsing and deterministic validation.
7. Validation constructs RTM rows and metrics for successful outputs and records categorized failures for unsuccessful outputs.
8. Storage atomically persists the comparison and condition files.
9. Streamlit reads the persisted results for display and download.

A provider context rejection or oversized request is recorded as a provider failure. The application never truncates source evidence to force a run to fit.

## 6. Storage

Local files are sufficient:

```text
runs/<comparison_id>/
  manifest.json
  chunks.json
  conditions/
    single_prompt/
      manifest.json
      requirements.json
      scenarios.json
      test_cases.json
      rtm.json
      metrics.json
      events.jsonl
    staged_single_agent/
      ...
    centralized_multi_agent/
      ...
```

The comparison ID includes a UTC timestamp and a short document-hash prefix. Writes use a file in the destination directory followed by atomic replacement. Completed condition artifacts are immutable. Re-running after any semantic completion or failure creates a new comparison ID; bounded transport retries remain events inside the same condition run.

Events contain timestamps, stage names, sanitized provider metadata, token usage, retry and revision counts, validation results, and categorized errors. They never contain API keys or hidden chain-of-thought.

## 7. Budgets, repairs, and errors

- Temperature is fixed at `0.0` for all conditions.
- The UI requires a positive per-condition token ceiling and applies the same value to every condition.
- Calls are refused before execution when estimated input tokens plus maximum output tokens would exceed the remaining ledger balance.
- Transient timeouts, connection failures, HTTP 429, and HTTP 500/502/503/504 failures receive at most two retries after the initial call, delayed by 1 then 2 seconds.
- A malformed single-prompt bundle receives at most one schema-repair request. In staged and centralized conditions, each malformed artifact response receives at most one schema-repair request.
- Single-prompt receives no semantic revision.
- Staged and centralized conditions receive at most one verifier-directed semantic revision per artifact.
- Validation never drops invalid or uncovered artifacts silently.
- Failures are categorized as parsing, configuration, provider rejection, transport exhaustion, timeout, budget exhaustion, schema failure, or semantic validation failure.
- A failed condition records its usage, latency, retries, validation findings, and failure category before the next condition begins.

## 8. Testing

Automated tests use deterministic mocked provider responses and temporary run directories. They cover:

- strict model validation and unknown-field rejection;
- PDF page extraction, stable chunk IDs, oversized paragraph splitting, and empty/scanned input rejection;
- source excerpt normalization and containment;
- Gemini and Ollama request mapping, response parsing, usage normalization, retry policy, and credential redaction;
- budget reservation, reconciliation, exhaustion, and thread safety;
- happy and failure paths for each pipeline;
- equal evidence and configuration across conditions;
- centralized worker isolation, assignment balancing, and failure propagation;
- referential validation, RTM construction, uncovered requirements, and metrics;
- duplicate-rate threshold behavior;
- atomic immutable storage and new-ID behavior on rerun; and
- a Streamlit smoke test covering upload, provider configuration, progress, side-by-side results, condition failures, and JSON downloads.

Manual smoke checks document one small PDF run with Gemini and one with Ollama. They are optional verification steps because live provider availability, credentials, cost, and nondeterminism make them unsuitable for the automated gate.

## 9. Acceptance criteria

The research core is ready when:

1. one text-extractable PDF can launch all three conditions from the English Streamlit UI using either Gemini or Ollama;
2. the chosen provider and exact model remain fixed across those conditions;
3. successful conditions produce strict, traceable requirements, scenarios, test cases, RTM rows, metrics, and usage as JSON;
4. every source excerpt is verified against a page-aware chunk;
5. invalid relationships, uncovered requirements, and failed conditions remain visible;
6. comparison and condition files are atomically persisted without credentials and are downloadable;
7. rerunning never overwrites a completed or failed semantic run; and
8. the complete mocked automated test suite passes without network access.

## 10. Deferred follow-up slices

After this slice is verified, separate specifications may add:

1. the frozen six-document, 54-run experiment manifest and resumable runner;
2. automated aggregate analysis and reporting;
3. blinded human evaluation and agreement analysis;
4. Excel exports; and
5. optional product-oriented improvements.

These features must reuse the research kernel rather than change the three condition definitions used by this slice.
