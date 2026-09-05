# End-to-End Research Documentation Design

Date: 2026-09-05  
Status: Approved

## Objective

Expand the existing HTML system guide into a research-oriented explanation of the complete experimental flow: source-document ingestion, generation strategies, validation and repair, evaluation, persistence, and the technologies that can affect results.

The primary audience is researchers and evaluators comparing the three run types. The page must make the controlled conditions, independent variables, intermediate artifacts, and interpretation limits easy to identify.

## Scope

The user-facing artifact remains one HTML page at `static/system-and-coverage.html`. It retains the existing overview workflow and adds one detailed embedded flow for each run type.

The page includes:

1. Research overview
2. End-to-end data pipeline
3. Run-type comparison matrix
4. Detailed Single Prompt flow
5. Detailed Staged Single Agent flow
6. Detailed Centralized Multi-Agent flow
7. Evaluation methodology
8. Result-affecting technology
9. Interpretation and reproducibility limits

Deployment instructions, Streamlit implementation details, and Docker operations are out of scope because they do not explain experimental outcomes.

## Information Architecture

The page uses a layered research narrative. It begins with the common experimental frame, exposes the differences between run types, then explains evaluation and implementation factors in progressively greater detail.

The existing overview diagram remains the first visual. Three smaller diagrams follow in the run-detail sections so readers can understand each strategy without decoding one oversized branch diagram.

## Shared End-to-End Pipeline

The common flow is:

`PDF → normalized evidence chunks → selected generation strategy → typed artifact bundle → citation canonicalization → deterministic validation → targeted link repair or one semantic revision → RTM and quality metrics → fixed Gemini Judge → persisted run result`

Each documented stage identifies:

- Input
- Transformation
- Output
- Configuration that can affect the result
- Failure behavior

### Intake and evidence preparation

The system creates an immutable running manifest and saves a configuration snapshot before generation. It extracts text from one text-readable PDF, normalizes it, and divides it into evidence chunks. Every chunk records a stable ID, page number, section, text, and SHA-256 content hash.

### Typed generation

The selected run strategy receives the same evidence chunks and core grounding rules. Provider calls must return structured JSON validated against Pydantic models. The final artifact bundle contains requirements, scenarios, and test cases connected by canonical IDs and source references.

### Correction and validation

Source references are canonicalized against extracted evidence. Deterministic validation checks schema relationships, ID uniqueness, dependencies, citations, orphans, and requirement coverage.

If uncovered requirements are the only problem, the Reviewer selects existing scenarios and test cases to repair links without creating new artifacts. Other failures may receive one whole-bundle semantic revision when the provider context permits it. Context-constrained llama.cpp runs do not attempt that large revision.

### Evaluation and persistence

The system builds the RTM and deterministic metrics. Every run that produced an artifact bundle is then offered to the fixed Judge for semantic coverage evaluation, including bundles that ultimately fail deterministic validation. The final manifest, artifacts, validation report, RTM, metrics, coverage result when available, events, and configuration snapshot are stored in PostgreSQL. Raw PDF bytes and provider credentials are not stored.

## Run-Type Designs

### Single Prompt

One provider call receives the complete evidence and returns one `ArtifactBundle` containing requirements, scenarios, and test cases.

Documented controls:

- Provider and model
- Run-specific prompt
- Gemini thinking level when supported
- Overall run token ceiling

The page describes this strategy as the minimal-orchestration baseline: it minimizes handoffs but asks one response to perform extraction, scenario design, case design, linking, and citation grounding together.

### Staged Single Agent

One shared provider and model make three sequential calls:

1. Complete evidence → `RequirementBatch`
2. Validated requirements plus evidence → `ScenarioBatch`
3. Validated requirements, scenarios, and evidence → `TestCaseBatch`

Each stage has an independent prompt, Gemini thinking level when supported, and output-token limit. The output of each stage becomes structured input to the next. A whole-bundle semantic revision inherits the Test cases output-token limit because it must reproduce the complete result.

This strategy isolates the effect of task decomposition while keeping the model/provider constant.

### Centralized Multi-Agent

The orchestrator partitions evidence into Analyst assignments. Analyst workers extract candidate requirements using reserved ID ranges. The application then deterministically merges duplicates, combines citations and dependencies, caps the canonical catalog, and renumbers requirements.

Canonical requirements are redistributed to Test Generator workers with relevant evidence and read-only dependency context. Workers generate scenarios and test cases in reserved ID ranges. The application namespaces IDs, validates worker boundaries, merges results, removes unusable orphan scenarios, and normalizes derived links and citations.

Remote-provider workers may run concurrently. Local-provider execution uses bounded, sequential assignments to control request size and model memory. The Reviewer participates only when targeted coverage-link repair or a final semantic revision is needed.

Documented controls:

- Separate provider, model, and prompt for Analyst, Test Generator, and Reviewer
- Shared run token ceiling
- Provider-dependent concurrency and request bounding

## Run-Type Comparison Matrix

The matrix compares:

- Orchestration pattern
- Number and shape of provider calls
- Evidence scope per call
- Intermediate structured artifacts
- Model and prompt configurability
- Thinking and output-token configurability
- Concurrency behavior
- Correction path
- Primary experimental strength
- Main interpretation risk

The matrix avoids declaring one strategy universally superior. It explains which architectural variable each strategy introduces.

## Evaluation Methodology

### Deterministic quality and traceability

The following measures are calculated directly from the generated bundle and validation report:

- Schema validity and completion
- Citation coverage
- Requirement-to-scenario coverage
- Requirement-to-test-case coverage
- Positive scenario coverage
- Non-positive scenario coverage
- RTM completeness
- Orphan rate
- Invalid-reference rate
- Duplicate test-case rate

The duplicate measure uses word-trigram Jaccard similarity with the existing threshold. No evaluator model decides these ratios.

### Fixed semantic Judge

The independent Judge is fixed across run types:

- Provider/model: Gemini 3.6 Flash
- Thinking level: medium
- Evaluation budget: separate 100,000-token ceiling
- Prompts: fixed by the application

Phase 1 extracts atomic coverage units from the full source evidence. Phase 2 maps each generated test case to units that its steps and expected results genuinely exercise. Unknown IDs are discarded before deterministic calculation of precision, recall, and F1.

Judge failure leaves semantic coverage unavailable but does not change the generation run's success or failure status.

### Fair-comparison frame

Controlled conditions are the source PDF, evidence chunking, schemas, core grounding rules, deterministic validation, quality calculations, and Judge policy.

Independent variables are the selected run type and the configured generation providers, models, prompts, thinking levels, output limits, and run token ceiling. The stored configuration snapshot supports interpretation and later comparison.

## Result-Affecting Technology

The technology section covers only components that shape inputs, generation, validation, evaluation, or reproducibility:

- PDF extraction, normalization, chunk identifiers, page metadata, and evidence hashing
- Pydantic artifact schemas and structured provider output
- Gemini, Ollama, LM Studio, and llama.cpp provider adapters
- Python prompt composition, orchestration, worker partitioning, and token ledgers
- Deterministic citation canonicalization, validation, normalization, RTM construction, and metric calculation
- Fixed Gemini Judge
- PostgreSQL configuration snapshots and normalized run results

## Failure Handling

The shared pipeline and run-specific diagrams distinguish these outcomes:

- Parsing failure: stops before generation and stores diagnostics without artifacts
- Transient provider failure: retries up to two times with bounded backoff
- Invalid structured output: attempts schema repair up to two times
- Output truncation: stops immediately because incomplete JSON cannot be repaired safely
- Token-budget exhaustion: stops when the ledger cannot reserve or settle the request
- Deterministic validation failure: attempts targeted link repair or one allowed semantic revision, then stores the remaining issues
- Judge failure: isolates the evaluation failure and preserves the generation outcome

Technical messages stored or displayed to users must continue to redact credentials and connection details.

## Documentation Components

The implementation updates the existing guide and overview workflow rather than creating a second guide. It adds three standalone diagram artifacts beside the existing diagram:

- Single Prompt detailed flow
- Staged Single Agent detailed flow
- Centralized Multi-Agent detailed flow

The diagrams use the same visual language as the overview and remain embedded in the main HTML page with descriptive titles.

## Verification

Implementation verification will include:

- Automated assertions that the HTML page contains all required sections, all three run types, the fixed-Judge policy, and references to all four diagrams
- Link checks for local documentation assets
- Desktop and mobile rendering checks for text, tables, and embedded diagrams
- A source audit against current pipeline functions, defaults, validation rules, provider behavior, and Judge configuration
- The repository's full automated test suite and HTML/build checks before deployment

## Acceptance Criteria

The design is complete when a researcher can use the page to:

1. Trace data from the uploaded PDF to stored results.
2. Explain the orchestration and intermediate artifacts of each run type.
3. Identify which conditions are controlled and which settings vary.
4. Distinguish deterministic quality metrics from Judge-assisted semantic F1.
5. Understand the result-affecting technology without deployment noise.
6. Interpret failure states and know which partial results remain available.
7. Avoid treating the scores as executed-code coverage, defect detection, or a human gold-standard evaluation.
