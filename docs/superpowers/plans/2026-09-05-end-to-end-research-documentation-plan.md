# End-to-End Research Documentation Implementation Plan

Design source: `docs/superpowers/specs/2026-09-05-end-to-end-research-documentation-design.md`

## Goal

Expand the existing HTML guide into a source-verified research reference covering the shared data pipeline, the three run strategies, evaluation controls, result-affecting technology, and failure behavior. Retain the overview diagram and add one detailed diagram per run type.

## Constraints

- The user-facing guide remains HTML.
- Preserve the existing home-page link and visual language.
- Describe only behavior present in the repository.
- Keep deployment technology out of the result-affecting stack section.
- Preserve unrelated uncommitted work.
- Use the installed Archify skill for diagram generation and validation.

## Task 1: Add documentation contract tests

Files:

- Add `tests/test_system_documentation.py`

Steps:

1. Read `static/system-and-coverage.html` with `pathlib.Path`.
2. Add assertions for the required research sections:
   - End-to-end data pipeline
   - Run-type comparison
   - Single Prompt
   - Staged Single Agent
   - Centralized Multi-Agent
   - Evaluation methodology
   - Result-affecting technology
   - Interpretation and reproducibility limits
3. Assert that the fixed Judge policy names Gemini 3.6 Flash, medium thinking, and the separate 100,000-token ceiling.
4. Assert that the main page references the overview plus all three run-specific diagram files and gives each iframe a descriptive title.
5. Assert that every referenced local HTML asset exists.
6. Run the new test and confirm it fails before the page is expanded:

   `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_system_documentation.py`

## Task 2: Define and generate the four source-backed diagrams

Files:

- Update `docs/system-and-coverage.workflow.json`
- Add diagram source files only if Archify requires them for repeatable generation
- Update `static/system-and-coverage-diagram.html`
- Add `static/single-prompt-flow.html`
- Add `static/staged-single-agent-flow.html`
- Add `static/centralized-multi-agent-flow.html`

Steps:

1. Read the Archify skill completely before diagram work.
2. Reconcile the overview diagram with the approved shared pipeline, including evaluation of every artifact-producing run.
3. Generate the Single Prompt flow:
   - Full evidence input
   - One `ArtifactBundle` call
   - Canonicalization and validation
   - Conditional repair/revision
   - Metrics, Judge, and persistence
4. Generate the Staged Single Agent flow:
   - `RequirementBatch → ScenarioBatch → TestCaseBatch`
   - Per-step prompts, thinking levels, and output limits
   - Whole-bundle revision using the Test cases limit
5. Generate the Centralized Multi-Agent flow:
   - Evidence partitioning
   - Analyst workers
   - Deterministic requirement reconciliation
   - Requirement redistribution with dependency context
   - Test Generator workers
   - Deterministic merge/normalization
   - Conditional Reviewer correction
   - Remote concurrency versus bounded local execution
6. Keep labels concise and use text—not color alone—to distinguish model calls, deterministic transformations, storage, and optional paths.
7. Run Archify validation for every generated diagram and fix all reported structural or rendering issues.

## Task 3: Expand the HTML research guide

Files:

- Update `static/system-and-coverage.html`

Steps:

1. Retain the hero, back link, and embedded overview diagram.
2. Add an experimental-frame introduction naming:
   - Controlled conditions
   - Independent variables
   - Generated artifacts
   - Evaluation outputs
3. Replace the short run-flow list with an end-to-end pipeline table. Each row must identify input, transformation, output, configurable influences, and failure behavior.
4. Add the run-type comparison matrix using the approved comparison fields.
5. Add one detailed section and embedded diagram for each run type.
6. Under each run type, explain agents or steps, evidence scope, intermediate schemas, prompts and controls, retry/correction behavior, and the experimental trade-off.
7. Expand evaluation methodology while preserving the existing metric formulas:
   - Deterministic quality and traceability
   - Fixed-Judge semantic coverage
   - Precision, recall, and F1 calculation
8. Add the result-affecting technology section covering extraction/chunking, Pydantic schemas, provider adapters, Python orchestration, validation, the Judge, and PostgreSQL snapshots.
9. Add failure and interpretation sections that distinguish parsing, provider, schema, truncation, budget, semantic-validation, and Judge outcomes.
10. Keep tables horizontally scrollable and text readable at narrow widths. Every iframe must have a descriptive `title`.

## Task 4: Verify content and rendering

Files:

- Update `tests/test_system_documentation.py` only if the implementation exposes a legitimate missing assertion

Steps:

1. Run the documentation contract test and confirm it passes.
2. Search the guide and diagrams for stale claims, including:
   - Coverage only running for valid bundles
   - Coverage Analyzer as a configurable generation agent
   - Staged revision fixed at 16,000 tokens
3. Start a local static server without changing project files.
4. Open the guide in a browser and inspect desktop and mobile widths.
5. Verify headings, tables, iframe loading, readable diagram labels, keyboard scrolling, and local links.
6. Run an HTML parser/link check over the main page and diagram files.

## Task 5: Run repository verification and deploy

Steps:

1. Read the verification-before-completion skill completely.
2. Run fresh verification:

   - `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q`
   - `rtk .venv/bin/python -m compileall -q app.py src tests`
   - `rtk git diff --check`

3. Review the final diff and confirm only intended documentation, diagram, and test changes were added during this implementation.
4. Rebuild and restart the app:

   `rtk docker compose up -d --build app`

5. Confirm the container is healthy and the Streamlit health endpoint returns `ok`.
6. Open the deployed home-page documentation link and verify the expanded HTML guide and all embedded diagrams load.

## Completion Evidence

- Documentation contract test passes.
- Every diagram passes Archify validation.
- Desktop and mobile rendering are inspected.
- Full repository verification passes.
- Deployed app is healthy and serves the expanded guide from the existing home-page link.
