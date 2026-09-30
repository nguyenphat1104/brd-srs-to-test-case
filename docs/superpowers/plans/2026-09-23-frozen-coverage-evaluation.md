# Frozen Coverage Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make coverage F1 comparable across runs by extracting one versioned coverage catalog per document, reusing it for every arm, persisting all judge decisions, and validating automated scores against independent human ratings.

**Architecture:** Split the current two-call judge into catalog extraction and run-specific mapping. Persist the typed catalog and mapping batches as JSONB snapshots because they are immutable audit records, not relational working data. Keep legacy ratings readable, add append-only multi-rater records, and expose evaluator failures instead of silently dropping them.

**Tech Stack:** Python 3.11, Pydantic, PostgreSQL/psycopg, Streamlit, pytest, Python standard library statistics.

---

## Scope and invariants

- This plan implements Sections 4, 8, 9, and the evaluation parts of Section 12 in `docs/superpowers/specs/2026-09-23-coverage-first-multi-agent-enhancement-design.md`.
- A `(document_hash, evaluator_version)` pair has exactly one catalog. Every run for that pair maps against the same `catalog_id`.
- Catalog units contain quoted `SourceReference` evidence, not only chunk IDs.
- Catalog extraction and test-case mapping are separate operations. Mapping failure never changes generation artifacts or their deterministic validation result.
- Evaluator failure is persisted and shown; it never becomes an implicit zero and is never silently ignored.
- Existing `human_coverage_ratings` data remains readable. New ratings are append-only and allow two raters, four dimensions, and adjudication.
- No new dependency is required.

## Task 1: Add the evaluator domain contracts

**Files:**
- Modify: `src/brd_srs_testgen/models.py`
- Modify: `tests/test_coverage.py`
- Modify: `tests/factories.py`

- [ ] **Step 1: Write model tests that describe the immutable records**

Add tests covering catalog identity, quoted evidence, evaluation state, and rating uniqueness fields:

```python
from datetime import UTC, datetime

from brd_srs_testgen.models import (
    CoverageCatalog,
    CoverageCatalogStatus,
    CoverageEvaluation,
    CoverageEvaluationStatus,
    CoverageUnit,
    HumanRating,
    HumanRatingDimension,
    SourceReference,
)


def test_coverage_catalog_keeps_version_and_quoted_evidence():
    catalog = CoverageCatalog(
        catalog_id="cat-doc-v1",
        document_hash="a" * 64,
        evaluator_version="coverage-v2:gemini-3.6-flash:medium",
        status=CoverageCatalogStatus.MACHINE_FROZEN,
        units=[
            CoverageUnit(
                unit_id="CU-001",
                title="Reject expired token",
                description="The API rejects an expired token.",
                unit_type="business_rule",
                source_references=[
                    SourceReference(
                        chunk_id="chunk-001",
                        page_number=2,
                        section="Authentication",
                        excerpt="Expired tokens must be rejected.",
                    )
                ],
            )
        ],
        created_at=datetime.now(UTC),
    )

    assert catalog.units[0].source_references[0].excerpt.startswith("Expired")


def test_human_rating_identifies_rater_dimension_and_round():
    rating = HumanRating(
        rating_id="rating-1",
        run_id="run-1",
        rater_id="rater-a",
        dimension=HumanRatingDimension.EXECUTABILITY,
        score=4,
        reason="Steps and outcomes are directly usable.",
        rubric_version="quality-rubric-v1",
        round=1,
        created_at=datetime.now(UTC),
    )

    assert rating.dimension is HumanRatingDimension.EXECUTABILITY
```

- [ ] **Step 2: Run the focused test and confirm it fails on missing models**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_coverage.py`

Expected: FAIL with import errors for the new catalog/evaluation/rating models.

- [ ] **Step 3: Add the minimum models**

Add these contracts next to the existing coverage models:

```python
class CoverageCatalogStatus(StrEnum):
    MACHINE_FROZEN = "machine_frozen"
    APPROVED = "approved"
    SUPERSEDED = "superseded"


class CoverageEvaluationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class HumanRatingDimension(StrEnum):
    COVERAGE = "coverage"
    GROUNDEDNESS = "groundedness"
    EXECUTABILITY = "executability"
    REDUNDANCY_CONTROL = "redundancy_control"


class CoverageUnit(StrictModel):
    unit_id: str = Field(pattern=r"^CU-\d{3,}$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    unit_type: str = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class CoverageCatalog(StrictModel):
    catalog_id: str = Field(min_length=1)
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_version: str = Field(min_length=1)
    status: CoverageCatalogStatus
    units: list[CoverageUnit] = Field(min_length=1)
    created_at: AwareDatetime
    approved_at: AwareDatetime | None = None


class CoverageEvaluation(StrictModel):
    run_id: str = Field(min_length=1)
    catalog_id: str = Field(min_length=1)
    status: CoverageEvaluationStatus
    mappings: CoverageMappingBatch | None = None
    score: CoverageScore | None = None
    error: str = Field(default="", max_length=2_000)
    evaluated_at: AwareDatetime


class HumanRating(StrictModel):
    rating_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    rater_id: str = Field(min_length=1, max_length=120)
    dimension: HumanRatingDimension
    score: int = Field(ge=1, le=4)
    reason: str = Field(default="", max_length=2_000)
    rubric_version: str = Field(min_length=1)
    round: int = Field(default=1, ge=1)
    created_at: AwareDatetime


class HumanAdjudication(StrictModel):
    adjudication_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    dimension: HumanRatingDimension
    score: int = Field(ge=1, le=4)
    reason: str = Field(min_length=1, max_length=2_000)
    rubric_version: str = Field(min_length=1)
    created_at: AwareDatetime
```

Add `catalog_id: str = Field(min_length=1)` to `CoverageScore`. Keep `CoverageUnitBatch` as the provider response shape and use its `units` to construct `CoverageCatalog`.

- [ ] **Step 4: Update coverage fixtures and run tests**

Replace `source_chunk_ids=["chunk-001"]` with a complete `source_references=[source_reference()]`, and supply `catalog_id="catalog-1"` to every `CoverageScore` fixture.

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_coverage.py tests/test_evaluation.py`

Expected: PASS.

- [ ] **Step 5: Commit the contracts**

```bash
rtk proxy git add src/brd_srs_testgen/models.py tests/test_coverage.py tests/test_evaluation.py tests/factories.py
rtk proxy git commit -m "feat: model frozen coverage evaluation"
```

## Task 2: Separate catalog extraction from run mapping

**Files:**
- Modify: `src/brd_srs_testgen/coverage.py`
- Modify: `tests/test_coverage.py`

- [ ] **Step 1: Add failing tests for a stable catalog and run-specific mapping**

```python
def test_extract_coverage_catalog_assigns_identity_and_version(fake_context, chunks):
    fake_context.responses = [coverage_unit_batch()]

    catalog = extract_coverage_catalog(
        fake_context,
        chunks,
        document_hash="a" * 64,
        evaluator_version="coverage-v2:test",
        catalog_id="catalog-1",
        created_at=datetime(2026, 9, 23, tzinfo=UTC),
    )

    assert catalog.catalog_id == "catalog-1"
    assert catalog.document_hash == "a" * 64
    assert catalog.status is CoverageCatalogStatus.MACHINE_FROZEN


def test_evaluate_against_catalog_never_extracts_units(fake_context, bundle):
    fake_context.responses = [coverage_mapping_batch()]

    evaluation = evaluate_against_catalog(
        fake_context,
        run_id="run-1",
        bundle=bundle,
        catalog=coverage_catalog(),
        evaluated_at=datetime(2026, 9, 23, tzinfo=UTC),
    )

    assert len(fake_context.prompts) == 1
    assert "extract every testable" not in fake_context.prompts[0]
    assert evaluation.score.catalog_id == "catalog-1"
```

- [ ] **Step 2: Run and observe the missing functions**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_coverage.py`

Expected: FAIL because `extract_coverage_catalog` and `evaluate_against_catalog` do not exist.

- [ ] **Step 3: Change the extraction prompt to require exact quotations**

In `extract_coverage_units_prompt`, replace the chunk-ID-only instruction with:

```text
Cite every unit with one or more SourceReference objects. Copy chunk_id, page_number,
section, and a short verbatim excerpt from the PDF evidence. A unit without quoted
support is invalid. Split compound statements into atomic units; do not invent behavior.
```

- [ ] **Step 4: Implement the two operations**

```python
def extract_coverage_catalog(
    context: PipelineContext,
    chunks: list[DocumentChunk],
    *,
    document_hash: str,
    evaluator_version: str,
    catalog_id: str,
    created_at: datetime,
) -> CoverageCatalog:
    batch = context.generate(
        [_user(extract_coverage_units_prompt(chunks, setup=context.agent_setup("coverage_analyzer")))],
        CoverageUnitBatch,
        max_output_tokens=16_000,
        agent="coverage_analyzer",
    )
    batch = canonicalize_source_references(batch, chunks)
    return CoverageCatalog(
        catalog_id=catalog_id,
        document_hash=document_hash,
        evaluator_version=evaluator_version,
        status=CoverageCatalogStatus.MACHINE_FROZEN,
        units=batch.units,
        created_at=created_at,
    )


def evaluate_against_catalog(
    context: PipelineContext,
    *,
    run_id: str,
    bundle: ArtifactBundle,
    catalog: CoverageCatalog,
    evaluated_at: datetime,
) -> CoverageEvaluation:
    units = CoverageUnitBatch(units=catalog.units)
    mappings = context.generate(
        [_user(map_test_cases_prompt(bundle, units, setup=context.agent_setup("coverage_analyzer")))],
        CoverageMappingBatch,
        max_output_tokens=16_000,
        agent="coverage_analyzer",
    )
    score = compute_f1(units, mappings, bundle, catalog_id=catalog.catalog_id)
    return CoverageEvaluation(
        run_id=run_id,
        catalog_id=catalog.catalog_id,
        status=CoverageEvaluationStatus.COMPLETED,
        mappings=mappings,
        score=score,
        evaluated_at=evaluated_at,
    )
```

Import `canonicalize_source_references` and add a required keyword-only `catalog_id` parameter to `compute_f1`.

- [ ] **Step 5: Add deterministic boundary tests**

Cover unknown test-case IDs, unknown unit IDs, omitted test cases, duplicate mapping entries, and an empty test-case bundle. Reject duplicate mapping entries with `ValueError`; do not let last-write-wins hide a judge defect.

- [ ] **Step 6: Run focused tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_coverage.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/coverage.py src/brd_srs_testgen/models.py tests/test_coverage.py
rtk proxy git commit -m "refactor: freeze coverage catalogs before mapping"
```

## Task 3: Persist catalogs, mappings, scores, and failures

**Files:**
- Modify: `src/brd_srs_testgen/schema.sql`
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Add failing repository round-trip tests**

Write tests for:

1. `save_coverage_catalog` then `load_coverage_catalog(document_hash, evaluator_version)` returns the same typed catalog.
2. Saving the same `(document_hash, evaluator_version)` twice returns the existing row and does not replace units.
3. `approve_coverage_catalog` changes only `status` and `approved_at`.
4. `save_coverage_evaluation` round-trips completed mappings and scores.
5. A failed evaluation round-trips `status="failed"`, an empty mapping, and a non-empty error.

- [ ] **Step 2: Run the storage tests and confirm the API is missing**

Run: `rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py`

Expected: FAIL on missing repository methods.

- [ ] **Step 3: Add idempotent schema objects**

Append these tables and columns to `schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS coverage_catalogs (
    catalog_id text PRIMARY KEY CHECK (catalog_id <> ''),
    document_hash text NOT NULL CHECK (document_hash ~ '^[0-9a-f]{64}$'),
    evaluator_version text NOT NULL CHECK (evaluator_version <> ''),
    status text NOT NULL CHECK (status IN ('machine_frozen', 'approved', 'superseded')),
    units jsonb NOT NULL CHECK (jsonb_typeof(units) = 'array'),
    created_at timestamptz NOT NULL,
    approved_at timestamptz,
    UNIQUE (document_hash, evaluator_version)
);

CREATE TABLE IF NOT EXISTS coverage_evaluations (
    run_id text PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    catalog_id text NOT NULL REFERENCES coverage_catalogs(catalog_id),
    status text NOT NULL CHECK (status IN ('completed', 'failed')),
    mappings jsonb,
    error text NOT NULL DEFAULT '' CHECK (length(error) <= 2000),
    evaluated_at timestamptz NOT NULL,
    CHECK (
        (status = 'completed' AND mappings IS NOT NULL AND error = '') OR
        (status = 'failed' AND mappings IS NULL AND error <> '')
    )
);

ALTER TABLE coverage_scores
    ADD COLUMN IF NOT EXISTS catalog_id text REFERENCES coverage_catalogs(catalog_id);
```

Use each model's `model_dump(mode="json")` inside `Jsonb(model.model_dump(mode="json"))`. Do not create normalized unit/mapping tables: the app loads complete immutable snapshots and never queries individual units across catalogs.

- [ ] **Step 4: Implement repository methods with typed validation**

Add `save_coverage_catalog`, `load_coverage_catalog`, `approve_coverage_catalog`, `save_coverage_evaluation`, and `load_coverage_evaluation` with the model types shown in Task 1. Use PostgreSQL's `ON CONFLICT (document_hash, evaluator_version) DO NOTHING`, then select and return the stored catalog. Raise `ImmutableRunError` if a caller attempts to replace a different stored snapshot.

- [ ] **Step 5: Make finalization atomic**

Extend `RunResult` with `coverage_evaluation: CoverageEvaluation | None`. In `RunRepository.finalize`, insert evaluation before score in the same transaction. Insert `coverage_scores.catalog_id` from `result.coverage.catalog_id`. Update `_load_coverage` and `load_run` accordingly.

- [ ] **Step 6: Run tests and commit**

Run: `rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py tests/test_coverage.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/schema.sql src/brd_srs_testgen/storage.py src/brd_srs_testgen/models.py tests/test_storage.py tests/test_coverage.py
rtk proxy git commit -m "feat: persist reproducible coverage evidence"
```

## Task 4: Reuse catalogs in the runner and expose evaluator failure

**Files:**
- Modify: `src/brd_srs_testgen/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Add failing runner tests**

Test these paths:

- an existing catalog causes one judge call (mapping only);
- a missing catalog causes extraction, persistence, then mapping;
- all three run types for the same document use the same `catalog_id`;
- judge failure leaves generation completed when deterministic validation passes, persists `CoverageEvaluationStatus.FAILED`, and reports the error through progress;
- judge failure does not create a `CoverageScore`.

- [ ] **Step 2: Run the focused runner tests**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_runner.py`

Expected: FAIL because the runner still invokes `run_coverage_analysis` and swallows all exceptions.

- [ ] **Step 3: Define an explicit evaluator version**

Next to the judge constants, add:

```python
COVERAGE_PROMPT_VERSION = "coverage-v2"
COVERAGE_SCHEMA_VERSION = "coverage-catalog-v1"
EVALUATOR_VERSION = (
    f"{COVERAGE_PROMPT_VERSION}:{COVERAGE_SCHEMA_VERSION}:"
    f"{JUDGE_PROVIDER}:{JUDGE_MODEL}:{JUDGE_THINKING_LEVEL}"
)
```

Any future prompt, schema, model, or thinking change must change one component and therefore create a new catalog.

- [ ] **Step 4: Replace the silent block with catalog reuse**

After building the judge context:

```python
catalog = repository.load_coverage_catalog(document_hash, EVALUATOR_VERSION)
if catalog is None:
    catalog = extract_coverage_catalog(
        judge_context,
        chunks,
        document_hash=document_hash,
        evaluator_version=EVALUATOR_VERSION,
        catalog_id=f"{document_hash[:16]}-{hashlib.sha256(EVALUATOR_VERSION.encode()).hexdigest()[:12]}",
        created_at=_now(),
    )
    catalog = repository.save_coverage_catalog(catalog)
coverage_evaluation = evaluate_against_catalog(
    judge_context,
    run_id=manifest.run_id,
    bundle=bundle,
    catalog=catalog,
    evaluated_at=_now(),
)
coverage = coverage_evaluation.score
```

Catch only `ConfigurationError`, `ProviderError`, `PipelineOutputError`, `ValueError`, and Pydantic validation errors. Build a failed `CoverageEvaluation` with a user-safe error string, notify progress, and persist it through `RunResult`. Unknown programming errors must still propagate.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_runner.py tests/test_coverage.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/runner.py src/brd_srs_testgen/models.py tests/test_runner.py
rtk proxy git commit -m "fix: reuse coverage catalog across runs"
```

## Task 5: Add independent ratings, adjudication, and agreement gates

**Files:**
- Modify: `src/brd_srs_testgen/schema.sql`
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `src/brd_srs_testgen/evaluation.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing agreement tests**

```python
def test_agreement_summary_is_per_dimension():
    ratings = human_ratings_for_two_raters()

    summary = summarize_human_agreement(ratings)

    assert summary[HumanRatingDimension.COVERAGE].pair_count == 3
    assert summary[HumanRatingDimension.COVERAGE].quadratic_weighted_kappa == 1.0


def test_agreement_gate_requires_every_dimension_at_point_seven():
    assert agreement_gate({dimension: 0.70 for dimension in HumanRatingDimension})
    assert not agreement_gate({
        **{dimension: 0.80 for dimension in HumanRatingDimension},
        HumanRatingDimension.EXECUTABILITY: 0.69,
    })
```

Also test exact agreement and adjacent agreement, a missing rater pair, and constant equal ratings. Preserve the existing `quadratic_weighted_kappa` tests.

- [ ] **Step 2: Run and confirm the summary API is absent**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_evaluation.py`

Expected: FAIL on missing `summarize_human_agreement` and `agreement_gate`.

- [ ] **Step 3: Add append-only tables**

```sql
CREATE TABLE IF NOT EXISTS human_ratings (
    rating_id text PRIMARY KEY CHECK (rating_id <> ''),
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    rater_id text NOT NULL CHECK (rater_id <> ''),
    dimension text NOT NULL CHECK (dimension IN ('coverage', 'groundedness', 'executability', 'redundancy_control')),
    score smallint NOT NULL CHECK (score BETWEEN 1 AND 4),
    reason text NOT NULL DEFAULT '' CHECK (length(reason) <= 2000),
    rubric_version text NOT NULL CHECK (rubric_version <> ''),
    round integer NOT NULL DEFAULT 1 CHECK (round > 0),
    created_at timestamptz NOT NULL,
    UNIQUE (run_id, rater_id, dimension, round)
);

CREATE TABLE IF NOT EXISTS human_adjudications (
    adjudication_id text PRIMARY KEY CHECK (adjudication_id <> ''),
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    dimension text NOT NULL CHECK (dimension IN ('coverage', 'groundedness', 'executability', 'redundancy_control')),
    score smallint NOT NULL CHECK (score BETWEEN 1 AND 4),
    reason text NOT NULL CHECK (reason <> '' AND length(reason) <= 2000),
    rubric_version text NOT NULL CHECK (rubric_version <> ''),
    created_at timestamptz NOT NULL,
    UNIQUE (run_id, dimension)
);
```

Do not delete or rewrite `human_coverage_ratings`; label it legacy in repository method names and UI.

- [ ] **Step 4: Implement append-only repository methods**

Add `save_human_rating`, `list_human_ratings`, `save_human_adjudication`, and `list_human_adjudications`. Let the database uniqueness constraints enforce immutability and translate `UniqueViolation` to `ImmutableRunError`.

- [ ] **Step 5: Implement agreement summaries from paired ratings**

Pair ratings by `(run_id, dimension, round)`, require exactly two distinct raters, and compute:

```python
exact = sum(left == right for left, right in pairs) / len(pairs)
adjacent = sum(abs(left - right) <= 1 for left, right in pairs) / len(pairs)
kappa = quadratic_weighted_kappa(pairs)
```

The operational gate is `kappa >= 0.70` for every dimension with at least two paired items. Report insufficient data separately; do not treat it as passing.

- [ ] **Step 6: Run evaluation and storage tests, then commit**

Run: `rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_evaluation.py tests/test_storage.py`

Expected: PASS.

```bash
rtk proxy git add src/brd_srs_testgen/schema.sql src/brd_srs_testgen/storage.py src/brd_srs_testgen/evaluation.py src/brd_srs_testgen/models.py tests/test_storage.py tests/test_evaluation.py
rtk proxy git commit -m "feat: add blinded multi-rater agreement"
```

## Task 6: Expose catalog review and evaluator evidence in Streamlit

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add failing UI-unit tests**

Extend the fake repository and test:

- the run result shows catalog ID/status, coverage units with quoted evidence, and each test-case mapping;
- an evaluator failure shows its message and never displays F1 `0.000`;
- only a machine-frozen catalog shows an Approve action;
- the rating form requires `rater_id` and all four dimensions;
- a saved rating cannot be edited;
- adjudication is separate from initial ratings;
- agreement is reported per dimension with exact, adjacent, and quadratic-weighted kappa values.

- [ ] **Step 2: Run the focused app tests**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_app.py`

Expected: FAIL on missing repository methods and render helpers.

- [ ] **Step 3: Replace the single coverage rating panel**

Replace `_render_human_coverage_rating` with `_render_human_ratings`, using a four-row form keyed by `HumanRatingDimension`. Keep a short rubric beside each score and store `QUALITY_RUBRIC_VERSION = "quality-rubric-v1"` in `evaluation.py`.

- [ ] **Step 4: Add catalog/evaluation panels**

Render the immutable catalog and mapping snapshots with `st.dataframe`. Put quoted evidence in a detail expander. Approval changes only catalog status; do not add in-place unit editing because edited evidence would require a new evaluator version and new catalog.

- [ ] **Step 5: Run tests and commit**

Run: `rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_app.py tests/test_evaluation.py`

Expected: PASS.

```bash
rtk proxy git add app.py src/brd_srs_testgen/evaluation.py tests/test_app.py
rtk proxy git commit -m "feat: show coverage audit and blinded ratings"
```

## Task 7: Document and verify the evaluator gate

**Files:**
- Modify: `README.md`
- Modify: `docs/research-methodology.md` if it exists; otherwise create it

- [ ] **Step 1: Document the operating sequence**

Document: upload document → extract catalog once → inspect quoted evidence → approve catalog → run comparison arms → collect two blind ratings → adjudicate disagreements → check per-dimension QWK ≥ 0.70.

- [ ] **Step 2: Document metric semantics**

State explicitly that test-case precision is the proportion of test cases mapped to at least one catalog unit, recall is the proportion of catalog units covered by at least one test case, and F1 is their harmonic mean. Label this a project-specific semantic coverage F1, not classifier micro-F1.

- [ ] **Step 3: Run fresh verification**

Run:

```bash
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_coverage.py tests/test_evaluation.py tests/test_runner.py tests/test_app.py
rtk env PYTHONPATH=src TEST_DATABASE_URL=postgresql://brd_srs:brd_srs_local@127.0.0.1:5432/brd_srs_test .venv/bin/python -m pytest -q tests/test_storage.py
rtk env PYTHONPATH=src .venv/bin/python -m pytest -q
```

Expected: all tests PASS. If the full suite requires the database, rerun the final command with `TEST_DATABASE_URL` set as above.

- [ ] **Step 4: Inspect the final diff for accidental metric drift**

Run: `rtk proxy git diff --check && rtk proxy git diff --stat`

Confirm the old F1 formula is unchanged except for `catalog_id`, and confirm no broad `except Exception` remains around coverage evaluation.

- [ ] **Step 5: Commit documentation**

```bash
rtk proxy git add README.md docs/research-methodology.md
rtk proxy git commit -m "docs: explain frozen coverage evaluation"
```

## Completion gate

- The same document and evaluator version resolve to the same catalog ID and unit snapshot.
- Every coverage score links to a persisted catalog and mapping batch.
- Evaluator failure is visible, persisted, and excluded from score comparisons.
- Two independent raters can score all four dimensions without seeing condition labels.
- Agreement is computed per dimension using quadratic-weighted Cohen kappa, exact agreement, and adjacent agreement.
- No comparison experiment starts until its catalog is approved; Plan 3 enforces this boundary.
