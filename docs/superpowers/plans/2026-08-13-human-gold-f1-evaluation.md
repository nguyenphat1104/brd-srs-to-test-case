# Human-Gold F1 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a human-reviewed gold-label workflow that measures requirement extraction F1 and test-coverage F1 without using an LLM as its own judge.

**Architecture:** Keep evaluation independent from generation. Pydantic models define the immutable gold-label and match-decision contracts, a new pure-Python module proposes candidates and computes deterministic one-to-one F1, and `RunRepository` stores validated label-set JSON by document hash. Comparison-run persistence and the review UI arrive in the later plans.

**Tech Stack:** Python 3.11, Pydantic 2, PostgreSQL/psycopg 3, pytest. No new dependencies.

---

## Scope boundaries

This plan implements only the evaluator foundation:

- strict `GoldLabelSet` JSON with document-hash binding;
- human-review candidate pairs for requirements and test intents;
- deterministic maximum-cardinality one-to-one matching;
- requirement precision/recall/F1 and test-coverage precision/recall/F1;
- append-only label-set persistence and retrieval.

It deliberately does not add the role-based generation graph, comparison records, aggregate macro/micro reports, or Streamlit screens. Those depend on this API and are covered by the next two plans.

## Metric contract

- Requirement predictions are every `Requirement` in a generated bundle.
- Test-coverage predictions are every `TestCase` in a generated bundle.
- Gold denominators are all gold requirements and all gold test intents.
- A true positive is an edge accepted by a human reviewer and selected by deterministic one-to-one matching.
- `precision = TP / predicted`, `recall = TP / gold`, and `F1 = 2PR / (P + R)`.
- A zero denominator produces `0.0`, never `NaN` or an exception.
- Candidate scores assist the reviewer only; they never count as matches by themselves.

## Task 1: Add strict gold-label and evaluation contracts

**Files:**

- Modify: `src/brd_srs_testgen/models.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_evaluation.py` with the shared fixtures and contract tests:

```python
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from brd_srs_testgen.models import (
    ArtifactKind,
    GoldLabelSet,
    GoldLabelStatus,
    GoldRequirement,
    GoldTestIntent,
    MatchDecision,
    RequirementPriority,
    RequirementType,
    ScenarioType,
)


def gold_labels(**changes: object) -> GoldLabelSet:
    values: dict[str, object] = {
        "label_set_id": "gold-001",
        "document_hash": "a" * 64,
        "page_count": 1,
        "version": 1,
        "status": GoldLabelStatus.APPROVED,
        "author": "qa@example.com",
        "created_at": datetime.now(UTC),
        "requirements": [
            GoldRequirement(
                gold_requirement_id="GREQ-001",
                title="Authenticate registered users",
                description="Registered users can sign in with valid credentials.",
                requirement_type=RequirementType.FUNCTIONAL,
                priority=RequirementPriority.HIGH,
                source_pages=[1],
            )
        ],
        "test_intents": [
            GoldTestIntent(
                gold_test_intent_id="GTI-001",
                requirement_ids=["GREQ-001"],
                title="Successful sign in",
                behavior="Submit valid credentials.",
                essential_assertion="The dashboard is displayed.",
                scenario_type=ScenarioType.POSITIVE,
            )
        ],
    }
    values.update(changes)
    return GoldLabelSet(**values)


def test_gold_labels_require_at_least_one_requirement_and_test_intent() -> None:
    with pytest.raises(ValidationError):
        gold_labels(requirements=[])
    with pytest.raises(ValidationError):
        gold_labels(test_intents=[])


def test_gold_labels_reject_duplicate_ids_and_unknown_requirement_links() -> None:
    labels = gold_labels()
    with pytest.raises(ValidationError, match="duplicate gold requirement"):
        gold_labels(requirements=labels.requirements * 2)
    with pytest.raises(ValidationError, match="unknown gold requirement"):
        gold_labels(
            test_intents=[
                labels.test_intents[0].model_copy(
                    update={"requirement_ids": ["GREQ-999"]}
                )
            ]
        )


def test_gold_labels_reject_source_pages_beyond_the_document() -> None:
    labels = gold_labels()
    with pytest.raises(ValidationError, match="outside the document"):
        gold_labels(
            requirements=[
                labels.requirements[0].model_copy(update={"source_pages": [2]})
            ]
        )


def test_match_decision_is_a_strict_accepted_or_rejected_pair() -> None:
    decision = MatchDecision(
        artifact_kind=ArtifactKind.REQUIREMENT,
        prediction_id="REQ-001",
        gold_id="GREQ-001",
        accepted=True,
    )
    assert decision.accepted is True
    with pytest.raises(ValidationError):
        MatchDecision.model_validate({**decision.model_dump(), "confidence": 0.9})
```

- [ ] **Step 2: Run the focused test and confirm it fails on missing imports**

Run:

```bash
rtk .venv/bin/pytest tests/test_evaluation.py -q
```

Expected: collection fails because the new model names do not exist yet.

- [ ] **Step 3: Add the model types**

Add these types after `TestPriority` in `src/brd_srs_testgen/models.py`:

```python
class ArtifactKind(StrEnum):
    REQUIREMENT = "requirement"
    TEST_INTENT = "test_intent"


class GoldLabelStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"


class GoldRequirement(StrictModel):
    gold_requirement_id: str = Field(pattern=r"^GREQ-\d{3,}$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requirement_type: RequirementType
    priority: RequirementPriority
    source_pages: list[int] = Field(min_length=1)


class GoldTestIntent(StrictModel):
    gold_test_intent_id: str = Field(pattern=r"^GTI-\d{3,}$")
    requirement_ids: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    behavior: str = Field(min_length=1)
    essential_assertion: str = Field(min_length=1)
    scenario_type: ScenarioType


class GoldLabelSet(StrictModel):
    label_set_id: str = Field(min_length=1)
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    version: int = Field(ge=1)
    status: GoldLabelStatus
    author: str = Field(min_length=1)
    created_at: AwareDatetime
    requirements: list[GoldRequirement] = Field(min_length=1)
    test_intents: list[GoldTestIntent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        requirement_ids = [item.gold_requirement_id for item in self.requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("duplicate gold requirement IDs")
        intent_ids = [item.gold_test_intent_id for item in self.test_intents]
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("duplicate gold test intent IDs")
        known = set(requirement_ids)
        if any(
            not 1 <= page <= self.page_count
            for requirement in self.requirements
            for page in requirement.source_pages
        ):
            raise ValueError("gold source page is outside the document")
        for intent in self.test_intents:
            unknown = set(intent.requirement_ids) - known
            if unknown:
                raise ValueError("test intent links an unknown gold requirement")
        return self


class MatchCandidate(StrictModel):
    artifact_kind: ArtifactKind
    prediction_id: str = Field(min_length=1)
    gold_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class MatchDecision(StrictModel):
    artifact_kind: ArtifactKind
    prediction_id: str = Field(min_length=1)
    gold_id: str = Field(min_length=1)
    accepted: bool


class F1Score(StrictModel):
    true_positives: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    gold_count: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class BundleEvaluation(StrictModel):
    matched_requirement_pairs: list[tuple[str, str]]
    matched_test_intent_pairs: list[tuple[str, str]]
    requirement_score: F1Score
    test_coverage_score: F1Score
```

Keep these contracts in the existing domain-model module; a separate model package would add indirection without reducing coupling.

- [ ] **Step 4: Run the model tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_evaluation.py -q
```

Expected: all model tests pass.

- [ ] **Step 5: Commit the contract**

```bash
rtk git add src/brd_srs_testgen/models.py tests/test_evaluation.py
rtk git commit -m "feat: define human gold evaluation models"
```

## Task 2: Propose review candidates without making decisions

**Files:**

- Create: `src/brd_srs_testgen/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Add failing candidate tests**

Append tests that use `tests.factories.bundle()`:

```python
from brd_srs_testgen.evaluation import (
    propose_requirement_matches,
    propose_test_intent_matches,
)
from tests.factories import bundle


def test_requirement_candidates_rank_matching_text_page_and_type_first() -> None:
    labels = gold_labels()

    candidates = propose_requirement_matches(bundle(), labels, limit=5)

    assert candidates[0].prediction_id == "REQ-001"
    assert candidates[0].gold_id == "GREQ-001"
    assert 0 < candidates[0].score <= 1


def test_test_intent_candidates_rank_behavior_and_scenario_type_first() -> None:
    labels = gold_labels()

    candidates = propose_test_intent_matches(bundle(), labels, limit=5)

    assert candidates[0].prediction_id == "TC-001"
    assert candidates[0].gold_id == "GTI-001"


def test_candidate_limit_is_applied_per_prediction() -> None:
    labels = gold_labels()
    extra = [
        labels.requirements[0].model_copy(
            update={
                "gold_requirement_id": f"GREQ-{index:03d}",
                "title": f"Alternative {index}",
            }
        )
        for index in range(2, 8)
    ]
    labels = labels.model_copy(
        update={"requirements": [labels.requirements[0], *extra]}
    )

    candidates = propose_requirement_matches(bundle(), labels, limit=3)

    assert len(candidates) == 3
```

- [ ] **Step 2: Confirm the tests fail because the module is absent**

Run:

```bash
rtk .venv/bin/pytest tests/test_evaluation.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `brd_srs_testgen.evaluation`.

- [ ] **Step 3: Implement normalized overlap and requirement candidates**

Create `src/brd_srs_testgen/evaluation.py`:

```python
from __future__ import annotations

import re
from collections.abc import Iterable

from .models import (
    ArtifactBundle,
    ArtifactKind,
    BundleEvaluation,
    F1Score,
    GoldLabelSet,
    MatchCandidate,
    MatchDecision,
)


def _tokens(*values: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", " ".join(values).casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _top_per_prediction(
    candidates: Iterable[MatchCandidate], limit: int
) -> list[MatchCandidate]:
    if limit < 1:
        raise ValueError("Candidate limit must be positive.")
    grouped: dict[str, list[MatchCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.prediction_id, []).append(candidate)
    return [
        candidate
        for prediction_id in sorted(grouped)
        for candidate in sorted(
            grouped[prediction_id], key=lambda item: (-item.score, item.gold_id)
        )[:limit]
    ]


def propose_requirement_matches(
    bundle: ArtifactBundle, labels: GoldLabelSet, *, limit: int = 5
) -> list[MatchCandidate]:
    candidates = []
    for prediction in bundle.requirements:
        prediction_tokens = _tokens(prediction.title, prediction.description)
        prediction_pages = {
            source.page_number for source in prediction.source_references
        }
        for gold in labels.requirements:
            text = _jaccard(
                prediction_tokens, _tokens(gold.title, gold.description)
            )
            page = float(bool(prediction_pages & set(gold.source_pages)))
            kind = float(prediction.requirement_type is gold.requirement_type)
            candidates.append(
                MatchCandidate(
                    artifact_kind=ArtifactKind.REQUIREMENT,
                    prediction_id=prediction.requirement_id,
                    gold_id=gold.gold_requirement_id,
                    score=round(0.7 * text + 0.2 * page + 0.1 * kind, 6),
                )
            )
    return _top_per_prediction(candidates, limit)
```

- [ ] **Step 4: Implement test-intent candidates**

Append to the same module:

```python
def propose_test_intent_matches(
    bundle: ArtifactBundle, labels: GoldLabelSet, *, limit: int = 5
) -> list[MatchCandidate]:
    scenarios = {item.scenario_id: item for item in bundle.scenarios}
    candidates = []
    for prediction in bundle.test_cases:
        scenario = scenarios.get(prediction.scenario_id)
        prediction_tokens = _tokens(
            prediction.title,
            *(step.action for step in prediction.steps),
            *(step.expected_result for step in prediction.steps),
        )
        for gold in labels.test_intents:
            text = _jaccard(
                prediction_tokens,
                _tokens(gold.title, gold.behavior, gold.essential_assertion),
            )
            kind = float(
                scenario is not None and scenario.scenario_type is gold.scenario_type
            )
            candidates.append(
                MatchCandidate(
                    artifact_kind=ArtifactKind.TEST_INTENT,
                    prediction_id=prediction.test_case_id,
                    gold_id=gold.gold_test_intent_id,
                    score=round(0.8 * text + 0.2 * kind, 6),
                )
            )
    return _top_per_prediction(candidates, limit)
```

The weights are deterministic reviewer aids, not learned thresholds and not metric inputs.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_evaluation.py -q
```

Expected: candidate tests and the earlier model tests pass.

- [ ] **Step 6: Commit candidate generation**

```bash
rtk git add src/brd_srs_testgen/evaluation.py tests/test_evaluation.py
rtk git commit -m "feat: propose human review match candidates"
```

## Task 3: Compute deterministic one-to-one F1

**Files:**

- Modify: `src/brd_srs_testgen/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Add failing matching and metric tests**

Append:

```python
from brd_srs_testgen.evaluation import evaluate_bundle, score_pairs


def test_score_pairs_uses_maximum_cardinality_one_to_one_matching() -> None:
    decisions = [
        MatchDecision(
            artifact_kind=ArtifactKind.REQUIREMENT,
            prediction_id="REQ-001",
            gold_id="GREQ-001",
            accepted=True,
        ),
        MatchDecision(
            artifact_kind=ArtifactKind.REQUIREMENT,
            prediction_id="REQ-001",
            gold_id="GREQ-002",
            accepted=True,
        ),
        MatchDecision(
            artifact_kind=ArtifactKind.REQUIREMENT,
            prediction_id="REQ-002",
            gold_id="GREQ-001",
            accepted=True,
        ),
    ]

    pairs, score = score_pairs(
        ["REQ-001", "REQ-002"],
        ["GREQ-001", "GREQ-002"],
        decisions,
        ArtifactKind.REQUIREMENT,
    )

    assert len(pairs) == 2
    assert score.true_positives == 2
    assert score.f1 == 1.0


def test_rejected_and_unknown_pairs_never_count() -> None:
    decisions = [
        MatchDecision(
            artifact_kind=ArtifactKind.REQUIREMENT,
            prediction_id="REQ-001",
            gold_id="GREQ-001",
            accepted=False,
        ),
        MatchDecision(
            artifact_kind=ArtifactKind.REQUIREMENT,
            prediction_id="REQ-999",
            gold_id="GREQ-001",
            accepted=True,
        ),
    ]

    pairs, score = score_pairs(
        ["REQ-001"], ["GREQ-001"], decisions, ArtifactKind.REQUIREMENT
    )

    assert pairs == []
    assert score.model_dump() == {
        "true_positives": 0,
        "predicted_count": 1,
        "gold_count": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_zero_denominators_return_zero_scores() -> None:
    pairs, score = score_pairs([], [], [], ArtifactKind.REQUIREMENT)
    assert pairs == []
    assert score.precision == score.recall == score.f1 == 0.0


def test_bundle_evaluation_keeps_requirement_and_test_f1_separate() -> None:
    decisions = [
        MatchDecision(
            artifact_kind=ArtifactKind.REQUIREMENT,
            prediction_id="REQ-001",
            gold_id="GREQ-001",
            accepted=True,
        )
    ]

    result = evaluate_bundle(bundle(), gold_labels(), decisions)

    assert result.requirement_score.f1 == 1.0
    assert result.test_coverage_score.f1 == 0.0
```

- [ ] **Step 2: Confirm the new tests fail on missing functions**

Run:

```bash
rtk .venv/bin/pytest tests/test_evaluation.py -q
```

Expected: import failure for `evaluate_bundle` and `score_pairs`.

- [ ] **Step 3: Implement deterministic augmenting-path matching**

Append to `evaluation.py`:

```python
def _maximum_matching(adjacency: dict[str, list[str]]) -> list[tuple[str, str]]:
    gold_to_prediction: dict[str, str] = {}

    def assign(prediction_id: str, seen: set[str]) -> bool:
        for gold_id in adjacency.get(prediction_id, []):
            if gold_id in seen:
                continue
            seen.add(gold_id)
            previous = gold_to_prediction.get(gold_id)
            if previous is None or assign(previous, seen):
                gold_to_prediction[gold_id] = prediction_id
                return True
        return False

    for prediction_id in sorted(adjacency):
        assign(prediction_id, set())
    return sorted(
        (prediction_id, gold_id)
        for gold_id, prediction_id in gold_to_prediction.items()
    )


def _f1(true_positives: int, predicted_count: int, gold_count: int) -> F1Score:
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return F1Score(
        true_positives=true_positives,
        predicted_count=predicted_count,
        gold_count=gold_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def score_pairs(
    prediction_ids: Iterable[str],
    gold_ids: Iterable[str],
    decisions: Iterable[MatchDecision],
    artifact_kind: ArtifactKind,
) -> tuple[list[tuple[str, str]], F1Score]:
    predictions = set(prediction_ids)
    gold = set(gold_ids)
    adjacency: dict[str, list[str]] = {}
    for decision in decisions:
        if (
            decision.artifact_kind is artifact_kind
            and decision.accepted
            and decision.prediction_id in predictions
            and decision.gold_id in gold
        ):
            adjacency.setdefault(decision.prediction_id, []).append(decision.gold_id)
    adjacency = {
        prediction_id: sorted(set(gold_ids))
        for prediction_id, gold_ids in adjacency.items()
    }
    pairs = _maximum_matching(adjacency)
    return pairs, _f1(len(pairs), len(predictions), len(gold))
```

- [ ] **Step 4: Implement bundle-level evaluation**

Append:

```python
def evaluate_bundle(
    bundle: ArtifactBundle,
    labels: GoldLabelSet,
    decisions: Iterable[MatchDecision],
) -> BundleEvaluation:
    decisions = list(decisions)
    requirement_pairs, requirement_score = score_pairs(
        (item.requirement_id for item in bundle.requirements),
        (item.gold_requirement_id for item in labels.requirements),
        decisions,
        ArtifactKind.REQUIREMENT,
    )
    test_pairs, test_score = score_pairs(
        (item.test_case_id for item in bundle.test_cases),
        (item.gold_test_intent_id for item in labels.test_intents),
        decisions,
        ArtifactKind.TEST_INTENT,
    )
    return BundleEvaluation(
        matched_requirement_pairs=requirement_pairs,
        matched_test_intent_pairs=test_pairs,
        requirement_score=requirement_score,
        test_coverage_score=test_score,
    )
```

- [ ] **Step 5: Run evaluator and model tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_evaluation.py tests/test_models.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit deterministic scoring**

```bash
rtk git add src/brd_srs_testgen/evaluation.py tests/test_evaluation.py
rtk git commit -m "feat: score human matches with deterministic f1"
```

## Task 4: Persist immutable label sets by document hash

**Files:**

- Modify: `src/brd_srs_testgen/schema.sql`
- Modify: `src/brd_srs_testgen/storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Add failing repository tests**

Append to `tests/test_storage.py`:

```python
from brd_srs_testgen.models import GoldLabelSet
from tests.test_evaluation import gold_labels


def test_gold_label_sets_round_trip_as_strict_json(repository: RunRepository) -> None:
    labels = gold_labels()

    repository.save_gold_label_set(labels)

    assert repository.load_gold_label_set(labels.label_set_id) == labels
    assert repository.list_gold_label_sets(labels.document_hash) == [labels]


def test_gold_label_sets_are_append_only(repository: RunRepository) -> None:
    labels = gold_labels()
    repository.save_gold_label_set(labels)

    with pytest.raises(ImmutableRunError, match="already exists"):
        repository.save_gold_label_set(labels)


def test_gold_label_list_is_isolated_by_document_hash(
    repository: RunRepository,
) -> None:
    repository.save_gold_label_set(gold_labels())

    assert repository.list_gold_label_sets("b" * 64) == []
```

- [ ] **Step 2: Run the storage tests and confirm missing methods**

Run:

```bash
rtk .venv/bin/pytest tests/test_storage.py -q
```

Expected: failures report that `RunRepository` has no gold-label methods.

- [ ] **Step 3: Add the storage table**

Append to `schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS gold_label_sets (
    label_set_id text PRIMARY KEY CHECK (label_set_id <> ''),
    document_hash text NOT NULL CHECK (document_hash ~ '^[0-9a-f]{64}$'),
    version integer NOT NULL CHECK (version > 0),
    page_count integer NOT NULL CHECK (page_count > 0),
    status text NOT NULL CHECK (status IN ('draft', 'approved')),
    author text NOT NULL CHECK (author <> ''),
    created_at timestamptz NOT NULL,
    payload jsonb NOT NULL,
    UNIQUE (document_hash, version)
);

CREATE INDEX IF NOT EXISTS gold_label_sets_document_idx
    ON gold_label_sets (document_hash, version DESC);
```

The JSONB payload is canonical only after `GoldLabelSet.model_validate`; metadata columns support filtering and database constraints without duplicating every label field into join tables.

- [ ] **Step 4: Implement repository methods**

Import `GoldLabelSet` in `storage.py`, then add:

```python
    def save_gold_label_set(self, labels: GoldLabelSet) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO gold_label_sets "
                    "(label_set_id, document_hash, page_count, version, status, author, created_at, payload) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        labels.label_set_id,
                        labels.document_hash,
                        labels.page_count,
                        labels.version,
                        labels.status.value,
                        labels.author,
                        labels.created_at,
                        Jsonb(labels.model_dump(mode="json")),
                    ),
                )
        except psycopg.errors.UniqueViolation as error:
            raise ImmutableRunError("Gold label set already exists.") from error
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error

    def load_gold_label_set(self, label_set_id: str) -> GoldLabelSet:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT payload FROM gold_label_sets WHERE label_set_id = %s",
                    (label_set_id,),
                ).fetchone()
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error
        if row is None:
            raise StorageError("Gold label set does not exist.")
        try:
            return GoldLabelSet.model_validate(row["payload"])
        except ValidationError as error:
            raise StorageError("Stored gold label set is invalid.") from error

    def list_gold_label_sets(self, document_hash: str) -> list[GoldLabelSet]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT payload FROM gold_label_sets "
                    "WHERE document_hash = %s ORDER BY version DESC",
                    (document_hash,),
                ).fetchall()
        except psycopg.Error as error:
            raise StorageError("Database operation failed.") from error
        try:
            return [GoldLabelSet.model_validate(row["payload"]) for row in rows]
        except ValidationError as error:
            raise StorageError("Stored gold label set is invalid.") from error
```

- [ ] **Step 5: Run storage tests**

Run:

```bash
rtk .venv/bin/pytest tests/test_storage.py -q
```

Expected: all storage tests pass against the local PostgreSQL fixture.

- [ ] **Step 6: Commit label persistence**

```bash
rtk git add src/brd_srs_testgen/schema.sql src/brd_srs_testgen/storage.py tests/test_storage.py
rtk git commit -m "feat: persist immutable human gold labels"
```

## Task 5: Verify the evaluator foundation

**Files:**

- No production changes expected.

- [ ] **Step 1: Run formatting-independent syntax checks**

Run:

```bash
rtk .venv/bin/python -m compileall -q src tests
```

Expected: exit code `0` with no output.

- [ ] **Step 2: Run the evaluator-focused suite**

Run:

```bash
rtk .venv/bin/pytest tests/test_evaluation.py tests/test_models.py tests/test_storage.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the full regression suite**

Run:

```bash
rtk .venv/bin/pytest -q
```

Expected: all tests pass with no regressions.

- [ ] **Step 4: Inspect the final diff**

Run:

```bash
rtk git diff --check
rtk git status --short
```

Expected: `git diff --check` produces no output; status contains only the intended evaluator files if commits were skipped, or is clean if the planned commits were made.

## Acceptance checklist

- [ ] Invalid, duplicate, or cross-document gold labels are rejected before scoring.
- [ ] Gold source-page anchors cannot exceed the declared document page count.
- [ ] Candidate scores never create automatic true positives.
- [ ] Human-accepted edges are reduced to a deterministic maximum-cardinality one-to-one match.
- [ ] Requirement F1 and test-coverage F1 remain distinct.
- [ ] Empty denominators return zero.
- [ ] Label sets are append-only and retrievable by document hash/version.
- [ ] No new dependency or LLM judge was added.
