from datetime import UTC, datetime, timedelta

import brd_srs_testgen
import pytest
from pydantic import ValidationError

import brd_srs_testgen.models as models
from brd_srs_testgen.models import (
    ArtifactBundle,
    FailureCategory,
    Requirement,
    RequirementPriority,
    RequirementType,
    RunHistoryItem,
    RunManifest,
    RunMetrics,
    RunResult,
    RunStatus,
    RunType,
    Scenario,
    ScenarioType,
    SourceReference,
    TestCase as Case,
    TestPriority as Priority,
    TestStep as Step,
)
from tests.factories import completed_run


def source() -> SourceReference:
    return SourceReference(
        chunk_id="p0001-c001-a1b2c3d4e5f6",
        page_number=1,
        section="Authentication",
        excerpt="The system shall authenticate registered users.",
    )


def test_bundle_accepts_many_to_many_traceability() -> None:
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Authenticate users",
        description="Registered users can sign in.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Authentication",
        priority=RequirementPriority.HIGH,
        source_references=[source()],
    )
    second_requirement = Requirement(
        requirement_id="REQ-002",
        title="Reject invalid credentials",
        description="Invalid credentials are rejected.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Authentication",
        priority=RequirementPriority.HIGH,
        source_references=[source()],
    )
    scenario = Scenario(
        scenario_id="SCN-001",
        title="Valid sign in",
        objective="Verify successful authentication.",
        scenario_type=ScenarioType.POSITIVE,
        requirement_ids=["REQ-001", "REQ-002"],
        source_references=[source()],
    )
    test_case = Case(
        test_case_id="TC-001",
        scenario_id="SCN-001",
        requirement_ids=["REQ-001", "REQ-002"],
        title="Sign in with valid credentials",
        priority=Priority.P1,
        preconditions=["A registered account exists."],
        test_data={"email": "user@example.com"},
        steps=[
            Step(
                step_number=1,
                action="Submit valid credentials.",
                expected_result="The dashboard is displayed.",
            )
        ],
        source_references=[source()],
    )

    bundle = ArtifactBundle(
        requirements=[requirement, second_requirement],
        scenarios=[scenario],
        test_cases=[test_case],
    )

    assert bundle.test_cases[0].requirement_ids == ["REQ-001", "REQ-002"]


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SourceReference.model_validate(
            {
                "chunk_id": "chunk",
                "page_number": 1,
                "section": "Section",
                "excerpt": "Evidence",
                "invented": True,
            }
        )


def test_artifact_ids_follow_canonical_patterns() -> None:
    with pytest.raises(ValidationError):
        Requirement(
            requirement_id="wrong",
            title="Title",
            description="Description",
            requirement_type=RequirementType.BUSINESS,
            module="Core",
            priority=RequirementPriority.MEDIUM,
            source_references=[source()],
        )


def test_test_case_accepts_json_compatible_test_data() -> None:
    test_case = Case(
        test_case_id="TC-001",
        scenario_id="SCN-001",
        requirement_ids=["REQ-001"],
        title="Sign in with structured data",
        priority=Priority.P1,
        test_data={"attempts": 3, "remember": True, "roles": ["user", None]},
        steps=[Step(step_number=1, action="Sign in.", expected_result="Signed in.")],
        source_references=[source()],
    )

    assert test_case.test_data["roles"] == ["user", None]


def metrics(**overrides: object) -> RunMetrics:
    values: dict[str, object] = {
        "completion": True,
        "schema_valid": True,
        "citation_coverage": 1.0,
        "requirement_scenario_coverage": 1.0,
        "requirement_test_case_coverage": 1.0,
        "positive_scenario_coverage": 1.0,
        "non_positive_scenario_coverage": 1.0,
        "rtm_completeness": 1.0,
        "orphan_rate": 0.0,
        "invalid_reference_rate": 0.0,
        "duplicate_test_case_rate": 0.0,
        "requirement_count": 1,
        "scenario_count": 1,
        "test_case_count": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "latency_seconds": 0.1,
        "retries": 0,
        "schema_repairs": 0,
        "semantic_revisions": 0,
        "budget_exhausted": False,
    }
    values.update(overrides)
    return RunMetrics(**values)


def test_metrics_default_charged_tokens_for_backward_compatibility() -> None:
    assert metrics().charged_tokens == 0


@pytest.mark.parametrize(
    "field, value",
    [
        ("citation_coverage", 1.1),
        ("requirement_scenario_coverage", 1.1),
        ("requirement_test_case_coverage", 1.1),
        ("positive_scenario_coverage", 1.1),
        ("non_positive_scenario_coverage", 1.1),
        ("rtm_completeness", 1.1),
        ("orphan_rate", -0.1),
        ("invalid_reference_rate", 1.1),
        ("duplicate_test_case_rate", 1.1),
        ("requirement_count", -1),
        ("scenario_count", -1),
        ("test_case_count", -1),
        ("input_tokens", -1),
        ("output_tokens", -1),
        ("charged_tokens", -1),
        ("latency_seconds", -0.1),
        ("retries", -1),
        ("schema_repairs", -1),
        ("semantic_revisions", -1),
    ],
)
def test_metrics_reject_out_of_range_or_negative_values(
    field: str, value: float | int
) -> None:
    with pytest.raises(ValidationError):
        metrics(**{field: value})


def run_manifest(**overrides: object) -> RunManifest:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "run_id": "20260812T120000000000Z-ecac9f035813-12345678",
        "source_filename": "sample.pdf",
        "document_hash": "a" * 64,
        "run_type": RunType.SINGLE_PROMPT,
        "status": RunStatus.RUNNING,
        "provider": "ollama",
        "model": "gemma4",
        "temperature": 0.0,
        "token_ceiling": 100_000,
        "prompt_version": "research-core-v1",
        "schema_version": "research-core-v1",
        "started_at": now,
    }
    values.update(overrides)
    return RunManifest(**values)


def test_running_run_has_no_terminal_fields() -> None:
    manifest = run_manifest()

    assert manifest.run_type is RunType.SINGLE_PROMPT
    assert manifest.completed_at is None


@pytest.mark.parametrize(
    "field", ["completed_at", "failure_category", "failure_message"]
)
def test_running_run_rejects_each_terminal_field(field: str) -> None:
    now = datetime.now(UTC)
    value: object = {
        "completed_at": now,
        "failure_category": FailureCategory.PARSING,
        "failure_message": "Unexpected failure.",
    }[field]

    with pytest.raises(ValidationError, match="running runs cannot have terminal fields"):
        run_manifest(started_at=now, **{field: value})


def test_completed_at_cannot_predate_started_at() -> None:
    now = datetime.now(UTC)

    with pytest.raises(
        ValidationError, match="completed_at cannot be earlier than started_at"
    ):
        run_manifest(started_at=now, completed_at=now - timedelta(seconds=1))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failure_category", FailureCategory.PARSING),
        ("failure_message", "Unexpected failure."),
    ],
)
def test_completed_run_rejects_failure_details(field: str, value: object) -> None:
    now = datetime.now(UTC)

    with pytest.raises(
        ValidationError, match="completed runs cannot have failure details"
    ):
        run_manifest(
            status=RunStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", ""),
        ("source_filename", ""),
        ("provider", ""),
        ("model", ""),
        ("prompt_version", ""),
        ("schema_version", ""),
        ("document_hash", "A" * 64),
        ("temperature", -0.1),
        ("token_ceiling", 0),
        ("started_at", datetime.now()),
    ],
)
def test_run_manifest_rejects_invalid_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        run_manifest(**{field: value})


def test_completed_run_requires_completion_time() -> None:
    with pytest.raises(ValidationError, match="completed runs require completed_at"):
        run_manifest(status=RunStatus.COMPLETED)


@pytest.mark.parametrize(
    "overrides",
    [
        {"failure_category": FailureCategory.PARSING},
        {
            "started_at": "2026-08-11T00:00:00+00:00",
            "completed_at": "2026-08-11T00:00:01+00:00",
        },
    ],
)
def test_failed_run_requires_category_and_completion_time(
    overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="failed runs require"):
        run_manifest(status=RunStatus.FAILED, **overrides)


def test_failed_run_accepts_safe_failure_details() -> None:
    now = datetime.now(UTC)
    manifest = run_manifest(
        status=RunStatus.FAILED,
        started_at=now,
        completed_at=now,
        failure_category=FailureCategory.PARSING,
        failure_message="PDF contains insufficient extractable text.",
    )

    assert manifest.failure_category is FailureCategory.PARSING


def test_run_result_download_bundle_handles_missing_artifacts() -> None:
    result = RunResult(manifest=run_manifest())

    assert result.download_bundle() == {
        "manifest": result.manifest.model_dump(mode="json"),
        "requirements": [],
        "scenarios": [],
        "test_cases": [],
        "validation": None,
        "rtm": [],
        "metrics": None,
        "coverage": None,
    }


def test_run_result_download_bundle_serializes_populated_result() -> None:
    payload = completed_run().download_bundle()

    assert payload["requirements"][0]["requirement_id"] == "REQ-001"
    assert payload["scenarios"][0]["scenario_id"] == "SCN-001"
    assert payload["test_cases"][0]["test_case_id"] == "TC-001"
    assert payload["validation"] == {
        "valid": True,
        "issues": [],
        "uncovered_requirement_ids": [],
        "orphan_scenario_ids": [],
        "orphan_test_case_ids": [],
    }
    assert payload["rtm"][0]["covered"] is True
    assert payload["metrics"]["charged_tokens"] == 30


def test_run_history_item_displays_running_as_interrupted() -> None:
    item = RunHistoryItem(
        run_id="run-001",
        source_filename="sample.pdf",
        run_type=RunType.SINGLE_PROMPT,
        status=RunStatus.RUNNING,
        provider="ollama",
        model="gemma4",
        started_at=datetime.now(UTC),
        completed_at=None,
    )

    assert item.display_status == "Interrupted"


@pytest.mark.parametrize(
    ("status", "expected"),
    [(RunStatus.COMPLETED, "Completed"), (RunStatus.FAILED, "Failed")],
)
def test_run_history_item_title_cases_terminal_statuses(
    status: RunStatus, expected: str
) -> None:
    item = RunHistoryItem(
        run_id="run-001",
        source_filename="sample.pdf",
        run_type=RunType.SINGLE_PROMPT,
        status=status,
        provider="ollama",
        model="gemma4",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )

    assert item.display_status == expected


def test_package_exports_run_type_without_comparison_models() -> None:
    assert brd_srs_testgen.RunType is RunType
    assert not hasattr(brd_srs_testgen, "Condition")
    assert not hasattr(models, "Condition")
    assert not hasattr(models, "ConditionManifest")
    assert not hasattr(models, "ComparisonManifest")
