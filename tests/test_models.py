from datetime import datetime

import pytest
from pydantic import ValidationError

from brd_srs_testgen.models import (
    ArtifactBundle,
    Condition,
    ConditionManifest,
    ComparisonManifest,
    FailureCategory,
    Requirement,
    RequirementPriority,
    RequirementType,
    RunMetrics,
    RunStatus,
    Scenario,
    ScenarioType,
    SourceReference,
    TestCase as Case,
    TestPriority as Priority,
    TestStep as Step,
)


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


def manifest(**overrides: object) -> ConditionManifest:
    values: dict[str, object] = {
        "condition": Condition.SINGLE_PROMPT,
        "status": RunStatus.COMPLETED,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "token_ceiling": 1,
        "started_at": "2026-08-11T00:00:00+00:00",
        "completed_at": "2026-08-11T00:00:01+00:00",
    }
    values.update(overrides)
    return ConditionManifest(**values)


def test_condition_manifest_parses_timestamps_and_enforces_status_invariants() -> None:
    completed = manifest()

    assert isinstance(completed.started_at, datetime)
    assert isinstance(completed.completed_at, datetime)

    for overrides in (
        {"token_ceiling": 0},
        {"temperature": -0.1},
        {"started_at": "not-a-timestamp"},
        {"started_at": "2026-08-11T00:00:00"},
        {"completed_at": None},
        {"completed_at": "2026-08-10T23:59:59+00:00"},
        {"failure_message": "unexpected"},
        {"status": RunStatus.FAILED, "failure_category": None},
        {
            "status": RunStatus.FAILED,
            "completed_at": None,
            "failure_category": FailureCategory.TIMEOUT,
        },
        {
            "status": RunStatus.RUNNING,
            "completed_at": "2026-08-11T00:00:01+00:00",
        },
    ):
        with pytest.raises(ValidationError):
            manifest(**overrides)


def comparison_manifest(**overrides: object) -> ComparisonManifest:
    values: dict[str, object] = {
        "comparison_id": "comparison-001",
        "document_hash": "a" * 64,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "token_ceiling": 1,
        "condition_order": list(Condition),
        "prompt_version": "v1",
        "schema_version": "v1",
        "started_at": "2026-08-11T00:00:00+00:00",
        "completed_at": "2026-08-11T00:00:01+00:00",
    }
    values.update(overrides)
    return ComparisonManifest(**values)


def test_comparison_manifest_validates_boundaries_and_condition_order() -> None:
    item = comparison_manifest()

    assert isinstance(item.started_at, datetime)
    assert isinstance(item.completed_at, datetime)

    for overrides in (
        {"document_hash": "A" * 64},
        {"document_hash": "a" * 63},
        {"temperature": -0.1},
        {"token_ceiling": 0},
        {"started_at": "not-a-timestamp"},
        {"started_at": "2026-08-11T00:00:00"},
        {"completed_at": "2026-08-10T23:59:59+00:00"},
        {"condition_order": []},
        {"condition_order": [Condition.SINGLE_PROMPT, Condition.STAGED_SINGLE_AGENT]},
        {"condition_order": [Condition.SINGLE_PROMPT] * 3},
    ):
        with pytest.raises(ValidationError):
            comparison_manifest(**overrides)
