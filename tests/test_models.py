import pytest
from pydantic import ValidationError

from brd_srs_testgen.models import (
    ArtifactBundle,
    Requirement,
    RequirementPriority,
    RequirementType,
    Scenario,
    ScenarioType,
    SourceReference,
    TestCase,
    TestPriority,
    TestStep,
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
    scenario = Scenario(
        scenario_id="SCN-001",
        title="Valid sign in",
        objective="Verify successful authentication.",
        scenario_type=ScenarioType.POSITIVE,
        requirement_ids=["REQ-001"],
        source_references=[source()],
    )
    test_case = TestCase(
        test_case_id="TC-001",
        scenario_id="SCN-001",
        requirement_ids=["REQ-001"],
        title="Sign in with valid credentials",
        priority=TestPriority.P1,
        preconditions=["A registered account exists."],
        test_data={"email": "user@example.com"},
        steps=[
            TestStep(
                step_number=1,
                action="Submit valid credentials.",
                expected_result="The dashboard is displayed.",
            )
        ],
        source_references=[source()],
    )

    bundle = ArtifactBundle(
        requirements=[requirement], scenarios=[scenario], test_cases=[test_case]
    )

    assert bundle.test_cases[0].requirement_ids == ["REQ-001"]


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
