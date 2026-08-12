from datetime import UTC, datetime

from brd_srs_testgen.models import (
    ArtifactBundle,
    DocumentChunk,
    Requirement,
    RequirementPriority,
    RequirementType,
    RunManifest,
    RunResult,
    RunStatus,
    RunType,
    Scenario,
    ScenarioType,
    SourceReference,
    TestCase,
    TestPriority,
    TestStep,
)
from brd_srs_testgen.validation import build_rtm, compute_metrics, validate_bundle


def chunk() -> DocumentChunk:
    text = "The system shall authenticate registered users."
    return DocumentChunk(
        chunk_id="p0001-c001-ecac9f035813",
        page_number=1,
        section="AUTHENTICATION",
        text=text,
        content_hash="ecac9f0358134f174bcbf0d60ddbc7c25bcb4f812ea8e4c57bfbd8c02edaa274",
    )


def source() -> SourceReference:
    item = chunk()
    return SourceReference(
        chunk_id=item.chunk_id,
        page_number=item.page_number,
        section=item.section,
        excerpt=item.text,
    )


def bundle() -> ArtifactBundle:
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
    return ArtifactBundle(
        requirements=[requirement], scenarios=[scenario], test_cases=[test_case]
    )


def completed_run(
    run_id: str = "20260812T120000000000Z-ecac9f035813-12345678",
    run_type: RunType = RunType.SINGLE_PROMPT,
) -> RunResult:
    now = datetime.now(UTC)
    artifacts = bundle()
    validation = validate_bundle(artifacts, [chunk()])
    metrics = compute_metrics(
        artifacts,
        validation,
        input_tokens=10,
        output_tokens=20,
        charged_tokens=30,
        latency_seconds=0.1,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )
    return RunResult(
        manifest=RunManifest(
            run_id=run_id,
            source_filename="sample.pdf",
            document_hash="a" * 64,
            run_type=run_type,
            status=RunStatus.COMPLETED,
            provider="ollama",
            model="gemma4",
            temperature=0,
            token_ceiling=100_000,
            prompt_version="research-core-v1",
            schema_version="research-core-v1",
            started_at=now,
            completed_at=now,
        ),
        bundle=artifacts,
        validation=validation,
        rtm=build_rtm(artifacts),
        metrics=metrics,
    )
