from brd_srs_testgen.models import Requirement, RequirementPriority, RequirementType
from brd_srs_testgen.validation import build_rtm, compute_metrics, validate_bundle
from tests.factories import bundle, chunk, source


def test_valid_bundle_builds_complete_rtm() -> None:
    artifacts = bundle()
    report = validate_bundle(artifacts, [chunk()])
    rows = build_rtm(artifacts)

    assert report.valid
    assert rows[0].requirement_id == "REQ-001"
    assert rows[0].scenario_ids == ["SCN-001"]
    assert rows[0].test_case_ids == ["TC-001"]
    assert rows[0].covered


def test_invalid_parent_and_excerpt_are_reported() -> None:
    artifacts = bundle()
    bad_case = artifacts.test_cases[0].model_copy(
        update={
            "scenario_id": "SCN-999",
            "source_references": [
                source().model_copy(update={"excerpt": "invented evidence"})
            ],
        }
    )
    artifacts = artifacts.model_copy(update={"test_cases": [bad_case]})

    report = validate_bundle(artifacts, [chunk()])

    assert not report.valid
    assert {issue.code for issue in report.issues} >= {
        "missing_scenario",
        "invalid_source_reference",
    }


def test_uncovered_requirement_remains_in_rtm() -> None:
    artifacts = bundle()
    second = Requirement(
        requirement_id="REQ-002",
        title="Audit sign in",
        description="Sign-in attempts are audited.",
        requirement_type=RequirementType.BUSINESS,
        module="Audit",
        priority=RequirementPriority.MEDIUM,
        source_references=[source()],
    )
    artifacts = artifacts.model_copy(
        update={"requirements": [*artifacts.requirements, second]}
    )

    report = validate_bundle(artifacts, [chunk()])
    rows = build_rtm(artifacts)

    assert report.uncovered_requirement_ids == ["REQ-002"]
    assert rows[1].requirement_id == "REQ-002"
    assert not rows[1].covered


def test_orphan_case_retains_requirement_test_case_coverage() -> None:
    artifacts = bundle()
    orphan = artifacts.test_cases[0].model_copy(update={"scenario_id": "SCN-999"})
    artifacts = artifacts.model_copy(update={"test_cases": [orphan]})

    report = validate_bundle(artifacts, [chunk()])
    metrics = compute_metrics(
        artifacts,
        report,
        input_tokens=0,
        output_tokens=0,
        latency_seconds=0,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )

    assert {issue.code for issue in report.issues} >= {
        "missing_scenario",
        "orphan_test_case",
    }
    assert report.uncovered_requirement_ids == []
    assert metrics.requirement_test_case_coverage == 1.0


def test_metrics_include_usage_and_duplicate_rate() -> None:
    artifacts = bundle()
    duplicate = artifacts.test_cases[0].model_copy(
        update={"test_case_id": "TC-002"}
    )
    artifacts = artifacts.model_copy(
        update={"test_cases": [*artifacts.test_cases, duplicate]}
    )
    report = validate_bundle(artifacts, [chunk()])

    metrics = compute_metrics(
        artifacts,
        report,
        input_tokens=100,
        output_tokens=50,
        latency_seconds=1.25,
        retries=1,
        schema_repairs=0,
        semantic_revisions=1,
        budget_exhausted=False,
    )

    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 50
    assert metrics.duplicate_test_case_rate == 1.0
