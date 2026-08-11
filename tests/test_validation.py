from brd_srs_testgen.models import (
    ArtifactBundle,
    Requirement,
    RequirementPriority,
    RequirementType,
)
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
        charged_tokens=175,
        latency_seconds=1.25,
        retries=1,
        schema_repairs=0,
        semantic_revisions=1,
        budget_exhausted=False,
    )

    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 50
    assert metrics.charged_tokens == 175
    assert metrics.duplicate_test_case_rate == 1.0


def test_empty_bundle_is_invalid_and_incomplete() -> None:
    artifacts = ArtifactBundle(requirements=[], scenarios=[], test_cases=[])
    report = validate_bundle(artifacts, [])
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

    assert not report.valid
    assert [(issue.code, issue.artifact_id) for issue in report.issues] == [
        ("empty_bundle", "bundle")
    ]
    assert not metrics.completion


def test_citation_coverage_counts_requirement_sources_only() -> None:
    artifacts = bundle()
    bad_scenario = artifacts.scenarios[0].model_copy(
        update={
            "source_references": [
                source().model_copy(update={"excerpt": "invented evidence"})
            ]
        }
    )
    scenario_invalid = artifacts.model_copy(update={"scenarios": [bad_scenario]})
    scenario_report = validate_bundle(scenario_invalid, [chunk()])

    scenario_metrics = compute_metrics(
        scenario_invalid,
        scenario_report,
        input_tokens=0,
        output_tokens=0,
        latency_seconds=0,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )

    bad_requirement = Requirement(
        requirement_id="REQ-002",
        title="Audit sign in",
        description="Sign-in attempts are audited.",
        requirement_type=RequirementType.BUSINESS,
        module="Audit",
        priority=RequirementPriority.MEDIUM,
        source_references=[source().model_copy(update={"excerpt": "invented evidence"})],
    )
    requirement_invalid = artifacts.model_copy(
        update={"requirements": [*artifacts.requirements, bad_requirement]}
    )
    requirement_report = validate_bundle(requirement_invalid, [chunk()])

    requirement_metrics = compute_metrics(
        requirement_invalid,
        requirement_report,
        input_tokens=0,
        output_tokens=0,
        latency_seconds=0,
        retries=0,
        schema_repairs=0,
        semantic_revisions=0,
        budget_exhausted=False,
    )

    assert scenario_metrics.citation_coverage == 1.0
    assert requirement_metrics.citation_coverage == 0.5


def test_duplicate_scenario_requirement_union_is_order_independent() -> None:
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
    duplicate_scenario = artifacts.scenarios[0].model_copy(
        update={"requirement_ids": ["REQ-002"]}
    )
    test_case = artifacts.test_cases[0].model_copy(
        update={"requirement_ids": ["REQ-002"]}
    )
    forward = artifacts.model_copy(
        update={
            "requirements": [*artifacts.requirements, second],
            "scenarios": [artifacts.scenarios[0], duplicate_scenario],
            "test_cases": [test_case],
        }
    )
    reverse = forward.model_copy(update={"scenarios": list(reversed(forward.scenarios))})

    forward_issues = [issue.model_dump() for issue in validate_bundle(forward, [chunk()]).issues]
    reverse_issues = [issue.model_dump() for issue in validate_bundle(reverse, [chunk()]).issues]

    assert forward_issues == reverse_issues
    assert {issue["code"] for issue in forward_issues} == {
        "duplicate_id",
        "uncovered_requirement",
    }


def test_validation_reports_link_and_orphan_issue_branches() -> None:
    artifacts = bundle()
    requirement = artifacts.requirements[0].model_copy(
        update={"dependency_ids": ["REQ-999"]}
    )
    invalid_scenario = artifacts.scenarios[0].model_copy(
        update={"requirement_ids": ["REQ-999"]}
    )
    orphan = invalid_scenario.model_copy(
        update={"scenario_id": "SCN-002", "requirement_ids": ["REQ-001"]}
    )
    invalid_case = artifacts.test_cases[0].model_copy(
        update={"requirement_ids": ["REQ-998"]}
    )
    artifacts = artifacts.model_copy(
        update={
            "requirements": [requirement, requirement.model_copy()],
            "scenarios": [invalid_scenario, orphan],
            "test_cases": [invalid_case],
        }
    )

    report = validate_bundle(artifacts, [chunk()])

    assert {issue.code for issue in report.issues} >= {
        "duplicate_id",
        "missing_dependency",
        "missing_requirement",
        "scenario_requirement_mismatch",
        "orphan_scenario",
    }
    assert {
        (issue.code, issue.artifact_id)
        for issue in report.issues
    } >= {
        ("missing_requirement", "SCN-001"),
        ("missing_requirement", "TC-001"),
    }
