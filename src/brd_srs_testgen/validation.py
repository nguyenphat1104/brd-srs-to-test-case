from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

from .documents import verify_source_reference
from .models import (
    ArtifactBundle,
    DocumentChunk,
    RTMRow,
    RunMetrics,
    ScenarioType,
    ValidationIssue,
    ValidationReport,
)


def _duplicates(values: list[str]) -> set[str]:
    return {value for value, count in Counter(values).items() if count > 1}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _issue(code: str, artifact_id: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, artifact_id=artifact_id, message=message)


def validate_bundle(
    bundle: ArtifactBundle, chunks: list[DocumentChunk]
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    requirements = {item.requirement_id: item for item in bundle.requirements}
    scenarios = {item.scenario_id: item for item in bundle.scenarios}
    scenario_requirement_ids_by_id: dict[str, set[str]] = {}
    for scenario in bundle.scenarios:
        scenario_requirement_ids_by_id.setdefault(scenario.scenario_id, set()).update(
            scenario.requirement_ids
        )

    if not (bundle.requirements or bundle.scenarios or bundle.test_cases):
        issues.append(_issue("empty_bundle", "bundle", "Bundle has no artifacts."))
    for duplicate in sorted(
        set().union(
            _duplicates([item.requirement_id for item in bundle.requirements]),
            _duplicates([item.scenario_id for item in bundle.scenarios]),
            _duplicates([item.test_case_id for item in bundle.test_cases]),
        )
    ):
        issues.append(_issue("duplicate_id", duplicate, "ID is not unique."))

    for requirement in bundle.requirements:
        for dependency_id in sorted(requirement.dependency_ids):
            if dependency_id not in requirements:
                issues.append(
                    _issue(
                        "missing_dependency",
                        requirement.requirement_id,
                        f"Unknown dependency {dependency_id}.",
                    )
                )
        for reference in requirement.source_references:
            if not verify_source_reference(reference, chunks):
                issues.append(
                    _issue(
                        "invalid_source_reference",
                        requirement.requirement_id,
                        f"Invalid source reference {reference.chunk_id}.",
                    )
                )

    for scenario in bundle.scenarios:
        for requirement_id in sorted(scenario.requirement_ids):
            if requirement_id not in requirements:
                issues.append(
                    _issue(
                        "missing_requirement",
                        scenario.scenario_id,
                        f"Unknown requirement {requirement_id}.",
                    )
                )
        for reference in scenario.source_references:
            if not verify_source_reference(reference, chunks):
                issues.append(
                    _issue(
                        "invalid_source_reference",
                        scenario.scenario_id,
                        f"Invalid source reference {reference.chunk_id}.",
                    )
                )

    for test_case in bundle.test_cases:
        scenario = scenarios.get(test_case.scenario_id)
        if scenario is None:
            issues.append(
                _issue(
                    "missing_scenario",
                    test_case.test_case_id,
                    f"Unknown scenario {test_case.scenario_id}.",
                )
            )
        for requirement_id in sorted(test_case.requirement_ids):
            if requirement_id not in requirements:
                issues.append(
                    _issue(
                        "missing_requirement",
                        test_case.test_case_id,
                        f"Unknown requirement {requirement_id}.",
                    )
                )
        if scenario and not set(test_case.requirement_ids).issubset(
            scenario_requirement_ids_by_id[test_case.scenario_id]
        ):
            issues.append(
                _issue(
                    "scenario_requirement_mismatch",
                    test_case.test_case_id,
                    "Test-case requirements are not linked to its scenario.",
                )
            )
        for reference in test_case.source_references:
            if not verify_source_reference(reference, chunks):
                issues.append(
                    _issue(
                        "invalid_source_reference",
                        test_case.test_case_id,
                        f"Invalid source reference {reference.chunk_id}.",
                    )
                )

    orphan_scenario_ids = sorted(
        scenario.scenario_id
        for scenario in bundle.scenarios
        if not any(case.scenario_id == scenario.scenario_id for case in bundle.test_cases)
    )
    orphan_test_case_ids = sorted(
        case.test_case_id
        for case in bundle.test_cases
        if case.scenario_id not in scenarios
    )
    scenario_requirement_ids = {
        requirement_id
        for scenario in bundle.scenarios
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirements
    }
    test_requirement_ids = {
        requirement_id
        for case in bundle.test_cases
        for requirement_id in case.requirement_ids
        if requirement_id in requirements
    }
    uncovered_requirement_ids = sorted(
        requirement_id
        for requirement_id in requirements
        if requirement_id not in scenario_requirement_ids
        or requirement_id not in test_requirement_ids
    )
    for scenario_id in orphan_scenario_ids:
        issues.append(_issue("orphan_scenario", scenario_id, "Scenario has no test cases."))
    for test_case_id in orphan_test_case_ids:
        issues.append(
            _issue("orphan_test_case", test_case_id, "Test case has no valid scenario.")
        )
    for requirement_id in uncovered_requirement_ids:
        issues.append(
            _issue(
                "uncovered_requirement",
                requirement_id,
                "Requirement lacks scenario or test-case coverage.",
            )
        )
    return ValidationReport(
        valid=not issues,
        issues=issues,
        uncovered_requirement_ids=uncovered_requirement_ids,
        orphan_scenario_ids=orphan_scenario_ids,
        orphan_test_case_ids=orphan_test_case_ids,
    )


def build_rtm(bundle: ArtifactBundle) -> list[RTMRow]:
    rows: list[RTMRow] = []
    for requirement in bundle.requirements:
        scenarios = [
            scenario
            for scenario in bundle.scenarios
            if requirement.requirement_id in scenario.requirement_ids
        ]
        test_cases = [
            case
            for case in bundle.test_cases
            if requirement.requirement_id in case.requirement_ids
        ]
        rows.append(
            RTMRow(
                requirement_id=requirement.requirement_id,
                scenario_ids=sorted({scenario.scenario_id for scenario in scenarios}),
                test_case_ids=sorted({case.test_case_id for case in test_cases}),
                source_chunk_ids=sorted(
                    {
                        reference.chunk_id
                        for artifact in [requirement, *scenarios, *test_cases]
                        for reference in artifact.source_references
                    }
                ),
                covered=bool(scenarios and test_cases),
            )
        )
    return rows


def _duplicates_rate(bundle: ArtifactBundle) -> float:
    fingerprints = []
    for case in bundle.test_cases:
        words = re.findall(
            r"\w+",
            " ".join(
                [case.title, *[f"{step.action} {step.expected_result}" for step in case.steps]]
            ).casefold(),
        )
        fingerprints.append(
            set(zip(words, words[1:], words[2:])) if len(words) >= 3 else set(words)
        )
    duplicate_pairs = pair_count = 0
    for left, right in combinations(fingerprints, 2):
        pair_count += 1
        duplicate_pairs += _ratio(len(left & right), len(left | right)) >= 0.85
    return _ratio(duplicate_pairs, pair_count)


def compute_metrics(
    bundle: ArtifactBundle,
    report: ValidationReport,
    *,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    retries: int,
    schema_repairs: int,
    semantic_revisions: int,
    budget_exhausted: bool,
    charged_tokens: int = 0,
) -> RunMetrics:
    requirements = {item.requirement_id for item in bundle.requirements}
    scenario_requirement_ids = {
        requirement_id
        for scenario in bundle.scenarios
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirements
    }
    test_requirement_ids = {
        requirement_id
        for case in bundle.test_cases
        for requirement_id in case.requirement_ids
        if requirement_id in requirements
    }
    positive_requirement_ids = {
        requirement_id
        for scenario in bundle.scenarios
        if scenario.scenario_type is ScenarioType.POSITIVE
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirements
    }
    non_positive_requirement_ids = {
        requirement_id
        for scenario in bundle.scenarios
        if scenario.scenario_type is not ScenarioType.POSITIVE
        for requirement_id in scenario.requirement_ids
        if requirement_id in requirements
    }
    artifacts = [*bundle.requirements, *bundle.scenarios, *bundle.test_cases]
    invalid_source_artifact_ids = {
        issue.artifact_id
        for issue in report.issues
        if issue.code == "invalid_source_reference"
    }
    reference_count = sum(len(artifact.source_references) for artifact in artifacts)
    invalid_reference_count = sum(
        issue.code == "invalid_source_reference" for issue in report.issues
    )
    rows = build_rtm(bundle)
    return RunMetrics(
        completion=report.valid,
        schema_valid=True,
        citation_coverage=_ratio(
            sum(
                requirement.requirement_id not in invalid_source_artifact_ids
                for requirement in bundle.requirements
            ),
            len(bundle.requirements),
        ),
        requirement_scenario_coverage=_ratio(
            len(scenario_requirement_ids), len(requirements)
        ),
        requirement_test_case_coverage=_ratio(
            len(test_requirement_ids), len(requirements)
        ),
        positive_scenario_coverage=_ratio(len(positive_requirement_ids), len(requirements)),
        non_positive_scenario_coverage=_ratio(
            len(non_positive_requirement_ids), len(requirements)
        ),
        rtm_completeness=_ratio(sum(row.covered for row in rows), len(rows)),
        orphan_rate=_ratio(
            len(report.orphan_scenario_ids) + len(report.orphan_test_case_ids),
            len(bundle.scenarios) + len(bundle.test_cases),
        ),
        invalid_reference_rate=_ratio(invalid_reference_count, reference_count),
        duplicate_test_case_rate=_duplicates_rate(bundle),
        requirement_count=len(bundle.requirements),
        scenario_count=len(bundle.scenarios),
        test_case_count=len(bundle.test_cases),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        charged_tokens=charged_tokens,
        latency_seconds=latency_seconds,
        retries=retries,
        schema_repairs=schema_repairs,
        semantic_revisions=semantic_revisions,
        budget_exhausted=budget_exhausted,
    )
