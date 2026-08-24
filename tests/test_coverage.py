from __future__ import annotations

import pytest

from brd_srs_testgen.coverage import (
    compute_f1,
    extract_coverage_units_prompt,
    map_test_cases_prompt,
)
from brd_srs_testgen.models import (
    ArtifactBundle,
    CoverageMappingBatch,
    CoverageMappingEntry,
    CoverageScore,
    CoverageUnit,
    CoverageUnitBatch,
    DocumentChunk,
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


def _chunk() -> DocumentChunk:
    text = "The system shall authenticate registered users."
    return DocumentChunk(
        chunk_id="p0001-c001-ecac9f035813",
        page_number=1,
        section="AUTHENTICATION",
        text=text,
        content_hash="ecac9f0358134f174bcbf0d60ddbc7c25bcb4f812ea8e4c57bfbd8c02edaa274",
    )


def _source() -> SourceReference:
    item = _chunk()
    return SourceReference(
        chunk_id=item.chunk_id,
        page_number=item.page_number,
        section=item.section,
        excerpt=item.text,
    )


def _bundle() -> ArtifactBundle:
    requirement = Requirement(
        requirement_id="REQ-001",
        title="Authenticate users",
        description="Registered users can sign in.",
        requirement_type=RequirementType.FUNCTIONAL,
        module="Authentication",
        priority=RequirementPriority.HIGH,
        source_references=[_source()],
    )
    scenario = Scenario(
        scenario_id="SCN-001",
        title="Valid sign in",
        objective="Verify successful authentication.",
        scenario_type=ScenarioType.POSITIVE,
        requirement_ids=["REQ-001"],
        source_references=[_source()],
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
        source_references=[_source()],
    )
    return ArtifactBundle(
        requirements=[requirement], scenarios=[scenario], test_cases=[test_case]
    )


def _units(count: int = 3) -> CoverageUnitBatch:
    return CoverageUnitBatch(
        units=[
            CoverageUnit(
                unit_id=f"CU-{i:03d}",
                title=f"Unit {i}",
                description=f"Testable behavior {i}.",
                unit_type="functional",
                source_chunk_ids=["p0001-c001-ecac9f035813"],
            )
            for i in range(1, count + 1)
        ]
    )


class TestComputeF1:
    def test_perfect_coverage(self) -> None:
        units = _units(2)
        bundle = _bundle()
        mappings = CoverageMappingBatch(
            mappings=[
                CoverageMappingEntry(
                    test_case_id="TC-001", covered_unit_ids=["CU-001", "CU-002"]
                ),
            ]
        )
        score = compute_f1(units, mappings, bundle)
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0
        assert score.true_positive_count == 1
        assert score.false_positive_count == 0
        assert score.false_negative_count == 0
        assert score.uncovered_unit_ids == []
        assert score.unmapped_test_case_ids == []

    def test_no_coverage(self) -> None:
        units = _units(3)
        bundle = _bundle()
        mappings = CoverageMappingBatch(
            mappings=[
                CoverageMappingEntry(test_case_id="TC-001", covered_unit_ids=[]),
            ]
        )
        score = compute_f1(units, mappings, bundle)
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.f1 == 0.0
        assert score.true_positive_count == 0
        assert score.false_positive_count == 1
        assert score.false_negative_count == 3
        assert len(score.uncovered_unit_ids) == 3
        assert score.unmapped_test_case_ids == ["TC-001"]

    def test_partial_coverage(self) -> None:
        units = _units(4)
        bundle = _bundle()
        mappings = CoverageMappingBatch(
            mappings=[
                CoverageMappingEntry(
                    test_case_id="TC-001", covered_unit_ids=["CU-001", "CU-002"]
                ),
            ]
        )
        score = compute_f1(units, mappings, bundle)
        assert score.precision == 1.0  # 1 mapped TC, 0 unmapped
        assert score.recall == 0.5  # 2 of 4 units covered
        assert score.f1 == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))
        assert score.true_positive_count == 1
        assert score.false_positive_count == 0
        assert score.false_negative_count == 2
        assert score.uncovered_unit_ids == ["CU-003", "CU-004"]

    def test_unmapped_test_case_in_batch(self) -> None:
        """Test cases not in the mapping batch are treated as unmapped."""
        units = _units(1)
        bundle = _bundle()
        # Empty mappings — TC-001 is not in the batch at all.
        mappings = CoverageMappingBatch(mappings=[])
        score = compute_f1(units, mappings, bundle)
        assert score.precision == 0.0
        assert score.false_positive_count == 1
        assert "TC-001" in score.unmapped_test_case_ids

    def test_invalid_unit_ids_ignored(self) -> None:
        """Mappings referencing unknown unit IDs are filtered out."""
        units = _units(1)
        bundle = _bundle()
        mappings = CoverageMappingBatch(
            mappings=[
                CoverageMappingEntry(
                    test_case_id="TC-001", covered_unit_ids=["CU-999"]
                ),
            ]
        )
        score = compute_f1(units, mappings, bundle)
        assert score.precision == 0.0  # CU-999 is not a real unit
        assert score.recall == 0.0

    def test_empty_units_and_test_cases(self) -> None:
        units = CoverageUnitBatch(units=[])
        bundle = ArtifactBundle(requirements=[], scenarios=[], test_cases=[])
        mappings = CoverageMappingBatch(mappings=[])
        score = compute_f1(units, mappings, bundle)
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.f1 == 0.0
        assert score.total_coverage_units == 0
        assert score.total_test_cases == 0


class TestPrompts:
    def test_extract_coverage_units_prompt_contains_evidence(self) -> None:
        chunks = [_chunk()]
        prompt = extract_coverage_units_prompt(chunks)
        assert "PDF EVIDENCE" in prompt
        assert "CU-001" in prompt
        assert "CoverageUnitBatch" in prompt
        assert chunks[0].chunk_id in prompt

    def test_map_test_cases_prompt_contains_both_catalogs(self) -> None:
        bundle = _bundle()
        units = _units(2)
        prompt = map_test_cases_prompt(bundle, units)
        assert "COVERAGE UNITS JSON" in prompt
        assert "TEST CASES JSON" in prompt
        assert "TC-001" in prompt
        assert "CU-001" in prompt
        assert "CoverageMappingBatch" in prompt

    def test_extract_prompt_with_agent_setup(self) -> None:
        from brd_srs_testgen.models import AgentSetup

        setup = AgentSetup(
            agent="coverage_analyzer",
            role="Senior QA analyst",
            instructions="Focus on boundary conditions.",
        )
        prompt = extract_coverage_units_prompt([_chunk()], setup=setup)
        assert "Senior QA analyst" in prompt
        assert "Focus on boundary conditions." in prompt

    def test_map_prompt_with_agent_setup(self) -> None:
        from brd_srs_testgen.models import AgentSetup

        setup = AgentSetup(
            agent="coverage_analyzer",
            role="Mapper",
            instructions="Be strict.",
        )
        prompt = map_test_cases_prompt(_bundle(), _units(1), setup=setup)
        assert "Mapper" in prompt
        assert "Be strict." in prompt
