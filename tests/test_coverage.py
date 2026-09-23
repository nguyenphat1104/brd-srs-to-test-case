from __future__ import annotations

from datetime import UTC, datetime

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
    CoverageCatalog,
    CoverageCatalogStatus,
    CoverageEvaluation,
    CoverageEvaluationStatus,
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
    HumanAdjudication,
    HumanRating,
    HumanRatingDimension,
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
                source_references=[_source()],
            )
            for i in range(1, count + 1)
        ]
    )


def _score() -> CoverageScore:
    return CoverageScore(
        catalog_id="cat-doc-v1",
        precision=1,
        recall=1,
        f1=1,
        true_positive_count=1,
        false_positive_count=0,
        false_negative_count=0,
        total_coverage_units=1,
        total_test_cases=1,
    )


class TestEvaluatorContracts:
    def test_coverage_catalog_keeps_version_and_quoted_evidence(self) -> None:
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

    def test_evaluation_accepts_completed_and_failed_shapes(self) -> None:
        now = datetime.now(UTC)
        completed = CoverageEvaluation(
            run_id="run-1",
            catalog_id="cat-doc-v1",
            status=CoverageEvaluationStatus.COMPLETED,
            mappings=CoverageMappingBatch(mappings=[]),
            score=_score(),
            evaluated_at=now,
        )
        failed = CoverageEvaluation(
            run_id="run-2",
            catalog_id="cat-doc-v1",
            status=CoverageEvaluationStatus.FAILED,
            error="Provider unavailable.",
            evaluated_at=now,
        )

        assert completed.score is not None
        assert failed.error == "Provider unavailable."

    def test_catalog_and_evaluation_enforce_status_fields(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(ValueError, match="approved_at"):
            CoverageCatalog(
                catalog_id="cat-doc-v1",
                document_hash="a" * 64,
                evaluator_version="coverage-v2",
                status=CoverageCatalogStatus.APPROVED,
                units=[_units(1).units[0]],
                created_at=now,
            )
        with pytest.raises(ValueError, match="non-approved"):
            CoverageCatalog(
                catalog_id="cat-doc-v1",
                document_hash="a" * 64,
                evaluator_version="coverage-v2",
                status=CoverageCatalogStatus.MACHINE_FROZEN,
                units=[_units(1).units[0]],
                created_at=now,
                approved_at=now,
            )
        with pytest.raises(ValueError, match="completed evaluations"):
            CoverageEvaluation(
                run_id="run-1",
                catalog_id="cat-doc-v1",
                status=CoverageEvaluationStatus.COMPLETED,
                mappings=CoverageMappingBatch(mappings=[]),
                score=_score(),
                error="unexpected",
                evaluated_at=now,
            )

    def test_contracts_reject_whitespace_required_strings(self) -> None:
        now = datetime.now(UTC)
        catalog = CoverageCatalog(
            catalog_id="cat-doc-v1",
            document_hash="a" * 64,
            evaluator_version="coverage-v2",
            status=CoverageCatalogStatus.MACHINE_FROZEN,
            units=[_units(1).units[0]],
            created_at=now,
        )
        failed = CoverageEvaluation(
            run_id="run-1",
            catalog_id="cat-doc-v1",
            status=CoverageEvaluationStatus.FAILED,
            error="Provider unavailable.",
            evaluated_at=now,
        )
        rating = HumanRating(
            rating_id="rating-1",
            run_id="run-1",
            rater_id="rater-a",
            dimension=HumanRatingDimension.COVERAGE,
            score=4,
            rubric_version="quality-rubric-v1",
            created_at=now,
        )
        adjudication = HumanAdjudication(
            adjudication_id="adj-1",
            run_id="run-1",
            dimension=HumanRatingDimension.COVERAGE,
            score=4,
            reason="Evidence supports the final rating.",
            rubric_version="quality-rubric-v1",
            created_at=now,
        )
        required_fields = (
            (CoverageCatalog, catalog, ("catalog_id", "evaluator_version")),
            (CoverageScore, _score(), ("catalog_id",)),
            (CoverageEvaluation, failed, ("run_id", "catalog_id", "error")),
            (HumanRating, rating, ("rating_id", "run_id", "rater_id", "rubric_version")),
            (
                HumanAdjudication,
                adjudication,
                ("adjudication_id", "run_id", "reason", "rubric_version"),
            ),
        )

        for model, instance, fields in required_fields:
            for field in fields:
                with pytest.raises(ValueError):
                    model.model_validate({**instance.model_dump(), field: " \t "})

    def test_evaluation_enforces_complete_and_failed_payloads(self) -> None:
        now = datetime.now(UTC)
        completed = {
            "run_id": "run-1",
            "catalog_id": "cat-doc-v1",
            "status": CoverageEvaluationStatus.COMPLETED,
            "mappings": CoverageMappingBatch(mappings=[]),
            "score": _score(),
            "evaluated_at": now,
        }
        failed = {
            "run_id": "run-1",
            "catalog_id": "cat-doc-v1",
            "status": CoverageEvaluationStatus.FAILED,
            "error": "Provider unavailable.",
            "evaluated_at": now,
        }

        for field in ("mappings", "score"):
            with pytest.raises(ValueError):
                CoverageEvaluation(**{**completed, field: None})
        for field, value in (("mappings", CoverageMappingBatch(mappings=[])), ("score", _score())):
            with pytest.raises(ValueError):
                CoverageEvaluation(**{**failed, field: value})
        for error in ("", " \t "):
            with pytest.raises(ValueError):
                CoverageEvaluation(**{**failed, "error": error})

    def test_human_rating_identifies_rater_dimension_and_round(self) -> None:
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
        assert (rating.rating_id, rating.rater_id, rating.round) == (
            "rating-1",
            "rater-a",
            1,
        )

    def test_human_adjudication_records_final_dimension_score(self) -> None:
        adjudication = HumanAdjudication(
            adjudication_id="adj-1",
            run_id="run-1",
            dimension=HumanRatingDimension.COVERAGE,
            score=3,
            reason="The final rating resolves the evidence disagreement.",
            rubric_version="quality-rubric-v1",
            created_at=datetime.now(UTC),
        )

        assert adjudication.score == 3


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
