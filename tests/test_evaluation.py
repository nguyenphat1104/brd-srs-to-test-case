from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brd_srs_testgen.evaluation import (
    agreement_gate,
    coverage_rating,
    quadratic_weighted_kappa,
    summarize_human_agreement,
)
from brd_srs_testgen.models import HumanRating, HumanRatingDimension


def rating(
    run_id: str,
    rater_id: str,
    score: int,
    *,
    dimension: HumanRatingDimension = HumanRatingDimension.COVERAGE,
) -> HumanRating:
    return HumanRating(
        rating_id=f"{run_id}-{rater_id}-{dimension.value}",
        run_id=run_id,
        rater_id=rater_id,
        dimension=dimension,
        score=score,
        rubric_version="quality-rubric-v1",
        created_at=datetime(2026, 9, 24, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("f1", "expected"),
    [
        (0, 1),
        (0.49, 1),
        (0.5, 2),
        (0.74, 2),
        (0.75, 3),
        (0.89, 3),
        (0.9, 4),
        (1, 4),
    ],
)
def test_coverage_rating_uses_shared_rubric(f1: float, expected: int) -> None:
    assert coverage_rating(f1) == expected


def test_quadratic_weighted_kappa_is_one_for_perfect_varied_agreement() -> None:
    assert quadratic_weighted_kappa([(1, 1), (2, 2), (3, 3), (4, 4)]) == 1


def test_quadratic_weighted_kappa_penalizes_distance() -> None:
    assert quadratic_weighted_kappa([(1, 1), (2, 2), (3, 4), (4, 4)]) == pytest.approx(
        11 / 12
    )


def test_quadratic_weighted_kappa_is_undefined_without_marginal_variation() -> None:
    assert quadratic_weighted_kappa([(4, 4), (4, 4)]) is None


def test_quadratic_weighted_kappa_rejects_scores_outside_rubric() -> None:
    with pytest.raises(ValueError, match="1 to 4"):
        quadratic_weighted_kappa([(0, 1)])


def test_agreement_summary_is_per_dimension() -> None:
    ratings = [
        rating(run_id, rater_id, score)
        for run_id, score in (("run-1", 1), ("run-2", 2), ("run-3", 4))
        for rater_id in ("rater-a", "rater-b")
    ]

    summary = summarize_human_agreement(ratings)

    assert summary[HumanRatingDimension.COVERAGE].pair_count == 3
    assert summary[HumanRatingDimension.COVERAGE].quadratic_weighted_kappa == 1.0
    assert summary[HumanRatingDimension.GROUNDEDNESS].pair_count == 0


def test_agreement_summary_reports_exact_adjacent_and_missing_pairs() -> None:
    ratings = [
        rating("run-1", "rater-a", 1),
        rating("run-1", "rater-b", 1),
        rating("run-2", "rater-a", 2),
        rating("run-2", "rater-b", 3),
        rating("run-3", "rater-a", 1),
        rating("run-3", "rater-b", 4),
        rating("unpaired", "rater-a", 4),
    ]

    coverage = summarize_human_agreement(ratings)[HumanRatingDimension.COVERAGE]

    assert coverage.pair_count == 3
    assert coverage.exact_agreement == pytest.approx(1 / 3)
    assert coverage.adjacent_agreement == pytest.approx(2 / 3)
    assert coverage.sufficient_data is True


def test_constant_equal_ratings_report_undefined_kappa() -> None:
    ratings = [
        rating(run_id, rater_id, 4)
        for run_id in ("run-1", "run-2")
        for rater_id in ("rater-a", "rater-b")
    ]

    coverage = summarize_human_agreement(ratings)[HumanRatingDimension.COVERAGE]

    assert coverage.pair_count == 2
    assert coverage.quadratic_weighted_kappa is None


def test_agreement_gate_requires_every_dimension_at_point_seven() -> None:
    assert agreement_gate(
        {dimension: 0.70 for dimension in HumanRatingDimension}
    )
    assert not agreement_gate(
        {
            **{dimension: 0.80 for dimension in HumanRatingDimension},
            HumanRatingDimension.EXECUTABILITY: 0.69,
        }
    )
    assert not agreement_gate(
        {
            dimension: 0.80
            for dimension in HumanRatingDimension
            if dimension is not HumanRatingDimension.COVERAGE
        }
    )
