from __future__ import annotations

import pytest

from brd_srs_testgen.evaluation import coverage_rating, quadratic_weighted_kappa


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
