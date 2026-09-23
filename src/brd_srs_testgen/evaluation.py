from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .models import HumanRating, HumanRatingDimension


COVERAGE_RATING_RUBRIC_VERSION = "coverage-ordinal-v1"
QUALITY_RUBRIC_VERSION = "quality-rubric-v1"
AGREEMENT_THRESHOLD = 0.70


@dataclass(frozen=True)
class AgreementSummary:
    pair_count: int
    exact_agreement: float | None
    adjacent_agreement: float | None
    quadratic_weighted_kappa: float | None
    sufficient_data: bool


def coverage_rating(f1: float) -> int:
    """Convert judge F1 to the shared four-point ordinal coverage rubric."""
    if not 0 <= f1 <= 1:
        raise ValueError("F1 must be between 0 and 1.")
    if f1 < 0.5:
        return 1
    if f1 < 0.75:
        return 2
    if f1 < 0.9:
        return 3
    return 4


def quadratic_weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Return quadratic-weighted Cohen's kappa for four-point ratings."""
    if not pairs:
        return None
    if any(
        human not in range(1, 5) or judge not in range(1, 5)
        for human, judge in pairs
    ):
        raise ValueError("Ratings must be integers from 1 to 4.")

    count = len(pairs)
    human_counts = [
        sum(human == score for human, _ in pairs) for score in range(1, 5)
    ]
    judge_counts = [
        sum(judge == score for _, judge in pairs) for score in range(1, 5)
    ]
    observed = sum(((human - judge) / 3) ** 2 for human, judge in pairs) / count
    expected = sum(
        ((human - judge) / 3) ** 2
        * human_counts[human - 1]
        * judge_counts[judge - 1]
        for human in range(1, 5)
        for judge in range(1, 5)
    ) / count**2
    return None if expected == 0 else 1 - observed / expected


def summarize_human_agreement(
    ratings: list[HumanRating],
) -> dict[HumanRatingDimension, AgreementSummary]:
    grouped: dict[
        tuple[str, HumanRatingDimension, int], list[tuple[str, int]]
    ] = {}
    for rating in ratings:
        grouped.setdefault(
            (rating.run_id, rating.dimension, rating.round), []
        ).append((rating.rater_id, rating.score))

    result = {}
    for dimension in HumanRatingDimension:
        pairs = [
            (items[0][1], items[1][1])
            for (_run_id, item_dimension, _round_number), items in grouped.items()
            if item_dimension is dimension
            and len(items) == 2
            and len({rater_id for rater_id, _score in items}) == 2
        ]
        count = len(pairs)
        result[dimension] = AgreementSummary(
            pair_count=count,
            exact_agreement=(
                sum(left == right for left, right in pairs) / count
                if count
                else None
            ),
            adjacent_agreement=(
                sum(abs(left - right) <= 1 for left, right in pairs) / count
                if count
                else None
            ),
            quadratic_weighted_kappa=quadratic_weighted_kappa(pairs),
            sufficient_data=count >= 2,
        )
    return result


def agreement_gate(
    kappas: Mapping[HumanRatingDimension, float | None],
    *,
    threshold: float = AGREEMENT_THRESHOLD,
) -> bool:
    if not 0 <= threshold <= 1:
        raise ValueError("Agreement threshold must be between 0 and 1.")
    return set(kappas) == set(HumanRatingDimension) and all(
        value is not None and value >= threshold for value in kappas.values()
    )
