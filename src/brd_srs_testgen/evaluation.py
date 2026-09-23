from __future__ import annotations


COVERAGE_RATING_RUBRIC_VERSION = "coverage-ordinal-v1"


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
