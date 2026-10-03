from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "sses2023_alignment" / "src"))

from pareto_depth import (  # noqa: E402
    equal_capacity_overlap, pareto_depth, weighted_bin_membership, weighted_midrank
)
from survey_variance_2023 import replicate_variance  # noqa: E402


def test_official_fay_brr_toy_formula() -> None:
    full = .25
    reps = np.array([.10, .20, .30, .40] * 20)
    expected = .05 * np.sum((reps - full) ** 2)
    assert math.isclose(replicate_variance(full, reps, .05), expected, rel_tol=0, abs_tol=1e-15)


def test_pareto_depth_known_order() -> None:
    values = np.array([[3, 3], [2, 2], [1, 4], [1, 1], [3, 3]], float)
    assert np.array_equal(pareto_depth(values), np.array([1, 2, 1, 3, 1]))


def test_exact_tie_symmetric_equal_capacity_selection() -> None:
    achievement = np.array([90, 90, 80, 70, 60], float)
    depth = np.array([1, 1, 2, 2, 3])
    weights = np.array([1, 3, 2, 1, 3], float)
    result = equal_capacity_overlap(achievement, depth, weights, .30)
    assert math.isclose(result["achievement_actual_rate"], .30, abs_tol=1e-12)
    assert math.isclose(result["development_actual_rate"], .30, abs_tol=1e-12)


def test_weighted_deciles_are_exact_and_tie_symmetric() -> None:
    value = np.array([3, 3, 2, 1], float)
    weight = np.array([4, 2, 3, 1], float)
    membership = weighted_bin_membership(value, weight, 10, True)
    assert np.allclose((membership * weight).sum(axis=1), weight.sum() / 10)
    assert np.allclose(membership[:, 0], membership[:, 1])


def test_weighted_rank_is_row_order_invariant() -> None:
    value = np.array([1, 2, 2, 4], float)
    weight = np.array([1, 5, 3, 2], float)
    first = weighted_midrank(value, weight)
    order = np.array([2, 0, 3, 1])
    second = weighted_midrank(value[order], weight[order])
    restored = np.empty_like(second)
    restored[order] = second
    assert np.allclose(first, restored)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"passed {name}")
    print("All unit tests passed")

