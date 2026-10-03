"""Exact rankings and tie-symmetric selection primitives for SSES alignment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _checked_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("weights must be a non-empty, finite, positive vector")
    return w


def pareto_depth(values: np.ndarray) -> np.ndarray:
    """Return exact strict Pareto depth for a finite maximisation matrix.

    Depth one is the strict nondominated frontier; equal objective vectors share
    a layer. The implementation is deterministic and invariant to row order.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Pareto input must be a non-empty finite 2D matrix")
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    # Descending lexicographic processing makes a point comparable only with
    # candidate members that can already dominate it.
    order = np.lexsort(tuple(-unique[:, j] for j in range(unique.shape[1] - 1, -1, -1)))
    fronts: list[list[int]] = []
    depth_unique = np.empty(len(unique), dtype=np.int32)
    for raw in order:
        idx = int(raw)
        point = unique[idx]
        low, high = 0, len(fronts)
        while low < high:
            middle = (low + high) // 2
            members = unique[np.asarray(fronts[middle], dtype=np.int64)]
            if bool(np.any(np.all(members >= point, axis=1))):
                low = middle + 1
            else:
                high = middle
        if low == len(fronts):
            fronts.append([])
        fronts[low].append(idx)
        depth_unique[idx] = low + 1
    return depth_unique[inverse]


def weighted_midrank(values: np.ndarray, weights: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Weighted symmetric fractional ranks on [0, 1], where high values are high ranks."""
    x, w = np.asarray(values, dtype=float), _checked_weights(weights)
    if len(x) != len(w) or not np.isfinite(x).all():
        raise ValueError("values and weights must be finite, matching vectors")
    unique, inverse = np.unique(x, return_inverse=True)
    mass = np.bincount(inverse, weights=w, minlength=len(unique))
    mid = (np.cumsum(mass) - 0.5 * mass) / mass.sum()
    result = mid[inverse]
    return result if higher_is_better else 1.0 - result


def development_rank_score(depth: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted fractional rank score for Pareto depth; lower depth is better."""
    return weighted_midrank(np.asarray(depth, dtype=float), weights, higher_is_better=False)


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x, y, w = np.asarray(x, float), np.asarray(y, float), _checked_weights(weights)
    if len(x) != len(y) or len(x) != len(w) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be finite matching vectors")
    mx, my = float(np.average(x, weights=w)), float(np.average(y, weights=w))
    dx, dy = x - mx, y - my
    denominator = math.sqrt(float(np.sum(w * dx * dx) * np.sum(w * dy * dy)))
    return float(np.sum(w * dx * dy) / denominator) if denominator else math.nan


def fractional_selection(values: np.ndarray, weights: np.ndarray, q: float, higher_is_better: bool) -> np.ndarray:
    """Tie-symmetric fractional membership in an exact weighted top/bottom q."""
    x, w = np.asarray(values, float), _checked_weights(weights)
    if len(x) != len(w) or not np.isfinite(x).all() or not 0 < q < 1:
        raise ValueError("selection needs finite matching values and 0 < q < 1")
    unique, inverse = np.unique(x, return_inverse=True)
    group_mass = np.bincount(inverse, weights=w, minlength=len(unique))
    order = np.arange(len(unique) - 1, -1, -1) if higher_is_better else np.arange(len(unique))
    cumulative = np.cumsum(group_mass[order])
    target = q * w.sum()
    boundary = int(np.searchsorted(cumulative, target, side="left"))
    before = float(cumulative[boundary - 1]) if boundary else 0.0
    prop = np.zeros(len(unique), dtype=float)
    prop[order[:boundary]] = 1.0
    prop[order[boundary]] = np.clip((target - before) / group_mass[order[boundary]], 0.0, 1.0)
    return prop[inverse]


def weighted_bin_membership(values: np.ndarray, weights: np.ndarray, bins: int, higher_is_better: bool) -> np.ndarray:
    """Return bin-by-person fractional memberships with exact weighted bin mass.

    A tied value that spans a boundary contributes the same fraction to each
    affected bin. Rows are ordered from the best to the worst bin.
    """
    x, w = np.asarray(values, float), _checked_weights(weights)
    if len(x) != len(w) or not np.isfinite(x).all() or bins < 2:
        raise ValueError("invalid weighted-bin inputs")
    unique, inverse = np.unique(x, return_inverse=True)
    group_mass = np.bincount(inverse, weights=w, minlength=len(unique))
    order = np.arange(len(unique) - 1, -1, -1) if higher_is_better else np.arange(len(unique))
    ordered_mass = group_mass[order]
    starts = np.r_[0.0, np.cumsum(ordered_mass)[:-1]]
    ends = np.cumsum(ordered_mass)
    edges = np.linspace(0.0, float(w.sum()), bins + 1)
    group_by_bin = np.zeros((bins, len(unique)), dtype=float)
    for b in range(bins):
        overlap = np.maximum(0.0, np.minimum(ends, edges[b + 1]) - np.maximum(starts, edges[b]))
        group_by_bin[b, order] = overlap / ordered_mass
    return group_by_bin[:, inverse]


def equal_capacity_overlap(achievement: np.ndarray, depth: np.ndarray, weights: np.ndarray, q: float) -> dict[str, float]:
    """Equal-capacity overlap and chance-adjusted alignment."""
    a = fractional_selection(achievement, weights, q, higher_is_better=True)
    d = fractional_selection(depth, weights, q, higher_is_better=False)
    w = _checked_weights(weights)
    overlap = float(np.sum(w * a * d) / (q * w.sum()))
    return {
        "overlap": overlap,
        "excess_overlap": overlap - q,
        "CAA": (overlap - q) / (1.0 - q),
        "achievement_actual_rate": float(np.sum(w * a) / w.sum()),
        "development_actual_rate": float(np.sum(w * d) / w.sum())
    }


def capture_requirement(achievement: np.ndarray, depth: np.ndarray, weights: np.ndarray, development_q: float, capture: float) -> float:
    """Minimum fractional achievement selection rate attaining a target recall."""
    if not 0 < capture < 1:
        raise ValueError("capture must lie strictly between zero and one")
    x, w = np.asarray(achievement, float), _checked_weights(weights)
    target = fractional_selection(depth, w, development_q, higher_is_better=False)
    total_target = float(np.sum(w * target))
    unique, inverse = np.unique(x, return_inverse=True)
    group_w = np.bincount(inverse, weights=w, minlength=len(unique))[::-1]
    group_target = np.bincount(inverse, weights=w * target, minlength=len(unique))[::-1]
    before_w = before_target = 0.0
    required = capture * total_target
    for mass, target_mass in zip(group_w, group_target):
        if before_target + target_mass + 1e-12 >= required and target_mass > 0:
            within = np.clip((required - before_target) / target_mass, 0.0, 1.0)
            return float((before_w + within * mass) / w.sum())
        before_w += float(mass)
        before_target += float(target_mass)
    return 1.0


def transition_matrix(achievement: np.ndarray, depth: np.ndarray, weights: np.ndarray, bins: int = 10) -> np.ndarray:
    """Weighted conditional development-bin distribution by achievement bin."""
    a = weighted_bin_membership(achievement, weights, bins, higher_is_better=True)
    d = weighted_bin_membership(depth, weights, bins, higher_is_better=False)
    w = _checked_weights(weights)
    matrix = (a * w) @ d.T
    denominators = (a * w).sum(axis=1)
    return matrix / denominators[:, None]


def near_match(achievement: np.ndarray, depth: np.ndarray, weights: np.ndarray, bins: int = 10) -> np.ndarray:
    """P(development decile is within one decile of achievement decile)."""
    a = weighted_bin_membership(achievement, weights, bins, higher_is_better=True)
    d = weighted_bin_membership(depth, weights, bins, higher_is_better=False)
    w = _checked_weights(weights)
    out = np.empty(bins, dtype=float)
    for b in range(bins):
        lower, upper = max(0, b - 1), min(bins, b + 2)
        out[b] = float(np.sum(w * a[b] * d[lower:upper].sum(axis=0)) / np.sum(w * a[b]))
    return out


def core_metrics(achievement: np.ndarray, development: np.ndarray, weights: np.ndarray, capacities: list[float], targets: list[float], captures: list[float]) -> dict[str, Any]:
    """Rebuild depth/ranks and calculate the prespecified scalar statistics."""
    w = _checked_weights(weights)
    x, dev = np.asarray(achievement, float), np.asarray(development, float)
    if len(x) != len(w) or dev.shape != (len(w), 4) or not np.isfinite(x).all() or not np.isfinite(dev).all():
        raise ValueError("core metric input must be complete with exactly four development variables")
    depth = pareto_depth(dev)
    r = weighted_correlation(weighted_midrank(x, w), development_rank_score(depth, w), w)
    overlaps = {float(q): equal_capacity_overlap(x, depth, w, float(q)) for q in capacities}
    captures_out: dict[tuple[float, float], float] = {}
    for target in targets:
        for capture in captures:
            captures_out[(float(target), float(capture))] = capture_requirement(x, depth, w, float(target), float(capture))
    return {"depth": depth, "r": r, "overlaps": overlaps, "capture_q": captures_out}

