"""Replicate-weight inference and random-effects meta-analysis."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import optimize, stats


def replicate_variance(full: float, replicates: np.ndarray, multiplier: float) -> float:
    reps = np.asarray(replicates, float)
    if not math.isfinite(full) or len(reps) == 0 or not np.isfinite(reps).all() or multiplier <= 0:
        raise ValueError("invalid replicate-variance inputs")
    return float(multiplier * np.sum((reps - full) ** 2))


def direct_summary(full: float, replicates: np.ndarray, multiplier: float, clip: tuple[float, float] | None = None) -> dict[str, float]:
    variance = replicate_variance(full, replicates, multiplier)
    se = math.sqrt(variance)
    low, high = full - 1.96 * se, full + 1.96 * se
    if clip is not None:
        low, high = max(low, clip[0]), min(high, clip[1])
    return {"estimate": float(full), "variance": variance, "se": se, "CI_low": float(low), "CI_high": float(high)}


def fisher_summary(correlation: float, replicates: np.ndarray, multiplier: float) -> dict[str, float]:
    clipped = float(np.clip(correlation, -0.999999, 0.999999))
    reps = np.clip(np.asarray(replicates, float), -0.999999, 0.999999)
    z, z_reps = float(np.arctanh(clipped)), np.arctanh(reps)
    result = direct_summary(z, z_reps, multiplier)
    return {
        "r": correlation, "z": z, "z_variance": result["variance"], "z_se": result["se"],
        "CI_low": float(np.tanh(result["CI_low"])), "CI_high": float(np.tanh(result["CI_high"]))
    }


def random_effects_reml(effects: Iterable[float], variances: Iterable[float]) -> dict[str, float]:
    y, v = np.asarray(list(effects), float), np.asarray(list(variances), float)
    if len(y) < 2 or len(y) != len(v) or not np.isfinite(y).all() or not np.isfinite(v).all():
        raise ValueError("REML requires at least two finite effect/variance pairs")
    v = np.maximum(v, 1e-12)
    fixed_w = 1.0 / v
    fixed_mean = float(np.sum(fixed_w * y) / fixed_w.sum())
    q = float(np.sum(fixed_w * (y - fixed_mean) ** 2))
    df = len(y) - 1

    def restricted_nll(tau2: float) -> float:
        w = 1.0 / (v + tau2)
        mean = float(np.sum(w * y) / w.sum())
        return float(0.5 * (np.log(v + tau2).sum() + math.log(w.sum()) + np.sum(w * (y - mean) ** 2)))

    upper = max(1e-8, float(100 * (np.var(y, ddof=1) + np.max(v))))
    fitted = optimize.minimize_scalar(restricted_nll, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-12})
    tau2 = 0.0 if restricted_nll(0.0) <= fitted.fun + 1e-10 else float(fitted.x)
    w = 1.0 / (v + tau2)
    pooled = float(np.sum(w * y) / w.sum())
    se = math.sqrt(float(1.0 / w.sum()))
    return {
        "K": int(len(y)), "pooled": pooled, "pooled_se": se,
        "pooled_CI_low": pooled - 1.96 * se, "pooled_CI_high": pooled + 1.96 * se,
        "tau2": tau2, "Q": q, "Q_df": int(df), "Q_p": float(stats.chi2.sf(q, df)),
        "I2": max(0.0, 100 * (q - df) / q) if q else 0.0
    }


def one_sided_negative_p(estimate: float, se: float) -> float:
    if se <= 0 or not math.isfinite(se):
        return math.nan
    return float(stats.norm.cdf(estimate / se))

