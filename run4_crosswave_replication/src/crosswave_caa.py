"""Chance-adjusted alignment curves and covariance-aware shape contrasts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import EXP, legacy_csv, primary_rows, two_sided_p


PRIMARY_K = (5, 10, 20, 30, 40, 50)
CONTRASTS = ((30, 10), (40, 10), (50, 10), (40, 50))


def _sources(wave: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if wave == 2019:
        return (
            primary_rows(pd.read_csv(EXP / "chance_adjusted_alignment_2019_reanalysis.csv")),
            primary_rows(pd.read_csv(EXP / "replicate_statistics_2019_reanalysis.csv")),
            .5,
        )
    if wave == 2023:
        return primary_rows(legacy_csv("chance_adjusted_alignment.csv")), primary_rows(legacy_csv("replicate_statistics_2023.csv")), .05
    raise ValueError("wave must be 2019 or 2023")


def _curve_summary(full: pd.DataFrame, reps: pd.DataFrame, multiplier: float, k: float) -> dict[str, float]:
    point = float(full[np.isclose(full["k"], k)]["CAA"].mean())
    r = reps[reps.metric.eq("CAA") & np.isclose(reps["k"], k)]
    replicate_mean = r.groupby("replicate", sort=True)["value"].mean().to_numpy(float)
    variance = float(multiplier * np.sum((replicate_mean - point) ** 2))
    se = math.sqrt(variance)
    return {"pooled_CAA": point, "SE": se, "CI_lower": point - 1.96 * se, "CI_upper": point + 1.96 * se}


def _contrast_summary(full: pd.DataFrame, reps: pd.DataFrame, multiplier: float, high: float, low: float) -> dict[str, float]:
    point_high = float(full[np.isclose(full["k"], high)]["CAA"].mean())
    point_low = float(full[np.isclose(full["k"], low)]["CAA"].mean())
    selected = reps[reps.metric.eq("CAA") & (np.isclose(reps["k"], high) | np.isclose(reps["k"], low))]
    matrix = selected.pivot_table(index="replicate", columns="k", values="value", aggfunc="mean")
    replicate_delta = matrix[high].to_numpy(float) - matrix[low].to_numpy(float)
    estimate = point_high - point_low
    variance = float(multiplier * np.sum((replicate_delta - estimate) ** 2))
    se = math.sqrt(variance)
    return {
        "CAA_high_minus_low": estimate, "SE": se, "CI_lower": estimate - 1.96 * se,
        "CI_upper": estimate + 1.96 * se, "p_two_sided": two_sided_p(estimate, se),
        "supported_positive": estimate - 1.96 * se > 0,
        "variance_method": "replicate differences; shared-wave covariance retained",
    }


def _shape_label(contrasts: pd.DataFrame) -> str:
    need = {(30, 10), (40, 10)}
    selected = contrasts.set_index(["k_high_percent", "k_low_percent"])
    if not need.issubset(set(selected.index)):
        return "not_estimable"
    c30, c40 = selected.loc[(30, 10)], selected.loc[(40, 10)]
    if bool(c30.supported_positive) and bool(c40.supported_positive):
        return "broader-selection alignment enhancement supported"
    if c30.CAA_high_minus_low > 0 and c40.CAA_high_minus_low > 0:
        return "directionally broader-selection enhancement"
    return "no broader-selection enhancement"


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows, contrast_rows = [], []
    for wave in (2019, 2023):
        full, reps, multiplier = _sources(wave)
        for percent in PRIMARY_K:
            estimate = _curve_summary(full, reps, multiplier, percent / 100)
            curve_rows.append({"wave": wave, "k_percent": percent, "k": percent / 100,
                               "number_sites": int(full["site"].nunique()), **estimate})
        for high, low in CONTRASTS:
            estimate = _contrast_summary(full, reps, multiplier, high / 100, low / 100)
            contrast_rows.append({"wave": wave, "k_high_percent": high, "k_low_percent": low, **estimate})
    curve, contrasts = pd.DataFrame(curve_rows), pd.DataFrame(contrast_rows)
    labels = []
    for wave, frame in contrasts.groupby("wave"):
        labels.append({"wave": wave, "within_wave_shape": _shape_label(frame)})
    label = pd.DataFrame(labels)
    label.to_csv(EXP / "caa_shape_classification.csv", index=False)
    curve.to_csv(EXP / "caa_crosswave_curve.csv", index=False)
    contrasts.to_csv(EXP / "caa_crosswave_contrasts.csv", index=False)
    return curve, contrasts, label


if __name__ == "__main__":
    run()
