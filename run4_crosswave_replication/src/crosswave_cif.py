"""Capacity-inflation replication using identical wave-specific definitions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import EXP, legacy_csv, primary_rows, replicate_mean_summary, two_sided_p


def _wave_sources(wave: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if wave == 2019:
        return (
            primary_rows(pd.read_csv(EXP / "capacity_inflation_2019_reanalysis.csv")),
            primary_rows(pd.read_csv(EXP / "replicate_statistics_2019_reanalysis.csv")),
            .5,
        )
    if wave == 2023:
        return primary_rows(legacy_csv("capacity_inflation.csv")), primary_rows(legacy_csv("replicate_statistics_2023.csv")), .05
    raise ValueError("wave must be 2019 or 2023")


def _filter(full: pd.DataFrame, replicates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = full[np.isclose(full["target_k"], .10) & np.isclose(full["capture"], .80)].copy()
    reps = replicates[
        replicates["metric"].eq("CIF") & np.isclose(replicates["target_k"], .10) & np.isclose(replicates["capture"], .80)
    ].copy()
    if full["site"].nunique() != reps["site"].nunique():
        raise ValueError("CIF full and replicate site sets differ")
    return full, reps


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, site_rows = [], []
    wave_estimates: dict[int, dict[str, float]] = {}
    for wave in (2019, 2023):
        full, reps, multiplier = _wave_sources(wave)
        full, reps = _filter(full, reps)
        estimate = replicate_mean_summary(full.rename(columns={"CIF": "value"}), reps, multiplier)
        q80 = float(full["achievement_selection_rate"].mean())
        values = full["CIF"]
        wave_estimates[wave] = estimate
        summary_rows.append({
            "wave": wave, "number_sites": int(full["site"].nunique()),
            "pooled_CIF80_10": estimate["estimate"], "SE": estimate["SE"],
            "CI_lower": estimate["CI_lower"], "CI_upper": estimate["CI_upper"],
            "q80_absolute": q80, "q80_absolute_percent": 100 * q80,
            "median": float(values.median()), "IQR": float(values.quantile(.75) - values.quantile(.25)),
            "min": float(values.min()), "max": float(values.max()),
            "definition": "equal-site mean CIF80_10; design variance from wave replicate weights",
        })
        for row in full.itertuples(index=False):
            site_rows.append({
                "wave": wave, "site": row.site, "site_code": row.site_code, "CIF80_10": row.CIF,
                "q80_absolute": row.achievement_selection_rate,
                "CIF_CI_lower": row.CIF_CI_low, "CIF_CI_upper": row.CIF_CI_high,
                "above_5": row.CIF > 5, "above_6": row.CIF > 6, "above_7": row.CIF > 7,
            })
    diff = wave_estimates[2023]["estimate"] - wave_estimates[2019]["estimate"]
    variance = wave_estimates[2023]["variance"] + wave_estimates[2019]["variance"]
    se = math.sqrt(variance)
    difference = pd.DataFrame([{
        "contrast": "CIF80_10_2023_minus_2019", "difference": diff, "SE": se,
        "CI_lower": diff - 1.96 * se, "CI_upper": diff + 1.96 * se,
        "p_two_sided": two_sided_p(diff, se),
        "variance_definition": "independent repeated cross-sectional wave estimates: Var(2023)+Var(2019)",
        "interpretation": "no evidence of wave difference" if diff - 1.96 * se <= 0 <= diff + 1.96 * se else "wave difference detected; not an equivalence test",
    }])
    summary, sites = pd.DataFrame(summary_rows), pd.DataFrame(site_rows)
    summary.to_csv(EXP / "cif_crosswave_summary.csv", index=False)
    sites.to_csv(EXP / "cif_site_level_replication.csv", index=False)
    difference.to_csv(EXP / "cif_crosswave_difference.csv", index=False)
    return summary, difference, sites


if __name__ == "__main__":
    run()
