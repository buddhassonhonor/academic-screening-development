"""Whole-distribution Pareto-depth alignment replication."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import EXP, legacy_csv, primary_rows, random_effects_reml, two_sided_p


def _source(wave: int) -> pd.DataFrame:
    name = "depth_alignment_2019_reanalysis.csv" if wave == 2019 else "depth_alignment_2023.csv"
    source = pd.read_csv(EXP / name) if wave == 2019 else legacy_csv(name)
    return primary_rows(source)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, effects = [], {}
    for wave in (2019, 2023):
        depth = _source(wave)
        meta = random_effects_reml(depth["z"], depth["z_variance"])
        effects[wave] = meta
        rows.append({
            "wave": wave, "pooled_r": float(np.tanh(meta["pooled"])), "pooled_z": meta["pooled"],
            "SE": meta["pooled_se"], "CI_lower_z": meta["pooled_CI_low"], "CI_upper_z": meta["pooled_CI_high"],
            "CI_lower_r": float(np.tanh(meta["pooled_CI_low"])), "CI_upper_r": float(np.tanh(meta["pooled_CI_high"])),
            "Q": meta["Q"], "I2": meta["I2"], "tau2": meta["tau2"], "n_sites": meta["K"],
            "site_r_min": float(depth["r"].min()), "site_r_max": float(depth["r"].max()),
        })
    delta = effects[2023]["pooled"] - effects[2019]["pooled"]
    se = math.sqrt(effects[2023]["pooled_se"] ** 2 + effects[2019]["pooled_se"] ** 2)
    difference = pd.DataFrame([{
        "contrast": "pooled_Fisher_z_2023_minus_2019", "delta_z": delta, "SE": se,
        "CI_lower_z": delta - 1.96 * se, "CI_upper_z": delta + 1.96 * se,
        "p_two_sided": two_sided_p(delta, se),
        "interpretation": "independent-wave pooled estimate comparison; differing site composition prevents a clean temporal causal interpretation",
    }])
    table = pd.DataFrame(rows)
    table.to_csv(EXP / "depth_alignment_crosswave.csv", index=False)
    difference.to_csv(EXP / "depth_alignment_wave_difference.csv", index=False)
    return table, difference


if __name__ == "__main__":
    run()
