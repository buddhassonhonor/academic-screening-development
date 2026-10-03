"""Cohort-specific differences with paired replicate estimates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from survey_variance_2023 import direct_summary, random_effects_reml


def cohort_differences(depth: pd.DataFrame, curve: pd.DataFrame, capture: pd.DataFrame, replicates: pd.DataFrame, multiplier: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return site-level Older-minus-Younger differences and REML summaries."""
    definitions = [
        ("r", depth[["site_code", "site", "cohort_scope", "r"]].rename(columns={"r": "value"}), "r", math.nan, math.nan),
        ("CAA10", curve[np.isclose(curve["k"], .10)][["site_code", "site", "cohort_scope", "CAA"]].rename(columns={"CAA": "value"}), "CAA", .10, math.nan),
        ("CIF80_10", capture[np.isclose(capture["target_k"], .10) & np.isclose(capture["capture"], .80)][["site_code", "site", "cohort_scope", "CIF"]].rename(columns={"CIF": "value"}), "CIF", math.nan, .10)
    ]
    site_rows, meta_rows = [], []
    for label, table, replicate_metric, k, target_k in definitions:
        per_metric = []
        for site, frame in table.groupby("site", sort=True):
            if set(frame["cohort_scope"]) != {"Younger", "Older"}:
                continue
            younger = float(frame.loc[frame["cohort_scope"] == "Younger", "value"].iloc[0])
            older = float(frame.loc[frame["cohort_scope"] == "Older", "value"].iloc[0])
            reps = replicates[(replicates["site"] == site) & (replicates["metric"] == replicate_metric)]
            if replicate_metric == "CAA":
                reps = reps[np.isclose(reps["k"], k)]
            elif replicate_metric == "CIF":
                reps = reps[np.isclose(reps["target_k"], target_k) & np.isclose(reps["capture"], .80)]
            young_rep = reps[reps["cohort_scope"] == "Younger"].sort_values("replicate")["value"].to_numpy(float)
            old_rep = reps[reps["cohort_scope"] == "Older"].sort_values("replicate")["value"].to_numpy(float)
            summary = direct_summary(older - younger, old_rep - young_rep, multiplier)
            row = {"metric": label, "site": site, "site_code": frame["site_code"].iloc[0], "younger": younger, "older": older, "delta_older_minus_younger": older - younger, **summary}
            site_rows.append(row)
            per_metric.append(row)
        if per_metric:
            frame = pd.DataFrame(per_metric)
            meta = random_effects_reml(frame["delta_older_minus_younger"], frame["variance"])
            meta_rows.append({"metric": label, "scope": "all_dual_cohort_sites", **meta})
    return pd.DataFrame(site_rows), pd.DataFrame(meta_rows)

