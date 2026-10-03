"""Transition-matrix aggregation helpers."""

from __future__ import annotations

import pandas as pd


def equal_site_transition(matrix: pd.DataFrame) -> pd.DataFrame:
    result = matrix.groupby(["wave", "analysis", "achievement_decile", "development_decile"], as_index=False).agg(
        cell_probability=("cell_probability", "mean"), sites=("site", "nunique")
    )
    result["site_code"] = "EQUAL_SITE_MEAN"
    result["site"] = "Equal-site mean"
    result["cohort_scope"] = "All"
    return result


def equal_site_diagonal(diagonal: pd.DataFrame) -> pd.DataFrame:
    result = diagonal.groupby(["wave", "analysis", "decile"], as_index=False).agg(
        diagonal_probability=("diagonal_probability", "mean"), diagonal_lift=("diagonal_lift", "mean"),
        near_match_probability=("near_match_probability", "mean"),
        near_match_permutation_baseline=("near_match_permutation_baseline", "mean"),
        sites=("site", "nunique")
    )
    result["site_code"] = "EQUAL_SITE_MEAN"
    result["site"] = "Equal-site mean"
    result["cohort_scope"] = "All"
    return result

