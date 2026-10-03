"""Paired cohort differences, kept separate for correlation and screening capacity."""

from __future__ import annotations

import math

import pandas as pd

from common import EXP, legacy_csv, two_sided_p


def _source(wave: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "2019_reanalysis" if wave == 2019 else "2023"
    site_name, meta_name = f"cohort_{suffix}.csv", f"cohort_meta_{suffix}.csv"
    return legacy_csv(site_name), legacy_csv(meta_name)


def _row(meta: pd.DataFrame, metric: str) -> pd.Series:
    selected = meta[meta.metric.eq(metric)]
    if len(selected) != 1:
        raise ValueError(f"Expected one cohort meta row for {metric}; found {len(selected)}")
    return selected.iloc[0]


def _significance(row: pd.Series) -> str:
    if row.pooled_CI_high < 0:
        return "supported_negative"
    if row.pooled_CI_low > 0:
        return "supported_positive"
    return "inconclusive"


def _depth_replication(a: pd.Series, b: pd.Series) -> str:
    if a.pooled_CI_high < 0 and b.pooled_CI_high < 0:
        return "STRONG REPLICATION"
    if a.pooled_CI_high < 0 and b.pooled < 0 and b.pooled_CI_low <= 0 <= b.pooled_CI_high:
        return "PARTIAL / DIRECTIONAL REPLICATION"
    if a.pooled_CI_high < 0 and b.pooled_CI_low > 0:
        return "CONTRADICTION"
    return "NON-REPLICATION"


def _cif_replication(a: pd.Series, b: pd.Series) -> str:
    if a.pooled_CI_low > 0 and b.pooled_CI_low > 0:
        return "STRONG REPLICATION"
    if a.pooled > 0 and b.pooled > 0:
        return "PARTIAL / DIRECTIONAL REPLICATION"
    if a.pooled > 0 and b.pooled < 0:
        return "CONTRADICTION"
    return "NON-REPLICATION"


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows, depth_rows, cif_rows = [], [], []
    meta_by_wave: dict[int, pd.DataFrame] = {}
    site_by_wave: dict[int, pd.DataFrame] = {}
    for wave in (2019, 2023):
        site, meta = _source(wave)
        meta_by_wave[wave], site_by_wave[wave] = meta, site
        r = _row(meta, "r")
        cif = _row(meta, "CIF80_10")
        site_r = site[site.metric.eq("r")]
        depth_rows.append({
            "wave": wave, "n_dual_sites": int(r.K), "n_negative": int((site_r.delta_older_minus_younger < 0).sum()),
            "pooled_delta_r": r.pooled, "SE": r.pooled_se, "CI_lower": r.pooled_CI_low,
            "CI_upper": r.pooled_CI_high, "p_two_sided": two_sided_p(r.pooled, r.pooled_se),
            "within_wave_status": _significance(r),
        })
        cif_rows.append({
            "wave": wave, "n_dual_sites": int(cif.K), "pooled_delta_CIF80_10": cif.pooled,
            "SE": cif.pooled_se, "CI_lower": cif.pooled_CI_low, "CI_upper": cif.pooled_CI_high,
            "p_two_sided": two_sided_p(cif.pooled, cif.pooled_se),
            "absolute_screening_capacity_gap_proportion": cif.pooled * .10,
            "absolute_screening_capacity_gap_percentage_points": cif.pooled * 10,
            "within_wave_status": _significance(cif),
        })
        for metric in ("r", "CAA10", "CIF80_10"):
            row = _row(meta, metric)
            detail_rows.append({"wave": wave, "metric": metric, **row.to_dict()})

    depth = pd.DataFrame(depth_rows)
    cif = pd.DataFrame(cif_rows)
    depth.loc[:, "crosswave_classification"] = _depth_replication(_row(meta_by_wave[2019], "r"), _row(meta_by_wave[2023], "r"))
    cif.loc[:, "crosswave_classification"] = _cif_replication(_row(meta_by_wave[2019], "CIF80_10"), _row(meta_by_wave[2023], "CIF80_10"))

    # A transparent independent-wave descriptive synthesis; it is not a common-site temporal effect.
    old, new = _row(meta_by_wave[2019], "CIF80_10"), _row(meta_by_wave[2023], "CIF80_10")
    difference = float(new.pooled - old.pooled)
    se = math.sqrt(float(old.pooled_se) ** 2 + float(new.pooled_se) ** 2)
    synthesis = pd.DataFrame([{
        "contrast": "cohort_delta_CIF80_10_2023_minus_2019", "difference": difference, "SE": se,
        "CI_lower": difference - 1.96 * se, "CI_upper": difference + 1.96 * se,
        "p_two_sided": two_sided_p(difference, se),
        "interpretation": "independent-wave meta-estimate comparison; no equivalence claim and no clean temporal interpretation",
    }])
    depth.to_csv(EXP / "cohort_depth_crosswave.csv", index=False)
    cif.to_csv(EXP / "cohort_cif_crosswave.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(EXP / "cohort_crosswave_metric_detail.csv", index=False)
    synthesis.to_csv(EXP / "cohort_cif_crosswave_synthesis.csv", index=False)
    return depth, cif, synthesis


if __name__ == "__main__":
    run()
