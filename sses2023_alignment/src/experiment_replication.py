"""Site-level calculation engine for the frozen SSES alignment analyses."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pareto_depth import (
    core_metrics, development_rank_score, fractional_selection, near_match,
    transition_matrix, weighted_bin_membership, weighted_correlation,
    weighted_midrank
)
from survey_variance_2023 import direct_summary, fisher_summary


def _profiles(achievement: np.ndarray, development: np.ndarray, weights: np.ndarray, bins: int) -> np.ndarray:
    membership = weighted_bin_membership(achievement, weights, bins, higher_is_better=True)
    result = np.empty((development.shape[1], bins), dtype=float)
    for j in range(development.shape[1]):
        mean = float(np.average(development[:, j], weights=weights))
        sd = math.sqrt(float(np.average((development[:, j] - mean) ** 2, weights=weights)))
        z = (development[:, j] - mean) / sd if sd else development[:, j] * math.nan
        for b in range(bins):
            bw = weights * membership[b]
            result[j, b] = float(np.sum(bw * z) / bw.sum())
    return result


def _rcs_design(x: np.ndarray) -> np.ndarray:
    """Linear term plus a four-knot restricted cubic spline basis."""
    knots = np.quantile(x, [0.05, 0.35, 0.65, 0.95])
    if np.unique(knots).size < 4:
        return np.column_stack([np.ones_like(x), x, x * x])
    penultimate, last = knots[-2], knots[-1]
    terms = [np.ones_like(x), x]
    for knot in knots[:-2]:
        term = np.maximum(x - knot, 0.0) ** 3
        term -= np.maximum(x - penultimate, 0.0) ** 3 * (last - knot) / (last - penultimate)
        term += np.maximum(x - last, 0.0) ** 3 * (penultimate - knot) / (last - penultimate)
        terms.append(term)
    return np.column_stack(terms)


def _nonlinearity(achievement: np.ndarray, development: np.ndarray, weights: np.ndarray) -> list[dict[str, float]]:
    rank = weighted_midrank(achievement, weights)
    linear = np.column_stack([np.ones(len(rank)), rank])
    spline = _rcs_design(rank)
    top = fractional_selection(achievement, weights, 0.10, higher_is_better=True)
    rows = []
    for j in range(development.shape[1]):
        y = development[:, j]
        sqrtw = np.sqrt(weights)
        beta_linear, *_ = np.linalg.lstsq(linear * sqrtw[:, None], y * sqrtw, rcond=None)
        beta_spline, *_ = np.linalg.lstsq(spline * sqrtw[:, None], y * sqrtw, rcond=None)
        residual_linear = y - linear @ beta_linear
        residual_spline = y - spline @ beta_spline
        ss_linear = float(np.sum(weights * residual_linear ** 2))
        ss_spline = float(np.sum(weights * residual_spline ** 2))
        numerator_df, denominator_df = spline.shape[1] - linear.shape[1], len(y) - spline.shape[1]
        f_value = ((ss_linear - ss_spline) / numerator_df) / (ss_spline / denominator_df) if ss_spline > 0 else math.nan
        p_value = float(stats.f.sf(f_value, numerator_df, denominator_df)) if math.isfinite(f_value) else math.nan
        selected_weight = weights * top
        elite_residual = float(np.sum(selected_weight * residual_linear) / selected_weight.sum())
        rows.append({"dimension_index": j, "linear_ss": ss_linear, "spline_ss": ss_spline, "F_approx": f_value, "p_approx": p_value, "elite_residual": elite_residual})
    return rows


def _near_match_permutation(achievement: np.ndarray, depth: np.ndarray, weights: np.ndarray, bins: int, repetitions: int, seed: int) -> np.ndarray:
    """Permutation baseline for near-match, retaining site weights and tie rules."""
    ach_bins = weighted_bin_membership(achievement, weights, bins, higher_is_better=True)
    dev_bins = weighted_bin_membership(depth, weights, bins, higher_is_better=False)
    denominators = (ach_bins * weights).sum(axis=1)
    values = np.zeros(bins, dtype=float)
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        permuted = dev_bins[:, rng.permutation(len(weights))]
        for b in range(bins):
            low, high = max(0, b - 1), min(bins, b + 2)
            values[b] += float(np.sum(weights * ach_bins[b] * permuted[low:high].sum(axis=0)) / denominators[b])
    return values / repetitions


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    achievement = np.asarray(payload["achievement"], float)
    development = np.asarray(payload["development"], float)
    weights = np.asarray(payload["weights"], float)
    repweights = np.asarray(payload["repweights"], float)
    capacities = [float(x) for x in payload["capacities"]]
    targets, captures = [float(x) for x in payload["targets"]], [float(x) for x in payload["captures"]]
    full = core_metrics(achievement, development, weights, capacities, targets, captures)
    depth = full["depth"]
    matrix = transition_matrix(achievement, depth, weights, int(payload["bins"]))
    near = near_match(achievement, depth, weights, int(payload["bins"]))
    permuted_near = _near_match_permutation(achievement, depth, weights, int(payload["bins"]), int(payload["near_permutations"]), int(payload["seed"]))
    profile = _profiles(achievement, development, weights, int(payload["bins"]))
    nonlinear = _nonlinearity(achievement, development, weights)

    reps: list[dict[str, Any]] = []
    profile_reps = []
    for index in range(repweights.shape[1]):
        rw = repweights[:, index]
        valid = np.isfinite(rw) & (rw > 0) & (rw < float(payload["missing_threshold"]))
        metric = core_metrics(achievement[valid], development[valid], rw[valid], capacities, targets, captures)
        reps.append(metric)
        profile_reps.append(_profiles(achievement[valid], development[valid], rw[valid], int(payload["bins"])))
    return {
        "site_code": payload["site_code"], "site": payload["site"], "cohort_scope": payload["cohort_scope"],
        "N": len(weights), "weighted_N": float(weights.sum()), "full": full, "replicates": reps,
        "matrix": matrix, "near": near, "permuted_near": permuted_near, "profile": profile,
        "profile_reps": np.asarray(profile_reps), "nonlinear": nonlinear
    }


def run_site_analyses(data: pd.DataFrame, config: dict[str, Any], wave: int, cohort_specific: bool, workers: int) -> list[dict[str, Any]]:
    """Run exact full and replicate estimates independently for every analysis group."""
    replicate_count = int(config["replicate_count_2023" if wave == 2023 else "replicate_count_2019"])
    replicate_cols = [f"{config['replicate_prefix']}{i}" for i in range(1, replicate_count + 1)]
    groups = data.groupby(["site_code", "site", "cohort"] if cohort_specific else ["site_code", "site"], sort=True)
    payloads = []
    for group_index, (keys, frame) in enumerate(groups):
        site_code, site = keys[:2]
        scope = keys[2] if cohort_specific else "All"
        payloads.append({
            "site_code": site_code, "site": site, "cohort_scope": scope,
            "achievement": frame["achievement"].to_numpy(float),
            "development": frame[[f"development_{i}" for i in range(1, 5)]].to_numpy(float),
            "weights": frame["weight"].to_numpy(float),
            "repweights": frame[replicate_cols].to_numpy(float),
            "capacities": config["selection_capacities"], "targets": config["development_target_capacities"],
            "captures": config["capture_targets"], "bins": config["deciles"],
            "near_permutations": config["near_match_permutations"], "seed": int(config["random_seed"]) + group_index,
            "missing_threshold": config["missing_value_threshold"]
        })
    results: list[dict[str, Any]] = []
    if not payloads:
        return results
    with ProcessPoolExecutor(max_workers=min(max(1, workers), len(payloads))) as pool:
        futures = {pool.submit(_worker, payload): payload for payload in payloads}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: (row["site"], row["cohort_scope"]))


def result_tables(results: list[dict[str, Any]], config: dict[str, Any], wave: int, analysis: str) -> dict[str, pd.DataFrame]:
    """Convert complete worker results to clean, design-inference output tables."""
    multiplier = float(config["fay_brr_variance_multiplier"] if wave == 2023 else 0.5)
    dim_names = ["Curiosity", "Creativity", "Sociability", "Well-being"]
    depth_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    diagonal_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    nonlinear_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []

    for item in results:
        full, reps = item["full"], item["replicates"]
        meta = {key: item[key] for key in ("site_code", "site", "cohort_scope", "N", "weighted_N")}
        r_reps = np.asarray([rep["r"] for rep in reps], float)
        r_summary = fisher_summary(float(full["r"]), r_reps, multiplier)
        depth_rows.append({**meta, "wave": wave, "analysis": analysis, **r_summary})

        for q in config["selection_capacities"]:
            q = float(q)
            value = full["overlaps"][q]
            caa_reps = np.asarray([rep["overlaps"][q]["CAA"] for rep in reps], float)
            overlap_reps = np.asarray([rep["overlaps"][q]["overlap"] for rep in reps], float)
            caa_summary = direct_summary(value["CAA"], caa_reps, multiplier, clip=(-1.0, 1.0))
            overlap_summary = direct_summary(value["overlap"], overlap_reps, multiplier, clip=(0.0, 1.0))
            curve_rows.append({**meta, "wave": wave, "analysis": analysis, "k": q, **value,
                               "CAA_SE": caa_summary["se"], "CAA_CI_low": caa_summary["CI_low"], "CAA_CI_high": caa_summary["CI_high"],
                               "overlap_SE": overlap_summary["se"], "overlap_CI_low": overlap_summary["CI_low"], "overlap_CI_high": overlap_summary["CI_high"]})
            replicate_rows.extend({**meta, "wave": wave, "analysis": analysis, "replicate": i + 1, "metric": "CAA", "k": q, "target_k": math.nan, "capture": math.nan, "value": rep["overlaps"][q]["CAA"]} for i, rep in enumerate(reps))

        for target in config["development_target_capacities"]:
            for capture in config["capture_targets"]:
                target, capture = float(target), float(capture)
                qfull = float(full["capture_q"][(target, capture)])
                qreps = np.asarray([rep["capture_q"][(target, capture)] for rep in reps], float)
                cif_full, cif_reps = qfull / target, qreps / target
                q_summary = direct_summary(qfull, qreps, multiplier, clip=(0.0, 1.0))
                cif_summary = direct_summary(cif_full, cif_reps, multiplier)
                capture_rows.append({**meta, "wave": wave, "analysis": analysis, "target_k": target, "capture": capture,
                                     "achievement_selection_rate": qfull, "selection_rate_SE": q_summary["se"], "selection_rate_CI_low": q_summary["CI_low"], "selection_rate_CI_high": q_summary["CI_high"],
                                     "CIF": cif_full, "CIF_SE": cif_summary["se"], "CIF_CI_low": cif_summary["CI_low"], "CIF_CI_high": cif_summary["CI_high"]})
                replicate_rows.extend({**meta, "wave": wave, "analysis": analysis, "replicate": i + 1, "metric": "CIF", "k": math.nan, "target_k": target, "capture": capture, "value": value} for i, value in enumerate(cif_reps))

        matrix = item["matrix"]
        for a in range(matrix.shape[0]):
            for d in range(matrix.shape[1]):
                transition_rows.append({**meta, "wave": wave, "analysis": analysis, "achievement_decile": a + 1, "development_decile": d + 1, "cell_probability": matrix[a, d]})
        for decile in range(matrix.shape[0]):
            diagonal_rows.append({**meta, "wave": wave, "analysis": analysis, "decile": decile + 1,
                                  "diagonal_probability": matrix[decile, decile], "diagonal_lift": matrix[decile, decile] / 0.10,
                                  "near_match_probability": item["near"][decile], "near_match_permutation_baseline": item["permuted_near"][decile]})

        reps_profile = item["profile_reps"]
        for dim, name in enumerate(dim_names):
            for decile in range(int(config["deciles"])):
                profile_summary = direct_summary(float(item["profile"][dim, decile]), reps_profile[:, dim, decile], multiplier)
                profile_rows.append({**meta, "wave": wave, "analysis": analysis, "dimension": name, "achievement_decile": decile + 1,
                                     "weighted_mean_standardized": item["profile"][dim, decile], "SE": profile_summary["se"], "CI_low": profile_summary["CI_low"], "CI_high": profile_summary["CI_high"]})
        for row in item["nonlinear"]:
            nonlinear_rows.append({**meta, "wave": wave, "analysis": analysis, "dimension": dim_names[row["dimension_index"]], **row})
        replicate_rows.extend({**meta, "wave": wave, "analysis": analysis, "replicate": i + 1, "metric": "r", "k": math.nan, "target_k": math.nan, "capture": math.nan, "value": rep["r"]} for i, rep in enumerate(reps))

    return {
        "depth": pd.DataFrame(depth_rows), "curve": pd.DataFrame(curve_rows), "capture": pd.DataFrame(capture_rows),
        "transition": pd.DataFrame(transition_rows), "diagonal": pd.DataFrame(diagonal_rows),
        "profiles": pd.DataFrame(profile_rows), "nonlinear": pd.DataFrame(nonlinear_rows),
        "replicates": pd.DataFrame(replicate_rows)
    }

