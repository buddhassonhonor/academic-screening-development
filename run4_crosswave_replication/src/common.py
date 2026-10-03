"""Shared, read-only input and inference helpers for the Run4 audit."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent
LEGACY = ROOT / "sses2023_alignment"
EXP = RUN / "exp"
FIGURES = RUN / "figures"
LOGS = RUN / "logs"

INPUT_DIRECTORIES = (ROOT / "run2", ROOT / "run2_1", ROOT / "run2_2", LEGACY)
KEY_FILENAMES = (
    "jrr_depth_site_overall.csv", "jrr_depth_heterogeneity_overall.csv",
    "depth_by_site_cohort.csv", "equal_capacity_overlap.csv",
    "equal_capacity_overlap_overall.csv", "global_cohort_difference.csv",
    "site_cohort_differences.csv", "equal_capacity_cohort_gap.csv",
    "equal_capacity_gap_meta.csv", "chance_adjusted_alignment.csv",
    "capacity_inflation.csv", "depth_alignment_2023.csv",
    "heterogeneity_2023.csv", "cohort_2023.csv", "cohort_meta_2023.csv",
    "helsinki_replication.csv", "bogota_replication.csv",
    "wave_comparison_2019_2023.csv", "RESULTS_SUMMARY_2023.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: Path) -> tuple[int, str]:
    """A deterministic content digest for a directory, including relative names."""
    digest = hashlib.sha256()
    total = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        file_hash = sha256(child).encode("ascii")
        digest.update(relative + b"\0" + file_hash + b"\n")
        total += child.stat().st_size
    return total, digest.hexdigest()


def _record(path: Path, kind: str, role: str) -> dict[str, object]:
    stat = path.stat()
    if kind == "directory":
        size, digest = directory_digest(path)
    else:
        size, digest = stat.st_size, sha256(path)
    return {
        "role": role, "kind": kind, "absolute_path": str(path.resolve()),
        "size_bytes": size,
        "mtime_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
        "sha256": digest,
    }


def freeze_inputs() -> pd.DataFrame:
    """Write the mandatory pre-analysis input manifest without modifying inputs."""
    rows = [_record(path, "directory", "required_source_directory") for path in INPUT_DIRECTORIES]
    found: set[Path] = set()
    for filename in KEY_FILENAMES:
        for folder in INPUT_DIRECTORIES:
            found.update(folder.rglob(filename))
    for path in sorted(found):
        rows.append(_record(path, "file", "required_key_result"))
    manifest = pd.DataFrame(rows).sort_values(["kind", "absolute_path"]).reset_index(drop=True)
    manifest.to_csv(RUN / "source_manifest.csv", index=False)
    manifest.to_csv(RUN / "hashes_before.csv", index=False)
    return manifest


def legacy_csv(name: str) -> pd.DataFrame:
    path = LEGACY / "exp" / name
    if not path.exists():
        raise FileNotFoundError(f"Required frozen source is absent: {path}")
    return pd.read_csv(path)


def run4_csv(name: str) -> pd.DataFrame:
    path = EXP / name
    if not path.exists():
        raise FileNotFoundError(f"Run4 intermediate is absent: {path}")
    return pd.read_csv(path)


def primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "analysis" in output.columns:
        output = output[output["analysis"].eq("site_all")]
    if "cohort_scope" in output.columns:
        output = output[output["cohort_scope"].eq("All")]
    return output.copy()


def two_sided_p(estimate: float, se: float) -> float:
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
        return math.nan
    return float(2 * stats.norm.sf(abs(estimate / se)))


def ci_text(low: float, high: float, digits: int = 3) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def random_effects_reml(effects: pd.Series | np.ndarray, variances: pd.Series | np.ndarray) -> dict[str, float]:
    """REML random-effects summary matching the frozen Run3 implementation."""
    y, v = np.asarray(effects, float), np.asarray(variances, float)
    if len(y) < 2 or len(y) != len(v) or not np.isfinite(y).all() or not np.isfinite(v).all():
        raise ValueError("REML requires at least two finite effect/variance pairs")
    v = np.maximum(v, 1e-12)
    fixed_w = 1 / v
    fixed_mean = float(np.sum(fixed_w * y) / fixed_w.sum())
    q = float(np.sum(fixed_w * (y - fixed_mean) ** 2))
    df = len(y) - 1

    def nll(tau2: float) -> float:
        w = 1 / (v + tau2)
        mean = float(np.sum(w * y) / w.sum())
        return float(.5 * (np.log(v + tau2).sum() + math.log(w.sum()) + np.sum(w * (y - mean) ** 2)))

    upper = max(1e-8, float(100 * (np.var(y, ddof=1) + np.max(v))))
    fit = optimize.minimize_scalar(nll, bounds=(0., upper), method="bounded", options={"xatol": 1e-12})
    tau2 = 0. if nll(0.) <= fit.fun + 1e-10 else float(fit.x)
    w = 1 / (v + tau2)
    pooled = float(np.sum(w * y) / w.sum())
    se = math.sqrt(float(1 / w.sum()))
    return {
        "K": int(len(y)), "pooled": pooled, "pooled_se": se,
        "pooled_CI_low": pooled - 1.96 * se, "pooled_CI_high": pooled + 1.96 * se,
        "tau2": tau2, "Q": q, "Q_df": int(df), "Q_p": float(stats.chi2.sf(q, df)),
        "I2": max(0., 100 * (q - df) / q) if q else 0.,
    }


def replicate_mean_summary(
    full: pd.DataFrame, replicates: pd.DataFrame, multiplier: float,
    value_col: str = "value",
) -> dict[str, float]:
    """Design-based SE for an equal-site mean, retaining shared replicate covariance."""
    full_estimate = float(full[value_col].mean())
    by_replicate = replicates.groupby("replicate", sort=True)[value_col].mean().to_numpy(float)
    if len(by_replicate) == 0 or not np.isfinite(by_replicate).all():
        raise ValueError("No finite replicate estimates available")
    variance = float(multiplier * np.sum((by_replicate - full_estimate) ** 2))
    se = math.sqrt(variance)
    return {
        "estimate": full_estimate, "variance": variance, "SE": se,
        "CI_lower": full_estimate - 1.96 * se, "CI_upper": full_estimate + 1.96 * se,
    }


def sign_label(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"
