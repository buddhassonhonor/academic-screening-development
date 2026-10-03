"""Load frozen SSES files and create auditable wave-specific complete cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat

SITE_NAMES_2023 = {
    "BOG": "Bogotá", "BGR": "Bulgaria", "CHL": "Chile", "DEL": "Delhi",
    "DUB": "Dubai", "EMR": "Emilia-Romagna", "ESP": "Spain", "GUN": "Gunma",
    "HEL": "Helsinki", "JIN": "Jinan", "KUN": "Kudus", "PER": "Peru",
    "SOB": "Sobral", "TUR": "Turin", "UKR": "Ukraine"
}
SITE_NAMES_2019 = {
    "01": "Ottawa", "02": "Houston", "03": "Bogotá", "04": "Manizales",
    "06": "Helsinki", "07": "Moscow", "08": "Istanbul", "09": "Daegu",
    "10": "Sintra", "11": "Suzhou"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(run: Path) -> dict[str, Any]:
    return json.loads((run / "analysis_config_2023.json").read_text(encoding="utf-8"))


def _normalise_site(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.round().astype(int).astype(str).str.zfill(2)
    return series.astype(str).str.strip().str.upper()


def _valid_score_frame(frame: pd.DataFrame, achievement: list[str], development: list[str], missing_threshold: float) -> pd.Series:
    values = frame[achievement + development].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(float)).all(axis=1)
    valid &= (values[achievement].to_numpy(float) < 900).all(axis=1)
    valid &= (values[development].to_numpy(float) < missing_threshold).all(axis=1)
    # WHO-5 uses 99 as a documented missing code in both waves.
    wellbeing = next(col for col in development if col.lower() in {"st_wellbeing", "st_wellbeing".lower()})
    valid &= values[wellbeing].to_numpy(float) < 99
    return pd.Series(valid, index=frame.index)


def prepare_wave(config: dict[str, Any], wave: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read an SSES wave and return analytic data plus site-by-cohort audit."""
    if wave not in (2019, 2023):
        raise ValueError("wave must be 2019 or 2023")
    is_2023 = wave == 2023
    path = Path(config[f"data_{wave}_path"])
    expected_hash = config[f"data_{wave}_sha256"]
    if sha256(path).lower() != expected_hash.lower():
        raise RuntimeError(f"{wave} raw data SHA-256 differs from frozen configuration")

    development = config["development_variables_2023" if is_2023 else "development_variables_2019"]
    weight = config["weight_2023" if is_2023 else "weight_2019"]
    count = int(config["replicate_count_2023" if is_2023 else "replicate_count_2019"])
    replicate_cols = [f"{config['replicate_prefix']}{i}" for i in range(1, count + 1)]
    core = [config["id_variable"], config["cohort_variable"], *config["achievement_variables"], *development, weight, *replicate_cols]
    if is_2023:
        core = [config["site_variable"], config["site_id_variable"], "PartCode", "InSSES_Std", "STQ_Empty", "Pct_STQ_complete", "Pct50_STA", "SchPart", *core]
    else:
        core = [config["site_id_variable"], *core]
    raw, _ = pyreadstat.read_dta(str(path), usecols=list(dict.fromkeys(core)))
    missing = set(core) - set(raw.columns)
    if missing:
        raise RuntimeError(f"{wave} raw file lacks required fields: {sorted(missing)}")

    site_source = raw[config["site_variable"]] if is_2023 else raw[config["site_id_variable"]]
    raw["site_code"] = _normalise_site(site_source)
    names = SITE_NAMES_2023 if is_2023 else SITE_NAMES_2019
    raw["site"] = raw["site_code"].map(names)
    raw["cohort_id"] = pd.to_numeric(raw[config["cohort_variable"]], errors="coerce")
    raw["cohort"] = raw["cohort_id"].map({1: "Younger", 2: "Older"})
    raw["_score_complete"] = _valid_score_frame(raw, config["achievement_variables"], development, float(config["missing_value_threshold"]))
    final_weight = pd.to_numeric(raw[weight], errors="coerce")
    raw["_valid_final_weight"] = np.isfinite(final_weight) & (final_weight > 0) & (final_weight < float(config["missing_value_threshold"]))
    if is_2023:
        raw["_valid_response"] = raw["PartCode"].astype(str).eq("R") & pd.to_numeric(raw["InSSES_Std"], errors="coerce").eq(1) & pd.to_numeric(raw["STQ_Empty"], errors="coerce").eq(0)
    else:
        raw["_valid_response"] = True
    raw["_analytic"] = raw["site"].notna() & raw["cohort"].notna() & raw["_valid_response"] & raw["_score_complete"] & raw["_valid_final_weight"]
    analytic = raw.loc[raw["_analytic"]].copy()
    analytic["achievement"] = analytic[config["achievement_variables"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    analytic["weight"] = pd.to_numeric(analytic[weight], errors="coerce")
    analytic["development_1"] = pd.to_numeric(analytic[development[0]], errors="coerce")
    analytic["development_2"] = pd.to_numeric(analytic[development[1]], errors="coerce")
    analytic["development_3"] = pd.to_numeric(analytic[development[2]], errors="coerce")
    analytic["development_4"] = pd.to_numeric(analytic[development[3]], errors="coerce")
    analytic["wave"] = wave

    audit = raw.groupby(["site_code", "site", "cohort"], dropna=False).agg(
        raw_N=(config["id_variable"], "size"),
        respondent_N=("_valid_response", "sum"),
        complete_score_N=("_score_complete", "sum"),
        valid_weight_N=("_valid_final_weight", "sum"),
        analytic_N=("_analytic", "sum")
    ).reset_index()
    weighted = analytic.groupby(["site_code", "site", "cohort"], dropna=False)["weight"].sum().rename("analytic_weight").reset_index()
    audit = audit.merge(weighted, how="left", on=["site_code", "site", "cohort"]).fillna({"analytic_weight": 0.0})
    audit["wave"] = wave
    return analytic.reset_index(drop=True), audit


def dual_cohort_sites(analytic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (code, site), group in analytic.groupby(["site_code", "site"], sort=True):
        counts = group.groupby("cohort").size()
        rows.append({
            "site_code": code, "site": site,
            "younger_available": bool(counts.get("Younger", 0)),
            "older_available": bool(counts.get("Older", 0)),
            "N_younger": int(counts.get("Younger", 0)),
            "N_older": int(counts.get("Older", 0))
        })
    return pd.DataFrame(rows)

