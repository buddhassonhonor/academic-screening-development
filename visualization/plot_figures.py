"""Publication-oriented rendering of already frozen Run4 result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "run4_crosswave_replication" / "exp"
LEGACY = ROOT / "sses2023_alignment" / "exp"
OUT = ROOT / "generated_figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 10, "legend.fontsize": 10,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=320, bbox_inches=None)
    plt.close(fig)


def capacity_figure() -> None:
    table = pd.read_csv(RESULTS / "cif_crosswave_summary.csv").sort_values("wave")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(left=.25, right=.96, top=.90, bottom=.17)
    x = np.arange(len(table))
    q = table["q80_absolute_percent"].to_numpy(float)
    err = (table["CI_upper"].to_numpy(float) - table["pooled_CIF80_10"].to_numpy(float)) * 10
    ax.bar(x, q, width=.55, color=["#3b6fb6", "#d77939"])
    ax.errorbar(x, q, yerr=err, fmt="none", capsize=5, color="black", lw=1.2)
    for value, label in ((10, "Development Top10 target"), (50, "50%"), (70, "70%")):
        ax.axhline(value, color="0.55", ls="--", lw=.9)
        ax.text(1.36, value + 1.1, label, color="0.35", ha="right", va="bottom", fontsize=9)
    ax.set_xticks(x, table["wave"].astype(str))
    ax.set_ylim(0, 85)
    ax.set_xlabel("Independent SSES wave")
    ax.set_ylabel("Achievement screen required for 80% recovery\nof Development Top10 (%)", labelpad=14)
    save(fig, "figure1_cif_replication.png")


def alignment_figure() -> None:
    table = pd.read_csv(RESULTS / "caa_crosswave_curve.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(left=.12, right=.97, top=.92, bottom=.17)
    for wave, color, marker in ((2019, "#3b6fb6", "o"), (2023, "#d77939", "s")):
        data = table[table["wave"] == wave].sort_values("k_percent")
        y = data["pooled_CAA"].to_numpy(float)
        error = np.vstack([y-data["CI_lower"].to_numpy(float),data["CI_upper"].to_numpy(float)-y])
        ax.errorbar(data["k_percent"], y, yerr=error, fmt=marker+'-', color=color, capsize=4, label=str(wave))
    ax.set_xlabel("Equal selection capacity (%)")
    ax.set_ylabel("Chance-adjusted alignment")
    ax.set_xticks([5,10,20,30,40,50])
    ax.legend(title="Wave")
    save(fig, "figure2_caa_crosswave_geometry.png")


def depth_figure() -> None:
    sources = {
        2019: pd.read_csv(RESULTS / "depth_alignment_2019_reanalysis.csv"),
        2023: pd.read_csv(LEGACY / "depth_alignment_2023.csv"),
    }
    pooled = pd.read_csv(RESULTS / "depth_alignment_crosswave.csv").set_index("wave")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.3), sharex=True)
    fig.subplots_adjust(left=.13, right=.98, top=.90, bottom=.12, wspace=.35)
    for ax, wave in zip(axes, (2019, 2023)):
        data = sources[wave].sort_values("r").reset_index(drop=True)
        y = np.arange(len(data))
        xerr = np.vstack([data["r"] - data["CI_low"], data["CI_high"] - data["r"]])
        ax.errorbar(data["r"], y, xerr=xerr, fmt="o", color="#3b6fb6", capsize=3, markersize=4.5)
        summary = pooled.loc[wave]
        pool_y = len(data) + .75
        pool_err = [[summary["pooled_r"] - summary["CI_lower_r"]], [summary["CI_upper_r"] - summary["pooled_r"]]]
        ax.errorbar([summary["pooled_r"]], [pool_y], xerr=pool_err, fmt="D", color="#d77939", capsize=4, markersize=6)
        ax.axvline(0, color="0.75", lw=.8)
        ax.set_yticks(np.r_[y, pool_y], [*data["site"], "Pooled (REML)"])
        ax.set_ylim(-.8, pool_y + .8)
        ax.invert_yaxis()
        ax.set_xlim(-.04, .34)
        ax.set_xlabel("Depth-rank correlation")
        ax.set_title(f"{wave}: $I^2$={summary['I2']:.1f}% (K={int(summary['n_sites'])})")
    axes[0].set_ylabel("Site")
    save(fig, "figure3_depth_alignment.png")


def cohort_figure() -> None:
    depth = pd.read_csv(RESULTS / "cohort_depth_crosswave.csv").sort_values("wave")
    cif = pd.read_csv(RESULTS / "cohort_cif_crosswave.csv").sort_values("wave")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    fig.subplots_adjust(left=.10, right=.98, top=.86, bottom=.20, wspace=.32)
    specs = (
        (axes[0], depth, "pooled_delta_r", "Depth correlation: Older - Younger", "Pooled difference"),
        (axes[1], cif, "pooled_delta_CIF80_10", "Screening capacity: Older - Younger", "CIF difference (x)"),
    )
    for panel, (ax, table, col, title, ylabel) in zip(("A", "B"), specs):
        y = table[col].to_numpy(float)
        error = np.vstack([y - table["CI_lower"].to_numpy(float), table["CI_upper"].to_numpy(float) - y])
        ax.errorbar(np.arange(len(table)), y, yerr=error, fmt="o", color="#7a4b9a", capsize=5, markersize=7)
        ax.axhline(0, color="0.55", lw=.9)
        ax.set_xticks(np.arange(len(table)), table["wave"].astype(str))
        ax.set_title(f"{panel}  {title}")
        ax.set_xlabel("Wave")
        ax.set_ylabel(ylabel)
    save(fig, "figure4_metric_dependent_cohort.png")


def transition_figure() -> None:
    table = pd.read_csv(RESULTS / "top10_transition_crosswave.csv").sort_values("wave")
    columns = [
        "achievement_top10_to_development_top10",
        "achievement_top10_to_development_top20",
        "achievement_top10_to_development_top30",
        "achievement_top10_to_development_top50",
        "achievement_top10_to_development_bottom50",
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    fig.subplots_adjust(left=.12, right=.98, top=.89, bottom=.17)
    x = np.arange(len(columns))
    width = .36
    for offset, (_, row) in zip((-.5, .5), table.iterrows()):
        ax.bar(x + offset * width, row[columns].to_numpy(float) * 100,
               width=width, label=str(int(row["wave"])))
    ax.set_xticks(x, ["Top10", "Top20", "Top30", "Top50", "Bottom50"])
    ax.set_xlabel("Developmental rank group")
    ax.set_ylabel("Academic top-decile students (%)")
    ax.legend(title="Wave")
    save(fig, "figure5_top10_transition.png")


if __name__ == "__main__":
    capacity_figure()
    alignment_figure()
    depth_figure()
    cohort_figure()
    transition_figure()
