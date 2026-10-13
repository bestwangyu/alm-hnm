from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "source_data"
OUT = ROOT / "reproduced_outputs"
BLUE, ORANGE, GREEN, GRAY = "#2F6B9A", "#D28B45", "#4D8B6A", "#8C939B"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ablation():
    data = pd.read_csv(SOURCE / "figure2_ablation_summary.csv")
    order = ["all", "random", "topk", "semi_hard", "semi_hard_topk"]
    data = data.set_index("mode").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(order))
    for ax, mean, sd, title in (
        (axes[0], "test_roc_auc_mean", "test_roc_auc_std", "ROC-AUC"),
        (axes[1], "test_auprc_mean", "test_auprc_std", "AUPRC"),
    ):
        ax.errorbar(x, data[mean], yerr=data[sd], fmt="o", color=BLUE, capsize=3)
        ax.set_xticks(x, ["All", "Random", "Top-k", "Semi-hard", "Mixed"], rotation=25, ha="right")
        ax.set_ylabel(title); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Mining ablation on CIFAR100 (mean +/- SD, n=20)")
    save(fig, "figure_ablation")


def robustness():
    data = pd.read_csv(SOURCE / "figure3_robustness_summary.csv")
    grid = data[(data["mode"] == "semi_hard_topk") & (data["random_k"] == 0)]
    baseline = float(data.loc[data["label"] == "all", "test_roc_auc_mean"].iloc[0])
    table = grid.pivot(index="semi_hard_upper", columns="topk", values="test_roc_auc_mean") - baseline
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    image = ax.imshow(table.values * 100, cmap="RdYlBu", aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns); ax.set_xlabel("Top-k")
    ax.set_yticks(range(len(table.index)), table.index); ax.set_ylabel("Semi-hard upper bound")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]): ax.text(j, i, f"{table.iloc[i,j]*100:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="ROC-AUC change vs original ALM (pp)")
    save(fig, "figure_parameter_robustness")


def generalization():
    data = pd.read_csv(SOURCE / "figure4_generalization_paired.csv")
    subset = data[data.metric.isin(["test_roc_auc", "test_fpr_at_tpr95"])].copy()
    subset["benefit_pp"] = np.where(
        subset.metric.eq("test_roc_auc"), -subset.control_minus_method_mean, subset.control_minus_method_mean
    ) * 100
    subset["condition"] = subset.dataset_label + " 1:" + subset.ratio.astype(str)
    conditions = list(dict.fromkeys(subset.condition))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for ax, metric, title in (
        (axes[0], "test_roc_auc", "ROC-AUC benefit"),
        (axes[1], "test_fpr_at_tpr95", "FPR95 reduction"),
    ):
        frame = subset[subset.metric == metric].set_index("condition").reindex(conditions)
        ax.axhline(0, color="black", lw=.7); ax.bar(range(len(frame)), frame.benefit_pp, color=np.where(frame.benefit_pp >= 0, BLUE, ORANGE))
        ax.set_ylabel("Benefit (pp)"); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[1].set_xticks(range(len(conditions)), conditions, rotation=35, ha="right")
    save(fig, "figure_generalization")


def losses():
    data = pd.read_csv(SOURCE / "figure5_loss_response_differences.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(data)); colors = np.where(data["ROC-AUC"] >= 0, BLUE, ORANGE)
    axes[0].bar(x, data["ROC-AUC"], color=colors); axes[0].axhline(0, color="black", lw=.7); axes[0].set_ylabel("ROC-AUC change (pp)")
    axes[1].bar(x, data["FPR95"], color=np.where(data["FPR95"] >= 0, GREEN, ORANGE)); axes[1].axhline(0, color="black", lw=.7); axes[1].set_ylabel("FPR95 reduction (pp)")
    for ax in axes:
        ax.set_xticks(x, data.loss, rotation=35, ha="right"); ax.grid(axis="y", alpha=.25)
    save(fig, "figure_loss_heterogeneity")


def convergence():
    data = pd.read_csv(SOURCE / "figure6_convergence_aggregate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    x = np.arange(len(data)); labels = data.point_tag.str.replace("_", " ")
    axes[0].errorbar(x, data.auc_mean, yerr=data.auc_std, fmt="o", color=BLUE, capsize=3); axes[0].set_ylabel("Ensemble ROC-AUC (%)")
    axes[1].errorbar(x, data.fpr98_mean, yerr=data.fpr98_std, fmt="o", color=ORANGE, capsize=3); axes[1].set_ylabel("FPR98 (%)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Independent 10-seed blocks (mean +/- SD, n=3 blocks)")
    save(fig, "figure_convergence")


def write_key_table():
    frames = []
    for path in sorted((ROOT / "results/summaries").rglob("*.csv")):
        frame = pd.read_csv(path)
        frames.append({"file": path.relative_to(ROOT).as_posix(), "rows": len(frame), "columns": len(frame.columns)})
    pd.DataFrame(frames).to_csv(OUT / "table_inventory.csv", index=False)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    ablation(); robustness(); generalization(); losses(); convergence(); write_key_table()
    print(f"Wrote reproducible outputs to {OUT}")


if __name__ == "__main__":
    main()
