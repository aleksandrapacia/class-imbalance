"""ROC-AUC comparison figures from the two cross-validated result tables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mla.data import OUTPUT_DIR

BCE_COLOR = "#2c5f8a"
FOCAL_COLOR = "#d4762c"
POSITIVE = "#2e7d4f"
NEGATIVE = "#a33b3b"


def save_roc_auc_figures():
    bce = pd.read_csv(OUTPUT_DIR / "cross_entropy_results.csv")
    focal = pd.read_csv(OUTPUT_DIR / "focal_loss_results.csv")
    df = bce.merge(focal, on="dataset", suffixes=("_bce", "_fl"))
    df["label"] = df["dataset"].str.replace("dataset_", "", regex=False)
    df["delta"] = df["roc_auc_mean_fl"] - df["roc_auc_mean_bce"]
    df = df.sort_values("imbalance_ratio_bce", ascending=True).reset_index(drop=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    y = np.arange(len(df))
    height = 0.38
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.barh(
        y - height / 2, df["roc_auc_mean_bce"], height,
        xerr=df["roc_auc_std_bce"], color=BCE_COLOR, label="Cross-entropy",
        error_kw={"ecolor": "#7f8c9b", "capsize": 1.5, "elinewidth": 0.6},
    )
    ax.barh(
        y + height / 2, df["roc_auc_mean_fl"], height,
        xerr=df["roc_auc_std_fl"], color=FOCAL_COLOR, label="Focal loss",
        error_kw={"ecolor": "#7f8c9b", "capsize": 1.5, "elinewidth": 0.6},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="#9aa3ad", lw=0.8, ls="--")
    ax.set_title(
        "ROC-AUC by dataset\n"
        "5-fold CV mean, error bars are the fold standard deviation\n"
        "sorted by imbalance ratio (highest at the top)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#e6e8eb", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc_by_dataset.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    sc = ax.scatter(
        df["roc_auc_mean_bce"], df["roc_auc_mean_fl"],
        c=df["imbalance_ratio_bce"], cmap="viridis",
        s=42, edgecolors="white", linewidths=0.4, zorder=3,
    )
    ax.plot([0.3, 1], [0.3, 1], color="#9aa3ad", ls="--", lw=1, label="Equal ROC-AUC")
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Cross-entropy ROC-AUC")
    ax.set_ylabel("Focal loss ROC-AUC")
    ax.set_title("Focal loss vs cross-entropy\npoints above the line score higher with focal loss")
    fig.colorbar(sc, ax=ax).set_label("Imbalance ratio")
    ax.legend(loc="upper left")
    ax.grid(color="#e6e8eb", lw=0.6)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc_scatter.png", dpi=160)
    plt.close()

    delta_df = df.sort_values("delta", ascending=True).reset_index(drop=True)
    colors = [POSITIVE if value >= 0 else NEGATIVE for value in delta_df["delta"]]
    y = np.arange(len(delta_df))
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.barh(y, delta_df["delta"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(delta_df["label"], fontsize=8)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Focal loss ROC-AUC − cross-entropy ROC-AUC")
    ax.set_title("Difference in ROC-AUC\ngreen: focal loss higher, red: cross-entropy higher")
    ax.grid(axis="x", color="#e6e8eb", lw=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc_difference.png", dpi=160)
    plt.close()
    print(f"Saved figures in {OUTPUT_DIR}")
