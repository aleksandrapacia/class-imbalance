import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mla.data import DATA_DIR, N_SPLITS, OUTPUT_DIR, SEED, load_dataset
from mla.linear_models import LogisticRegression, LogisticRegressionFocal
from mla.metrics import compute_alphas

SUMMARY_COLUMNS = ["roc_auc_mean", "pr_auc_mean", "f1_mean", "recall_mean"]


def eval_fold(model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr, y_tr)
    proba = np.array(model._predict(X_val)).flatten()
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_val, proba),
        "pr_auc": average_precision_score(y_val, proba),
        "f1": f1_score(y_val, pred, zero_division=0),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


def _run(model_factory, output_name, desc, extra_row=None):
    logging.basicConfig(level=logging.ERROR)
    np.random.seed(1000)
    results = []
    datasets = sorted(DATA_DIR.glob("dataset_*.csv"))
    print(f"Found {len(datasets)} datasets\n")

    for csv_path in tqdm(datasets, desc=desc):
        try:
            X, y, target_col, ir = load_dataset(csv_path)
        except Exception as e:
            print(f"[SKIP] {csv_path.stem} — {e}")
            continue

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_metrics = []
        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            scaler = StandardScaler()
            X_tr = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)
            X_val = np.nan_to_num(scaler.transform(X_val), nan=0.0)
            try:
                fold_metrics.append(eval_fold(model_factory(), X_tr, y_tr, X_val, y_val))
            except Exception as e:
                print(f"  [ERR]  {csv_path.stem} fold {fold_i}: {e}")

        if not fold_metrics:
            continue

        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        row = {
            "dataset": csv_path.stem,
            "target_col": target_col,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "imbalance_ratio": ir,
            **{f"{k}_mean": round(avg[k], 4) for k in avg},
            **{f"{k}_std": round(std[k], 4) for k in std},
        }
        if extra_row is not None:
            row.update(extra_row(y))
        results.append(row)

    if not results:
        print("\nNo results.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / output_name
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    return results_df


def _print_summary(results_df, columns, title):
    print(f"\n{title}")
    print(results_df[columns].to_string(index=False))
    print("\nMean across datasets")
    for col in SUMMARY_COLUMNS:
        print(f"  {col}: {results_df[col].mean():.4f}")


def run_cross_entropy():
    """5-fold cross-entropy logistic regression. Writes output/cross_entropy_results.csv."""
    results_df = _run(
        lambda: LogisticRegression(lr=0.01, max_iters=500, penalty="l2", C=0.01),
        "cross_entropy_results.csv",
        "Cross-entropy",
    )
    if not results_df.empty:
        _print_summary(
            results_df,
            ["dataset", "imbalance_ratio", "roc_auc_mean", "pr_auc_mean", "f1_mean", "recall_mean"],
            "Cross-entropy",
        )
    return results_df


def run_focal_loss(gamma=2.0):
    """5-fold focal-loss logistic regression. Writes output/focal_loss_results.csv."""
    results_df = _run(
        lambda: LogisticRegressionFocal(gamma=gamma, lr=0.01, max_iters=500, penalty="l2", C=0.01),
        "focal_loss_results.csv",
        "Focal loss",
        extra_row=lambda y: {
            "gamma": gamma,
            "alpha_pos": round(compute_alphas(y)[0], 4),
            "alpha_neg": round(compute_alphas(y)[1], 4),
        },
    )
    if not results_df.empty:
        _print_summary(
            results_df,
            ["dataset", "imbalance_ratio", "alpha_pos", "roc_auc_mean", "pr_auc_mean", "f1_mean", "recall_mean"],
            f"Focal loss (gamma={gamma})",
        )
    return results_df
