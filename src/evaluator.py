"""
Model Evaluator for AI Health Risk Predictor.
Computes discrimination, calibration, and fairness metrics.
Generates publication-quality plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve

from src.utils import get_logger, ensure_dir, bootstrap_confidence_interval

logger = get_logger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {"logistic_regression": "#3498db", "xgboost": "#e74c3c", "neural_net": "#2ecc71"}
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Core metrics ─────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 500,
) -> dict:
    """
    Compute a comprehensive metrics dictionary.
    
    Returns:
        dict with AUC-ROC, AUC-PR, Brier Score, and their 95% CIs.
    """
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    brier   = brier_score_loss(y_true, y_prob)

    # Bootstrap CIs
    roc_lo, roc_hi = bootstrap_confidence_interval(y_true, y_prob, roc_auc_score, n_bootstrap)
    pr_lo,  pr_hi  = bootstrap_confidence_interval(y_true, y_prob, average_precision_score, n_bootstrap)

    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "auc_roc": auc_roc,
        "auc_roc_ci": (roc_lo, roc_hi),
        "auc_pr": auc_pr,
        "auc_pr_ci": (pr_lo, pr_hi),
        "brier_score": brier,
        "threshold": threshold,
        "precision": report.get("1", {}).get("precision", 0.0),
        "recall":    report.get("1", {}).get("recall", 0.0),
        "f1":        report.get("1", {}).get("f1-score", 0.0),
        "n_positive": int(y_true.sum()),
        "n_total":    int(len(y_true)),
        "prevalence": float(y_true.mean()),
    }


def print_metrics_table(results: dict[str, dict]) -> None:
    """Pretty-print a comparison table of model metrics."""
    rows = []
    for model_name, m in results.items():
        lo, hi = m["auc_roc_ci"]
        rows.append({
            "Model": model_name,
            "AUC-ROC": f"{m['auc_roc']:.3f} ({lo:.3f}–{hi:.3f})",
            "AUC-PR":  f"{m['auc_pr']:.3f}",
            "Brier":   f"{m['brier_score']:.4f}",
            "F1":      f"{m['f1']:.3f}",
        })
    df = pd.DataFrame(rows).set_index("Model")
    logger.info("\n" + df.to_string())
    return df


# ── Fairness evaluation ───────────────────────────────────────────────────────
def evaluate_fairness(
    test_raw: pd.DataFrame,
    y_prob: np.ndarray,
    subgroup_cols: list[str] = ["sex", "ethnicity"]
) -> pd.DataFrame:
    """
    Compute AUC-ROC per subgroup (sex, ethnicity).
    """
    test_df = test_raw.copy().reset_index(drop=True)
    test_df["y_prob"] = y_prob
    test_df["y_true"] = test_df["event_within_5yrs"]

    rows = []
    for col in subgroup_cols:
        for group_val in sorted(test_df[col].dropna().unique()):
            mask = test_df[col] == group_val
            sub = test_df[mask]
            if sub["y_true"].nunique() < 2 or len(sub) < 20:
                continue
            try:
                auc = roc_auc_score(sub["y_true"], sub["y_prob"])
                rows.append({
                    "Subgroup": col,
                    "Value": group_val,
                    "N": len(sub),
                    "N_events": int(sub["y_true"].sum()),
                    "Prevalence": f"{sub['y_true'].mean():.1%}",
                    "AUC-ROC": round(auc, 3),
                })
            except Exception as e:
                logger.warning(f"Fairness eval skipped {col}={group_val}: {e}")

    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_roc_curves(
    results: dict[str, dict],
    y_trues: dict[str, np.ndarray],
    y_probs: dict[str, np.ndarray],
    save_path: str = "reports/figures/roc_curves.png"
) -> None:
    """Plot ROC curves for all models on the same axis."""
    ensure_dir(Path(save_path).parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

    for model_name in results:
        fpr, tpr, _ = roc_curve(y_trues[model_name], y_probs[model_name])
        auc = results[model_name]["auc_roc"]
        color = PALETTE.get(model_name, "gray")
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{model_name.replace('_', ' ').title()} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC plot → {save_path}")


def plot_pr_curves(
    y_trues: dict[str, np.ndarray],
    y_probs: dict[str, np.ndarray],
    results: dict[str, dict],
    save_path: str = "reports/figures/pr_curves.png"
) -> None:
    """Precision-Recall curves."""
    ensure_dir(Path(save_path).parent)
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name in y_probs:
        prec, rec, _ = precision_recall_curve(y_trues[model_name], y_probs[model_name])
        auc_pr = results[model_name]["auc_pr"]
        color = PALETTE.get(model_name, "gray")
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{model_name.replace('_', ' ').title()} (AP={auc_pr:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"PR plot → {save_path}")


def plot_calibration_curves(
    y_trues: dict[str, np.ndarray],
    y_probs: dict[str, np.ndarray],
    save_path: str = "reports/figures/calibration.png"
) -> None:
    """Reliability (calibration) diagrams."""
    ensure_dir(Path(save_path).parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for model_name in y_probs:
        frac_pos, mean_pred = calibration_curve(
            y_trues[model_name], y_probs[model_name], n_bins=10, strategy="quantile"
        )
        color = PALETTE.get(model_name, "gray")
        ax.plot(mean_pred, frac_pos, "o-", lw=2, color=color,
                label=model_name.replace("_", " ").title())

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves (Reliability Diagrams)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Calibration plot → {save_path}")


def plot_fairness_heatmap(
    fairness_df: pd.DataFrame,
    save_path: str = "reports/figures/fairness_heatmap.png"
) -> None:
    """Heatmap of AUC-ROC by subgroup."""
    ensure_dir(Path(save_path).parent)
    pivot = fairness_df.pivot(index="Subgroup", columns="Value", values="AUC-ROC")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title("Fairness: AUC-ROC by Demographic Subgroup")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Fairness heatmap → {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = "reports/figures/confusion_matrix.png"
) -> None:
    ensure_dir(Path(save_path).parent)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Event", "Event"],
                yticklabels=["No Event", "Event"])
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
