"""
SHAP Explainer for AI Health Risk Predictor.
Generates SHAP values, waterfall charts, beeswarm plots, and top-5 feature contributions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import shap

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class RiskExplainer:
    """
    Wraps SHAP explainers for XGBoost and sklearn-compatible models.
    Provides per-patient and global explanations.
    """

    def __init__(self, model, model_name: str, X_background: pd.DataFrame | np.ndarray):
        """
        Args:
            model: Fitted model (BaseModel subclass)
            model_name: 'logistic_regression', 'xgboost', or 'neural_net'
            X_background: Background dataset for KernelExplainer (use ~100-500 rows)
        """
        self.model_name = model_name
        self.feature_names = (
            list(X_background.columns)
            if isinstance(X_background, pd.DataFrame)
            else None
        )

        logger.info(f"Building SHAP explainer for {model_name}...")

        if model_name == "xgboost":
            base_estimator = getattr(model, "_base_model", model._model)
            self.explainer = shap.TreeExplainer(base_estimator)
            self._uses_tree = True
        else:
            # For LR and NN, use linear or kernel explainer
            # Use a summarised background with k-means
            bg = shap.sample(np.asarray(X_background, dtype=float), min(200, len(X_background)))
            predict_fn = lambda x: model.predict_proba(x)
            self.explainer = shap.KernelExplainer(predict_fn, bg)
            self._uses_tree = False

        logger.info("  SHAP explainer ready.")

    def compute_shap_values(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch of samples.
        
        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        X_arr = np.asarray(X, dtype=float)
        if self._uses_tree:
            sv = self.explainer.shap_values(X_arr)
            # XGBoost returns array of shape (n, f) for binary classification
            if isinstance(sv, list):
                sv = sv[1]
            return sv
        else:
            sv = self.explainer.shap_values(X_arr)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            return np.asarray(sv)

    def get_top_features(
        self,
        X_row: pd.DataFrame | np.ndarray,
        n: int = 5
    ) -> list[dict]:
        """
        Get the top-N contributing features for a single patient.
        
        Returns:
            List of dicts: {feature, shap_value, feature_value, direction}
            sorted by abs(shap_value) descending.
        """
        sv = self.compute_shap_values(X_row[:1] if hasattr(X_row, '__len__') else X_row)
        sv = sv.flatten()

        if self.feature_names and len(self.feature_names) == len(sv):
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(sv))]

        rows = X_row.iloc[0] if isinstance(X_row, pd.DataFrame) else X_row.flatten()

        contributions = []
        for name, val, feat_val in zip(names, sv, rows):
            contributions.append({
                "feature": name,
                "shap_value": float(val),
                "feature_value": float(feat_val),
                "direction": "↑ Risk" if val > 0 else "↓ Risk",
                "abs_shap": abs(float(val)),
            })

        contributions.sort(key=lambda x: x["abs_shap"], reverse=True)
        return contributions[:n]

    # ── Plots ─────────────────────────────────────────────────────────────────
    def plot_waterfall(
        self,
        X_row: pd.DataFrame | np.ndarray,
        base_value: float = None,
        save_path: str = "reports/figures/shap_waterfall.png"
    ) -> plt.Figure:
        """
        Waterfall chart for a single patient showing SHAP contributions.
        """
        ensure_dir(Path(save_path).parent)
        sv = self.compute_shap_values(X_row[:1] if hasattr(X_row, '__len__') else X_row).flatten()
        
        if self.feature_names and len(self.feature_names) == len(sv):
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(sv))]

        # Keep top 10 by magnitude
        idx = np.argsort(np.abs(sv))[-10:][::-1]
        top_sv = sv[idx]
        top_names = [names[i] for i in idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_sv]
        y_pos = range(len(top_names))

        bars = ax.barh(list(y_pos), top_sv, color=colors, edgecolor="white", height=0.65)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([n.replace("_", " ").title() for n in top_names], fontsize=10)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP Value (impact on risk prediction)")
        ax.set_title(f"Feature Contributions — {self.model_name.replace('_', ' ').title()}")

        # Labels
        for bar, val in zip(bars, top_sv):
            ax.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=9
            )

        red_patch = mpatches.Patch(color="#e74c3c", label="Increases Risk")
        blue_patch = mpatches.Patch(color="#3498db", label="Decreases Risk")
        ax.legend(handles=[red_patch, blue_patch], loc="lower right")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Waterfall plot → {save_path}")
        return fig

    def plot_beeswarm(
        self,
        X: pd.DataFrame | np.ndarray,
        max_display: int = 15,
        save_path: str = "reports/figures/shap_beeswarm.png"
    ) -> None:
        """Global SHAP beeswarm (summary) plot."""
        ensure_dir(Path(save_path).parent)
        sv = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(9, 6))
        fn = self.feature_names if self.feature_names else None

        shap.summary_plot(
            sv, np.asarray(X, dtype=float),
            feature_names=fn,
            max_display=max_display,
            show=False,
            plot_size=(9, 6),
        )
        plt.title(f"SHAP Feature Importance — {self.model_name.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Beeswarm plot → {save_path}")

    def plot_feature_importance_bar(
        self,
        X: pd.DataFrame | np.ndarray,
        max_display: int = 15,
        save_path: str = "reports/figures/shap_importance_bar.png"
    ) -> pd.Series:
        """Bar chart of mean |SHAP| values."""
        ensure_dir(Path(save_path).parent)
        sv = self.compute_shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)

        fn = self.feature_names if self.feature_names else [f"f{i}" for i in range(len(mean_abs))]
        importance = pd.Series(mean_abs, index=fn).sort_values(ascending=False)

        top = importance.head(max_display)
        fig, ax = plt.subplots(figsize=(9, 5))
        top.plot.bar(ax=ax, color="#3498db", edgecolor="white")
        ax.set_title("Mean |SHAP| Feature Importance")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Importance bar → {save_path}")
        return importance
