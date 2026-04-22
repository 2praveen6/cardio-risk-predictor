"""
Model definitions for AI Health Risk Predictor.
Three model tiers: Logistic Regression (baseline), XGBoost, Neural Net (MLP).
All models expose a consistent interface: fit, predict_proba, save, load.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import xgboost as xgb

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_logger

logger = get_logger(__name__)


class BaseModel:
    """Shared interface for all models."""
    name: str = "base"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Model '{self.name}' saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return joblib.load(path)


# ── Logistic Regression ───────────────────────────────────────────────────────
class ClinicalLogisticRegression(BaseModel):
    """
    L2-regularized Logistic Regression with Platt calibration.
    Interpretable baseline appropriate for clinical settings.
    """
    name = "logistic_regression"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._model = CalibratedClassifierCV(
            LogisticRegression(C=C, max_iter=max_iter, random_state=seed,
                               class_weight="balanced", solver="lbfgs"),
            method="sigmoid", cv=5
        )

    def fit(self, X, y) -> "ClinicalLogisticRegression":
        logger.info(f"Fitting {self.name} (C={self.C})...")
        self._model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(np.asarray(X))[:, 1]

    @property
    def coef_(self):
        """Return coefficients from the base estimator (first fold)."""
        try:
            return self._model.calibrated_classifiers_[0].estimator.coef_[0]
        except Exception:
            return None


# ── XGBoost ───────────────────────────────────────────────────────────────────
class XGBoostRiskModel(BaseModel):
    """
    Gradient-boosted tree model with early stopping and isotonic calibration.
    Typically best-performing model in tabular clinical data.
    """
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        scale_pos_weight: float = 1.0,
        seed: int = 42,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric=["logloss", "auc"],
            tree_method="hist",
            early_stopping_rounds=30,
        )
        self._base_model = xgb.XGBClassifier(**self.params)
        # Wrap XGBoost in a probability calibrator to ensure extreme cases aren't squashed
        self._model = CalibratedClassifierCV(estimator=self._base_model, cv="prefit", method="isotonic")
        self.feature_names_: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None) -> "XGBoostRiskModel":
        logger.info(f"Fitting {self.name}...")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)

        eval_set = [(X_arr, y_arr)]
        if X_val is not None and y_val is not None:
            eval_set.append((np.asarray(X_val, dtype=float), np.asarray(y_val)))

        self._base_model.fit(
            X_arr, y_arr,
            eval_set=eval_set,
            verbose=False,
        )
        best = self._base_model.best_iteration
        logger.info(f"  Best iteration: {best}")

        if X_val is not None and y_val is not None:
            logger.info("  Applying isotonic calibration on validation set...")
            self._model = CalibratedClassifierCV(estimator=FrozenEstimator(self._base_model), method="isotonic")
            self._model.fit(np.asarray(X_val, dtype=float), np.asarray(y_val))
        else:
            self._model = self._base_model

        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


# ── Neural Network (MLP) ──────────────────────────────────────────────────────
class NeuralNetRiskModel(BaseModel):
    """
    Multi-layer perceptron using scikit-learn MLPClassifier.
    Lightweight neural baseline without requiring deep-learning framework.
    """
    name = "neural_net"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64, 32),
        activation: str = "relu",
        dropout: float = 0.2,
        learning_rate_init: float = 1e-3,
        max_iter: int = 300,
        seed: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.seed = seed
        # Isotonic calibration wraps the MLP
        self._model = CalibratedClassifierCV(
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed,
                solver="adam",
            ),
            method="isotonic", cv=3
        )

    def fit(self, X, y) -> "NeuralNetRiskModel":
        logger.info(f"Fitting {self.name} {self.hidden_layer_sizes}...")
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(np.asarray(X, dtype=float))[:, 1]


# ── Factory ───────────────────────────────────────────────────────────────────
def get_model(name: str, **kwargs) -> BaseModel:
    """Instantiate a model by name."""
    registry = {
        "logistic_regression": ClinicalLogisticRegression,
        "xgboost": XGBoostRiskModel,
        "neural_net": NeuralNetRiskModel,
    }
    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Choose from {list(registry)}")
    return registry[name](**kwargs)


if __name__ == "__main__":
    # Quick smoke-test
    import os, sys
    sys.path.insert(0, os.path.abspath("."))
    from src.data_generator import generate_synthetic_ehr
    from src.preprocessor import run_preprocessing_pipeline
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "ehr.csv")
        generate_synthetic_ehr(500).to_csv(csv, index=False)
        splits = run_preprocessing_pipeline(
            input_csv=csv, output_dir=tmp+"/splits",
            preprocessor_path=tmp+"/prep.pkl"
        )
        X_tr, y_tr = splits["X_train"], splits["y_train"]
        X_v, y_v   = splits["X_val"],   splits["y_val"]

        for name in ["logistic_regression", "xgboost", "neural_net"]:
            m = get_model(name)
            if name == "xgboost":
                m.fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
            else:
                m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_v)
            print(f"Success {name}: proba range [{proba.min():.3f}, {proba.max():.3f}]")
