"""Shared utility functions for AI Health Risk Predictor."""

import logging
from pathlib import Path
import json
import joblib
import numpy as np


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: str | Path) -> None:
    """Save a dictionary to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_model(model, path: str | Path) -> None:
    """Save a sklearn-compatible model with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """Load a joblib-serialized model."""
    return joblib.load(path)


def risk_category(score: float) -> tuple[str, str]:
    """
    Convert a numeric risk probability to a category label and color.
    
    Returns:
        (category, hex_color)
    """
    if score < 0.10:
        return "Low", "#27ae60"
    elif score < 0.20:
        return "Moderate", "#f39c12"
    else:
        return "High", "#e74c3c"


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a given metric.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric_fn: Function(y_true, y_prob) -> float
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval width (0.95 = 95%)
        seed: Random seed
    
    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except Exception:
            pass
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))
