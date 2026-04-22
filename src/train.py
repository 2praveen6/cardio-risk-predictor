"""
Model Training Script for AI Health Risk Predictor.
Trains all three models, evaluates them, generates SHAP explanations, and saves artefacts.
Run directly: python src/train.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import generate_synthetic_ehr
from src.preprocessor import run_preprocessing_pipeline
from src.models import get_model
from src.evaluator import (
    compute_metrics, print_metrics_table,
    evaluate_fairness,
    plot_roc_curves, plot_pr_curves, plot_calibration_curves,
    plot_fairness_heatmap, plot_confusion_matrix
)
from src.explainer import RiskExplainer
from src.utils import get_logger, ensure_dir, save_json, save_model

logger = get_logger(__name__)


def run_training_pipeline(
    data_csv: str = "data/synthetic_ehr.csv",
    generate_data: bool = True,
    n_patients: int = 5000,
    seed: int = 42,
    models_to_train: list[str] | None = None,
    figures_dir: str = "reports/figures",
    models_dir: str = "models",
):
    """End-to-end training and evaluation pipeline."""
    ensure_dir(figures_dir)
    ensure_dir(models_dir)

    if models_to_train is None:
        models_to_train = ["logistic_regression", "xgboost", "neural_net"]

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    if generate_data or not Path(data_csv).exists():
        logger.info(f"Generating {n_patients} synthetic patients...")
        ensure_dir(Path(data_csv).parent)
        df = generate_synthetic_ehr(n_patients=n_patients, seed=seed)
        df.to_csv(data_csv, index=False)
        logger.info(f"✓ Data saved → {data_csv}")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    splits = run_preprocessing_pipeline(
        input_csv=data_csv,
        output_dir="data/splits",
        preprocessor_path=f"{models_dir}/preprocessor.pkl",
        seed=seed,
    )
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    test_raw          = splits["test_raw"]

    # ── Step 2.5: Data Analysis ──────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Data Analysis (Class Distribution & Feature Means)")
    target_dist = y_train.value_counts(normalize=True) * 100
    logger.info(f"Target Distribution (Train): {target_dist.to_dict()}")
    
    # Load raw dataframe for analysis
    df_raw = pd.read_csv("data/cleaned_synthetic_ehr.csv")
    raw_dist = df_raw["event_within_5yrs"].value_counts(normalize=True) * 100
    logger.info(f"Target Distribution (Raw Before SMOTE): {raw_dist.to_dict()}")
    
    logger.info("\nMean features for Event=0 (Low Risk):")
    logger.info(f"\n{df_raw[df_raw['event_within_5yrs']==0][['age', 'systolic_bp', 'ldl', 'hba1c', 'bmi']].mean().to_string()}")
    logger.info("\nMean features for Event=1 (High Risk):")
    logger.info(f"\n{df_raw[df_raw['event_within_5yrs']==1][['age', 'systolic_bp', 'ldl', 'hba1c', 'bmi']].mean().to_string()}")

    # ── Step 3: Train Models ─────────────────────────────────────────────────
    trained_models = {}
    probas = {}

    for model_name in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name}")

        model = get_model(model_name)

        if model_name == "xgboost":
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)

        # Evaluate on test set
        proba = model.predict_proba(X_test)
        probas[model_name] = proba

        metrics = compute_metrics(y_test.values, proba)
        logger.info(
            f"  AUC-ROC: {metrics['auc_roc']:.3f} "
            f"| AUC-PR: {metrics['auc_pr']:.3f} "
            f"| Brier: {metrics['brier_score']:.4f}"
        )

        # Save model
        model_path = f"{models_dir}/{model_name}.pkl"
        model.save(model_path)
        trained_models[model_name] = {"model": model, "metrics": metrics}

        # Save metrics
        save_json(metrics, f"{models_dir}/{model_name}_metrics.json")

        # Confusion matrix
        y_pred = model.predict(X_test)
        plot_confusion_matrix(
            y_test.values, y_pred, model_name,
            save_path=f"{figures_dir}/confusion_{model_name}.png"
        )

    # ── Step 4: Comparative Evaluation ───────────────────────────────────────
    logger.info("\nComparative evaluation:")
    all_metrics = {name: d["metrics"] for name, d in trained_models.items()}
    y_trues = {name: y_test.values for name in trained_models}
    metrics_df = print_metrics_table(all_metrics)
    metrics_df.to_csv("reports/model_comparison.csv")

    plot_roc_curves(all_metrics, y_trues, probas,
                    save_path=f"{figures_dir}/roc_curves.png")
    plot_pr_curves(y_trues, probas, all_metrics,
                   save_path=f"{figures_dir}/pr_curves.png")
    plot_calibration_curves(y_trues, probas,
                            save_path=f"{figures_dir}/calibration.png")

    # ── Step 5: Fairness Evaluation ──────────────────────────────────────────
    logger.info("\nFairness evaluation (XGBoost)...")
    best_model_name = "xgboost" if "xgboost" in trained_models else models_to_train[-1]
    fairness_df = evaluate_fairness(test_raw, probas[best_model_name])
    fairness_df.to_csv("reports/fairness_results.csv", index=False)
    logger.info(f"\n{fairness_df.to_string(index=False)}")
    if len(fairness_df) > 0:
        plot_fairness_heatmap(fairness_df, save_path=f"{figures_dir}/fairness_heatmap.png")

    # ── Step 6: SHAP Explainability ─────────────────────────────────────────
    logger.info(f"\nGenerating SHAP explanations for {best_model_name}...")
    best_model = trained_models[best_model_name]["model"]

    # Use 200-row background for KernelExplainer
    bg_sample = X_train.sample(min(200, len(X_train)), random_state=seed)
    explainer = RiskExplainer(best_model, best_model_name, bg_sample)

    try:
        explainer.plot_beeswarm(
            X_test.sample(min(300, len(X_test)), random_state=seed),
            save_path=f"{figures_dir}/shap_beeswarm.png"
        )
    except Exception as e:
        logger.warning(f"Beeswarm plot failed: {e}")

    try:
        importance = explainer.plot_feature_importance_bar(
            X_test.sample(min(300, len(X_test)), random_state=seed),
            save_path=f"{figures_dir}/shap_importance_bar.png"
        )
        importance.head(15).to_csv("reports/shap_importance.csv")
    except Exception as e:
        logger.warning(f"Importance bar plot failed: {e}")

    # Single patient waterfall example
    try:
        sample_patient = X_test.iloc[[0]]
        explainer.plot_waterfall(
            sample_patient,
            save_path=f"{figures_dir}/shap_waterfall_sample.png"
        )
    except Exception as e:
        logger.warning(f"Waterfall plot failed: {e}")

    # Save explainer
    import joblib
    joblib.dump(explainer, f"{models_dir}/explainer_{best_model_name}.pkl")

    # ── Step 7: Sanity Testing (Extreme Cases) ──────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Sanity Testing: Extreme Cases Validation")
    preprocessor = splits["preprocessor"]
    
    extreme_high_risk = pd.DataFrame([{
        "age": 80,
        "sex": "M",
        "ethnicity": "White",
        "smoking_status": "Current",
        "systolic_bp": 170,
        "diastolic_bp": 100,
        "total_chol": 250,
        "hdl": 30,
        "ldl": 180,
        "hba1c": 7.5,
        "bmi": 35,
        "diabetes_flag": 1,
        "meds_list": "none",
        "heart_rate": 90
    }])
    
    extreme_low_risk = pd.DataFrame([{
        "age": 30,
        "sex": "F",
        "ethnicity": "White",
        "smoking_status": "Never",
        "systolic_bp": 110,
        "diastolic_bp": 70,
        "total_chol": 150,
        "hdl": 60,
        "ldl": 80,
        "hba1c": 5.0,
        "bmi": 22,
        "diabetes_flag": 0,
        "meds_list": "none",
        "heart_rate": 60
    }])
    
    hr_features = preprocessor.transform(extreme_high_risk)
    lr_features = preprocessor.transform(extreme_low_risk)
    
    logger.info("\nModel predictions for synthetic patients:")
    for model_name in models_to_train:
        m = trained_models[model_name]["model"]
        hr_prob = m.predict_proba(hr_features)[0]
        lr_prob = m.predict_proba(lr_features)[0]
        logger.info(f"  {model_name}:")
        logger.info(f"    Low-Risk Patient Prob:  {lr_prob:.4f} (< 0.1 expected)")
        logger.info(f"    High-Risk Patient Prob: {hr_prob:.4f} (> 0.5 expected)")

    logger.info("\n" + "="*50)
    logger.info("✓ Training pipeline complete!")
    logger.info(f"  Models saved to: {models_dir}/")
    logger.info(f"  Figures saved to: {figures_dir}/")
    logger.info(f"  Reports saved to: reports/")
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"  AUC-ROC: {all_metrics[best_model_name]['auc_roc']:.3f}")
    logger.info(f"  AUC-PR:  {all_metrics[best_model_name]['auc_pr']:.3f}")
    logger.info(f"  Brier:   {all_metrics[best_model_name]['brier_score']:.4f}")

    return trained_models, explainer, splits


def main():
    parser = argparse.ArgumentParser(description="Train AI Health Risk Predictor")
    parser.add_argument("--data-csv", default="data/synthetic_ehr.csv")
    parser.add_argument("--generate-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-patients", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+",
                        default=["logistic_regression", "xgboost", "neural_net"],
                        choices=["logistic_regression", "xgboost", "neural_net"])
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    run_training_pipeline(
        data_csv=args.data_csv,
        generate_data=args.generate_data,
        n_patients=args.n_patients,
        seed=args.seed,
        models_to_train=args.models,
        figures_dir=args.figures_dir,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
