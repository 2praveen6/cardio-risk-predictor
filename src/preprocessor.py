"""
Data Preprocessor for AI Health Risk Predictor
Handles cleaning, imputation, feature engineering, scaling, and train/val/test splitting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import argparse

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

# ── Column definitions ────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "systolic_bp", "diastolic_bp", "total_chol",
    "hdl", "ldl", "hba1c", "bmi", "followup_time"
]

CATEGORICAL_FEATURES = ["sex", "ethnicity", "smoking_status"]
BINARY_FEATURES = ["diabetes_flag"]
TARGET = "event_within_5yrs"
ID_COL = "patient_id"

ENGINEERED_FEATURES = [
    "pulse_pressure",      # SBP - DBP
    "chol_ratio",          # total_chol / HDL
    "bmi_category",        # 0=Underweight, 1=Normal, 2=Overweight, 3=Obese
    "hba1c_elevated",      # 1 if HbA1c > 6.5
    "med_count",           # number of distinct medications
    "bp_elevated",         # 1 if SBP>=140 or DBP>=90
    "is_male",             # binary sex encoding
    "age_decade",          # age / 10
]


class EHRPreprocessor:
    """Full preprocessing pipeline for synthetic EHR data."""

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self.num_cols_: list[str] = []
        self.label_encoders_: dict = {}
        self._fitted = False

    # ── Helper: feature engineering ──────────────────────────────────────────
    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Pulse pressure
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

        # Cholesterol ratio (handle division by zero)
        df["chol_ratio"] = df["total_chol"] / df["hdl"].replace(0, np.nan)

        # BMI category
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3]
        ).astype(float)

        # HbA1c threshold
        df["hba1c_elevated"] = (df["hba1c"] > 6.5).astype(float)

        # Medication count
        df["med_count"] = df["meds_list"].apply(
            lambda x: 0 if pd.isna(x) or x == "none" else len(str(x).split("|"))
        )

        # Blood pressure flag
        df["bp_elevated"] = (
            (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)
        ).astype(float)

        # Binary sex
        df["is_male"] = (df["sex"] == "M").astype(float)

        # Age in decades
        df["age_decade"] = df["age"] / 10.0

        return df

    @staticmethod
    def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features."""
        df = pd.get_dummies(df, columns=["sex", "ethnicity", "smoking_status"],
                            drop_first=False)
        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return all feature columns (drop ID, target, meds_list string)."""
        drop_cols = {ID_COL, TARGET, "meds_list", "followup_time"}
        return [c for c in df.columns if c not in drop_cols]

    def _get_scaler_columns(self) -> list[str]:
        """
        Return the columns expected by the fitted scaler.

        Older saved preprocessors may have fit the scaler on numeric columns
        only, while newer ones fit it on the full engineered feature matrix.
        Use the scaler's fitted schema so inference does not pass unseen
        one-hot columns to a numeric-only scaler.
        """
        scaler_feature_names = getattr(self.scaler, "feature_names_in_", None)
        if scaler_feature_names is not None:
            scaler_cols = list(scaler_feature_names)
            missing = [c for c in scaler_cols if c not in self.feature_names_]
            if missing:
                raise RuntimeError(
                    "Saved scaler expects columns that are not in the preprocessor schema: "
                    f"{missing}"
                )
            return scaler_cols

        n_features = getattr(self.scaler, "n_features_in_", None)
        if n_features is None or n_features == len(self.feature_names_):
            return list(self.feature_names_)
        if n_features == len(self.num_cols_):
            return list(self.num_cols_)

        raise RuntimeError(
            "Saved scaler feature count does not match the preprocessor schema "
            f"({n_features} scaler features, {len(self.feature_names_)} total features, "
            f"{len(self.num_cols_)} numeric features). Retrain the preprocessor and models."
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor on the training data and transform it.
        Call this ONLY on the training split.
        """
        df = self._engineer_features(df)
        df = self._encode_categoricals(df)

        feature_cols = self._get_feature_columns(df)
        self.feature_names_ = feature_cols

        X = df[feature_cols].copy()

        # Impute numeric
        self.num_cols_ = [c for c in feature_cols if X[c].dtype in [np.float64, np.float32, float, int, np.int64]]
        X[self.num_cols_] = self.numeric_imputer.fit_transform(X[self.num_cols_])

        # Coerce all to float
        X = X.astype(float)

        # Scale
        if self.scale:
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform validation/test data using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        df = self._engineer_features(df)
        df = self._encode_categoricals(df)

        # Align columns (add missing one-hot cols with 0, drop extras)
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names_].copy()

        # Impute using fitted imputer
        df[self.num_cols_] = self.numeric_imputer.transform(df[self.num_cols_])
        df = df.astype(float)

        # Scale
        if self.scale:
            scaler_cols = self._get_scaler_columns()
            df.loc[:, scaler_cols] = self.scaler.transform(df[scaler_cols])

        return df

    def save(self, path: str | Path) -> None:
        """Persist the fitted preprocessor."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EHRPreprocessor":
        """Load a persisted preprocessor."""
        return joblib.load(path)


def run_preprocessing_pipeline(
    input_csv: str = "data/synthetic_ehr.csv",
    output_dir: str = "data/splits",
    preprocessor_path: str = "models/preprocessor.pkl",
    val_size: float = 0.20,
    test_size: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Full preprocessing pipeline: load → split → preprocess → save.
    
    Returns a dict with train/val/test DataFrames and the fitted preprocessor.
    """
    ensure_dir(output_dir)
    ensure_dir(Path(preprocessor_path).parent)

    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    logger.info(f"  Target prevalence: {df[TARGET].mean():.1%}")

    # ── Train / (Val + Test) split  ─────────────────────────────────────────
    relative_test = test_size / (1 - val_size)
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size),
        stratify=df[TARGET], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df[TARGET], random_state=seed
    )

    logger.info(f"  Split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    preprocessor = EHRPreprocessor(scale=True)

    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df[TARGET].reset_index(drop=True)

    X_val = preprocessor.transform(val_df)
    y_val = val_df[TARGET].reset_index(drop=True)

    X_test = preprocessor.transform(test_df)
    y_test = test_df[TARGET].reset_index(drop=True)

    # Save splits
    for name, X, y in [("train", X_train, y_train),
                        ("val",   X_val,   y_val),
                        ("test",  X_test,  y_test)]:
        X.assign(event_within_5yrs=y.values).to_csv(
            f"{output_dir}/{name}.csv", index=False
        )
        logger.info(f"  Saved {name} split → {output_dir}/{name}.csv")

    # Save raw test for fairness analysis (includes demographics)
    test_df.to_csv(f"{output_dir}/test_raw.csv", index=False)

    preprocessor.save(preprocessor_path)

    logger.info("✓ Preprocessing complete")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,   "y_val": y_val,
        "X_test": X_test,  "y_test": y_test,
        "test_raw": test_df,
        "preprocessor": preprocessor,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess EHR data")
    parser.add_argument("--input", default="data/synthetic_ehr.csv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--preprocessor-path", default="models/preprocessor.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_preprocessing_pipeline(
        input_csv=args.input,
        output_dir=args.output_dir,
        preprocessor_path=args.preprocessor_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
