"""
Synthetic EHR Data Generator for AI Health Risk Predictor
Generates realistic patient data with clinically plausible correlations
inspired by Framingham Heart Study risk factors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

RANDOM_SEED = 42

def generate_synthetic_ehr(n_patients: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic EHR dataset with clinically plausible correlations.
    
    Returns:
        pd.DataFrame with all required fields.
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ──────────────────────────────────────────────────────────
    patient_ids = [f"P{str(i).zfill(6)}" for i in range(1, n_patients + 1)]
    
    # Age distribution: 30–85, skewed toward middle age
    age = rng.integers(30, 86, size=n_patients).astype(float)
    # Add some missing
    missing_mask = rng.random(n_patients) < 0.01
    age[missing_mask] = np.nan

    sex = rng.choice(["M", "F"], size=n_patients, p=[0.52, 0.48])
    ethnicity = rng.choice(
        ["White", "Black", "Hispanic", "Asian", "Other"],
        size=n_patients,
        p=[0.60, 0.13, 0.18, 0.06, 0.03]
    )

    # ── Blood Pressure ────────────────────────────────────────────────────────
    age_filled = np.where(np.isnan(age), 55, age)
    sbp_base = 90 + 0.6 * age_filled + rng.normal(0, 12, n_patients)
    sbp_base = np.clip(sbp_base, 80, 220)
    
    dbp_base = 60 + 0.25 * age_filled + 0.35 * (sbp_base - 120) + rng.normal(0, 8, n_patients)
    dbp_base = np.clip(dbp_base, 40, 130)

    # Males tend slightly higher BP
    sbp_base += np.where(sex == "M", 3, 0)
    dbp_base += np.where(sex == "M", 2, 0)

    # Introduce missing values (~3%)
    sbp = sbp_base.copy()
    dbp = dbp_base.copy()
    sbp[rng.random(n_patients) < 0.03] = np.nan
    dbp[rng.random(n_patients) < 0.03] = np.nan

    # ── Cholesterol ───────────────────────────────────────────────────────────
    total_chol = 150 + 0.4 * age_filled + rng.normal(0, 25, n_patients)
    total_chol = np.clip(total_chol, 100, 350)

    hdl = rng.normal(55, 12, n_patients)
    hdl += np.where(sex == "F", 8, 0)  # Women tend higher HDL
    hdl = np.clip(hdl, 20, 100)

    ldl = total_chol - hdl - rng.uniform(20, 40, n_patients)
    ldl = np.clip(ldl, 40, 250)

    # Missing cholesterol (~5%)
    chol_missing = rng.random(n_patients) < 0.05
    total_chol[chol_missing] = np.nan
    ldl[chol_missing] = np.nan
    hdl[rng.random(n_patients) < 0.04] = np.nan

    # ── Metabolic ─────────────────────────────────────────────────────────────
    # HbA1c: baseline ~5.4%, elevated with age and BMI
    bmi = rng.normal(27, 5, n_patients)
    bmi += np.where(sex == "M", 0.5, 0)
    bmi = np.clip(bmi, 16, 50)

    hba1c = 4.8 + 0.015 * (age_filled - 40) + 0.03 * (bmi - 25) + rng.normal(0, 0.4, n_patients)
    hba1c = np.clip(hba1c, 4.0, 12.0)

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    smoking_status = np.where(
        sex == "M",
        rng.choice(["Never", "Former", "Current"], size=n_patients, p=[0.55, 0.25, 0.20]),
        rng.choice(["Never", "Former", "Current"], size=n_patients, p=[0.65, 0.15, 0.20])
    )

    # ── Comorbidities ─────────────────────────────────────────────────────────
    # Diabetes: ~10.5% prevalence, correlated with BMI/HbA1c
    diabetes_prob = 0.05 + 0.003 * (bmi - 25).clip(0) + 0.04 * (hba1c > 6.5).astype(float)
    diabetes_flag = (rng.random(n_patients) < diabetes_prob).astype(int)

    # ── Medications ──────────────────────────────────────────────────────────
    med_options = [
        "statin", "ace_inhibitor", "arb", "beta_blocker",
        "calcium_channel_blocker", "aspirin", "metformin", "insulin",
        "diuretic", "anticoagulant"
    ]

    def assign_meds(i):
        meds = []
        bp_elevated = (sbp_base[i] > 140 or dbp_base[i] > 90)
        if bp_elevated:
            possible = ["ace_inhibitor", "arb", "beta_blocker", "calcium_channel_blocker", "diuretic"]
            meds += list(rng.choice(possible, size=rng.integers(1, 3), replace=False))
        if total_chol[i] > 200 or (not np.isnan(total_chol[i]) and ldl[i] > 130):
            meds.append("statin")
        if diabetes_flag[i]:
            meds += list(rng.choice(["metformin", "insulin"], size=1))
        if age_filled[i] > 50 and rng.random() < 0.4:
            meds.append("aspirin")
        meds = list(set(meds))
        return "|".join(meds) if meds else "none"

    meds_list = [assign_meds(i) for i in range(n_patients)]

    # ── Follow-up Time ────────────────────────────────────────────────────────
    followup_time = rng.uniform(0.5, 5.0, n_patients).round(1)

    # ── Outcome: 5-year CVD Event ─────────────────────────────────────────────
    # Framingham-inspired log-odds
    log_odds = (
        -10.0
        + 0.065 * age_filled
        + 0.8 * (sex == "M").astype(float)
        + 0.012 * sbp_base
        + 0.008 * (total_chol - hdl)
        + 0.7 * (smoking_status == "Current").astype(float)
        + 0.3 * (smoking_status == "Former").astype(float)
        + 0.6 * diabetes_flag
        + 0.02 * (bmi - 25).clip(0)
        + 0.8 * (hba1c > 7.0).astype(float)
        - 0.015 * hdl
        # Ethnicity adjustments (modest)
        + 0.3 * (ethnicity == "Black").astype(float)
    )

    prob_event = 1 / (1 + np.exp(-log_odds))
    # Scale to realistic ~15% prevalence
    prob_event = np.clip(prob_event * 0.6, 0, 1)
    event_within_5yrs = (rng.random(n_patients) < prob_event).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "patient_id":       patient_ids,
        "age":              age.round(1),
        "sex":              sex,
        "ethnicity":        ethnicity,
        "systolic_bp":      sbp.round(1),
        "diastolic_bp":     dbp.round(1),
        "total_chol":       total_chol.round(1),
        "hdl":              hdl.round(1),
        "ldl":              ldl.round(1),
        "hba1c":            hba1c.round(2),
        "smoking_status":   smoking_status,
        "diabetes_flag":    diabetes_flag,
        "bmi":              bmi.round(1),
        "meds_list":        meds_list,
        "event_within_5yrs": event_within_5yrs,
        "followup_time":    followup_time,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EHR dataset")
    parser.add_argument("--n", type=int, default=5000, help="Number of patients")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--out", type=str, default="data/synthetic_ehr.csv", help="Output path")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.n} synthetic patient records...")
    df = generate_synthetic_ehr(n_patients=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    
    print(f"✓ Dataset saved → {args.out}")
    print(f"  Shape: {df.shape}")
    print(f"  Event prevalence: {df['event_within_5yrs'].mean():.1%}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


if __name__ == "__main__":
    main()
