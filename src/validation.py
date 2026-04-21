import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def validate_and_correct_ehr(df: pd.DataFrame, is_streamlit: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates and corrects the clinical rules in the EHR dataset.
    Returns the corrected DataFrame and a list of warnings (if any corrections were made).
    """
    df = df.copy()
    warnings = []
    
    # 1. Remove dual RAAS blockade
    # ACE inhibitor + ARB should not co-exist; keep only one randomly
    if "meds_list" in df.columns:
        def fix_raas(meds_str):
            if pd.isna(meds_str):
                return meds_str
            meds = meds_str.split("|")
            has_ace = "ace_inhibitor" in meds
            has_arb = "arb" in meds
            if has_ace and has_arb:
                # Randomly drop one, keep the other
                drop_med = np.random.choice(["ace_inhibitor", "arb"])
                meds.remove(drop_med)
                if is_streamlit:
                    warnings.append(f"Dual RAAS blockade detected. Removed {drop_med} to correct medication list.")
                return "|".join(meds)
            return meds_str

        # To track if any changes happen for batch processing warnings
        original_meds = df["meds_list"].copy()
        df["meds_list"] = df["meds_list"].apply(fix_raas)
        if not is_streamlit and (original_meds != df["meds_list"]).any():
            warnings.append("Corrected dual RAAS blockade (ACE inhibitor + ARB) in some records.")

    # 2. Fix lipid inconsistency: Total Cholesterol (TC) >= HDL + LDL
    if all(c in df.columns for c in ["total_chol", "hdl", "ldl"]):
        lipid_mask = df["total_chol"] < (df["hdl"] + df["ldl"])
        if lipid_mask.any():
            # Adjust Total Cholesterol to be at least HDL + LDL + 20% buffer for VLDL
            df.loc[lipid_mask, "total_chol"] = (df.loc[lipid_mask, "hdl"] + df.loc[lipid_mask, "ldl"]) * 1.2
            
            if is_streamlit:
                warnings.append("Lipid inconsistency detected (Total Cholesterol < HDL + LDL). Corrected Total Cholesterol.")
            else:
                warnings.append(f"Corrected lipid inconsistency (TC < HDL + LDL) for {lipid_mask.sum()} records.")

    # 3. Correct diabetes labels: if HbA1c >= 6.5, set diabetes_flag = 1
    if "hba1c" in df.columns and "diabetes_flag" in df.columns:
        diabetes_mask = (df["hba1c"] >= 6.5) & (df["diabetes_flag"] == 0)
        if diabetes_mask.any():
            df.loc[diabetes_mask, "diabetes_flag"] = 1
            if is_streamlit:
                warnings.append("HbA1c level is ≥ 6.5. Automatically diagnosed with Diabetes.")
            else:
                warnings.append(f"Corrected diabetes_flag for {diabetes_mask.sum()} records based on HbA1c ≥ 6.5.")

    return df, warnings
