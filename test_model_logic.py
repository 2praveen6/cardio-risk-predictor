import os
from pathlib import Path
import pandas as pd
import joblib
from pprint import pprint

# Emulate Streamlit Inputs Structure
def patient_input_to_df(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "patient_id":     "APP_PATIENT",
        "age":            inputs["age"],
        "sex":            inputs["sex"],
        "ethnicity":      inputs["ethnicity"],
        "systolic_bp":    inputs["sys_bp"],
        "diastolic_bp":   inputs["dia_bp"],
        "heart_rate":     inputs["heart_rate"],
        "total_chol":     inputs["chol"],
        "hdl":            inputs["hdl"],
        "ldl":            inputs["ldl"],
        "hba1c":          inputs["hba1c"],
        "smoking_status": inputs["smoking"],
        "diabetes_flag":  int(inputs.get("diabetes", False)),
        "bmi":            inputs["bmi"],
        "meds_list":      inputs.get("meds_list", "none"),
        "event_within_5yrs": 0,
        "followup_time":  5.0,
    }])

def get_risk_label(score):
    if score < 0.10: return "Low Risk"
    elif score < 0.20: return "Moderate Risk"
    else: return "High Risk"

def run_local_model_test():
    print("Loading models...")
    models_dir = Path("models")
    prep_path = models_dir / "preprocessor.pkl"
    xgb_path = models_dir / "xgboost.pkl"

    preprocessor = joblib.load(prep_path)
    model = joblib.load(xgb_path)
    print("Models loaded successfully.\n")

    # 1. Healthy Persona
    healthy = {
        'age': 30, 'sex': 'M', 'ethnicity': 'White',
        'sys_bp': 110, 'dia_bp': 70, 'heart_rate': 60, 'bmi': 21.0,
        'chol': 140, 'hdl': 65, 'ldl': 80, 
        'hba1c': 5.0, 'smoking': 'Never'
    }

    # 2. High-Risk Persona
    high_risk = {
        'age': 82, 'sex': 'M', 'ethnicity': 'White',
        'sys_bp': 190, 'dia_bp': 110, 'heart_rate': 95, 'bmi': 38.0,
        'chol': 310, 'hdl': 25, 'ldl': 190, 
        'hba1c': 10.0, 'smoking': 'Current'
    }

    scenarios = [
        ("Young & Healthy Profile", healthy, "Low Risk"),
        ("Elderly & High-Risk Profile", high_risk, "High Risk")
    ]

    for name, data, expected in scenarios:
        print(f"--- Scenario: {name} ---")
        df = patient_input_to_df(data)
        
        try:
            X_transformed = preprocessor.transform(df)
            score = float(model.predict_proba(X_transformed)[0])
            label = get_risk_label(score)
            
            print(f"Computed Score: {score*100:.1f}%")
            print(f"Computed Label: {label}")
            if label == expected:
                print(f"[OK] VERIFIED: Model accurately predicted '{expected}' based on logical inputs.\n")
            else:
                print(f"[WARN] Expected '{expected}' but got '{label}'\n")
        except Exception as e:
            print(f"[ERROR] Failed during prediction: {e}\n")

if __name__ == "__main__":
    run_local_model_test()
