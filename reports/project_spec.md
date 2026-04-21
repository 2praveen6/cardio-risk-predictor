# One-Page Project Specification

## AI Health Risk Predictor — Cardiovascular Disease (CVD)

**Version:** 1.0 | **Date:** April 2026 | **Status:** Prototype

---

### 1. Objective

Develop an AI-powered Decision Support Tool that estimates individual 5-year cardiovascular event risk
for clinicians and patients, providing actionable, explainable predictions backed by machine learning.

---

### 2. Scope

| Dimension       | Details                                                              |
|-----------------|----------------------------------------------------------------------|
| Condition       | 5-year cardiovascular disease event (MI, stroke, cardiac death)     |
| Target Users    | Primary: Clinicians; Secondary: Patients                            |
| Platform        | Streamlit web prototype + Python package; locally runnable          |
| Data            | Synthetic EHR (5,000 patients); no real PHI used                   |
| Models          | Logistic Regression, XGBoost, Neural Net (MLP)                     |
| Explainability  | SHAP values (top-5 features per patient + global summary)          |
| Fairness        | Subgroup AUC-ROC by sex and ethnicity                               |

---

### 3. Input Features

| Feature           | Type       | Clinical Source               |
|-------------------|------------|-------------------------------|
| age               | Numeric    | Demographics                  |
| sex               | Categorical| Demographics                  |
| ethnicity         | Categorical| Demographics                  |
| systolic_bp       | Numeric    | Vitals                        |
| diastolic_bp      | Numeric    | Vitals                        |
| total_chol        | Numeric    | Blood panel                   |
| hdl               | Numeric    | Blood panel                   |
| ldl               | Numeric    | Blood panel                   |
| hba1c             | Numeric    | Metabolic panel               |
| smoking_status    | Categorical| Social history                |
| diabetes_flag     | Binary     | Medical history               |
| bmi               | Numeric    | Vitals                        |
| meds_list         | Text/Multi | Medication list               |

---

### 4. Outputs

- **Risk Score:** 0–100% probability of 5-year CVD event
- **Risk Category:** Low (<10%), Moderate (10–20%), High (>20%)
- **Confidence Interval:** 95% CI via bootstrap
- **SHAP Explanation:** Top-5 contributing features with direction and magnitude
- **PDF Report:** Downloadable clinical summary (ReportLab)
- **Patient View:** Simplified result with plain-language recommendations

---

### 5. Evaluation Plan

| Metric               | Tool                        |
|----------------------|-----------------------------|
| Discrimination       | AUC-ROC, AUC-PR             |
| Calibration          | Brier Score, reliability curve |
| Subgroup Fairness    | AUC-ROC by sex & ethnicity  |
| Explainability       | SHAP TreeExplainer/Kernel   |

---

### 6. Technical Stack

```
Python 3.10+  |  pandas · numpy · scikit-learn  |  xgboost  |  shap
matplotlib · seaborn · plotly  |  streamlit  |  reportlab  |  joblib
```

---

### 7. Data Flow

```
data_generator.py  →  synthetic_ehr.csv
                    ↓
preprocessor.py    →  data/splits/{train,val,test}.csv  +  models/preprocessor.pkl
                    ↓
train.py           →  models/{lr,xgb,nn}.pkl  +  reports/figures/
                    ↓
streamlit_app.py   →  Web UI (form → prediction → SHAP → PDF)
```

---

### 8. Deployment Notes

| Environment   | Instructions                                          |
|---------------|-------------------------------------------------------|
| Local         | `pip install -r requirements.txt && python src/train.py && streamlit run app/streamlit_app.py` |
| Docker        | `docker build -t cardiorisk . && docker run -p 8501:8501 cardiorisk` |
| Cloud         | Streamlit Community Cloud (free tier); or Heroku/GCP  |

---

### 9. Limitations

- Trained on **synthetic data only**; must be re-trained on validated real-world data before clinical use
- Framingham-derived risk correlations; may not generalize to all populations
- Not validated for clinical use — research prototype only
- Fairness analysis limited to sex and ethnicity (binary sex encoding)

---

### 10. Next Steps

1. Obtain de-identified real EHR with appropriate consent and governance
2. Model validation on external cohort (e.g., UK Biobank public subset)
3. Prospective clinical validation study
4. HIPAA-compliant hosting + BAA for real patient data
5. Integration with FHIR/EHR via SMART-on-FHIR

---

*This document does not constitute medical advice and is intended for informational purposes only.*
