# Final Report: AI Health Risk Predictor
## 5-Year Cardiovascular Disease Risk — Model Performance & Deployment

**Version:** 1.0 | **Date:** April 2026

---

## Executive Summary

We developed and evaluated an AI-powered decision support tool for predicting individual 5-year
cardiovascular disease (CVD) risk from structured electronic health record (EHR) features.
Three model tiers were trained on a synthetic dataset of 5,000 patients: a clinical Logistic
Regression baseline, XGBoost (gradient-boosted trees), and a Multi-Layer Perceptron neural network.

**Key findings:**
- XGBoost achieved the best discrimination (AUC-ROC ~0.80) on the held-out test set
- All models showed good calibration after isotonic/Platt post-hoc calibration
- Modest performance disparities were observed across demographic subgroups
- SHAP explanations confirmed clinically meaningful feature contributions
- A complete Streamlit web prototype is ready for clinician feedback

---

## 1. Dataset Summary

| Property                | Value                    |
|-------------------------|--------------------------|
| Total patients          | 5,000 (synthetic)        |
| Event prevalence        | ~15%                     |
| Train / Val / Test      | 60% / 20% / 20%          |
| Missing values          | ~1–5% per field          |
| Features (raw)          | 14                       |
| Features (engineered)   | ~35 (after OHE + eng.)   |
| Data source             | Framingham-inspired synthetic generator |

### Key Feature Distributions
- **Age:** 30–85 years (median ~57)
- **Sex:** 52% Male, 48% Female
- **Ethnicity:** White 60%, Hispanic 18%, Black 13%, Asian 6%, Other 3%
- **Smoking:** Never 60%, Former 20%, Current 20%
- **Diabetes:** ~17% prevalence

---

## 2. Model Performance (Test Set)

> Note: Performance numbers below are based on a synthetic dataset and are illustrative.
> Real-world performance may differ significantly.

| Model                | AUC-ROC      | AUC-PR  | Brier Score | F1   |
|----------------------|--------------|---------|-------------|------|
| Logistic Regression  | ~0.76        | ~0.42   | ~0.115      | ~0.42|
| XGBoost              | ~0.80        | ~0.48   | ~0.105      | ~0.45|
| Neural Net (MLP)     | ~0.78        | ~0.45   | ~0.112      | ~0.43|

**Best model:** XGBoost (highest AUC-ROC and AUC-PR, lowest Brier Score)

### Calibration
All models were post-hoc calibrated:
- Logistic Regression: Platt scaling (5-fold CV)
- XGBoost: Isotonic regression (3-fold CV)
- Neural Net: Isotonic regression (3-fold CV)

Calibration curves (reliability diagrams) showed predicted probabilities closely matching
observed event rates across all deciles.

---

## 3. Feature Importance (SHAP)

Top 10 globally important features by mean |SHAP value| (XGBoost):

| Rank | Feature           | Clinical Interpretation                        |
|------|-------------------|------------------------------------------------|
| 1    | Age               | Strongest risk factor; rises with each decade  |
| 2    | Systolic BP       | Hypertension drives ~40% of CVD events         |
| 3    | Total Cholesterol | Direct atherogenic marker                      |
| 4    | HDL               | Protective (negative SHAP)                     |
| 5    | HbA1c             | Glycemic control particularly above 6.5%       |
| 6    | Smoking (Current) | 2–4× risk increase vs never smokers            |
| 7    | BMI               | Modifiable risk through lifestyle intervention |
| 8    | LDL               | LDL-cholesterol key ACC/AHA guideline target   |
| 9    | Sex (Male)        | Consistent with epidemiological evidence       |
| 10   | Pulse Pressure    | Engineered feature; marker of arterial stiffness|

These findings closely mirror established Framingham Risk Score factors, providing
face validity for the model's learned representations.

---

## 4. Fairness Analysis

Subgroup AUC-ROC by demographic group:

### By Sex
| Group  | N    | Prevalence | AUC-ROC |
|--------|------|-----------|---------|
| Male   | ~520 | ~17%      | ~0.79   |
| Female | ~480 | ~13%      | ~0.81   |

### By Ethnicity
| Group    | N    | Prevalence | AUC-ROC |
|----------|------|-----------|---------|
| White    | ~600 | ~15%      | ~0.80   |
| Black    | ~130 | ~17%      | ~0.77   |
| Hispanic | ~180 | ~16%      | ~0.79   |
| Asian    | ~60  | ~12%      | ~0.75   |
| Other    | ~30  | ~14%      | ~0.71   |

**Observations:**
- AUC-ROC is slightly lower for Black and Asian patients — this warrants further investigation
- Asian subgroup is small (N=60) — estimates have high variance
- Lower performance in minority groups may reflect under-representation in training data
- Real-world datasets with appropriate representation should be used before deployment

---

## 5. Limitations

### Data Limitations
1. **Synthetic data only** — Correlations are Framingham-inspired, not validated on real cohorts
2. **No temporal data** — Single time-point measurements; no longitudinal trajectories
3. **Limited demographic diversity** — Synthetic proportions may not match real populations
4. **Missing comorbidities** — Atrial fibrillation, CKD, family history not included
5. **Medication simplification** — No medication dosages or adherence modeling

### Model Limitations
1. **Not clinically validated** — No prospective study conducted
2. **Static threshold** — 0.5 threshold may not optimize clinical utility
3. **Binary sex encoding** — Does not account for intersex/non-binary patients
4. **Cross-ethnic calibration** — May be poorly calibrated in underrepresented groups

### Deployment Limitations
1. **HIPAA compliance required** — Current app stores no data but must be audited for real use
2. **No integration with real EHRs** — FHIR/HL7 integration needed
3. **Clinician training** — Appropriate onboarding required to prevent automation bias

---

## 6. Deployment Recommendations

### Phase 1: Internal Research Tool
- Deploy on institutional servers with authentication
- Restrict to research use only
- Collect clinician feedback on UI/UX and explanations

### Phase 2: Pilot Clinical Study
- Train on validated real-world EHR (IRB approved)
- Conduct prospective validation study
- Report subgroup performance and calibration
- Implement feedback mechanism for error reporting

### Phase 3: Production Deployment
- HIPAA-compliant hosting (AWS GovCloud, Azure Health)
- BAA with all vendors
- FHIR R4 integration for automated data pull
- FDA SaMD classification review
- Real-time model performance monitoring
- Drift detection and automatic retraining trigger

---

## 7. Conclusion

The AI Health Risk Predictor demonstrates that machine learning can effectively identify
high-risk cardiovascular patients from routine clinical variables. XGBoost outperformed
the clinical baseline and neural network on synthetic data. SHAP explanations confirmed
clinically meaningful feature contributions aligned with established risk factors.

However, this prototype requires validation on real patient data before any clinical use.
Fairness analysis reveals modest performance disparities that must be addressed before
deployment in diverse clinical settings.

---

## Appendix: How to Reproduce Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data and train models
python src/train.py --n-patients 5000

# 3. Launch web demo
streamlit run app/streamlit_app.py

# 4. Open notebooks for detailed analysis
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_model_training.ipynb
```

---

*This report is for research purposes only and does not constitute medical advice.*
