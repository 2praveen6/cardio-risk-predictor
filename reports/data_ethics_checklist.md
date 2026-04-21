# Data Ethics Checklist
## AI Health Risk Predictor — Cardiovascular Disease

**Version:** 1.0 | **Date:** April 2026 | **Reviewer:** Project Lead

---

## 1. Data De-identification & PHI

| # | Check                                                              | Status    | Notes                                           |
|---|--------------------------------------------------------------------|-----------|-------------------------------------------------|
| 1 | All data is synthetic / fully de-identified                       | ✅ DONE   | `data_generator.py` generates synthetic data    |
| 2 | No real patient names, DOB, SSN, MRN present                     | ✅ DONE   | Synthetic IDs used (`P000001` format)           |
| 3 | Patient IDs are not linkable to real individuals                  | ✅ DONE   | Random sequential IDs                           |
| 4 | Geographic data limited to high-level region (no ZIP/address)     | ✅ DONE   | No geographic data included                     |
| 5 | Dates not present in data (only follow-up duration in years)      | ✅ DONE   | `followup_time` in years only                   |
| 6 | No free-text clinical notes that could contain identifiers        | ✅ DONE   | Structured fields only                          |
| 7 | HIPAA Safe Harbor / Expert Determination applied if using real PHI | ⚠️ N/A   | Applies only when real data is used             |

---

## 2. Data Provenance & Consent

| # | Check                                                              | Status    | Notes                                           |
|---|--------------------------------------------------------------------|-----------|-------------------------------------------------|
| 8 | Data source and collection method documented                      | ✅ DONE   | Synthetic — documented in project spec          |
| 9 | If real data: IRB/ethics board approval obtained                  | ⚠️ N/A   | Not required for synthetic data                 |
|10 | Patient consent obtained (if real data)                           | ⚠️ N/A   | Not required for synthetic data                 |
|11 | Data use agreement in place (if external data)                    | ⚠️ N/A   | Not required for synthetic data                 |
|12 | Data provenance tracked (version, hash, source)                   | ✅ DONE   | Random seed documented for reproducibility      |

---

## 3. Bias & Fairness

| # | Check                                                              | Status    | Notes                                           |
|---|--------------------------------------------------------------------|-----------|-------------------------------------------------|
|13 | Dataset includes diverse demographic groups                       | ✅ DONE   | Sex: M/F 52/48%; 5 ethnicity groups             |
|14 | Class imbalance assessed and reported                             | ✅ DONE   | ~15% event prevalence; class_weight="balanced"  |
|15 | Subgroup performance evaluated by sex                             | ✅ DONE   | AUC-ROC reported per sex group                  |
|16 | Subgroup performance evaluated by ethnicity                       | ✅ DONE   | AUC-ROC reported per ethnicity group            |
|17 | Performance disparities identified and documented                 | ✅ DONE   | Fairness heatmap in `reports/figures/`          |
|18 | Mitigation strategies considered if disparities found             | 📋 TODO  | Re-weighting or calibration per subgroup        |
|19 | Age-based subgroup analysis included                              | 📋 TODO  | Planned for next iteration                      |

---

## 4. Model Transparency

| # | Check                                                              | Status    | Notes                                           |
|---|--------------------------------------------------------------------|-----------|-------------------------------------------------|
|20 | Model architecture and choices documented                         | ✅ DONE   | `reports/project_spec.md` + README              |
|21 | Feature engineering steps documented                              | ✅ DONE   | `src/preprocessor.py` + Notebook 01             |
|22 | SHAP explainability implemented for each prediction               | ✅ DONE   | `src/explainer.py`                              |
|23 | Confidence intervals reported with predictions                    | ✅ DONE   | Bootstrap 95% CI shown in UI                    |
|24 | Model limitations explicitly stated in UI                         | ✅ DONE   | Disclaimer in app and PDF                       |
|25 | Calibration of predictions verified                               | ✅ DONE   | Reliability curves + Brier score computed       |

---

## 5. Security & Access Control

| # | Check                                                              | Status    | Notes                                           |
|---|--------------------------------------------------------------------|-----------|-------------------------------------------------|
|26 | Application does not store patient data entered via UI            | ✅ DONE   | Stateless app; no persistence of inputs         |
|27 | API keys / credentials not hardcoded                              | ✅ DONE   | No external API keys required                   |
|28 | HTTPS enforced in production deployment                           | 📋 TODO  | Required for cloud deployment                   |
|29 | Access logs implemented for PHI access                            | ⚠️ N/A   | No real PHI; required for production            |
|30 | Authentication required before patient data access                | ⚠️ N/A   | Add OAuth/SSO before production use             |

---

## 6. Clinical Safety

| # | Check                                                              | Status    | Notes                                           |
|---|--------------------------------------------------------------------|-----------|-------------------------------------------------|
|31 | Prominent disclaimer: not for clinical use                        | ✅ DONE   | In UI, PDF, README, and docs                    |
|32 | Outputs framed as "decision support" not "diagnosis"              | ✅ DONE   | Language reviewed throughout                    |
|33 | Clinical validation study designed/planned                        | 📋 TODO  | Required before real-world deployment           |
|34 | Clinician review required before action on prediction             | ✅ DONE   | Stated explicitly in UI and PDF                 |
|35 | Emergency pathways not bypassed by tool                           | ✅ DONE   | Tool is advisory only                           |

---

## 7. Regulatory Considerations (Pre-Production Checklist)

> ⚠️ The following checks apply when deploying with **real patient data** in a clinical setting.

| # | Check                                                              |
|---|-------------------------------------------------------------------|
|36 | HIPAA Business Associate Agreement (BAA) with cloud vendor        |
|37 | FDA Software as a Medical Device (SaMD) classification review     |
|38 | EU MDR / CE marking review if deployed in EU                      |
|39 | IRB / clinical trial registration if used in clinical research    |
|40 | Incident response plan for model failure events                   |

---

## Legend

- ✅ DONE — Implemented and verified
- ⚠️ N/A — Not applicable to this prototype stage
- 📋 TODO — Required before production deployment

---

*This checklist should be reviewed and updated at each project milestone.*
