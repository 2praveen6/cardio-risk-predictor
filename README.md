# 🫀 AI Health Risk Predictor

> **AI-Powered 5-Year Cardiovascular Disease Risk Prediction with Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)

---

## Overview

This project provides a complete, reproducible implementation of an AI health risk predictor for
5-year cardiovascular disease (CVD) events. It includes:

- 🧪 **Realistic synthetic EHR dataset generator** (Framingham-inspired, 5,000 patients)
- 🤖 **Three ML models**: Logistic Regression, XGBoost, Neural Network (MLP)
- 📊 **Full evaluation**: AUC-ROC, AUC-PR, Brier Score, calibration curves, fairness by sex/ethnicity
- 🧠 **SHAP explainability**: Waterfall charts, beeswarm plots, top-5 features per patient
- 🌐 **Streamlit web demo**: Clinician form + patient-friendly result page
- 📄 **PDF report export**: Clinical summary with ReportLab
- 📓 **Jupyter notebooks**: Step-by-step preprocessing and training walkthroughs
- 📋 **Documentation**: Project spec, data ethics checklist, final report

> ⚠️ **Research prototype only.** Not validated for clinical use. All data is synthetic.

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Install & Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-health-risk-predictor.git
cd ai-health-risk-predictor

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train all models (generates data + trains LR, XGBoost, NN)
python src/train.py

# 5. Launch the Streamlit demo
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
ai-health-risk-predictor/
├── 📁 data/
│   ├── synthetic_ehr.csv           # Generated dataset (5,000 patients)
│   └── splits/                     # Train / Val / Test CSVs
├── 📁 notebooks/
│   ├── 01_data_preprocessing.ipynb # Data cleaning & feature engineering walkthrough
│   └── 02_model_training.ipynb     # Training, evaluation & SHAP walkthrough
├── 📁 src/
│   ├── data_generator.py           # Synthetic EHR generator
│   ├── preprocessor.py             # Imputation, feature engineering, scaling
│   ├── models.py                   # LR, XGBoost, Neural Net definitions
│   ├── evaluator.py                # Metrics, calibration, fairness, plots
│   ├── explainer.py                # SHAP explanations (waterfall, beeswarm)
│   ├── pdf_report.py               # Clinical PDF generator (ReportLab)
│   ├── train.py                    # End-to-end training script
│   └── utils.py                    # Shared utilities
├── 📁 app/
│   └── streamlit_app.py            # Full web demo
├── 📁 models/                      # Saved model artefacts (.pkl)
├── 📁 reports/
│   ├── project_spec.md             # One-page project specification
│   ├── data_ethics_checklist.md    # Data ethics & compliance checklist
│   ├── final_report.md             # Full results report
│   ├── model_comparison.csv        # Model comparison table
│   ├── fairness_results.csv        # Subgroup fairness metrics
│   ├── shap_importance.csv         # SHAP feature importance
│   └── figures/                    # All generated plots
├── requirements.txt
└── README.md
```

---

## Training Scripts

```bash
# Generate only the synthetic dataset
python src/data_generator.py --n 5000 --out data/synthetic_ehr.csv

# Run only preprocessing (requires dataset)
python src/preprocessor.py

# Train specific models only (faster)
python src/train.py --models logistic_regression xgboost

# Customize training
python src/train.py \
  --n-patients 10000 \
  --seed 123 \
  --models xgboost neural_net \
  --figures-dir reports/figures \
  --models-dir models
```

---

## Running Notebooks

```bash
# Install Jupyter if needed
pip install jupyter

# Launch notebooks
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_model_training.ipynb
```

---

## Input Features

| Feature         | Type       | Unit       | Description                          |
|-----------------|------------|------------|--------------------------------------|
| age             | Numeric    | years      | Patient age (30–85)                  |
| sex             | Categorical| M/F        | Biological sex                       |
| ethnicity       | Categorical| —          | Self-reported ethnicity              |
| systolic_bp     | Numeric    | mmHg       | Systolic blood pressure              |
| diastolic_bp    | Numeric    | mmHg       | Diastolic blood pressure             |
| total_chol      | Numeric    | mg/dL      | Total cholesterol                    |
| hdl             | Numeric    | mg/dL      | HDL cholesterol (protective)         |
| ldl             | Numeric    | mg/dL      | LDL cholesterol                      |
| hba1c           | Numeric    | %          | Glycated hemoglobin                  |
| smoking_status  | Categorical| —          | Never / Former / Current             |
| diabetes_flag   | Binary     | 0/1        | Diagnosed diabetes                   |
| bmi             | Numeric    | kg/m²      | Body mass index                      |
| meds_list       | Text       | —          | Pipe-separated medication list       |

---

## Model Summary

| Model               | Calibration    | Explainability          | Speed  |
|---------------------|----------------|-------------------------|--------|
| Logistic Regression | Platt (5-fold) | Coefficients + SHAP     | Fast   |
| XGBoost             | Isotonic (3-fold)| TreeSHAP (exact)      | Medium |
| Neural Net (MLP)    | Isotonic (3-fold)| KernelSHAP (approx) | Slow   |

---

## Evaluation Metrics

| Metric       | Description                                      |
|--------------|--------------------------------------------------|
| AUC-ROC      | Area Under ROC Curve (discrimination)            |
| AUC-PR       | Area Under Precision-Recall Curve                |
| Brier Score  | Calibration metric (lower is better)             |
| Reliability  | Calibration curve (predicted vs observed)        |
| Fairness     | AUC-ROC by sex and ethnicity subgroups           |

---

## Streamlit App Features

- 📋 **Clinician View**: Full input form, risk gauge, SHAP waterfall chart, feature table, PDF export
- 👤 **Patient View**: Plain-language risk summary, visual risk meter, 3 personalized recommendations
- 🤖 **Model switcher**: Compare predictions across all three models
- 📄 **One-click PDF**: Clinical PDF report download (ReportLab)

---

## Cloud Deployment

### Streamlit Community Cloud (Free)
```
1. Push repo to GitHub
2. Go to share.streamlit.io
3. Connect your repo
4. Set main file: app/streamlit_app.py
```

### Docker
```bash
docker build -t cardiorisk .
docker run -p 8501:8501 cardiorisk
# Open http://localhost:8501
```

---

## Deliverables Checklist

- [x] One-page project spec (`reports/project_spec.md`)
- [x] Data ethics checklist (`reports/data_ethics_checklist.md`)
- [x] Preprocessing notebook (`notebooks/01_data_preprocessing.ipynb`)
- [x] Model training notebook (`notebooks/02_model_training.ipynb`)
- [x] Streamlit web demo (`app/streamlit_app.py`)
- [x] PDF report generator (`src/pdf_report.py`)
- [x] SHAP explanations (`src/explainer.py`)
- [x] Fairness evaluation (`src/evaluator.py`)
- [x] Final report (`reports/final_report.md`)
- [x] Synthetic dataset (`data/synthetic_ehr.csv`)
- [x] Complete code repository with README

---

## Privacy & Compliance Notes

- All data used in this prototype is **fully synthetic** — no real PHI
- For real patient data use:
  - De-identify per HIPAA Safe Harbor method
  - Obtain IRB approval and patient consent
  - Use HIPAA-compliant hosting (AWS GovCloud, Azure Health)
  - Execute BAA with cloud vendor
  - Implement audit logging for all data access

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Disclaimer

> This tool is a **research prototype** for educational and demonstration purposes only.
> It has NOT been clinically validated and must NOT be used for real patient care decisions.
> Always consult a qualified healthcare professional.

---

*Built with ❤️ using Python, scikit-learn, XGBoost, SHAP, and Streamlit.*
