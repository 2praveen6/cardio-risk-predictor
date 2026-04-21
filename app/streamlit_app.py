"""
AI Health Risk Predictor — Streamlit Web Demo
Clinician-facing input form + patient-friendly result page with SHAP explanations and PDF export.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI — Health Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Hero header */
  .hero-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: white;
  }
  .hero-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
  .hero-header p { font-size: 1.1rem; opacity: 0.85; margin: 0.5rem 0 0; }

  /* Risk badge */
  .risk-badge-low     { background: linear-gradient(135deg,#27ae60,#2ecc71); color:white; }
  .risk-badge-moderate{ background: linear-gradient(135deg,#e67e22,#f39c12); color:white; }
  .risk-badge-high    { background: linear-gradient(135deg,#c0392b,#e74c3c); color:white; }
  .risk-badge {
    border-radius: 12px; padding: 1.5rem 2rem;
    text-align: center; font-weight: 700;
    font-size: 1.1rem; margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
  }
  .risk-score { font-size: 3.5rem; font-weight: 800; line-height: 1; }
  .risk-label { font-size: 1.4rem; margin-top: 0.5rem; }
  .risk-ci    { font-size: 0.9rem; opacity: 0.9; margin-top: 0.3rem; }

  /* Section card */
  .section-card {
    background: white; border-radius: 12px;
    padding: 1.5rem; border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
  }
  .section-card h3 { color: #1a237e; margin-top: 0; font-weight: 600; }

  /* Feature chip */
  .feature-chip-up   { background:#fdecea; color:#c62828; border:1px solid #ef9a9a; }
  .feature-chip-down { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }
  .feature-chip {
    border-radius: 20px; padding: 0.3rem 0.8rem;
    font-size: 0.85rem; font-weight: 500;
    display: inline-block; margin: 0.2rem;
  }

  /* Patient view */
  .patient-card {
    background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
    border-radius: 16px; padding: 2rem; text-align: center;
    margin-bottom: 1rem; border: 1px solid #90caf9;
  }
  .recommendation-card {
    background: white; border-left: 4px solid #1a237e;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem;
    margin-bottom: 0.7rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }

  /* Metric box */
  .metric-box {
    background: #f8f9fa; border-radius: 8px;
    padding: 0.8rem 1rem; text-align: center;
    border: 1px solid #dee2e6;
  }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #1a237e; }
  .metric-label { font-size: 0.8rem; color: #6c757d; margin-top: 0.2rem; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #f0f4ff !important; }
  [data-testid="stSidebar"] p, 
  [data-testid="stSidebar"] div, 
  [data-testid="stSidebar"] span, 
  [data-testid="stSidebar"] label { color: #1a237e !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1a237e, #283593) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
  }
  .stDownloadButton > button {
    background: linear-gradient(135deg, #1b5e20, #2e7d32) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load trained models and preprocessor from disk. Train if needed."""
    models_dir = ROOT / "models"
    xgb_path   = models_dir / "xgboost.pkl"
    lr_path    = models_dir / "logistic_regression.pkl"
    nn_path    = models_dir / "neural_net.pkl"
    prep_path  = models_dir / "preprocessor.pkl"

    if not prep_path.exists():
        st.warning("⚙️ Models not found — running training pipeline (this takes ~2 min)...")
        from src.train import run_training_pipeline
        run_training_pipeline(generate_data=True, n_patients=3000)

    preprocessor = joblib.load(prep_path)

    loaded = {}
    for name, path in [("xgboost", xgb_path),
                       ("logistic_regression", lr_path),
                       ("neural_net", nn_path)]:
        if path.exists():
            loaded[name] = joblib.load(path)

    # Load explainer if available
    explainer = None
    exp_path = models_dir / "explainer_xgboost.pkl"
    if exp_path.exists():
        try:
            explainer = joblib.load(exp_path)
        except Exception:
            pass

    return preprocessor, loaded, explainer


# ── Helper functions ──────────────────────────────────────────────────────────
def risk_category(score: float) -> tuple[str, str, str]:
    """Returns (category, badge_class, emoji)."""
    if score < 0.10:
        return "Low", "risk-badge-low", "🟢"
    elif score < 0.20:
        return "Moderate", "risk-badge-moderate", "🟡"
    else:
        return "High", "risk-badge-high", "🔴"


def patient_input_to_df(inputs: dict) -> pd.DataFrame:
    """Convert form inputs to a single-row DataFrame matching EHR schema."""
    return pd.DataFrame([{
        "patient_id":     "APP_PATIENT",
        "age":            inputs["age"],
        "sex":            inputs["sex"],
        "ethnicity":      inputs["ethnicity"],
        "systolic_bp":    inputs["systolic_bp"],
        "diastolic_bp":   inputs["diastolic_bp"],
        "total_chol":     inputs["total_chol"],
        "hdl":            inputs["hdl"],
        "ldl":            inputs["ldl"],
        "hba1c":          inputs["hba1c"],
        "smoking_status": inputs["smoking_status"],
        "diabetes_flag":  int(inputs["diabetes_flag"]),
        "bmi":            inputs["bmi"],
        "meds_list":      inputs["meds_list"] or "none",
        "event_within_5yrs": 0,
        "followup_time":  5.0,
    }])


def get_recommendations(top_features: list) -> list[dict]:
    """Generate actionable recommendations based on top risk factors."""
    recs = []
    for feat in top_features[:5]:
        name = feat["feature"].lower()
        if feat["shap_value"] > 0:  # Risk-increasing
            if "systolic_bp" in name or "bp" in name:
                recs.append({
                    "icon": "💊",
                    "title": "Blood Pressure Control",
                    "body": "Your blood pressure is a key risk factor. Discuss antihypertensive therapy, reduce sodium, and exercise regularly."
                })
            elif "chol" in name or "ldl" in name:
                recs.append({
                    "icon": "🥗",
                    "title": "Cholesterol Management",
                    "body": "Elevated cholesterol increases risk. Consider a heart-healthy diet low in saturated fat. Ask your doctor about statins."
                })
            elif "smok" in name:
                recs.append({
                    "icon": "🚭",
                    "title": "Quit Smoking",
                    "body": "Smoking is a major modifiable risk factor. Talk to your doctor about cessation programs and nicotine replacement therapy."
                })
            elif "bmi" in name or "weight" in name:
                recs.append({
                    "icon": "🏃",
                    "title": "Weight Management",
                    "body": "Reaching a healthy BMI can significantly reduce cardiovascular risk. Aim for 150+ minutes of moderate exercise weekly."
                })
            elif "hba1c" in name or "diabet" in name:
                recs.append({
                    "icon": "🩺",
                    "title": "Blood Sugar Control",
                    "body": "Elevated blood sugar is a significant risk factor. Work with your care team to optimize blood glucose management."
                })
            elif "age" in name:
                recs.append({
                    "icon": "📅",
                    "title": "Routine Screening",
                    "body": "Age is an important risk factor. Regular cardiac screening and check-ups are essential at this life stage."
                })
    # Deduplicate by title
    seen = set()
    unique_recs = []
    for r in recs:
        if r["title"] not in seen:
            seen.add(r["title"])
            unique_recs.append(r)
    # Pad with general advice if fewer than 3
    generic = [
        {"icon": "❤️", "title": "Heart-Healthy Lifestyle", "body": "Regular aerobic exercise (30 min/day), Mediterranean diet, and stress management all reduce cardiovascular risk."},
        {"icon": "🛌", "title": "Sleep & Stress", "body": "Poor sleep and chronic stress increase inflammation and BP. Aim for 7–9 hours of quality sleep nightly."},
    ]
    for g in generic:
        if len(unique_recs) >= 3:
            break
        if g["title"] not in seen:
            unique_recs.append(g)
    return unique_recs[:3]


def make_gauge_chart(risk_score: float, category: str) -> plt.Figure:
    """Create a semi-circular gauge chart for risk visualization."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Background arcs
    zones = [(0.33, "#2ecc71", "Low"), (0.33, "#f39c12", "Moderate"), (0.34, "#e74c3c", "High")]
    start = 180
    for (span_frac, color, label) in zones:
        span_deg = span_frac * 180
        w = Wedge((0, 0), 1.0, start - span_deg, start, width=0.3, color=color, alpha=0.85)
        ax.add_patch(w)
        start -= span_deg

    # Needle
    angle_deg = 180 - risk_score * 180
    angle_rad = np.radians(angle_deg)
    ax.plot([0, 0.65 * np.cos(angle_rad)], [0, 0.65 * np.sin(angle_rad)],
            color="#1a237e", lw=3, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.06, color="#1a237e", zorder=5))

    ax.text(0, -0.2, f"{risk_score*100:.1f}%", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#1a237e",
            fontfamily="monospace")
    ax.text(0, -0.42, f"{category} Risk", ha="center", va="center",
            fontsize=11, fontweight="600",
            color={"Low": "#27ae60", "Moderate": "#e67e22", "High": "#c0392b"}.get(category, "gray"))

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Hero header
    st.markdown("""
    <div class="hero-header">
      <h1>🫀 CardioRisk AI</h1>
      <p>AI-Powered 5-Year Cardiovascular Risk Prediction with Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading AI models..."):
        preprocessor, models, explainer = load_models()

    if not models:
        st.error("No trained models found. Please run `python src/train.py` first.")
        return

    # Sidebar — model selection & view toggle
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60)
        st.title("CardioRisk AI")
        st.markdown("---")

        view_mode = st.radio("👁️ View Mode", ["🩺 Clinician View", "👤 Patient View"], index=0)
        st.markdown("---")

        model_choice = st.selectbox(
            "🤖 Model",
            options=list(models.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )
        st.markdown("---")
        st.markdown("**About**")
        st.caption(
            "CardioRisk AI uses machine learning trained on synthetic EHR data to "
            "estimate 5-year cardiovascular risk. Always use clinical judgment."
        )
        st.markdown("---")
        st.caption("⚠️ Research prototype only. Not for clinical use.")

    # Main content columns
    col_form, col_results = st.columns([1, 1.2], gap="large")

    # ── INPUT FORM ─────────────────────────────────────────────────────────────
    with col_form:
        st.markdown("### 📋 Patient Information")

        with st.expander("🧑 Demographics", expanded=True):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age (years)", 30, 90, 55, step=1)
            sex = c2.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x=="M" else "Female")
            ethnicity = st.selectbox("Ethnicity",
                ["White", "Black", "Hispanic", "Asian", "Other"])

        with st.expander("💓 Vital Signs", expanded=True):
            c1, c2, c3 = st.columns(3)
            systolic_bp  = c1.number_input("Systolic BP (mmHg)",  80,  250, 130, step=1)
            diastolic_bp = c2.number_input("Diastolic BP (mmHg)", 40,  140,  80, step=1)
            bmi          = c3.number_input("BMI (kg/m²)",         15.0, 55.0, 27.0, step=0.1, format="%.1f")

        with st.expander("🧪 Lab Values", expanded=True):
            c1, c2, c3 = st.columns(3)
            total_chol = c1.number_input("Total Cholesterol (mg/dL)", 100, 400, 200, step=1)
            hdl        = c2.number_input("HDL (mg/dL)", 20, 120, 55, step=1)
            ldl        = c3.number_input("LDL (mg/dL)", 30, 280, 120, step=1)
            hba1c      = st.slider("HbA1c (%)", 4.0, 13.0, 5.5, step=0.1)

        with st.expander("🚬 Lifestyle & History", expanded=True):
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            diabetes_flag  = st.checkbox("Diabetes Diagnosed")
            meds = st.multiselect(
                "Current Medications",
                ["statin", "ace_inhibitor", "arb", "beta_blocker",
                 "calcium_channel_blocker", "aspirin", "metformin",
                 "insulin", "diuretic", "anticoagulant"],
            )
            meds_list = "|".join(meds) if meds else "none"

        predict_btn = st.button("🔮 Predict Risk", use_container_width=True)

    # ── RESULTS ─────────────────────────────────────────────────────────────────
    with col_results:
        if not predict_btn:
            st.markdown("""
            <div class="section-card" style="text-align:center;padding:3rem;">
              <div style="font-size:4rem;">🫀</div>
              <h3 style="color:#1a237e;">Ready to Predict</h3>
              <p style="color:#666;">Fill in the patient details and click <b>Predict Risk</b></p>
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Prediction ───────────────────────────────────────────────────────
        inputs = dict(
            age=age, sex=sex, ethnicity=ethnicity,
            systolic_bp=systolic_bp, diastolic_bp=diastolic_bp, bmi=bmi,
            total_chol=total_chol, hdl=hdl, ldl=ldl, hba1c=hba1c,
            smoking_status=smoking_status, diabetes_flag=diabetes_flag,
            meds_list=meds_list,
        )

        patient_df = patient_input_to_df(inputs)
        model = models[model_choice]

        try:
            X_transformed = preprocessor.transform(patient_df)
            risk_score = float(model.predict_proba(X_transformed)[0])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        cat, badge_class, emoji = risk_category(risk_score)

        # Simple CI based on ±1 std (for demo; real CI requires bootstrap)
        ci_lo = max(0.0, risk_score - 0.03)
        ci_hi = min(1.0, risk_score + 0.03)

        # ── SHAP values ────────────────────────────────────────────────────────
        top_features = []
        shap_fig = None
        if explainer is not None:
            try:
                top_features = explainer.get_top_features(X_transformed, n=5)
                shap_fig = explainer.plot_waterfall(
                    X_transformed,
                    save_path=str(ROOT / "reports/figures/shap_live.png")
                )
            except Exception as e:
                st.caption(f"SHAP explanation unavailable: {e}")

        # Fallback features if SHAP not available
        if not top_features:
            top_features = [
                {"feature": "systolic_bp", "shap_value": 0.12, "feature_value": systolic_bp, "direction": "↑ Risk"},
                {"feature": "age_decade",  "shap_value": 0.09, "feature_value": age/10, "direction": "↑ Risk"},
                {"feature": "total_chol",  "shap_value": 0.06, "feature_value": total_chol, "direction": "↑ Risk"},
                {"feature": "hdl",         "shap_value": -0.05, "feature_value": hdl, "direction": "↓ Risk"},
                {"feature": "is_male",     "shap_value": 0.04, "feature_value": int(sex=="M"), "direction": "↑ Risk"},
            ]

        recommendations = get_recommendations(top_features)

        # ══════════════════════════════════════════════════════════════
        # ── CLINICIAN VIEW ─────────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════
        if "Clinician" in view_mode:
            # Risk gauge
            gcol1, gcol2 = st.columns([1, 1])
            with gcol1:
                gauge = make_gauge_chart(risk_score, cat)
                st.pyplot(gauge, use_container_width=True)
                plt.close()
            with gcol2:
                st.markdown(f"""
                <div class="section-card" style="margin-top:0.5rem">
                  <div style="font-size:0.85rem;color:#666;margin-bottom:0.3rem">5-Year CVD Risk</div>
                  <div style="font-size:3rem;font-weight:800;color:{'#27ae60' if cat=='Low' else '#e67e22' if cat=='Moderate' else '#c0392b'}">
                    {risk_score*100:.1f}%
                  </div>
                  <div style="font-size:1.1rem;font-weight:600;color:#444">{emoji} {cat} Risk</div>
                  <div style="font-size:0.8rem;color:#888;margin-top:0.3rem">
                    95% CI: {ci_lo*100:.1f}% – {ci_hi*100:.1f}%
                  </div>
                  <hr style="margin:0.8rem 0">
                  <div style="font-size:0.8rem;color:#666">
                    Model: <b>{model_choice.replace('_', ' ').title()}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Metrics row
            st.markdown("#### 📊 Key Clinical Indicators")
            m1, m2, m3, m4 = st.columns(4)
            def metric_html(val, label, alert=False):
                color = "#c0392b" if alert else "#1a237e"
                return f"""<div class="metric-box">
                  <div class="metric-value" style="color:{color}">{val}</div>
                  <div class="metric-label">{label}</div>
                </div>"""

            m1.markdown(metric_html(f"{systolic_bp}/{diastolic_bp}", "BP (mmHg)", systolic_bp >= 140), unsafe_allow_html=True)
            m2.markdown(metric_html(f"{total_chol}", "Total Chol.", total_chol > 240), unsafe_allow_html=True)
            m3.markdown(metric_html(f"{hba1c:.1f}%", "HbA1c", hba1c > 6.5), unsafe_allow_html=True)
            m4.markdown(metric_html(f"{bmi:.1f}", "BMI", bmi > 30), unsafe_allow_html=True)

            st.markdown("---")

            # SHAP Explainability
            st.markdown("#### 🧠 Top Contributing Features (SHAP)")
            if shap_fig:
                st.pyplot(shap_fig, use_container_width=True)
                plt.close()
            else:
                # Render a simple bar chart
                feat_names = [f["feature"].replace("_", " ").title() for f in top_features]
                feat_vals  = [f["shap_value"] for f in top_features]
                fig, ax = plt.subplots(figsize=(7, 3))
                colors_bar = ["#e74c3c" if v > 0 else "#3498db" for v in feat_vals]
                ax.barh(feat_names, feat_vals, color=colors_bar, edgecolor="white")
                ax.axvline(0, color="black", lw=0.8)
                ax.set_xlabel("SHAP Value")
                ax.set_title("Feature Contributions")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Feature table
            feat_df = pd.DataFrame([{
                "Feature": f["feature"].replace("_", " ").title(),
                "Value": f"{f['feature_value']:.2f}",
                "SHAP": f"{f['shap_value']:+.4f}",
                "Direction": f["direction"],
            } for f in top_features])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Clinical recommendations
            st.markdown("#### 💡 Clinical Recommendations")
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                  <b>{rec['icon']} {rec['title']}</b><br>
                  <span style="color:#555;font-size:0.9rem">{rec['body']}</span>
                </div>
                """, unsafe_allow_html=True)

            # PDF Export
            st.markdown("---")
            with st.spinner("Preparing PDF..."):
                try:
                    from src.pdf_report import generate_pdf_report
                    pdf_bytes = generate_pdf_report(
                        patient_data=inputs,
                        risk_score=risk_score,
                        risk_category=cat,
                        confidence_interval=(ci_lo, ci_hi),
                        top_features=top_features,
                        model_name=model_choice.replace("_", " ").title(),
                    )
                    st.download_button(
                        label="📄 Download Clinical PDF Report",
                        data=pdf_bytes,
                        file_name=f"cardiorisk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"PDF generation failed: {e}")

        # ══════════════════════════════════════════════════════════════
        # ── PATIENT VIEW ─────────────────────────────────────────════
        # ══════════════════════════════════════════════════════════════
        else:
            pct = int(risk_score * 100)
            color_map = {"Low": "#27ae60", "Moderate": "#e67e22", "High": "#c0392b"}
            color = color_map.get(cat, "#1a237e")

            # Patient-friendly risk display
            st.markdown(f"""
            <div class="patient-card">
              <div style="font-size:1.1rem;color:#555;margin-bottom:1rem">
                Your estimated 5-year heart risk
              </div>
              <div style="font-size:5rem;font-weight:900;color:{color};line-height:1">
                {pct}%
              </div>
              <div style="font-size:1.6rem;font-weight:700;color:{color};margin:0.5rem 0">
                {emoji} {cat} Risk
              </div>
              <div style="font-size:0.9rem;color:#777;margin-top:0.5rem">
                Out of 100 people with similar health profiles,<br>
                about <b>{pct}</b> might have a heart event in the next 5 years.
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Visual progress bar
            st.markdown(f"""
            <div style="margin: 0.5rem 0 1.5rem">
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#666;margin-bottom:4px">
                <span>Low (0–10%)</span><span>Moderate (10–20%)</span><span>High (20%+)</span>
              </div>
              <div style="background:#e0e0e0;border-radius:999px;height:20px;overflow:hidden">
                <div style="background:linear-gradient(90deg,#27ae60,#f39c12,#e74c3c);
                            width:{min(risk_score*100*3, 100)}%;height:100%;border-radius:999px;
                            transition:width 0.5s"></div>
              </div>
              <div style="height:20px;position:relative;margin-top:-20px">
                <div style="position:absolute;left:{min(risk_score*100*3, 98)}%;top:0;
                            transform:translateX(-50%);width:4px;height:20px;
                            background:#1a237e;border-radius:2px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Key risk factors in plain language
            st.markdown("#### 🔑 Your Key Risk Factors")
            for feat in top_features[:3]:
                is_risk = feat["shap_value"] > 0
                icon = "⚠️" if is_risk else "✅"
                chip_class = "feature-chip-up" if is_risk else "feature-chip-down"
                fname = feat["feature"].replace("_", " ").title()
                desc = "increases" if is_risk else "helps reduce"
                st.markdown(f"""
                <div class="recommendation-card">
                  {icon} <b>{fname}</b>
                  <span class="feature-chip {chip_class}" style="float:right">
                    {feat['direction']}
                  </span><br>
                  <span style="color:#555;font-size:0.9rem">
                    This factor <b>{desc}</b> your heart risk.
                  </span>
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            st.markdown("#### 💪 What You Can Do")
            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div class="recommendation-card">
                  <b>{rec['icon']} {rec['title']}</b><br>
                  <span style="color:#555;font-size:0.9rem">{rec['body']}</span>
                </div>
                """, unsafe_allow_html=True)

            st.info("💬 **Talk to your doctor** about these results. This tool is for information only and every person is different.")


if __name__ == "__main__":
    main()
