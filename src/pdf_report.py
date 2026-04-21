"""
PDF Report Generator for AI Health Risk Predictor.
Creates a clinician-ready clinical summary using ReportLab.
"""

from __future__ import annotations
import io
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas

from src.utils import get_logger

logger = get_logger(__name__)

# ── Color constants ──────────────────────────────────────────────────────────
COLOR_PRIMARY   = colors.HexColor("#1a237e")  # Deep blue
COLOR_LOW       = colors.HexColor("#27ae60")
COLOR_MODERATE  = colors.HexColor("#f39c12")
COLOR_HIGH      = colors.HexColor("#e74c3c")
COLOR_LIGHT_BG  = colors.HexColor("#eef2f7")
COLOR_BORDER    = colors.HexColor("#bdc3c7")
COLOR_TEXT      = colors.HexColor("#2c3e50")


def _risk_color(category: str) -> Any:
    return {"Low": COLOR_LOW, "Moderate": COLOR_MODERATE, "High": COLOR_HIGH}.get(category, COLOR_TEXT)


def generate_pdf_report(
    patient_data: dict,
    risk_score: float,
    risk_category: str,
    confidence_interval: tuple[float, float],
    top_features: list[dict],
    model_name: str = "XGBoost",
    output_path: str | None = None,
) -> bytes:
    """
    Generate a full clinical PDF report.

    Args:
        patient_data: Dict of input features (displayed in demographics table).
        risk_score: Predicted 5-year CVD risk probability (0–1).
        risk_category: 'Low', 'Moderate', or 'High'.
        confidence_interval: (lower, upper) 95% CI.
        top_features: List of dicts from RiskExplainer.get_top_features().
        model_name: Name of model used for prediction.
        output_path: If provided, save to file. Otherwise return bytes.

    Returns:
        PDF bytes (also saved to output_path if given).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
        title="Cardiovascular Risk Assessment Report",
        author="AI Health Risk Predictor",
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    header_style = ParagraphStyle(
        "Header", parent=styles["Title"],
        fontSize=18, textColor=COLOR_PRIMARY,
        spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=10, textColor=colors.grey,
        spaceAfter=2, alignment=TA_CENTER
    )
    story.append(Paragraph("Cardiovascular Risk Assessment Report", header_style))
    story.append(Paragraph("AI-Assisted 5-Year Event Probability", sub_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}  |  Model: {model_name}",
        sub_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY, spaceAfter=12))

    # ── Risk Score Box ────────────────────────────────────────────────────────
    risk_pct = f"{risk_score * 100:.1f}%"
    ci_lo, ci_hi = confidence_interval
    ci_str = f"95% CI: {ci_lo*100:.1f}% – {ci_hi*100:.1f}%"

    box_color = _risk_color(risk_category)
    risk_data = [
        [
            Paragraph(f'<font color="white" size="28"><b>{risk_pct}</b></font>', ParagraphStyle("rc", alignment=TA_CENTER)),
            Paragraph(f'<font color="white" size="16"><b>{risk_category} Risk</b></font><br/>'
                      f'<font color="white" size="10">{ci_str}</font>',
                      ParagraphStyle("rl", alignment=TA_CENTER, leading=20)),
        ]
    ]
    risk_table = Table(risk_data, colWidths=["35%", "65%"])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [box_color]),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Patient Demographics ──────────────────────────────────────────────────
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, textColor=COLOR_PRIMARY,
        spaceBefore=12, spaceAfter=4, fontName="Helvetica-Bold"
    )
    story.append(Paragraph("Patient Information", section_style))

    demo_fields = [
        ("Age", patient_data.get("age", "—")),
        ("Sex", patient_data.get("sex", "—")),
        ("Ethnicity", patient_data.get("ethnicity", "—")),
        ("BMI", f"{patient_data.get('bmi', '—')} kg/m²"),
        ("Systolic BP", f"{patient_data.get('systolic_bp', '—')} mmHg"),
        ("Diastolic BP", f"{patient_data.get('diastolic_bp', '—')} mmHg"),
        ("Total Cholesterol", f"{patient_data.get('total_chol', '—')} mg/dL"),
        ("HDL", f"{patient_data.get('hdl', '—')} mg/dL"),
        ("LDL", f"{patient_data.get('ldl', '—')} mg/dL"),
        ("HbA1c", f"{patient_data.get('hba1c', '—')}%"),
        ("Smoking Status", patient_data.get("smoking_status", "—")),
        ("Diabetes", "Yes" if patient_data.get("diabetes_flag") else "No"),
        ("Current Medications", patient_data.get("meds_list", "None")),
    ]

    demo_data = [["Field", "Value"]] + [[k, str(v)] for k, v in demo_fields]
    demo_table = Table(demo_data, colWidths=["40%", "60%"])
    demo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(demo_table)

    # ── SHAP Feature Contributions ────────────────────────────────────────────
    story.append(Paragraph("Top Contributing Factors", section_style))
    story.append(Paragraph(
        "The following factors had the greatest influence on this patient's risk prediction "
        "(SHAP values indicate magnitude and direction of impact).",
        ParagraphStyle("note", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=6)
    ))

    shap_data = [["Rank", "Feature", "Value", "SHAP Impact", "Direction"]]
    for i, feat in enumerate(top_features[:5], 1):
        direction_color = "#e74c3c" if feat["direction"].startswith("↑") else "#27ae60"
        shap_data.append([
            str(i),
            feat["feature"].replace("_", " ").title(),
            f"{feat['feature_value']:.2f}",
            f"{feat['shap_value']:+.4f}",
            feat["direction"],
        ])

    shap_table = Table(shap_data, colWidths=["8%", "35%", "17%", "20%", "20%"])
    shap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLOR_LIGHT_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(shap_table)

    # ── Clinical Interpretation ───────────────────────────────────────────────
    story.append(Paragraph("Clinical Notes", section_style))

    risk_interpretations = {
        "Low": (
            "This patient is estimated to have a LOW 5-year cardiovascular event risk (<10%). "
            "Standard preventive care and lifestyle counseling are recommended. "
            "Annual reassessment advised."
        ),
        "Moderate": (
            "This patient is estimated to have a MODERATE 5-year cardiovascular event risk (10–20%). "
            "Consider initiating statin therapy per ACC/AHA guidelines. "
            "Lifestyle modification (diet, exercise, smoking cessation) is strongly recommended. "
            "Reassess in 6–12 months."
        ),
        "High": (
            "This patient is estimated to have a HIGH 5-year cardiovascular event risk (>20%). "
            "Immediate clinical review is recommended. Consider high-intensity statin therapy, "
            "blood pressure optimization, and referral to cardiology. "
            "Reassess in 3–6 months."
        ),
    }

    story.append(Paragraph(
        risk_interpretations.get(risk_category, "Please review patient data carefully."),
        ParagraphStyle("interp", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8)
    ))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER, spaceAfter=6))
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, leading=11, spaceBefore=4
    )
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI decision-support tool for research "
        "and clinical aid purposes only. It is NOT a substitute for professional clinical judgment. "
        "All risk estimates should be interpreted in the context of the full clinical picture. "
        "This tool has not been validated for clinical use and should not be used as the sole basis "
        "for clinical decisions. Data is de-identified. Comply with applicable privacy regulations.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"PDF report saved → {output_path}")

    return pdf_bytes
