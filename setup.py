from setuptools import setup, find_packages

setup(
    name="ai-health-risk-predictor",
    version="1.0.0",
    description="AI-powered 5-year cardiovascular risk predictor with SHAP explainability",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.44.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit>=1.30.0",
        "reportlab>=4.0.0",
        "joblib>=1.3.0",
        "scipy>=1.11.0",
        "plotly>=5.18.0",
        "Pillow>=10.0.0",
    ],
)
