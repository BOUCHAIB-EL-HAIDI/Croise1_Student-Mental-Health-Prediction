
import streamlit as st
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

st.title("Model Performance")

# Load evaluation reports
metrics = pd.read_csv(REPORTS_DIR / "evaluation_metrics.csv")
report = pd.read_csv(REPORTS_DIR / "classification_report.csv")

# Display evaluation metrics
st.subheader("Evaluation Metrics")

metric_values = metrics.iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{metric_values['Accuracy']:.3f}")
col2.metric("Precision", f"{metric_values['Precision']:.3f}")
col3.metric("Recall", f"{metric_values['Recall']:.3f}")
col4.metric("F1-Score", f"{metric_values['F1-Score']:.3f}")
col5.metric("ROC-AUC", f"{metric_values['ROC-AUC']:.3f}")

# Display classification report
st.subheader("Classification Report")

st.dataframe(
    report,
    use_container_width=True
)

# Display confusion matrix
st.subheader("Confusion Matrix")

confusion_matrix_path = REPORTS_DIR / "confusion_matrix.png"

st.image(
    str(confusion_matrix_path),
    use_container_width=True
)
