
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import joblib

from src.data.data_loader import load_dataset


st.title("Student Risk Prediction")

# Load model and dataset
model_path = PROJECT_ROOT / "models" / "best_model_pipeline.joblib"

model = joblib.load(model_path)
df = load_dataset()

# Get expected model features
features = list(model.feature_names_in_)
X = df[features]

st.write("Enter the student's information below.")

input_data = {}

with st.form("prediction_form"):

    for column in features:

        if pd.api.types.is_numeric_dtype(X[column]):

            min_value = float(X[column].min())
            max_value = float(X[column].max())
            default_value = float(X[column].median())

            if pd.api.types.is_integer_dtype(X[column]):
                input_data[column] = st.number_input(
                    column,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(default_value),
                    step=1
                )
            else:
                input_data[column] = st.number_input(
                    column,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value
                )

        else:

            options = X[column].dropna().unique().tolist()

            input_data[column] = st.selectbox(
                column,
                options=options
            )

    submitted = st.form_submit_button("Predict")

if submitted:

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    st.success(f"Predicted Risk Level: {prediction}")

    st.subheader("Class Probabilities")

    for class_name, probability in zip(model.classes_, probabilities):
        st.write(f"{class_name}: {probability:.2%}")
        st.progress(float(probability))
