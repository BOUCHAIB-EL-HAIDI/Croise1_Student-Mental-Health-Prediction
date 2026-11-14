import streamlit as st

st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("Student Mental Health Risk Prediction")

st.write("""
This application is designed to predict student mental health
risk levels using Machine Learning.

It also provides an overview of the trained model's performance.
""")

st.info("""
The model predicts one of three classes:

- High
- Medium
- Low
""")