import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Loan Approval Prediction System",
    layout="wide"
)
st.title("🏦 Loan Approval Prediction System")
st.markdown(
    """
    This application predicts **loan approval probability** using a machine learning model.
    Adjust inputs to perform **scenario analysis** and understand predictions using **SHAP explanations**.
    """
)
@st.cache_resource
def load_artifacts():
    best_model = joblib.load("best_model.joblib")
    num_imputer = joblib.load("num_imputer.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    num_cols = joblib.load("num_cols.joblib")
    cat_cols = joblib.load("cat_cols.joblib")
    return best_model, num_imputer, feature_cols, num_cols, cat_cols
best_model, num_imputer, feature_cols, num_cols, cat_cols = load_artifacts()

st.sidebar.header("📌 Applicant Details")

Gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"],
    help="Applicant gender (used for demographic risk patterns)"
)

Marital_Status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married"],
    help="Marital status can influence financial stability"
)

Number_of_Dependents = st.sidebar.slider(
    "Number of Dependents",
    0, 4, 1,
    help="More dependents may increase financial burden"
)

Education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"],
    help="Higher education may correlate with income stability"
)

Annual_Income = st.sidebar.slider(
    "Annual Income",
    1500.0, 12000.0, 1000.0,
    step=500.0,
    help="Total annual income of the applicant"
)

Credit_Score = st.sidebar.slider(
    "Credit Score",
    0.0, 5000.0, 50.0,
    help="Higher credit score indicates better creditworthiness"
)

Loan_Amount = st.sidebar.slider(
    "Loan Amount",
    0.0, 10000.0, 3000.0,
    step=500.0,
    help="Requested loan amount"
)

Term = st.sidebar.selectbox(
    "Loan Term",
    [180,360],
    help="Loan repayment duration (months)"
)

Property_Area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"],
    help="Location of property associated with loan risk"
)
input_df = pd.DataFrame({
    "Gender": [Gender],
    "Marital_Status": [Marital_Status],
    "Number_of_Dependents": [Number_of_Dependents],
    "Education": [Education],
    "Annual_Income": [Annual_Income],
    "Credit_Score": [Credit_Score],
    "Loan_Amount": [Loan_Amount],
    "Term": [Term],
    "Property_Area": [Property_Area]
})
st.write("Imputer expects:", len(num_cols))
st.write("Input numeric cols:", input_df[num_cols].shape[1])
st.write("Numeric columns:", num_cols)

input_df[num_cols] = num_imputer.transform(input_df[num_cols])

input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

input_df = input_df.reindex(columns=feature_cols, fill_value=0)

proba = best_model.predict_proba(input_df)[0][1]

st.subheader("📊 Prediction Result")

st.metric(
    label="Loan Approval Probability",
    value=f"{proba:.2%}"
)

if proba >= 0.6:
    st.success("✅ High likelihood of loan approval")
elif proba >= 0.45:
    st.warning("⚠️ Moderate approval chance")
else:
    st.error("❌ Low likelihood of approval")

st.subheader("🔍 Model Explainability (SHAP)")

@st.cache_resource
def shap_explainer():
    return shap.TreeExplainer(best_model)

explainer = shap_explainer()
shap_values = explainer.shap_values(input_df)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔹 Individual Prediction Explanation")
    fig, ax = plt.subplots()
    if isinstance(shap_values, list) and len(shap_values) > 1:
    class_idx = 1
    else: 
    class_idx = 0
    shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[class_idx][0]
        if isinstance(shap_values, list) else shap_values[0],
        base_values=explainer.expected_value[class_idx]
        if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    ),
    show=False
)
st.pyplot(fig)

with col2:
    st.markdown("### 🔹 Global Feature Importance")
    X_sample = pd.DataFrame(
        np.repeat(input_df.values, 50, axis=0),
        columns=input_df.columns
    )
    shap_vals = explainer.shap_values(X_sample)
    fig2, ax2 = plt.subplots()
    shap.summary_plot(
        shap_vals[1],
        X_sample,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig2)

st.markdown("---")
st.markdown(
    "Educational Note: This tool demonstrates how machine learning predictions can be interpreted using SHAP values,enabling transparency and trust in automated loan decision systems."
)
