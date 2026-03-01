import pickle
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    with open("full_pipeline", "rb") as f:
        return pickle.load(f)


st.title("Loan Approval: Quick Demo")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "error" not in st.session_state:
    st.session_state.error = None


with st.form("loan_form"):
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    applicant_income = st.text_input("Applicant Income")
    loan_amount = st.text_input("Loan Amount")
    credit_history = st.selectbox("Credit History", ["1", "0"])
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame(
        [[married, education, applicant_income, loan_amount, credit_history]],
        columns=[
            "Married",
            "Education",
            "ApplicantIncome",
            "LoanAmount",
            "Credit_History",
        ],
    )

    df[["ApplicantIncome", "LoanAmount", "Credit_History"]] = (
        df[["ApplicantIncome", "LoanAmount", "Credit_History"]]
        .apply(pd.to_numeric, errors="coerce")
    )

    if df.isna().any().any():
        st.session_state.error = "Invalid numeric input"
        st.session_state.prediction = None
    else:
        model = load_model()
        pred = float(model.predict(df)[0])
        pred = max(0.0, min(1.0, pred))
        st.session_state.prediction = pred
        st.session_state.error = None


if st.session_state.error:
    st.error(st.session_state.error)

if st.session_state.prediction is not None:
    st.metric(
        label="Approval Score",
        value=f"{st.session_state.prediction:.2f}",
    )
