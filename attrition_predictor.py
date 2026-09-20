import streamlit as st
import numpy as np
import pickle

#loading model and scaler
try:
    model = pickle.load(open("attrition_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("❌Model files not found! Please run train_model.py first.")
    st.stop()

#Web UI using streamlit
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("👔Employee Attrition Predictor")
st.markdown("Use this tool to predict whether an employee is likely to **Stay** or **Leave** the company based on key details.")

st.divider()

#input section
Age = st.slider("Age", 18, 60, 30)
MonthlyIncome = st.number_input("Monthly Income (₹)", 1000, 20000, 5000)
JobSatisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
WorkLifeBalance = st.selectbox("Work-Life Balance (1=Poor, 4=Excellent)", [1, 2, 3, 4])
YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
OverTime = st.selectbox("OverTime", ["No", "Yes"])

OverTime = 1 if OverTime == "Yes" else 0

#converting input data
input_data = np.array([[Age, MonthlyIncome, JobSatisfaction, WorkLifeBalance, YearsAtCompany, OverTime]])

#Prediction
if st.button("🔍 Predict Attrition"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100  # chance of leaving

    if prediction == 1:
        st.error(f"❌Employee is \"Likely to leave!!\" ({probability:.1f}% chance).")
    else:
        st.success(f"✅Employee is \"Likely to stay!!\" ({100 - probability:.1f}% chance).")
