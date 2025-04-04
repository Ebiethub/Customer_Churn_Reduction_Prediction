import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('xgboost_churn_model.pkl')

# Define the app title
st.title("Customer Churn Prediction")

# Input fields for user
st.header("Enter Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=120)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Mailed Check", "Electronic Check"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0)

# Map categorical inputs to numeric values
gender_map = {"Male": 0, "Female": 1}
senior_citizen_map = {"No": 0, "Yes": 1}
partner_map = {"No": 0, "Yes": 1}
dependents_map = {"No": 0, "Yes": 1}
phone_service_map = {"No": 0, "Yes": 1}
multiple_lines_map = {"No": 0, "Yes": 1, "No phone service": 2}
internet_service_map = {"DSL": 0, "Fiber Optic": 1, "No": 2}
online_security_map = {"No": 0, "Yes": 1, "No internet service": 2}
online_backup_map = {"No": 0, "Yes": 1, "No internet service": 2}
device_protection_map = {"No": 0, "Yes": 1, "No internet service": 2}
tech_support_map = {"No": 0, "Yes": 1, "No internet service": 2}
streaming_tv_map = {"No": 0, "Yes": 1, "No internet service": 2}
streaming_movies_map = {"No": 0, "Yes": 1, "No internet service": 2}
contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
paperless_billing_map = {"No": 0, "Yes": 1}
payment_method_map = {"Credit Card": 0, "Bank Transfer": 1, "Mailed Check": 2, "Electronic Check": 3}

# Prepare input data with all features
input_data = pd.DataFrame({
    'gender': [gender_map[gender]],
    'SeniorCitizen': [senior_citizen_map[senior_citizen]],
    'Partner': [partner_map[partner]],
    'Dependents': [dependents_map[dependents]],
    'tenure': [tenure],
    'PhoneService': [phone_service_map[phone_service]],
    'MultipleLines': [multiple_lines_map[multiple_lines]],
    'InternetService': [internet_service_map[internet_service]],
    'OnlineSecurity': [online_security_map[online_security]],
    'OnlineBackup': [online_backup_map[online_backup]],
    'DeviceProtection': [device_protection_map[device_protection]],
    'TechSupport': [tech_support_map[tech_support]],
    'StreamingTV': [streaming_tv_map[streaming_tv]],
    'StreamingMovies': [streaming_movies_map[streaming_movies]],
    'Contract': [contract_map[contract]],
    'PaperlessBilling': [paperless_billing_map[paperless_billing]],
    'PaymentMethod': [payment_method_map[payment_method]],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Predict churn
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"The customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"The customer is unlikely to churn (Probability: {1 - probability:.2f})")