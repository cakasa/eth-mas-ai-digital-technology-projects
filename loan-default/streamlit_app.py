import streamlit as st
import joblib
import os

path = './work/eth-mas-ai-digital-technology-projects/loan-default'

model = joblib.load(path + '/model/logistic_regression_model.pkl')
imputer_dict = joblib.load(path + '/model/imputer.pkl')
scaler = joblib.load(path + '/model/scaler.pkl')

loan_request_form = st.form('loan request')

loan_amount = loan_request_form.number_input('Loan Amount', min_value=100, max_value=100_000, value=1000, step=100)
annual_income = loan_request_form.number_input('Annual Income', min_value=0, value=12000, step=100)
if loan_request_form.form_submit_button('Request loan'):
    st.markdown("Hello Streamlit-er")

if st.button("Send balloons!"):
    st.balloons()
