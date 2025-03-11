import streamlit as st
import joblib
import os

path = './work/eth-mas-ai-digital-technology-projects/loan-default'

model = joblib.load(path + '/model/logistic_regression_model.pkl')
imputer_dict = joblib.load(path + '/model/imputer.pkl')
scaler = joblib.load(path + '/model/scaler.pkl')
def request_loan():
    return 0.8

loan_request_form = st.form('loan request')

loan_amount = loan_request_form.number_input('Loan Amount', min_value=100, max_value=100_000, value=1000, step=100)
annual_income = loan_request_form.number_input('Annual Income', min_value=0, value=12000, step=100)

duration = loan_request_form.radio(
    'Loan Duration',
    [36, 60],
    format_func=lambda duration: f'{duration} months'
)

installment = round(loan_amount / duration, 2)
loan_request_form.markdown(f'Per month: {installment}')

if loan_request_form.form_submit_button('Request loan'):
    probability_granted = request_loan()
    confidence = 0
    
    if probability_granted >= 0.5:
        st.markdown(':green[Loan granted]')
        confidence = probability_granted
    else:
        st.markdown(':red[Loan rejected]')
        confidence = 1 - probability_granted

    st.markdown(f'Confidence: {round(confidence * 100)}%')

if st.button("Send balloons!"):
    st.balloons()
