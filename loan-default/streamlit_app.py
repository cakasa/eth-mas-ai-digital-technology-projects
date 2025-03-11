import streamlit as st
import pandas as pd
import joblib
import os

path = './work/eth-mas-ai-digital-technology-projects/loan-default'

model = joblib.load(path + '/model/logistic_regression_model.pkl')
imputer_dict = joblib.load(path + '/model/imputer.pkl')
scaler = joblib.load(path + '/model/scaler.pkl')
features_list = joblib.load(path + '/model/features.pkl')

def impute(df):
    num_cols = imputer_dict['num_cols']
    cat_cols = imputer_dict['cat_cols']
    cat_marker = imputer_dict['cat_marker']
    num_impute_values = imputer_dict['num_impute_values']
    
    new_df = df.copy()
    new_df[cat_cols] = new_df[cat_cols].fillna(cat_marker)
    for col in num_cols:
        new_df[col] = new_df[col].fillna(num_impute_values[col])
    
    return new_df

def scale(df):
    return pd.DataFrame(scaler.transform(df), 
                        columns=df.columns,
                        index=df.index)

def request_loan(datapoint):
    df = pd.DataFrame(columns=features_list)
    df = pd.concat([df, pd.DataFrame([datapoint])], ignore_index=True)
    st.write(df)
    
    df = impute(df)
    df = scale(df)
    prob_of_grant = model.predict_proba(df)[0][1]
    st.write(prob_of_grant)
    
    return prob_of_grant
    
loan_request_form = st.form('loan request')

loan_amount = loan_request_form.number_input('Loan Amount', min_value=100, max_value=100_000, value=1000, step=100)
annual_income = loan_request_form.number_input('Annual Income', min_value=0, value=12000, step=100)
employment_duration = loan_request_form.number_input('Number of years at current employment position', min_value=0, max_value=10, step=1)
home_type = loan_request_form.radio(
    'Home Ownership',
    ['Rent', 'Own', 'Mortgage', 'Other', 'None']
)

loan_duration = loan_request_form.radio(
    'Loan Duration',
    [36, 60],
    format_func=lambda duration: f'{duration} months'
)

installment = round(loan_amount / loan_duration, 2)
st.write(f'Per month: {installment}')

if loan_request_form.form_submit_button('Request loan'):
    datapoint = {
        'annual_inc': annual_income,
        'emp_length': employment_duration,
        'loan_amnt': loan_amount,
        'installment': round(loan_amount / loan_duration, 2),
        'home_ownership_MORTGAGE': home_type == 'Mortgage',
        'home_ownership_NONE': home_type == 'None',
        'home_ownership_OWN': home_type == 'Own',
        'home_ownership_RENT': home_type == 'Rent',
        'home_ownership_OTHER': home_type == 'Other'
    }
    
    probability_granted = request_loan(datapoint)
    confidence = 0
    
    if probability_granted >= 0.5:
        st.markdown(':green[Loan granted]')
        confidence = probability_granted
    else:
        st.markdown(':red[Loan rejected]')
        confidence = 1 - probability_granted

    st.markdown(f'Confidence: {round(confidence * 100)}%')