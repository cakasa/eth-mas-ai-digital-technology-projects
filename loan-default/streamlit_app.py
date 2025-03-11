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
    
    return prob_of_grant
    
loan_request_form = st.form('loan request')

loan_amount = loan_request_form.number_input('Loan Amount', min_value=100, max_value=100_000, value=1000, step=100)
annual_income = loan_request_form.number_input('Annual Income', min_value=0, value=12000, step=100)
employment_duration = loan_request_form.number_input('Number of years at current employment position', min_value=0, max_value=10, step=1)

home_types = ['Rent', 'Own', 'Mortgage', 'Other', 'None']
home_type = loan_request_form.radio(
    'Home Ownership',
    home_types
)

loan_purposes = ['credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'house', 'major_purchase', 'medical', 'moving', 'renewable_energy', 'small_business', 'vacation', 'wedding', 'other']
purpose = loan_request_form.pills(
    'Purpose',
    loan_purposes,
    format_func=lambda option: ' '.join([word.capitalize() for word in option.split('_')])
)

loan_duration = loan_request_form.radio(
    'Loan Duration',
    [36, 60],
    format_func=lambda duration: f'{duration} months'
)

education_options = ['College degree', 'GED/High school', 'None']
education = loan_request_form.radio(
    'Completed Education Level',
    education_options
)

advanced_options_form = loan_request_form.expander('Expanded options')
fico_range_low = advanced_options_form.number_input('FICO Lower Bound', min_value=100, max_value=1000, value=500, step=1)
fico_range_high = advanced_options_form.number_input('FICO Upper Bound', min_value=100, max_value=1000, value=600, step=1)

installment = round(loan_amount / loan_duration, 2)
st.write(f'Per month: {installment}')

if loan_request_form.form_submit_button('Request loan'):
    datapoint = {
        'annual_inc': annual_income,
        'emp_length': employment_duration,
        'loan_amnt': loan_amount,
        'installment': round(loan_amount / loan_duration, 2),
        'fico_range_low': fico_range_low,
        'fico_range_high': fico_range_high,
    }

    for loan_purpose in loan_purposes:
        datapoint[f'purpose_{loan_purpose}'] = 1 if loan_purpose == purpose else 0

    for home in home_types:
        datapoint[f'home_ownership_{home.upper()}'] = 1 if home == home_type else 0

    for education_level in education_options:
        datapoint[f'education_{education_level}'] = 1 if education == education_level else 0
    
    probability_granted = request_loan(datapoint)
    confidence = 0
    
    if probability_granted >= 0.5:
        st.markdown(':green[Loan request should be accepted]')
        confidence = probability_granted
    else:
        st.markdown(':red[Loan request should be rejected]')
        confidence = 1 - probability_granted

    st.markdown(f'Confidence: {round(confidence * 100)}%')