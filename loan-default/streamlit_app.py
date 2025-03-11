import streamlit as st
import joblib
import os

path = './work/eth-mas-ai-digital-technology-projects/loan-default'

model = joblib.load(path + '/model/logistic_regression_model.pkl')
imputer_dict = joblib.load(path + '/model/imputer.pkl')
scaler = joblib.load(path + '/model/scaler.pkl')
print(path + '/model/logistic_regression_model.pkl')
print(model, imputer_dict, scaler)

st.title("Hello Streamlit-er")
columns = st.columns(2)

for column in columns:
    with column:
        st.markdown('I am a column.')



if st.button("Send balloons!"):
    st.balloons()
