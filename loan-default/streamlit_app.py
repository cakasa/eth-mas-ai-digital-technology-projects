import streamlit as st
import joblib
import os

st.title(os.getcwd())

# model = joblib.load('./model/logistic_regression_model.pkl')
# imputer_dict = joblib.load('./model/imputer.pkl')
# scaler = joblib.load('./model/scaler.pkl')
# print(model, imputer_dict, scaler)

st.title("Hello Streamlit-er")
columns = st.columns(2)

for column in columns:
    with column:
        st.markdown('I am a column.')



if st.button("Send balloons!"):
    st.balloons()
