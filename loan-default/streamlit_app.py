import streamlit as st
import joblib

model = joblib.load('logistic_regression_model.pkl')
print(model)

st.title("Hello Streamlit-er")
columns = st.columns(2)

for column in columns:
    with column:
        st.markdown('I am a column.')



if st.button("Send balloons!"):
    st.balloons()
