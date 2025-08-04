import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.markdown("""
<style>
body {
  background:#ff0099; 
}
</style>
    """, unsafe_allow_html=True)

model = joblib.load('houseprice.pkl')
st.title("house price prediction")
st.markdown("Enter Details")

#feature i used X=df[['NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','LSTAT']]


nox = st.number_input("NOX", 0.0, 1.0, 0.5)
rm = st.number_input("RM", 1.0, 10.0, 6.0)
age = st.number_input("AGE", 0.0, 100.0, 60.0)
dis = st.number_input("DIS", 1.0, 15.0, 4.0)
rad= st.selectbox("RAD", list(range(1, 25)))
tax= st.number_input("TAX", 100.0, 800.0, 300.0)
ptratio = st.number_input("PTRATIO", 10.0, 25.0, 15.0)
lstat = st.number_input("LSTA", 1.0, 40.0, 12.0)

if st.button("Predict Price"):
    input_data = np.array([[ nox,rm, age,dis,rad, dis,tax,ptratio,]])
    prediction = model.predict(input_data)[0]
    st.success(f"HousePrice:{prediction:.2f}")
   
