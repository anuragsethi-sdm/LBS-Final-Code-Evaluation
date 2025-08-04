import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)



with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("House Price Prediction")

# Get user input
LSTAT = st.number_input("Enter the LSTAT Value:")
INDUS = st.number_input("Enter the INDUS Value:")
NOX = st.number_input("Enter the NOX Value:")
PTRATIO = st.number_input("Enter the PTRATIO Value:")
RM = st.number_input("Enter the RM Value:")
TAX = st.number_input("Enter the TAX Value:")
DIS = st.number_input("Enter the DIS Value:")
AGE = st.number_input("Enter the AGE Value:")

# Make prediction
if st.button("Predict"):
    house_data = np.array([LSTAT, INDUS, NOX, PTRATIO, RM, TAX, DIS, AGE])
    house_data = house_data.reshape(1, -1)
    house_data = scaler.transform(house_data)
    house_price = model.predict(house_data)
    st.success(f"The predicted house price is: {house_price[0]:.2f}")
