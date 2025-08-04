import streamlit as st # type: ignore 
import pandas as pd
import pickle

def load_resources():
    with open('RFC_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_resources()

st.title("Boston Housing Price Prediction")

st.header("Area Details Input")

input_features = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
]

input_data = {}

# Corrected assignments - removed the trailing commas
input_data['CRIM'] = st.number_input("Insert a CRIM")
input_data['ZN'] = st.number_input("Insert a ZN")
input_data['INDUS'] = st.number_input("Insert a INDUS")
input_data['CHAS'] = st.number_input("Insert a CHAS")
input_data['NOX'] = st.number_input("Insert a NOX")
input_data['RM'] = st.number_input("Insert a RM")
input_data['AGE'] = st.number_input("Insert a AGE")
input_data['DIS'] = st.number_input("Insert a DIS")
input_data['RAD'] = st.number_input("Insert a RAD")
input_data['TAX'] = st.number_input("Insert a TAX")
input_data['PTRATIO'] = st.number_input("Insert a PTRATION")
input_data['B'] = st.number_input("Insert a B")
input_data['LSTAT'] = st.number_input("Insert a LSTAT")

input_df = pd.DataFrame([input_data])
input_df = input_df[input_features]

input_scaled = scaler.transform(input_df)

if st.button("Predict House Price"):
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"Predicted price is:- {predicted_price:,.2f}")