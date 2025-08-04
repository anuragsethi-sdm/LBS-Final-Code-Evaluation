import streamlit as st
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open ("SentModel.pkl","rb") as file:
    model=pickle.load(file)

with open ("SentVectorizer.pkl","rb") as file:
    tfidf=pickle.load(file)

lemmatizer=WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

def predict_sentiment(text):
    clean_text = preprocess(text)
    vector = tfidf.transform([clean_text])
    prediction = model.predict(vector)[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[prediction]

st.set_page_config(page_title="Product Review Sentiment Analyzer", layout="centered")
st.title("Sentiment Analysis on Product Reviews")
st.write("Enter your review and the sentiment will be provided: ")

user_input=st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment of your review: **{result}**")
