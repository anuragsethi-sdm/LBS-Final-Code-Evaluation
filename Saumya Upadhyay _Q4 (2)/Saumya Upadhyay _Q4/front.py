import streamlit as st
import joblib
import nltk
from nltk import word_tokenize
import pandas as pd

crf = joblib.load("ner_crf_model.pkl")

def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True 

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True  

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Streamlit UI
st.set_page_config(page_title="NER with CRF", layout="centered")
st.title("Named Entity Recognition using CRF model")
st.markdown("Enter a sentence below to identify named entities:")

sentence = st.text_input("Input Sentence:", "Modiji is very popular .")

if st.button("Predict Entities"):
    tokens = word_tokenize(sentence)
    features = sent2features(tokens)
    pred_tags = crf.predict([features])[0]

    st.markdown("Predicted Entities")
    for word, tag in zip(tokens, pred_tags):
        st.write(f"**{word}** → `{tag}`")

joblib.dump(crf, 'crf_model.pkl')