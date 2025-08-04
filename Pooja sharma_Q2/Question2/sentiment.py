import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("senti.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# Preprocess function
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(tokens)

# Streamlit UI
st.title("Sentimen Analysis App")

user_input = st.text_area("Enter your review here:")

if st.button("Predict"):
    cleaned = preprocess(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    st.write(f"Predicted Sentiment:{prediction}")