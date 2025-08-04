import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the saved model and TF-IDF vectorizer
with open("sentiment_model.pkl", "rb") as file:
    model, tfidf = pickle.load(file)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    
    # Custom stopword removal
    sw = set(stopwords.words("english"))
    important_words = {"bad", "good", "excellent", "poor", "great", "worst", "terrible", "awful", "amazing"}
    custom_sw = sw - important_words
    tokens = [word for word in tokens if word not in custom_sw]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(lemmatized)

# Streamlit UI
st.set_page_config(page_title="Amazon Review Sentiment", layout="centered")

st.title("Amazon Review Sentiment Classifier")

user_input = st.text_area("Enter a product review below:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess(user_input)
        transformed = tfidf.transform([processed])
        prediction = model.predict(transformed)[0]
        if (prediction>3):
            st.success(f"Predicted Sentiment: Positive Review")
        elif (prediction<3 & prediction>=2):
            st.success(f"Predicted Sentiment: Neutral Review")
        else:
            st.success(f"Predcited Sentiment: Negative Review")
