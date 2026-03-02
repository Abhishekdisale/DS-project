import streamlit as st
import joblib
import re
import nltk
from text_cleaner import TextCleaner   

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# download nltk data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# LOAD MODEL AFTER CLASS DEFINITION
model = joblib.load("drug_condition_model (1).pkl")


# Streamlit UI
st.title("Drug Review Condition Predictor")

drug = st.text_input("Drug Name")

review = st.text_area("Enter Review")

if st.button("Predict"):

    if drug.strip() == "" or review.strip() == "":
        st.warning("Please enter both Drug Name and Review.")
    else:
        text = drug + " " + review
        prediction = model.predict([text])[0]
        st.success(f"Predicted Condition: {prediction}")