import streamlit as st
import joblib
import nltk
import difflib
from text_cleaner import TextCleaner   

# nltk resources
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# LOAD MODEL AND DRUG LIST
model = joblib.load("drug_condition_model (1).pkl")
drug_list = joblib.load("drug_list.pkl")


# Convert drug list to lowercase for easier comparison
drug_list_lower = [d.lower() for d in drug_list]


# STREAMLIT UI
st.title("💊 Drug Review Condition Predictor")

st.write("Enter a drug name and review to predict the medical condition.")


# Drug Input
drug = st.text_input("Enter Drug Name")


# Review Input
review = st.text_area("Enter Review")


# Prediction Button
if st.button("Predict"):

    # -------- Validation -------- #

    if drug.strip() == "":
        st.warning("Please enter the drug name.")

    elif review.strip() == "":
        st.warning("Please enter the review.")

    elif len(review.split()) < 3:
        st.warning("Review must contain at least 3 words.")

    else:

        # -------- Drug Validation -------- #

        if drug.lower() not in drug_list_lower:

            matches = difflib.get_close_matches(drug, drug_list, n=1, cutoff=0.6)

            st.warning("Drug not found in dataset.")

            if matches:
                st.info(f"Did you mean: **{matches[0]}** ?")

        else:

            # Give more importance to review text
            text = drug + " " + review + " " + review + " " + review

            prediction = model.predict([text])[0]

            st.success(f"Predicted Condition: **{prediction}**")