import streamlit as st
import joblib

# Load model
model = joblib.load("drug_condition_model.pkl")

st.title("Drug Review Disease Predictor")

st.write("Enter drug name and review to predict the condition.")

drug = st.text_input("Drug Name")

review = st.text_area("Enter Review")

if st.button("Predict"):

    if drug and review:

        text = drug + " " + review

        prediction = model.predict([text])[0]

        st.success(f"Predicted Condition: {prediction}")

    else:
        st.warning("Please enter both drug name and review")