import streamlit as st
import joblib

# Load models
clf = joblib.load("models/baseline_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("IMDB Sentiment Classifier")
review = st.text_area("Enter a Movie Review:")

if st.button("Predict Sentiment"):
    review_tfidf = tfidf.transform([review])
    prediction = clf.predict(review_tfidf)
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"Predicted Sentiment: {sentiment}")
