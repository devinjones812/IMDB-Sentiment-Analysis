import shap
import joblib
from preprocessing import preprocess_data, vectorize_data

# Load the baseline model and vectorizer
clf = joblib.load("models/baseline_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Load and preprocess a subset of data
pos_folder = "data/train/pos"
neg_folder = "data/train/neg"
reviews, labels, _ = preprocess_data(pos_folder, neg_folder, limit=1000)
X, _ = vectorize_data(reviews)

# Explain predictions
explainer = shap.LinearExplainer(clf, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)
