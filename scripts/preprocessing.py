import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def load_reviews(folder):
    reviews = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading reviews from {folder}"):
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as file:
            reviews.append(file.read())
    return reviews

def preprocess_data(pos_folder, neg_folder, unsup_folder=None, limit=None):
    pos_reviews = load_reviews(pos_folder)
    neg_reviews = load_reviews(neg_folder)
    unsup_reviews = load_reviews(unsup_folder) if unsup_folder else []

    if limit:
        pos_reviews = pos_reviews[:limit // 2]
        neg_reviews = neg_reviews[:limit // 2]

    reviews = pos_reviews + neg_reviews
    labels = [1] * len(pos_reviews) + [0] * len(neg_reviews)

    return reviews, labels, unsup_reviews

def vectorize_and_save(reviews, labels, filename="preprocessed_data.pkl", max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = tfidf.fit_transform(reviews)
    data = {"X": X, "labels": labels, "tfidf": tfidf}
    
    # Save preprocessed data
    joblib.dump(data, filename)
    print(f"Preprocessed data saved to {filename}")
    return data

if __name__ == "__main__":
    # Define dataset paths
    pos_folder = "data/train/pos"
    neg_folder = "data/train/neg"
    unsup_folder = "data/train/unsup"

    # Preprocess and save data
    reviews, labels, _ = preprocess_data(pos_folder, neg_folder, unsup_folder, limit=5000)
    vectorize_and_save(reviews, labels)
