import os
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tqdm import tqdm
from pathlib import Path

# Create processed directory if it doesn't exist
Path("processed").mkdir(exist_ok=True)

def load_reviews(folder):
    """Load all reviews from a specified folder."""
    reviews = []
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Skipping.")
        return reviews
    
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    with tqdm(total=len(files), desc=f"Loading {os.path.basename(folder)} reviews") as pbar:
        for filename in files:
            file_path = os.path.join(folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    reviews.append(file.read())
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}")
            pbar.update(1)
    return reviews

def lemmatize_texts(nlp, texts):
    """Lemmatize a list of texts using spaCy."""
    lemmatized_texts = []
    print("Starting lemmatization...")
    docs = nlp.pipe(texts, batch_size=500)
    for doc in tqdm(docs, total=len(texts), desc="Lemmatizing texts"):
        lemmatized = ' '.join([token.lemma_ for token in doc 
                             if not token.is_punct and not token.is_space])
        lemmatized_texts.append(lemmatized)
    return lemmatized_texts

def get_tfidf_vectorizer():
    """Get TF-IDF vectorizer with optimal or default parameters."""
    try:
        with open("processed/tfidf_optimal_params.json", 'r') as f:
            params = json.load(f)
            tfidf_params = params["tfidf_parameters"]
            
            # Convert ngram_range from list to tuple if needed
            if 'ngram_range' in tfidf_params:
                if isinstance(tfidf_params['ngram_range'], list):
                    tfidf_params['ngram_range'] = tuple(tfidf_params['ngram_range'])
                elif isinstance(tfidf_params['ngram_range'], str):
                    tfidf_params['ngram_range'] = eval(tfidf_params['ngram_range'])
            
            print("Using optimized TF-IDF parameters:")
            for param, value in tfidf_params.items():
                print(f"  {param}: {value}")
            
            return TfidfVectorizer(stop_words='english', **tfidf_params)
            
    except FileNotFoundError:
        print("No optimized parameters found. Using default parameters.")
        return TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            stop_words='english'
        )

def load_or_process_data(force_reprocess=False):
    """Smart loading/processing of data with checks for existing files."""
    
    if not force_reprocess:
        try:
            # First try to load raw data
            print("Checking for existing raw data...")
            raw_data = joblib.load("processed/raw_data.pkl")
            print("Found existing raw data!")
            
            try:
                # Then try to load lemmatized data
                print("Checking for existing lemmatized data...")
                lemmatized_data = joblib.load("processed/lemmatized_reviews.pkl")
                print("Found existing lemmatized data!")
                return (lemmatized_data["labeled"], 
                       lemmatized_data["unlabeled"], 
                       lemmatized_data["labels"])
            except FileNotFoundError:
                print("No lemmatized data found. Will lemmatize existing raw data.")
                nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                nlp.max_length = 2000000
                
                # Lemmatize existing raw data
                lemmatized_reviews = lemmatize_texts(nlp, raw_data["reviews"])
                lemmatized_unsup = lemmatize_texts(nlp, raw_data["unsupervised_reviews"])
                
                # Save lemmatized data
                lemmatized_data = {
                    "labeled": lemmatized_reviews,
                    "unlabeled": lemmatized_unsup,
                    "labels": raw_data["labels"]
                }
                joblib.dump(lemmatized_data, "processed/lemmatized_reviews.pkl")
                
                return lemmatized_reviews, lemmatized_unsup, raw_data["labels"]
                
        except FileNotFoundError:
            print("No existing data found. Will process from scratch.")
    else:
        print("Force reprocessing flag set. Will process from scratch.")
    
    # If we're here, we need to process from scratch
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nlp.max_length = 2000000
    
    print("\nLoading reviews...")
    pos_reviews = load_reviews("data/train/pos")
    neg_reviews = load_reviews("data/train/neg")
    unsup_reviews = load_reviews("data/train/unsup")
    
    # Combine labeled reviews and create labels
    print("\nPreparing datasets...")
    labeled_reviews = pos_reviews + neg_reviews
    labels = [1] * len(pos_reviews) + [0] * len(neg_reviews)
    
    # Save raw data
    print("\nSaving raw data...")
    raw_data = {
        "reviews": labeled_reviews,
        "labels": labels,
        "unsupervised_reviews": unsup_reviews
    }
    joblib.dump(raw_data, "processed/raw_data.pkl")
    
    # Lemmatize reviews
    lemmatized_reviews = lemmatize_texts(nlp, labeled_reviews)
    lemmatized_unsup = lemmatize_texts(nlp, unsup_reviews)
    
    # Save lemmatized data
    print("\nSaving lemmatized data...")
    lemmatized_data = {
        "labeled": lemmatized_reviews,
        "unlabeled": lemmatized_unsup,
        "labels": labels
    }
    joblib.dump(lemmatized_data, "processed/lemmatized_reviews.pkl")
    
    return lemmatized_reviews, lemmatized_unsup, labels

def process_data_with_tfidf(lemmatized_reviews, lemmatized_unsup):
    """Process both labeled and unlabeled data using same vectorizer."""
    print("\nCreating TF-IDF features...")
    
    # Get and fit vectorizer on all data (labeled + unlabeled)
    vectorizer = get_tfidf_vectorizer()
    
    # Transform data
    X = vectorizer.fit_transform(lemmatized_reviews)
    X_unsup = vectorizer.transform(lemmatized_unsup)
    
    return X, X_unsup, vectorizer

if __name__ == "__main__":
    print("Starting preprocessing pipeline...")
    
    # Load or process data
    lemmatized_reviews, lemmatized_unsup, labels = load_or_process_data(force_reprocess=False)
    
    # Process all data with TF-IDF
    X, X_unsup, vectorizer = process_data_with_tfidf(lemmatized_reviews, lemmatized_unsup)
    
    # Save processed data
    print("\nSaving final processed data...")
    processed_data = {
        "X_train": X,
        "y_train": np.array(labels),
        "X_unsup": X_unsup,
        "vectorizer": vectorizer,
        "vocab_size": len(vectorizer.vocabulary_),
        "feature_size": X.shape[1],
        "class_distribution": np.bincount(labels)
    }
    joblib.dump(processed_data, "processed/processed_data.pkl")
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total labeled samples: {X.shape[0]}")
    print(f"Total unlabeled samples: {X_unsup.shape[0]}")
    print(f"Feature size: {processed_data['feature_size']}")
    print(f"Vocabulary size: {processed_data['vocab_size']}")
    print(f"Class distribution: {processed_data['class_distribution']}")