import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tqdm import tqdm
from pathlib import Path
import json

def load_optimized_tfidf_params():
    """Load previously optimized TF-IDF parameters."""
    try:
        with open("processed/tfidf_optimal_params.json", 'r') as f:
            params = json.load(f)
        print("\nLoaded optimized TF-IDF parameters:")
        print(f"max_features: {params['tfidf_parameters']['max_features']}")
        print(f"min_df: {params['tfidf_parameters']['min_df']}")
        print(f"max_df: {params['tfidf_parameters']['max_df']}")
        print(f"ngram_range: {params['tfidf_parameters']['ngram_range']}")
        return params['tfidf_parameters']
    except FileNotFoundError:
        print("\nOptimized TF-IDF parameters not found!")
        print("Please run optimize_parameters.py first")
        exit(1)

def load_reviews(folder, desc):
    """Load all reviews from a specified folder."""
    reviews = []
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Skipping.")
        return reviews
    
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    with tqdm(total=len(files), desc=desc) as pbar:
        for filename in files:
            file_path = os.path.join(folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    reviews.append(file.read())
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}")
            pbar.update(1)
    return reviews

def process_dataset():
    """Process both training and test datasets using optimized parameters."""
    print("=== Processing IMDB Dataset with Optimized Parameters ===")
    
    # Load optimized parameters
    tfidf_params = load_optimized_tfidf_params()
    
    # Load training data
    print("\nLoading training data...")
    train_pos = load_reviews("data/train/pos", "Training positive")
    train_neg = load_reviews("data/train/neg", "Training negative")
    train_unsup = load_reviews("data/train/unsup", "Unsupervised")
    
    # Load test data
    print("\nLoading test data...")
    test_pos = load_reviews("data/test/pos", "Test positive")
    test_neg = load_reviews("data/test/neg", "Test negative")
    
    # Create labels
    train_labels = np.concatenate([
        np.ones(len(train_pos)), 
        np.zeros(len(train_neg))
    ])
    test_labels = np.concatenate([
        np.ones(len(test_pos)), 
        np.zeros(len(test_neg))
    ])
    
    # Combine reviews
    train_reviews = train_pos + train_neg
    test_reviews = test_pos + test_neg
    
    print("\nInitializing TF-IDF vectorizer with optimized parameters...")
    vectorizer = TfidfVectorizer(
        max_features=tfidf_params['max_features'],
        min_df=tfidf_params['min_df'],
        max_df=tfidf_params['max_df'],
        ngram_range=tuple(tfidf_params['ngram_range']),  # Convert list back to tuple
        sublinear_tf=tfidf_params['sublinear_tf'],
        stop_words='english'
    )
    
    # Transform all datasets
    print("Transforming training data...")
    X_train = vectorizer.fit_transform(train_reviews)
    
    print("Transforming test data...")
    X_test = vectorizer.transform(test_reviews)
    
    print("Transforming unsupervised data...")
    X_unsup = vectorizer.transform(train_unsup)
    
    # Create data dictionaries
    train_data = {
        "X": X_train,
        "y": train_labels,
        "X_unsup": X_unsup,
        "vectorizer": vectorizer,
        "class_distribution": {
            0: len(train_neg),
            1: len(train_pos)
        }
    }
    
    test_data = {
        "X": X_test,
        "y": test_labels,
        "class_distribution": {
            0: len(test_neg),
            1: len(test_pos)
        }
    }
    
    # Save processed data
    print("\nSaving processed data...")
    Path("processed").mkdir(exist_ok=True)
    
    joblib.dump(train_data, "processed/train_data.pkl")
    joblib.dump(test_data, "processed/test_data.pkl")
    
    # Save dataset statistics
    stats = {
        "dataset_sizes": {
            "train_positive": len(train_pos),
            "train_negative": len(train_neg),
            "train_unsupervised": len(train_unsup),
            "test_positive": len(test_pos),
            "test_negative": len(test_neg)
        },
        "vocabulary_size": len(vectorizer.get_feature_names_out()),
        "tfidf_parameters": tfidf_params
    }
    
    with open("processed/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print("\nPreprocessing complete!")
    print(f"Training examples: {X_train.shape[0]}")
    print(f"Test examples: {X_test.shape[0]}")
    print(f"Unsupervised examples: {X_unsup.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")

if __name__ == "__main__":
    process_dataset()