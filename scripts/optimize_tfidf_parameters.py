import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import json
from pathlib import Path
from tqdm import tqdm
import spacy
import os

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

def create_initial_data():
    """Create initial raw_data.pkl if it doesn't exist"""
    print("Creating initial raw data...")
    
    # Load reviews
    pos_reviews = load_reviews("data/train/pos")
    neg_reviews = load_reviews("data/train/neg")
    unsup_reviews = load_reviews("data/train/unsup")
    
    # Combine labeled reviews and create labels
    labeled_reviews = pos_reviews + neg_reviews
    labels = [1] * len(pos_reviews) + [0] * len(neg_reviews)
    
    # Save raw data
    raw_data = {
        "reviews": labeled_reviews,
        "labels": labels,
        "unsupervised_reviews": unsup_reviews
    }
    
    # Create processed directory if it doesn't exist
    Path("processed").mkdir(exist_ok=True)
    
    # Save the data
    joblib.dump(raw_data, "processed/raw_data.pkl")
    print("Raw data saved to processed/raw_data.pkl")
    
    return raw_data

def optimize_tfidf_advanced(reviews, labels):
    """Two-stage optimization with better regularization and more conservative features"""
    print("\nStarting advanced TF-IDF parameter optimization...")
    
    # Create base pipeline with stronger regularization defaults
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            penalty='l2'
        ))
    ])
    
    # Stage 1: More conservative random search
    random_param_dist = {
        'tfidf__max_features': randint(12000, 20000),
        'tfidf__min_df': randint(5, 10),
        'tfidf__max_df': uniform(0.7, 0.2),
        'tfidf__ngram_range': [(1,2)],
        'tfidf__sublinear_tf': [True],
        'clf__C': uniform(0.5, 0.5)
    }
    
    print("\nStage 1: Random Search")
    n_iter = 50
    n_cv_folds = 5
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=random_param_dist,
        n_iter=n_iter,
        cv=n_cv_folds,
        verbose=1,
        n_jobs=-1,
        scoring=['f1', 'accuracy'],
        refit='f1',
        random_state=42
    )
    
    print(f"\nRunning random search with {n_iter} iterations...")
    random_search.fit(reviews, labels)
    
    print(f"\nBest parameters from random search: {random_search.best_params_}")
    print(f"Best F1 score from random search: {random_search.best_score_:.4f}")
    
    # Stage 2: Focused grid search
    best_max_features = random_search.best_params_['tfidf__max_features']
    best_min_df = random_search.best_params_['tfidf__min_df']
    best_max_df = random_search.best_params_['tfidf__max_df']
    best_C = random_search.best_params_['clf__C']
    
    focused_param_grid = {
        'tfidf__max_features': [
            max(12000, best_max_features - 1000),
            best_max_features,
            best_max_features + 1000
        ],
        'tfidf__min_df': [
            best_min_df - 1,
            best_min_df,
            best_min_df + 1
        ],
        'tfidf__max_df': [
            max(0.7, best_max_df - 0.05),
            best_max_df,
            min(0.9, best_max_df + 0.05)
        ],
        'tfidf__ngram_range': [(1,2)],
        'tfidf__sublinear_tf': [True],
        'clf__C': [
            max(0.5, best_C - 0.1),
            best_C,
            min(1.0, best_C + 0.1)
        ]
    }
    
    print("\nStage 2: Grid Search")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=focused_param_grid,
        cv=n_cv_folds,
        verbose=1,
        n_jobs=-1,
        scoring=['f1', 'accuracy', 'precision', 'recall'],
        refit='f1'
    )
    
    n_combinations = np.prod([len(v) for v in focused_param_grid.values()])
    print(f"\nRunning grid search with {n_combinations} parameter combinations...")
    grid_search.fit(reviews, labels)
    
    # Process results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    cv_results = grid_search.cv_results_
    best_idx = grid_search.best_index_
    
    tfidf_params = {k.replace('tfidf__', ''): v 
                   for k, v in best_params.items() 
                   if k.startswith('tfidf__')}
    
    results = {
        "tfidf_parameters": tfidf_params,
        "best_f1_score": float(best_score),
        "best_accuracy": float(cv_results['mean_test_accuracy'][best_idx]),
        "best_precision": float(cv_results['mean_test_precision'][best_idx]),
        "best_recall": float(cv_results['mean_test_recall'][best_idx]),
        "logistic_regression_C": float(best_params['clf__C']),
        "optimization_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_optimization_results(results)
    
    print("\nOptimization Results:")
    print(f"Best F1 score: {best_score:.4f}")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Best precision: {results['best_precision']:.4f}")
    print(f"Best recall: {results['best_recall']:.4f}")
    
    print("\nBest TF-IDF parameters:")
    for param, value in tfidf_params.items():
        print(f"  {param}: {value}")
    
    print("\nAnalyzing parameter importance...")
    analyze_parameter_importance(grid_search)
    
    return results

def save_optimization_results(results: dict):
    """Save optimization results to JSON file"""
    # Convert non-serializable types
    results = {k: str(v) if isinstance(v, tuple) else v 
              for k, v in results.items()}
    
    # Create directory and save
    Path("processed").mkdir(exist_ok=True)
    with open("processed/tfidf_optimal_params.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nOptimal parameters saved to processed/tfidf_optimal_params.json")

def analyze_parameter_importance(grid_search_results):
    """Analyze how different parameters affect the F1 score"""
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    Path("analysis").mkdir(exist_ok=True)
    analysis_results = {"parameter_impacts": {}}
    
    print("\nParameter Impact Analysis:")
    params = ['tfidf__max_features', 'tfidf__min_df', 'tfidf__max_df', 
             'tfidf__ngram_range', 'clf__C']
    
    for param in params:
        try:
            param_values = results_df['param_' + param].unique()
            param_impacts = {}
            
            print(f"\n{param}:")
            for value in param_values:
                mask = results_df['param_' + param] == value
                mean_f1 = results_df.loc[mask, 'mean_test_f1'].mean()
                param_impacts[str(value)] = float(mean_f1)
                print(f"  Value: {value}, Mean F1: {mean_f1:.4f}")
            
            analysis_results["parameter_impacts"][param] = param_impacts
        except KeyError:
            print(f"Parameter {param} not found in grid search results")
    
    with open("analysis/parameter_impact_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=4)

if __name__ == "__main__":
    # Check for raw_data.pkl and create if needed
    try:
        print("Looking for raw data...")
        raw_data = joblib.load("processed/raw_data.pkl")
        print("Found existing raw data!")
    except FileNotFoundError:
        print("No raw data found. Creating initial dataset...")
        raw_data = create_initial_data()
    
    # Create optimization subset
    n_samples = 5000
    print(f"\nCreating optimization subset with {n_samples} samples...")
    indices = np.random.choice(len(raw_data["reviews"]), n_samples, replace=False)
    reviews_subset = [raw_data["reviews"][i] for i in indices]
    labels_subset = [raw_data["labels"][i] for i in indices]
    
    # Run optimization
    results = optimize_tfidf_advanced(reviews_subset, labels_subset)