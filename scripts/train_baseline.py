import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score
from tqdm import tqdm
import json
from pathlib import Path

def load_data():
    """Load processed train and test data."""
    print("Loading processed data...")
    try:
        train_data = joblib.load("processed/train_data.pkl")
        test_data = joblib.load("processed/test_data.pkl")
        
        # Calculate class distribution from training data
        unique, counts = np.unique(train_data["y"], return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        return (
            train_data["X"], 
            train_data["y"],
            test_data["X"],
            test_data["y"],
            train_data.get("X_unsup", None),
            class_distribution
        )
    except FileNotFoundError:
        raise FileNotFoundError("Processed data not found. Run preprocessing.py first.")

def optimize_baseline_model(X_train, y_train, class_distribution):
    """Optimize hyperparameters with thorough grid search."""
    print("\nOptimizing baseline model hyperparameters...")
    
    # Calculate class weights based on distribution
    n_samples = sum(class_distribution.values())
    custom_class_weights = {
        0: n_samples / (2 * class_distribution[0]),
        1: n_samples / (2 * class_distribution[1])
    }
    
    # Thorough parameter grid
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'class_weight': [None, 'balanced', custom_class_weights],
        'tol': [1e-4, 1e-3],
        'max_iter': [2000, 3000]  # Increased from original to handle convergence
    }
    
    # Initialize base model
    base_model = LogisticRegression(
        solver='saga',
        n_jobs=-1,
        random_state=42
    )
    
    # Create F1 scorer
    f1_scorer = make_scorer(f1_score, average='macro')
    
    # Calculate total combinations
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTesting {total_combinations} parameter combinations...")
    
    # Perform grid search with progress bar
    with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring=f1_scorer,
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        pbar.update(total_combinations)
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # Save optimal parameters for semi-supervised learning
    optimal_params = {
        "logistic_regression_C": float(grid_search.best_params_['C']),
        "class_weight": str(grid_search.best_params_['class_weight']),
        "max_iter": int(grid_search.best_params_['max_iter']),
        "tol": float(grid_search.best_params_['tol']),
        "best_f1_score": float(grid_search.best_score_)
    }
    
    Path("processed").mkdir(exist_ok=True)
    with open("processed/optimal_logreg_params.json", "w") as f:
        json.dump(optimal_params, f, indent=4)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    
    metrics = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics

def save_results(model, metrics):
    """Save model and results."""
    print("\nSaving results...")
    Path("models").mkdir(exist_ok=True)
    
    # Save model
    joblib.dump(model, "models/baseline_model.pkl")
    
    # Save results
    results = {
        "model_parameters": model.get_params(),
        "metrics": metrics
    }
    
    with open("models/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)

def main():
    print("=== Training Baseline Model with Thorough Grid Search ===")
    
    # Load data
    X_train, y_train, X_test, y_test, X_unsup, class_distribution = load_data()
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Class distribution: {class_distribution}")
    
    # Optimize model with thorough grid search
    model = optimize_baseline_model(X_train, y_train, class_distribution)
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save everything
    save_results(model, metrics)
    
    print("\nBaseline training complete!")
    print(f"Test set F1 score: {metrics['classification_report']['macro avg']['f1-score']:.4f}")

if __name__ == "__main__":
    main()