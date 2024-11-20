import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from pathlib import Path

# Create models directory if it doesn't exist
Path("models").mkdir(exist_ok=True)

def load_data():
    """Load the preprocessed data."""
    print("Loading processed data...")
    try:
        data = joblib.load("processed/processed_data.pkl")
        return (
            data["X_train"], 
            data["y_train"],
            data["feature_size"],
            data["class_distribution"]
        )
    except FileNotFoundError:
        raise FileNotFoundError("Processed data not found. Run preprocessing.py first.")

def train_baseline_model(X, y, class_distribution):
    """Train and evaluate the baseline logistic regression model."""
    # Calculate class weights based on distribution
    n_samples = sum(class_distribution)
    class_weights = {
        0: n_samples / (2 * class_distribution[0]),
        1: n_samples / (2 * class_distribution[1])
    }
    
    # Initialize model with optimal parameters for text classification
    model = LogisticRegression(
        C=1.0,  # Inverse of regularization strength
        class_weight=class_weights,  # Handle class balance
        max_iter=1000,
        n_jobs=-1,  # Use all available cores
        random_state=42,
        solver='saga',  # Efficient solver for large datasets
        tol=1e-4,
    )
    
    return model

def evaluate_model(model, X, y):
    """Perform cross-validation and detailed evaluation."""
    # Perform stratified k-fold cross-validation
    print("\nPerforming cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train final model
    print("\nTraining final model...")
    with tqdm(total=1, desc="Training Logistic Regression") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "cv_scores_mean": float(cv_scores.mean()),
        "cv_scores_std": float(cv_scores.std()),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, metrics, (X_test, y_test, y_pred)

def save_results(model, metrics, feature_size):
    """Save the model, metrics, and parameters."""
    print("\nSaving results...")
    
    # Save model
    joblib.dump(model, "models/baseline_model.pkl")
    
    # Save metrics and parameters
    results = {
        "model_parameters": model.get_params(),
        "feature_size": feature_size,
        "metrics": metrics
    }
    
    with open("models/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)

def print_results(metrics):
    """Print the evaluation results."""
    print("\nModel Performance:")
    print(f"Cross-validation accuracy: {metrics['cv_scores_mean']:.4f} (+/- {metrics['cv_scores_std']*2:.4f})")
    print("\nDetailed Performance Metrics:")
    clf_report = metrics['classification_report']
    
    # Print formatted classification report
    print(f"\nPrecision, Recall, F1-Score:")
    print(f"{'Class':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 55)
    for label in ['0', '1']:
        print(f"{label:>10} {clf_report[label]['precision']:>10.4f} "
              f"{clf_report[label]['recall']:>10.4f} "
              f"{clf_report[label]['f1-score']:>10.4f} "
              f"{clf_report[label]['support']:>10}")

def main():
    print("Starting baseline model training...")
    
    # Load data
    X, y, feature_size, class_distribution = load_data()
    
    # Initialize and train model
    model = train_baseline_model(X, y, class_distribution)
    
    # Evaluate model
    model, metrics, eval_data = evaluate_model(model, X, y)
    
    # Save results
    save_results(model, metrics, feature_size)
    
    # Print results
    print_results(metrics)

if __name__ == "__main__":
    main()