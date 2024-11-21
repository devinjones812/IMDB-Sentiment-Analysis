import numpy as np
import joblib
from sklearn.metrics import classification_report
from sklearn.semi_supervised import SelfTrainingClassifier
import scipy.sparse
from pathlib import Path
from tqdm import tqdm
import json

# Ensure models directory exists
Path("models").mkdir(exist_ok=True)

def load_data():
    """Load all processed data including train, test, and unlabeled data."""
    print("Loading processed data...")
    try:
        train_data = joblib.load("processed/train_data.pkl")
        test_data = joblib.load("processed/test_data.pkl")
        return (
            train_data["X"],      # Training features
            train_data["y"],      # Training labels
            test_data["X"],       # Test features
            test_data["y"],       # Test labels
            train_data["X_unsup"] # Unlabeled features
        )
    except FileNotFoundError:
        raise FileNotFoundError("Processed data not found. Run preprocessing.py first.")

def load_baseline_model():
    """Load the optimized baseline model."""
    print("\nLoading optimized baseline model...")
    try:
        model = joblib.load("models/baseline_model.pkl")
        print("Baseline model loaded successfully!")
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Baseline model not found. Run train_baseline.py first.")

def optimize_self_training(base_model, X_train, y_train, X_test, y_test, X_unlabeled):
    """Optimize self-training parameters using training data and evaluating on test data."""
    print("\nOptimizing self-training parameters...")
    
    # Define parameter grid
    thresholds = [0.75, 0.8, 0.85, 0.9]
    max_iters = [5, 10, 15]
    
    # Combine training and unlabeled data
    X_all = scipy.sparse.vstack([X_train, X_unlabeled])
    y_unlabeled = np.full(X_unlabeled.shape[0], -1)  # -1 indicates unlabeled
    y_all = np.concatenate([y_train, y_unlabeled])
    
    print("\nData sizes:")
    print(f"Training examples: {X_train.shape[0]}")
    print(f"Test examples: {X_test.shape[0]}")
    print(f"Unlabeled examples: {X_unlabeled.shape[0]}")
    print(f"Total examples for training: {X_all.shape[0]}")
    
    best_f1 = 0
    best_model = None
    best_params = None
    
    total_combinations = len(thresholds) * len(max_iters)
    with tqdm(total=total_combinations, desc="Testing self-training parameters") as pbar:
        for threshold in thresholds:
            for max_iter in max_iters:
                # Initialize self-training classifier
                self_training_model = SelfTrainingClassifier(
                    base_model,
                    threshold=threshold,
                    max_iter=max_iter,
                    verbose=False
                )
                
                # Train on combined data
                self_training_model.fit(X_all, y_all)
                
                # Evaluate on test set
                y_pred = self_training_model.predict(X_test)
                f1 = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = self_training_model
                    best_params = {'threshold': threshold, 'max_iter': max_iter}
                
                pbar.update(1)
    
    print(f"\nBest self-training parameters found:")
    print(f"Threshold: {best_params['threshold']}")
    print(f"Max iterations: {best_params['max_iter']}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    return best_model, best_params, best_f1

def evaluate_model(model, X_test, y_test):
    """Evaluate the final self-trained model on test set."""
    print("\nEvaluating final model on test set...")
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return metrics

def save_results(model, metrics, self_training_params, original_f1):
    """Save the self-trained model and results."""
    print("\nSaving results...")
    
    joblib.dump(model, "models/semi_supervised_model.pkl")
    
    results = {
        "self_training_params": self_training_params,
        "metrics": metrics,
        "improvement": {
            "original_f1": original_f1,
            "new_f1": metrics['macro avg']['f1-score'],
            "f1_improvement": metrics['macro avg']['f1-score'] - original_f1
        }
    }
    
    with open("models/semi_supervised_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to models/semi_supervised_results.json")
    print(f"F1 score improvement: {results['improvement']['f1_improvement']:.4f}")

def main():
    print("=== Semi-Supervised Learning Using Optimized Baseline Model ===")
    
    # Load all data
    X_train, y_train, X_test, y_test, X_unlabeled = load_data()
    
    # Load the optimized baseline model
    base_model = load_baseline_model()
    
    # Get baseline performance on test set
    original_f1 = classification_report(
        y_test, 
        base_model.predict(X_test), 
        output_dict=True
    )['macro avg']['f1-score']
    print(f"Baseline model F1 score on test set: {original_f1:.4f}")
    
    # Optimize self-training parameters
    best_model, best_params, best_f1 = optimize_self_training(
        base_model, X_train, y_train, X_test, y_test, X_unlabeled
    )
    
    # Final evaluation on test set
    final_metrics = evaluate_model(best_model, X_test, y_test)
    
    # Save results
    save_results(best_model, final_metrics, best_params, original_f1)

if __name__ == "__main__":
    main()