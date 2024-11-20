from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm

# Load preprocessed data
print("Loading preprocessed data...")
data = joblib.load("preprocessed_data.pkl")
X, labels, tfidf = data["X"], data["labels"], data["tfidf"]

# Split data
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train baseline classifier with progress tracking
print("Training Logistic Regression baseline model...")
clf = LogisticRegression(max_iter=1000)
with tqdm(total=1, desc="Training Logistic Regression") as pbar:
    clf.fit(X_train, y_train)
    pbar.update(1)

# Save the trained model
print("Saving trained model...")
joblib.dump(clf, "models/baseline_model.pkl")
print("Baseline model saved as 'models/baseline_model.pkl'.")

# Evaluate the model
print("Evaluating the baseline model...")
y_pred = clf.predict(X_val)
print("Baseline Model Performance:")
print(classification_report(y_val, y_pred))
