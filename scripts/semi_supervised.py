from sklearn.semi_supervised import LabelSpreading
import joblib
from sklearn.metrics import classification_report
import numpy as np

# Load preprocessed data
data = joblib.load("preprocessed_data.pkl")
X, labels, tfidf = data["X"], data["labels"], data["tfidf"]

# Simulate unlabeled data (e.g., last 20% of data as "unlabeled")
pseudo_labels = np.full(len(labels), -1)  # -1 for unlabeled
labeled_indices = np.random.choice(len(labels), int(len(labels) * 0.8), replace=False)
pseudo_labels[labeled_indices] = labels[labeled_indices]

# Apply Label Spreading
label_spread = LabelSpreading(kernel='knn', alpha=0.8, max_iter=100)
print("Running Label Spreading...")
label_spread.fit(X, pseudo_labels)

# Evaluate the new model on labeled data
predicted_labels = label_spread.transduction_
print("Classification Report on Labeled Data:")
print(classification_report(labels[labeled_indices], predicted_labels[labeled_indices]))

# Save the label spreading model
joblib.dump(label_spread, "models/semi_supervised_model.pkl")
