# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("water_potability.csv")

# Split features and target
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Train-test split
X_train_full, X_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val = train_test_split(X_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

# Define and calibrate SVC
svc = SVC(probability=True, class_weight='balanced', C=1, kernel='rbf')
calibrated_svc = CalibratedClassifierCV(estimator=svc, cv=5)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', calibrated_svc)
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities on validation set
probs = pipeline.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, probs)
pr_auc = auc(recall, precision)

# Select best threshold (e.g., max F1)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_thr = thresholds[np.argmax(f1_scores)]

# Evaluate on test set using best threshold
test_probs = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (test_probs >= best_thr).astype(int)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Best Threshold: {best_thr:.4f}")
print(f"Validation AUC-PR: {pr_auc:.4f}")
print(f"Test Accuracy (Threshold {best_thr:.2f}): {test_accuracy:.4f}")

# Save the model and threshold
joblib.dump({
    "pipeline": pipeline,
    "threshold": best_thr
}, "best_model_svc_calibrated.pkl")

print("\n✅ Calibrated SVC model saved as 'best_model_svc_calibrated.pkl'")
