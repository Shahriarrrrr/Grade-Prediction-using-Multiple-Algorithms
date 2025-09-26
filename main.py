import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("StudentsPerformance.csv")  # replace with your dataset filename

print("Data Shape:", df.shape)
print(df.head())

# -----------------------
# Features & Target
# -----------------------
X = df.drop(columns=["GRADE", "STUDENT ID", "COURSE ID"])  # drop IDs
y = df["GRADE"]
# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Random Forest Classifier
# -----------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# -----------------------
# Predictions & Evaluation
# -----------------------
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# Feature Importance
# -----------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(importances.head(10))
