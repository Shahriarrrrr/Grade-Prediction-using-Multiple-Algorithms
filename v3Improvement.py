# v3Improvement_collapsed_XGBoost.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("StudentsPerformance.csv")  # replace with your CSV filename

# -----------------------
# Collapse grades into 3 groups
# -----------------------
def collapse_grades(grade):
    if grade in [0, 1, 2]:      # Fail, DD, DC
        return "Low"
    elif grade in [3, 4, 5]:    # CC, CB, BB
        return "Medium"
    else:                        # BA, AA
        return "High"

df["GRADE_COLLAPSED"] = df["GRADE"].apply(collapse_grades)

# -----------------------
# Features and Target
# -----------------------
X = df.drop(columns=["GRADE", "STUDENT ID", "COURSE ID", "GRADE_COLLAPSED"])
y = df["GRADE_COLLAPSED"]

# Encode labels to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Low->0, Medium->1, High->2

# -----------------------
# Balance Dataset using SMOTE
# -----------------------
print("Before Resampling:\n", pd.Series(y_encoded).value_counts())
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y_encoded)
print("\nAfter Resampling:\n", pd.Series(y_res).value_counts())

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# -----------------------
# XGBoost Model
# -----------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# -----------------------
# Predictions
# -----------------------
y_pred = model.predict(X_test)

# Decode numeric labels back to original strings
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# -----------------------
# Evaluation
# -----------------------
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred_labels)
print(cm)

# -----------------------
# Plot Confusion Matrix
# -----------------------
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Collapsed Grades")
plt.colorbar()
tick_marks = range(len(le.classes_))
plt.xticks(tick_marks, le.classes_)
plt.yticks(tick_marks, le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Annotate numbers
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.show()

# -----------------------
# Feature Importance
# -----------------------
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(feat_importances.head(10))
