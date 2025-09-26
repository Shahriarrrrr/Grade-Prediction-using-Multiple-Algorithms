import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv("StudentsPerformance.csv")

# Features & Target
X = df.drop(columns=["GRADE", "STUDENT ID", "COURSE ID"])
y = df["GRADE"]

# -----------------------
# Handle Class Imbalance (SMOTE)
# -----------------------
print("Before Resampling:", y.value_counts().to_dict())
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("After Resampling:", pd.Series(y_res).value_counts().to_dict())

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
# Evaluation
# -----------------------
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(set(y)),
            yticklabels=sorted(set(y)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost + SMOTE")
plt.show()

# -----------------------
# Feature Importance
# -----------------------
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(feat_importances.head(10))
