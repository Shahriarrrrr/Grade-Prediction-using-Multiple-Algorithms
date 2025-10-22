# ----------------------------
# Random Forest Grade Prediction with SMOTE
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("StudentsPerformance.csv")

# Separate Features and Target
X = df.drop(columns=["GRADE", "STUDENT ID", "COURSE ID"])
y = df["GRADE"]

# ----------------------------
# Balance Dataset using SMOTE
# ----------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("Before Resampling:", y.value_counts().to_dict())
print("After Resampling:", pd.Series(y_res).value_counts().to_dict())

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ----------------------------
# Train Random Forest Model
# ----------------------------
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Make Predictions
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Evaluation Metrics
# ----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Confusion Matrix Visualization
# ----------------------------
cm = confusion_matrix(y_test, y_pred)

# Normalized Confusion Matrix (row-wise percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True,
    xticklabels=np.unique(y), yticklabels=np.unique(y),
    annot_kws={"size": 12}
)

plt.title("Normalized Confusion Matrix - Random Forest + SMOTE", fontsize=14, weight='bold')
plt.xlabel("Predicted Grade", fontsize=12)
plt.ylabel("Actual Grade", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
