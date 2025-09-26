import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("StudentsPerformance.csv")

# Features & Target
X = df.drop(columns=["GRADE", "STUDENT ID", "COURSE ID"])
y = df["GRADE"]

# Balance Dataset with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("Before Resampling:", y.value_counts().to_dict())
print("After Resampling:", pd.Series(y_res).value_counts().to_dict())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Random Forest Model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
