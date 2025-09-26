# v4_numeric_CGPA_regression.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("StudentsPerformance.csv")  # replace with your CSV filename

# -----------------------
# Features and Target
# -----------------------
# Assuming the last semester CGPA column is 'Cumulative grade point average in the last semester (/4.00)'
# and you want to predict 'Expected Cumulative grade point average in the graduation (/4.00)'
X = df.drop(columns=[
    "STUDENT ID", "COURSE ID", 
    "Expected Cumulative grade point average in the graduation (/4.00)", "GRADE"
])
y = df["Expected Cumulative grade point average in the graduation (/4.00)"]

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# XGBoost Regressor
# -----------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------
# Predictions
# -----------------------
y_pred = model.predict(X_test)

# -----------------------
# Evaluation
# -----------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.5f}")
print(f"R2 Score: {r2:.5f}")

# -----------------------
# Compare predicted vs actual
# -----------------------
comparison = pd.DataFrame({
    "Actual_CGPA": y_test,
    "Predicted_CGPA": y_pred
}).reset_index(drop=True)
print("\nPredicted vs Actual CGPA:")
print(comparison.head(10))

# -----------------------
# Plot Actual vs Predicted
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual CGPA")
plt.ylabel("Predicted CGPA")
plt.title("Actual vs Predicted CGPA - XGBoost Regressor")
plt.show()

# -----------------------
# Feature Importance
# -----------------------
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(feat_importances.head(10))
