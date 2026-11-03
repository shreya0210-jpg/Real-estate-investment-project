
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/feature_engineered_data.csv")

# Separate features and target
y = df["Price_in_Lakhs"]

X = df.drop(
    columns=[
        "Price_in_Lakhs",
        "Price_Per_BHK",
        "Price_per_SqFt",
        "ID",
        "City",
        "Locality"
    ],
    errors="ignore"
)

# Convert categorical data
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    max_features="sqrt",
    max_samples=0.5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate residuals
residuals = y_test - predictions

# Evaluation metrics
print("\nRESIDUAL ANALYSIS")
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Residual:", residuals.mean())

# Actual vs Predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.4)
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted Property Prices")
plt.tight_layout()
plt.savefig("data/actual_vs_predicted.png", dpi=300)
plt.show()

# Residual distribution graph
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30)
plt.xlabel("Residuals (Lakhs)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig("data/residual_distribution.png", dpi=300)
plt.show()

# Residual vs Predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, alpha=0.4)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Price (Lakhs)")
plt.ylabel("Residuals (Lakhs)")
plt.title("Residuals vs Predicted Prices")
plt.tight_layout()
plt.savefig("data/residuals_vs_predicted.png", dpi=300)
plt.show()

print("\nResidual analysis completed!")