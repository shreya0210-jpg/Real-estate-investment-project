import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading dataset...")

df = pd.read_csv("data/feature_engineered_data.csv")

# Select target variable
y = df["Price_in_Lakhs"]

# Remove target and leakage features
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

# Convert categorical columns into numerical values
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data preparation completed!")

# Train Decision Tree
print("\nTraining Decision Tree...")

model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Decision Tree training completed!")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nDecision Tree Results:")

print("MAE:", mean_absolute_error(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

print("R2 Score:", r2_score(y_test, y_pred))