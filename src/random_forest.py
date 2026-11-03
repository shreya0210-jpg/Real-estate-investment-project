
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading dataset...", flush=True)

df = pd.read_csv("data/feature_engineered_data.csv")

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

X = pd.get_dummies(X, drop_first=True, dtype=int)

# Use a smaller sample for initial testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data preparation completed!", flush=True)
print("Training Random Forest...", flush=True)

model = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    max_features="sqrt",
    max_samples=0.5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!", flush=True)

y_pred = model.predict(X_test)

print("\nRandom Forest Results:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
