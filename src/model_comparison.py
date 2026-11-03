print("Model Comparison Script Started!", flush=True)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
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

X = pd.get_dummies(X, drop_first=True, dtype=int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10, min_samples_split=20,
        min_samples_leaf=10, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=20, max_depth=10,
        max_features="sqrt", max_samples=0.5,
        random_state=42, n_jobs=-1
    )
}

results = []

# Train and evaluate
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("data/model_comparison.csv", index=False)

print("\nComparison saved successfully!")

# Model Comparison Graph

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.title("R2 Score Comparison of Regression Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")

plt.xticks(rotation=15)
plt.axhline(y=0, linestyle="--")

plt.tight_layout()

plt.savefig("data/model_comparison_r2.png", dpi=300, bbox_inches="tight")
print("Graph saved at:", __import__("os").path.abspath("data/model_comparison_r2.png"))

plt.show()

print("Model comparison graph saved successfully!")