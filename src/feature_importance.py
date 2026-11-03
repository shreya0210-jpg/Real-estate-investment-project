import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/feature_engineered_data.csv")

# Separate target and features
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

# Convert categorical columns into numeric
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

# Calculate feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance.head(10).to_string(index=False))

# Save results
importance.to_csv("data/feature_importance.csv", index=False)

# Plot graph
plt.figure(figsize=(10, 6))

top_features = importance.head(10)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nFeature importance analysis completed!")