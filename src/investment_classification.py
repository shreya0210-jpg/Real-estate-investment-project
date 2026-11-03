
import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv("data/feature_engineered_data.csv")

# Remove records with invalid price per square foot
df = df[df["Price_per_SqFt"] > 0].copy()

# Calculate median price per sq. ft. for each city and property type
group_median = df.groupby(
    ["City", "Property_Type"]
)["Price_per_SqFt"].transform("median")

# Create investment classification target
df["Good_Investment"] = (
    df["Price_per_SqFt"] <= group_median
).astype(int)

# Save classification dataset
df.to_csv("data/investment_classification_data.csv", index=False)

# Display results
print("Classification dataset created successfully!")
print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["Good_Investment"].value_counts())

print("\nClass percentages:")
print(
    (df["Good_Investment"].value_counts(normalize=True) * 100).round(2)
)

print("\nFirst five records:")
print(df.head())
