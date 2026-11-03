
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_housing_data.csv")

print("Original Dataset Shape:", df.shape)

# Create new features

# 1. Price per BHK
df["Price_Per_BHK"] = df["Price_in_Lakhs"] / df["BHK"]

# 2. Area per BHK
df["Area_Per_BHK"] = df["Size_in_SqFt"] / df["BHK"]

# 3. Property Age Category
df["Property_Age_Category"] = pd.cut(
    df["Age_of_Property"],
    bins=[-1, 5, 15, 30, float("inf")],
    labels=["New", "Moderate", "Old", "Very Old"]
)

# Display new features
print("\nNew Features:")
print(df[[
    "Price_Per_BHK",
    "Area_Per_BHK",
    "Property_Age_Category"
]].head())

print("\nUpdated Dataset Shape:", df.shape)

# Save feature-engineered dataset
df.to_csv("data/feature_engineered_data.csv", index=False)

print("\nFeature Engineering Completed!")


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize Area Per BHK
plt.figure(figsize=(10, 5))

sns.histplot(
    df["Area_Per_BHK"],
    bins=50,
    kde=True
)

plt.title("Distribution of Area Per BHK")
plt.xlabel("Area Per BHK (Sq. Ft.)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("data/area_per_bhk_distribution.png")
plt.show()

# Property Age Category Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Property_Age_Category",
    order=["New", "Moderate", "Old", "Very Old"]
)

plt.title("Property Age Category Distribution")
plt.xlabel("Property Age Category")
plt.ylabel("Number of Properties")

plt.tight_layout()
plt.savefig("data/property_age_category.png")
plt.show()

