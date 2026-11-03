import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_housing_data.csv")
print("\nChecking Price and Area:")
print(df[["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]].head(10))

# Display basic information
print("Dataset Shape:", df.shape)
print("\nPrice Statistics:")
print(df["Price_in_Lakhs"].describe())

# Plot property price distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    df["Price_in_Lakhs"],
    bins=50,
    kde=True
)

plt.title("Distribution of Property Prices")
plt.xlabel("Price (in Lakhs)")
plt.ylabel("Number of Properties")

plt.tight_layout()
plt.savefig("data/property_price_distribution.png")
plt.show()

# Property Type Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Property_Type",
    order=df["Property_Type"].value_counts().index
)

plt.title("Distribution of Property Types")
plt.xlabel("Property Type")
plt.ylabel("Number of Properties")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("data/property_type_distribution.png")
plt.show()


# BHK Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="BHK",
    order=sorted(df["BHK"].unique())
)

plt.title("Distribution of BHK")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Number of Properties")

plt.tight_layout()
plt.savefig("data/bhk_distribution.png")
plt.show()


# Print counts
print("\nProperty Type Counts:")
print(df["Property_Type"].value_counts())

print("\nBHK Counts:")
print(df["BHK"].value_counts().sort_index())


# Area vs Property Price (Scatter Plot)
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    alpha=0.5
)

plt.title("Property Area vs Price")
plt.xlabel("Property Area (Sq. Ft.)")
plt.ylabel("Price (in Lakhs)")

plt.tight_layout()
plt.savefig("data/area_vs_price.png")
plt.show()

# Average Price by Property Type
plt.figure(figsize=(10, 6))

avg_price = df.groupby("Property_Type")["Price_in_Lakhs"].mean().sort_values(ascending=False)

sns.barplot(
    x=avg_price.index,
    y=avg_price.values
)

plt.title("Average Property Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Average Price (in Lakhs)")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("data/average_price_by_property_type.png")
plt.show()


# Top 10 Cities by Average Property Price
plt.figure(figsize=(12, 6))

city_avg_price = (
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

sns.barplot(
    x=city_avg_price.index,
    y=city_avg_price.values
)

plt.title("Top 10 Cities by Average Property Price")
plt.xlabel("City")
plt.ylabel("Average Price (in Lakhs)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("data/top_10_cities_by_average_price.png")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12, 8))

numerical_df = df.select_dtypes(include=["number"])

correlation = numerical_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()
plt.savefig("data/correlation_heatmap.png")
plt.show()

# Correlation of features with property price
print("\nCorrelation with Property Price:")

correlation = numerical_df.corr()

print(
    correlation["Price_in_Lakhs"]
    .sort_values(ascending=False)
)

# Average price by categorical features

categorical_features = [
    "State",
    "Property_Type",
    "Furnished_Status",
    "Owner_Type",
    "Availability_Status"
]

for col in categorical_features:
    print(f"\nAverage Price by {col}:")
    print(
        df.groupby(col)["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    # Check relationship between property price and property area

print("\nPrice and Area Analysis:")

df["Area_Category"] = pd.cut(
    df["Size_in_SqFt"],
    bins=5
)

print(
    df.groupby("Area_Category", observed=False)["Price_in_Lakhs"]
    .agg(["mean", "min", "max", "count"])
)

print("\nPrice and BHK Analysis:")

print(
    df.groupby("BHK")["Price_in_Lakhs"]
    .mean()
)
