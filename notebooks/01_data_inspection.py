import pandas as pd

# Load the real estate dataset
df = pd.read_csv(
    r"C:\Users\shrey\Downloads\real estate investment project\data\india_housing_prices.csv"
)

# Display first five rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check duplicate rows
print("\nTotal Duplicate Rows:")
print(df.duplicated().sum())

# Check duplicate IDs
print("\nDuplicate IDs:")
print(df["ID"].duplicated().sum())

# Check statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check unique values in important numerical columns
print("\nUnique BHK Values:")
print(sorted(df["BHK"].unique()))

print("\nUnique Property Types:")
print(df["Property_Type"].unique())

print("\nPrice Range:")
print(df["Price_in_Lakhs"].min(), "to", df["Price_in_Lakhs"].max())

print("\nProperty Size Range:")
print(df["Size_in_SqFt"].min(), "to", df["Size_in_SqFt"].max())

print("\nYear Built Range:")
print(df["Year_Built"].min(), "to", df["Year_Built"].max())

print("\nAge of Property Range:")
print(df["Age_of_Property"].min(), "to", df["Age_of_Property"].max())

# Check consistency between property year and age
print("\nYear and Age Consistency:")
print(
    (2025 - df["Year_Built"] == df["Age_of_Property"]).value_counts()
)

# Check whether price per square foot is consistent
calculated_price = (
    df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
)

price_difference = (
    df["Price_per_SqFt"] - calculated_price
).abs()

print("\nPrice per SqFt Difference:")
print(price_difference.describe())

print("\nRecords with Price Difference Above 1:")
print((price_difference > 1).sum())

# Compare original and calculated price per square foot
comparison = df[
    ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]
].head(5).copy()

comparison["Calculated_Price_per_SqFt"] = (
    comparison["Price_in_Lakhs"] * 100000
    / comparison["Size_in_SqFt"]
)

comparison["Difference"] = (
    comparison["Price_per_SqFt"]
    - comparison["Calculated_Price_per_SqFt"]
).abs()

print("\nPrice Per SqFt Comparison:")
print(comparison)

# Verify Price_per_SqFt unit conversion

comparison = df[
    ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]
].head(5).copy()

comparison["Converted_Price_per_SqFt"] = (
    comparison["Price_per_SqFt"] * 100000
)

comparison["Calculated_Price_per_SqFt"] = (
    comparison["Price_in_Lakhs"] * 100000
    / comparison["Size_in_SqFt"]
)

print("\nUnit Conversion Verification:")
print(comparison)

# Check unique categorical values

categorical_columns = df.select_dtypes(
    include=["str", "object"]
).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())