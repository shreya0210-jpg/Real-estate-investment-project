import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_housing_data.csv")

print("Dataset Shape:", df.shape)

# 1. Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# 2. Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 3. Check data types
print("\nData Types:")
print(df.dtypes)

# 4. Check invalid values
print("\nInvalid Values:")

print("Invalid BHK:", (~df["BHK"].between(1, 5)).sum())
print("Invalid Size:", (df["Size_in_SqFt"] <= 0).sum())
print("Invalid Price:", (df["Price_in_Lakhs"] <= 0).sum())
print("Invalid Year Built:", (~df["Year_Built"].between(1990, 2023)).sum())

# 5. Check year and age consistency
inconsistent_age = (
    (2025 - df["Year_Built"]) != df["Age_of_Property"]
).sum()

print("Inconsistent Property Age:", inconsistent_age)

print("\nData Validation Completed!")