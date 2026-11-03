
import pandas as pd
import os

# Load original dataset
df = pd.read_csv(
    r"C:\Users\shrey\Downloads\real estate investment project\data\india_housing_prices.csv"
)

# Remove unnecessary ID column
df = df.drop(columns=["ID"])

# Standardize column names
df.columns = df.columns.str.strip()

# Remove leading and trailing spaces from text columns
categorical_columns = df.select_dtypes(include=["str", "object"]).columns

for column in categorical_columns:
    df[column] = df[column].str.strip()

# Save cleaned dataset
output_path = os.path.join("data", "cleaned_housing_data.csv")

df.to_csv(output_path, index=False)

print("Data preprocessing completed successfully!")
print("Cleaned dataset shape:", df.shape)
print("Saved at:", output_path)
