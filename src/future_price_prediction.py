
import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv("data/feature_engineered_data.csv")

# Annual appreciation assumption
annual_growth_rate = 0.08

# Projection period
years = 5

# Calculate future property price
df["Future_Price_5Y"] = (
    df["Price_in_Lakhs"] *
    (1 + annual_growth_rate) ** years
)

# Save projected dataset
df.to_csv("data/future_price_projection.csv", index=False)

# Display results
print("Five-Year Property Price Projection")
print("-----------------------------------")

print("Annual growth rate:", annual_growth_rate * 100, "%")
print("Projection period:", years, "years")

print("\nSample results:")
print(
    df[["City", "Property_Type", "Price_in_Lakhs",
        "Future_Price_5Y"]].head(10).round(2)
)

print("\nProjection summary:")
print(df["Future_Price_5Y"].describe().round(2))

print("\nDataset saved successfully!")
