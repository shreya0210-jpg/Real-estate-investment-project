
import pandas as pd

# Load dataset
df = pd.read_csv("data/feature_engineered_data.csv")

# Define appreciation scenarios
growth_rates = {
    "Conservative_5Y": 0.05,
    "Moderate_5Y": 0.08,
    "Optimistic_5Y": 0.10
}

# Calculate five-year projected prices
for scenario, rate in growth_rates.items():
    df[scenario] = (
        df["Price_in_Lakhs"] * (1 + rate) ** 5
    )

# Save scenario dataset
df.to_csv("data/future_price_scenarios.csv", index=False)

# Display average projected prices
print("Five-Year Appreciation Scenarios")
print("--------------------------------")

for scenario in growth_rates:
    print(
        scenario,
        ": ₹",
        round(df[scenario].mean(), 2),
        "Lakhs"
    )

print("\nSample comparison:")
print(
    df[[
        "Price_in_Lakhs",
        "Conservative_5Y",
        "Moderate_5Y",
        "Optimistic_5Y"
    ]].head(10).round(2)
)

print("\nScenario dataset saved successfully!")
