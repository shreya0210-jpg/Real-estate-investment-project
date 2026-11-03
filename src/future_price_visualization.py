import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/future_price_scenarios.csv")

scenarios = [
    "Conservative_5Y",
    "Moderate_5Y",
    "Optimistic_5Y"
]

average_prices = [df[col].mean() for col in scenarios]

plt.figure(figsize=(9, 5))

plt.bar(
    ["Conservative (5%)", "Moderate (8%)", "Optimistic (10%)"],
    average_prices
)

plt.title("Average Five-Year Property Price Projection")
plt.xlabel("Appreciation Scenario")
plt.ylabel("Average Projected Price (Lakhs)")

plt.tight_layout()
plt.savefig("data/future_price_scenarios.png", dpi=300)
plt.show()

print("Scenario visualization saved successfully!")