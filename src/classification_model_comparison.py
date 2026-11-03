
import pandas as pd

# Classification model results
results = {
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [0.7241, 0.7232],
    "Precision": [0.7335, 0.7379],
    "Recall": [0.7353, 0.7236],
    "F1 Score": [0.7344, 0.7307]
}

# Create DataFrame
comparison = pd.DataFrame(results)

# Sort by F1 Score
comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

# Save comparison
comparison.to_csv(
    "data/classification_model_comparison.csv",
    index=False
)

print("Classification Model Comparison")
print("--------------------------------")
print(comparison.to_string(index=False))

print("\nBest model based on F1 Score:")
print(comparison.iloc[0]["Model"])

print("\nComparison saved to data/classification_model_comparison.csv")
