
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load dataset
df = pd.read_csv("data/investment_classification_data.csv")

# Separate features and target
X = df.drop(columns=[
    "Good_Investment",
    "Price_per_SqFt",
    "Price_in_Lakhs",
    "Price_Per_BHK",
    "Locality"
])

y = df["Good_Investment"]

# Split dataset (same settings used during training)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained Logistic Regression model
model = joblib.load("models/good_investment_classifier.pkl")

# Predict
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

os.makedirs("data", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Good Investment", "Good Investment"]
)

display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

ax.set_title("Logistic Regression - Confusion Matrix")

plt.tight_layout()
plt.savefig("data/classification_confusion_matrix.png", dpi=300)
plt.show()
plt.close()
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig("data/classification_confusion_matrix.png", dpi=300)
plt.show()

# Save classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=["Not Good Investment", "Good Investment"],
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    "data/classification_report.csv"
)

print("Confusion Matrix:")
print(cm)

print("\nClassification report saved successfully!")
print("Confusion matrix saved to data/classification_confusion_matrix.png")
