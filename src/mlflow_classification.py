
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained Random Forest model
model = joblib.load(
    "models/good_investment_random_forest.pkl"
)

# Generate predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Set MLflow experiment
mlflow.set_experiment(
    "Real Estate Investment Classification"
)

# Log Random Forest experiment
with mlflow.start_run(run_name="Random_Forest"):

    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_leaf", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log trained model
    mlflow.sklearn.log_model(
        model,
        name="investment_random_forest",
        skops_trusted_types=["numpy.dtype"]
    )

# Display results
print("Random Forest MLflow Tracking Completed!")
print("-----------------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
