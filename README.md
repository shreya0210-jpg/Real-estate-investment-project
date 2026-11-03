# 🏠 Real Estate Investment Advisor

**Predicting Property Profitability & Future Value**

An interactive data analytics and machine learning application designed to analyze the Indian housing market, explore property price trends, and support informed real estate investment decisions.

## 🌐 Live Application

**[Click Here to Open Real Estate Investment Advisor](https://real-estate-investment-project-4mhxvdya8pwuauxarxs8fm.streamlit.app/)**

## 📌 Project Overview

The Real Estate Investment Advisor is a Python-based project that uses data analytics, machine learning, and interactive visualizations to explore the Indian real estate market.

It helps users analyze housing prices, compare properties, evaluate relative affordability, and explore possible future property values under different growth scenarios.

## ✨ Key Features

- **Dashboard Overview:** Displays important housing market statistics and insights.
- **Market Analysis:** Explore housing prices using filters such as state, property type, and BHK.
- **Model Performance:** Compare machine learning models and evaluate their performance.
- **Investment Advisor:** Enter property details to receive a relative affordability prediction.
- **Future Price Projection:** Explore estimated property values using different annual growth assumptions (5%, 8%, and 10%).
- **Data Visualization:** Interactive charts and visual representations of housing market trends.
- **Feature Importance:** Understand which features contribute to model predictions.
- **MLflow Tracking:** Track machine learning experiments and evaluation metrics.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Web Framework | Streamlit |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib, Pickle |

## 📂 Project Structure

```text
Real-estate-investment-project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── india_housing_prices.csv
│   ├── cleaned_housing_data.csv
│   ├── feature_engineered_data.csv
│   ├── investment_classification_data.csv
│   └── other analysis outputs
│
├── models/
│   ├── good_investment_classifier.pkl
│   └── good_investment_random_forest.pkl
│
├── notebooks/
│   ├── 01_data_inspection.py
│   └── 02_eda.py
│
├── src/
│   ├── model_training.py
│   ├── random_forest.py
│   ├── decision_tree.py
│   ├── model_comparison.py
│   ├── investment_classification.py
│   ├── train_classification.py
│   └── future_price_prediction.py
│
└── mlruns/
    └── MLflow experiment tracking
```

*The structure highlights the main project components. Additional scripts, datasets, and generated outputs are also included in the repository.*

## 📊 Machine Learning Approach

The project explores both regression and classification techniques.

### Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### Classification Models
- Logistic Regression
- Random Forest Classifier

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Logistic Regression achieved approximately **72.41% accuracy** on the classification task.

## 🔮 Future Price Projection

The application provides illustrative future property price scenarios based on assumed annual growth rates:

- 5% annual growth
- 8% annual growth
- 10% annual growth

These projections help users compare possible future values over a five-year period.

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/shreya0210-jpg/Real-estate-investment-project.git
```

### 2. Navigate to the Project Directory

```bash
cd Real-estate-investment-project
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ⚠️ Important Notes

- The investment classification represents relative affordability based on price per square foot within a city and property type. It does not guarantee actual investment profitability.
- Future price projections are illustrative scenarios based on assumed growth rates, not guaranteed market forecasts.
- Model predictions depend on the available dataset and its features.

## 🚀 Future Enhancements

- Integration of real-time real estate market data.
- Improved machine learning model performance.
- Location-based property recommendations.
- Advanced investment analysis and visualization.
- Additional features for comparing properties.

## 👩‍💻 Author

**Shreya Tripathi**

B.Tech – Computer Science and Engineering (AI & ML)

## 🔗 Project Links

- **GitHub Repository:** [Real Estate Investment Advisor](https://github.com/shreya0210-jpg/Real-estate-investment-project)
- **Live Application:** [Open Project](https://real-estate-investment-project-4mhxvdya8pwuauxarxs8fm.streamlit.app/)

---

*Developed as part of an internship project to explore data analytics, machine learning, and real estate investment insights.*
