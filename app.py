
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Real Estate Investment Analysis",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# LOAD DATA AND MODELS
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("data/feature_engineered_data.csv")


@st.cache_data
def load_results():
    results = pd.read_csv("data/model_comparison.csv")
    importance = pd.read_csv("data/feature_importance.csv")
    return results, importance


@st.cache_data
def load_classification_data():
    return pd.read_csv("data/investment_classification_data.csv")


@st.cache_resource
def load_classifier():
    return joblib.load("models/good_investment_classifier.pkl")


df = load_data()
results, importance = load_results()
classification_df = load_classification_data()
classifier = load_classifier()

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1 {
    color: #16324f;
    font-weight: 700;
}
h2, h3 {
    color: #24527a;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
[data-testid="stSidebar"] {
    background-color: #eaf0f6;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🏠 Real Estate Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Market Analysis",
        "Model Performance",
        "Investment Advisor"
    ]
)

st.sidebar.divider()

# ==========================================
# PROPERTY FILTERS
# ==========================================

st.sidebar.header("Property Filters")

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(df["State"].dropna().unique().tolist())
)

selected_type = st.sidebar.selectbox(
    "Property Type",
    ["All"] + sorted(df["Property_Type"].dropna().unique().tolist())
)

selected_bhk = st.sidebar.selectbox(
    "Select BHK",
    ["All"] + sorted(df["BHK"].dropna().unique().tolist())
)

filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["State"] == selected_state
    ]

if selected_type != "All":
    filtered_df = filtered_df[
        filtered_df["Property_Type"] == selected_type
    ]

if selected_bhk != "All":
    filtered_df = filtered_df[
        filtered_df["BHK"] == selected_bhk
    ]

# ==========================================
# DASHBOARD HEADER
# ==========================================

st.title("🏠 Real Estate Investment Analysis")

st.write(
    "An interactive platform to explore housing prices, "
    "property characteristics, investment suitability "
    "and future price scenarios."
)

# ==========================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard Overview":

    st.header("Property Overview")

    if filtered_df.empty:
        st.warning("No properties found for the selected filters.")

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Properties",
            f"{len(filtered_df):,}"
        )

        col2.metric(
            "Average Property Price",
            f"₹{filtered_df['Price_in_Lakhs'].mean():.2f} Lakhs"
        )

        col3.metric(
            "Average Property Size",
            f"{filtered_df['Size_in_SqFt'].mean():.0f} Sq.Ft."
        )

        st.subheader("Property Dataset")

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True
        )

        st.caption("Showing the first 100 filtered properties.")

# ==========================================
# PAGE 2: MARKET ANALYSIS
# ==========================================

elif page == "Market Analysis":

    st.header("📊 Property Market Analysis")

    if filtered_df.empty:
        st.warning("No properties found for the selected filters.")

    else:

        st.subheader("Property Price Distribution")

        price_bins = pd.cut(
            filtered_df["Price_in_Lakhs"],
            bins=20
        ).value_counts().sort_index()

        price_chart = pd.DataFrame({
            "Price Range": [
                str(interval) for interval in price_bins.index
            ],
            "Number of Properties": price_bins.to_numpy()
        })

        st.bar_chart(
            price_chart,
            x="Price Range",
            y="Number of Properties"
        )

        st.subheader("Average Price by Property Type")

        avg_price = filtered_df.groupby(
            "Property_Type"
        )["Price_in_Lakhs"].mean()

        st.bar_chart(avg_price)

        st.subheader("Property Distribution by BHK")

        bhk_count = filtered_df["BHK"].value_counts().sort_index()

        st.bar_chart(bhk_count)

        st.subheader("Average Property Price by State")

        state_price = filtered_df.groupby(
            "State"
        )["Price_in_Lakhs"].mean().sort_values(
            ascending=False
        )

        st.bar_chart(state_price)

# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("🤖 Machine Learning Model Performance")

    st.subheader("Regression Model Comparison")

    st.dataframe(
        results,
        use_container_width=True
    )

    best_model = results.loc[results["MAE"].idxmin()]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Lowest MAE Model",
        best_model["Model"]
    )

    col2.metric(
        "MAE",
        f"{best_model['MAE']:.2f} Lakhs"
    )

    col3.metric(
        "R² Score",
        f"{best_model['R2 Score']:.4f}"
    )

    st.subheader("R² Score Comparison")

    st.bar_chart(
        results.set_index("Model")["R2 Score"]
    )

    st.subheader("Top 10 Important Features")

    st.bar_chart(
        importance.head(10).set_index("Feature")["Importance"]
    )

    st.header("📉 Residual Analysis")

    st.subheader("Actual vs Predicted Prices")

    st.image(
        "data/actual_vs_predicted.png",
        use_container_width=True
    )

    st.subheader("Residual Distribution")

    st.image(
        "data/residual_distribution.png",
        use_container_width=True
    )

    st.subheader("Residuals vs Predicted Prices")

    st.image(
        "data/residuals_vs_predicted.png",
        use_container_width=True
    )

    st.info(
        "The current regression models have very low R² scores. "
        "Their predictions should not be treated as reliable "
        "property investment estimates."
    )

# ==========================================
# PAGE 4: INVESTMENT ADVISOR
# ==========================================

elif page == "Investment Advisor":

    st.header("🏡 Real Estate Investment Advisor")

    st.write(
        "Enter basic property details to estimate whether "
        "the property is relatively affordable in its market."
    )

    st.info(
        "This classifier predicts relative affordability "
        "based on the dataset's labeling rule. It does not "
        "guarantee investment profitability."
    )

    with st.form("investment_form"):

        col1, col2 = st.columns(2)

        with col1:

            state = st.selectbox(
                "State",
                sorted(classification_df["State"].dropna().unique())
            )

            city_options = sorted(
                classification_df["City"].dropna().unique()
            )

            city = st.selectbox(
                "City",
                city_options
            )

            property_type = st.selectbox(
                "Property Type",
                sorted(
                    classification_df["Property_Type"].dropna().unique()
                )
            )

        with col2:

            bhk = st.selectbox(
                "Number of BHK",
                sorted(classification_df["BHK"].dropna().unique())
            )

            size = st.number_input(
                "Property Size (Sq. Ft.)",
                min_value=100,
                max_value=10000,
                value=1200,
                step=50
            )

        submitted = st.form_submit_button(
            "Check Investment Suitability"
        )

    if submitted:

        excluded_columns = [
            "Good_Investment",
            "Price_per_SqFt",
            "Price_in_Lakhs",
            "Price_Per_BHK",
            "Locality"
        ]

        feature_columns = [
            col for col in classification_df.columns
            if col not in excluded_columns
        ]

        input_data = classification_df[
            feature_columns
        ].iloc[[0]].copy()

        input_data["State"] = state
        input_data["City"] = city
        input_data["Property_Type"] = property_type
        input_data["BHK"] = bhk
        input_data["Size_in_SqFt"] = size

        if "Area_Per_BHK" in input_data.columns:
            input_data["Area_Per_BHK"] = size / bhk

        prediction = classifier.predict(input_data)[0]

        st.subheader("Investment Suitability Result")

        if prediction == 1:
            st.success(
                "Good Investment: The property is predicted "
                "to be relatively affordable."
            )
        else:
            st.warning(
                "Not a Good Investment: The property is not "
                "predicted to be relatively affordable."
            )

        st.caption(
            "This is a model-based classification, not a "
            "financial recommendation."
        )

    # ======================================
    # FIVE-YEAR PRICE PROJECTION
    # ======================================

    st.divider()

    st.header("📈 Five-Year Property Price Projection")

    st.write(
        "Estimate a property's possible future value "
        "using different annual appreciation assumptions."
    )

    current_price = st.number_input(
        "Current Property Price (Lakhs)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=5.0
    )

    growth_scenario = st.selectbox(
        "Select Appreciation Scenario",
        [
            "Conservative (5%)",
            "Moderate (8%)",
            "Optimistic (10%)"
        ]
    )

    growth_rates = {
        "Conservative (5%)": 0.05,
        "Moderate (8%)": 0.08,
        "Optimistic (10%)": 0.10
    }

    rate = growth_rates[growth_scenario]

    future_price = current_price * (1 + rate) ** 5

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Price",
        f"₹{current_price:.2f} Lakhs"
    )

    col2.metric(
        "Annual Appreciation",
        f"{rate * 100:.0f}%"
    )

    col3.metric(
        "Estimated Price After 5 Years",
        f"₹{future_price:.2f} Lakhs"
    )

    scenario_results = pd.DataFrame({
        "Scenario": [
            "Conservative (5%)",
            "Moderate (8%)",
            "Optimistic (10%)"
        ],
        "Projected Price (Lakhs)": [
            current_price * (1.05 ** 5),
            current_price * (1.08 ** 5),
            current_price * (1.10 ** 5)
        ]
    })

    st.subheader("Compare All Three Scenarios")

    st.dataframe(
        scenario_results.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        scenario_results.set_index("Scenario")
    )

    st.caption(
        "These are illustrative compound-growth scenarios "
        "based on assumed appreciation rates, not ML predictions "
        "or guaranteed future prices."
    )

# ==========================================
# SIDEBAR FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "Real Estate Investment Analysis | Internship Project"
)
