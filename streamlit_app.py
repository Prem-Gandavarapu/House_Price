import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

train = pd.read_csv("train.csv")
features = train.drop(columns=["SalePrice"]).columns.tolist()

st.title("House Price Prediction")
st.caption("AI-powered house price prediction using multiple regression models")

st.subheader("Enter House Details")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    overall_qual = st.slider(
        "Overall Quality",
        1, 10, 5
    )

with col2:
    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=6000,
        value=1500
    )

with col3:
    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000
    )

with col4:
    garage_cars = st.slider(
        "Garage Capacity",
        0, 4, 2
    )

with col5:
    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

if st.button("Predict House Price", type="primary"):

    # Start with constant values
    data = {}

    for feature in features:

        if train[feature].dtype == "object":
            data[feature] = train[feature].mode()[0]
        else:
            data[feature] = train[feature].median()

    # User inputs
    data["OverallQual"] = overall_qual
    data["GrLivArea"] = gr_liv_area
    data["TotalBsmtSF"] = total_bsmt_sf
    data["GarageCars"] = garage_cars
    data["YearBuilt"] = year_built

    try:

        response = requests.post(
            API_URL,
            json=data
        )

        if response.status_code == 200:

            predictions = response.json()

            st.subheader("Model Predictions")

            cols = st.columns(5)

            for col, (model, price) in zip(
                cols,
                predictions.items()
            ):

                with col:
                    st.metric(
                        model,
                        f"${price:,.0f}"
                    )

        else:
            st.error(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Flask API is not running.")

    except Exception as e:
        st.error(str(e))