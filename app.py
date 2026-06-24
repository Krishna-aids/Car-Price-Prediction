import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# load trained model
model = pickle.load(open("car_price_pred.pkl","rb"))

# recreate polynomial transformer
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit([[0]*11])   # 11 original features

st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AI Car Price Valuation Studio")
st.markdown("Estimate the market value of a used car using Machine Learning")

st.divider()

# layout
col1, col2 = st.columns(2)

with col1:

    st.subheader("📋 Vehicle Information")

    model_car = st.number_input("Car Model Code", min_value=0, value=1)

    year = st.slider(
        "Manufacturing Year",
        2000, 2024, 2018
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )

    mileage = st.number_input(
        "Mileage (km)",
        min_value=0,
        value=20000
    )

with col2:

    st.subheader("⚙️ Engine & Fuel Details")

    tax = st.number_input("Road Tax", value=150)

    mpg = st.slider(
        "Fuel Efficiency (MPG)",
        10.0, 100.0, 40.0
    )

    engineSize = st.slider(
        "Engine Size (L)",
        0.8, 6.0, 2.0
    )

    fuel = st.selectbox(
        "Fuel Type",
        ["Diesel", "Electric", "Hybrid", "Other", "Petrol"]
    )

st.divider()

# encoding
transmission_val = 1 if transmission == "Automatic" else 0

fuelType_Electric = 1 if fuel == "Electric" else 0
fuelType_Hybrid = 1 if fuel == "Hybrid" else 0
fuelType_Other = 1 if fuel == "Other" else 0
fuelType_Petrol = 1 if fuel == "Petrol" else 0

# prediction
if st.button("Estimate Car Price 🚀"):

    features = np.array([[model_car,
                          year,
                          transmission_val,
                          mileage,
                          tax,
                          mpg,
                          engineSize,
                          fuelType_Electric,
                          fuelType_Hybrid,
                          fuelType_Other,
                          fuelType_Petrol]])

    # polynomial transform
    features_poly = poly.transform(features)

    prediction = model.predict(features_poly)[0]

    st.success("### Estimated Car Market Value")

    st.metric(
        label="Predicted Price",
        value=f"${prediction:,.2f}"
    )

    st.balloons()

st.markdown("---")
st.caption("AI Vehicle Valuation System • Powered by Machine Learning")