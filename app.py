import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Product Return Risk Predictor", layout="centered")

st.title("📦 Product Return Risk Prediction")

with open("return_risk_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]
encoders = artifacts["encoders"]

st.subheader("Enter Order Details")

price = st.number_input("Product Price", 10.0, 1000.0)
discount = st.slider("Discount (%)", 0, 50)
shipping_days = st.slider("Shipping Days", 1, 20)
rating = st.slider("Customer Rating", 1.0, 5.0)
previous_returns = st.number_input("Previous Returns", 0, 20)
account_age = st.number_input("Account Age (days)", 30, 3000)

category = st.selectbox("Product Category", encoders["category"].classes_)
payment = st.selectbox("Payment Method", encoders["payment"].classes_)
day = st.selectbox("Order Day", encoders["day"].classes_)

if st.button("Predict Return Risk"):
    input_data = pd.DataFrame([{
        "product_price": price,
        "discount_percent": discount,
        "order_hour": 12,
        "shipping_days": shipping_days,
        "customer_reviews": 10,
        "customer_rating": rating,
        "previous_returns": previous_returns,
        "account_age_days": account_age,
        "product_category_encoded": encoders["category"].transform([category])[0],
        "payment_method_encoded": encoders["payment"].transform([payment])[0],
        "order_day_encoded": encoders["day"].transform([day])[0],
        "is_expensive": int(price > 300),
        "price_per_discount": price / (discount + 1),
        "return_rate": previous_returns / (account_age/30 + 1),
        "is_frequent_returner": int(previous_returns > 2),
        "customer_engagement": rating * 10,
        "is_weekend": int(day in ["Saturday","Sunday"]),
        "is_peak_hour": 1,
        "is_delayed_shipping": int(shipping_days > 7),
        "high_discount": int(discount > 20),
        "is_cod": int(payment == "COD")
    }])

    X = scaler.transform(input_data[features])
    prob = model.predict_proba(X)[0][1]

    st.success(f"🔮 Return Probability: **{prob:.2%}**")
