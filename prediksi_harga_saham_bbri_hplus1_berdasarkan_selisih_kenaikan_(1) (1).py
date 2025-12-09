import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ======================================
# LOAD MODEL & SCALER
# ======================================
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ======================================
#   STREAMLIT UI
# ======================================
st.title("Prediksi Harga Saham BBRI H+1")

st.write("Masukkan data hari ini untuk memprediksi harga besok.")

open_val  = st.number_input("Open Hari Ini", min_value=0.0, value=5000.0)
high_val  = st.number_input("High Hari Ini", min_value=0.0, value=5050.0)
low_val   = st.number_input("Low Hari Ini", min_value=0.0, value=4980.0)
close_val = st.number_input("Close Hari Ini", min_value=0.0, value=5020.0)
vol_val   = st.number_input("Volume Hari Ini", min_value=0.0, value=1000000.0)

st.subheader("Data Tambahan")
prev_close = st.number_input("Close Kemarin", min_value=0.0, value=5000.0)
prev_vol   = st.number_input("Volume Kemarin", min_value=0.0, value=1000000.0)
ma5_val    = st.number_input("MA 5 Hari", min_value=0.0, value=5010.0)

if st.button("Prediksi Harga Besok (H+1)"):

    # Log transform
    log_open  = np.log(open_val)
    log_high  = np.log(high_val)
    log_low   = np.log(low_val)
    log_close = np.log(close_val)
    log_vol   = np.log(vol_val + 1)

    log_prev_close = np.log(prev_close)
    log_prev_vol   = np.log(prev_vol + 1)
    log_ma5        = np.log(ma5_val)

    # Feature engineering
    feat_return_1d  = log_close - log_prev_close
    feat_close_open = log_close - log_open
    feat_high_low   = log_high - log_low
    feat_dist_ma5   = log_close - log_ma5
    feat_vol_change = log_vol - log_prev_vol

    input_df = pd.DataFrame([{
        "Feat_Return_1d": feat_return_1d,
        "Feat_Close_Open": feat_close_open,
        "Feat_High_Low": feat_high_low,
        "Feat_Dist_MA5": feat_dist_ma5,
        "Feat_Vol_Change": feat_vol_change
    }])

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Predict returns
    pred_lr = models["Linear Regression"].predict(input_scaled)[0]
    pred_rf = models["Random Forest"].predict(input_df)[0]
    pred_svr = models["SVR"].predict(input_scaled)[0]

    # Convert to price
    price_lr  = close_val * np.exp(pred_lr)
    price_rf  = close_val * np.exp(pred_rf)
    price_svr = close_val * np.exp(pred_svr)

    st.subheader("Hasil Prediksi")

    st.success(f"Linear Regression: Rp {price_lr:,.0f}")
    st.info(f"Random Forest: Rp {price_rf:,.0f}")
    st.warning(f"SVR: Rp {price_svr:,.0f}")
