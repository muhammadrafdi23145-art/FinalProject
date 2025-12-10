import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===============================
# LOAD MODEL & SCALER
# ===============================
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===============================
# STREAMLIT UI
# ===============================

st.title("Prediksi Harga Saham BBRI H+1")
st.write("Website ini memprediksi harga saham BBRI untuk H+1 menggunakan Linear Regression, Random Forest, dan SVR.")

st.subheader("Input Data Hari Ini")

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

    # Logaritma
    log_open  = np.log(open_val)
    log_high  = np.log(high_val)
    log_low   = np.log(low_val)
    log_close = np.log(close_val)
    log_prev_close = np.log(prev_close)
    log_vol   = np.log(vol_val + 1)
    log_prev_vol = np.log(prev_vol + 1)
    log_ma5 = np.log(ma5_val)

    # Fitur
    features = pd.DataFrame([{
        'Feat_Return_1d': log_close - log_prev_close,
        'Feat_Close_Open': log_close - log_open,
        'Feat_High_Low': log_high - log_low,
        'Feat_Dist_MA5': log_close - log_ma5,
        'Feat_Vol_Change': log_vol - log_prev_vol
    }])

    scaled = scaler.transform(features)

    # Prediksi return
    pred_lr  = models['Linear Regression'].predict(input_scaled)[0]
    pred_rf  = models['Random Forest'].predict(input_df)[0]   # PENTING
    pred_svr = models['SVR'].predict(input_scaled)[0]

    # Konversi return → harga besok
    price_lr  = close_val * math.exp(pred_lr)
    price_rf  = close_val * math.exp(pred_rf)
    price_svr = close_val * math.exp(pred_svr)

    st.subheader("Hasil Prediksi")

    st.success(f"Linear Regression: Rp {price_lr:,.0f}")
    st.info(f"Random Forest: Rp {price_rf:,.0f}")
    st.warning(f"SVR: Rp {price_svr:,.0f}")

# ============================================
# GRAFIK DAN EVALUASI (VERSI GOOGLE COLAB ASLI)
# ============================================

st.subheader("📉 Grafik Prediksi Model vs Harga Asli (200 Hari Terakhir)")

try:
    # Load dataset lagi (pastikan sama)
    df = pd.read_csv("BBRI.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Close_Log'] = np.log(df['close'])

    # Feature-engineering ulang (harus sama 1:1 dengan training & Colab)
    df['MA5_Log'] = df['Close_Log'].rolling(5).mean()
    df['Target_Return'] = df['Close_Log'].shift(-1) - df['Close_Log']
    df['Feat_Return_1d'] = df['Close_Log'] - df['Close_Log'].shift(1)
    df['Feat_Close_Open'] = df['Close_Log'] - np.log(df['open'])
    df['Feat_High_Low'] = np.log(df['high']) - np.log(df['low'])
    df['Feat_Dist_MA5'] = df['Close_Log'] - df['MA5_Log']
    df['Feat_Vol_Change'] = np.log(df['volume']+1) - np.log(df['volume'].shift(1)+1)
    df.dropna(inplace=True)

    features = ['Feat_Return_1d','Feat_Close_Open','Feat_High_Low','Feat_Dist_MA5','Feat_Vol_Change']
    X = df[features]

    # Sama seperti Colab: split TIDAK di-shuffle
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

    # Ambil log harga actual besok
    actual_price = np.exp(df.loc[X_test.index, 'Close_Log'].shift(-1))
    valid_idx = ~np.isnan(actual_price)
    actual_price = actual_price[valid_idx]
    test_dates = df.loc[X_test.index, 'timestamp'][valid_idx]

    # Prediksi return dari 3 model
    X_test_scaled = scaler.transform(X_test)

    pred_lr = models['Linear Regression'].predict(X_test_scaled)[valid_idx]
    pred_rf = models['Random Forest'].predict(X_test)[valid_idx]
    pred_svr = models['SVR'].predict(X_test_scaled)[valid_idx]

    # Konversi ke harga
    close_today = np.exp(df.loc[X_test.index, 'Close_Log'])[valid_idx]

    price_lr  = close_today * np.exp(pred_lr)
    price_rf  = close_today * np.exp(pred_rf)
    price_svr = close_today * np.exp(pred_svr)

    last_n = 200
    plt.figure(figsize=(12,5))
    plt.plot(test_dates[-last_n:], actual_price[-last_n:], label='Harga Asli (Actual)', linewidth=2, color='black')
    plt.plot(test_dates[-last_n:], price_lr[-last_n:], '--', label='Linear Regression')
    plt.plot(test_dates[-last_n:], price_rf[-last_n:], '-', label='Random Forest')
    plt.plot(test_dates[-last_n:], price_svr[-last_n:], ':', label='SVR')

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

    # ============================
    # EVALUASI — Sama Dengan Colab
    # ============================
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    st.subheader("📊 Evaluasi Model (RMSE, MAE, R²) — Versi Test Set (Colab)")

    eval_data = [
        ["Linear Regression",
         mean_squared_error(actual_price, price_lr, squared=False),
         mean_absolute_error(actual_price, price_lr),
         r2_score(actual_price, price_lr)],
         
        ["Random Forest",
         mean_squared_error(actual_price, price_rf, squared=False),
         mean_absolute_error(actual_price, price_rf),
         r2_score(actual_price, price_rf)],

        ["SVR",
         mean_squared_error(actual_price, price_svr, squared=False),
         mean_absolute_error(actual_price, price_svr),
         r2_score(actual_price, price_svr)]
    ]

    eval_df = pd.DataFrame(eval_data, columns=["Model", "RMSE", "MAE", "R²"])
    st.dataframe(eval_df)

except Exception as e:
    st.warning("Grafik & evaluasi tidak bisa ditampilkan: {e}")

