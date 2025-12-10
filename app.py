import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math

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
    pred_lr  = models['Linear Regression'].predict(scaled)[0]
    pred_rf  = models['Random Forest'].predict(features)[0]
    pred_svr = models['SVR'].predict(scaled)[0]

    # Konversi return → harga besok
    price_lr  = close_val * math.exp(pred_lr)
    price_rf  = close_val * math.exp(pred_rf)
    price_svr = close_val * math.exp(pred_svr)

    st.subheader("Hasil Prediksi")

    st.success(f"Linear Regression: Rp {price_lr:,.0f}")
    st.info(f"Random Forest: Rp {price_rf:,.0f}")
    st.warning(f"SVR: Rp {price_svr:,.0f}")

# =======================
# GRAFIK PERBANDINGAN
# =======================
st.subheader("Grafik Prediksi Model vs Harga Asli (200 Hari Terakhir)")

try:
    df = pd.read_csv("BBRI.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['Close_Log'] = np.log(df['close'])
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
    X_scaled_hist = scaler.transform(X)

    preds_hist = {
        'Linear Regression': models['Linear Regression'].predict(X_scaled_hist),
        'Random Forest': models['Random Forest'].predict(X),
        'SVR': models['SVR'].predict(X_scaled_hist)
    }

    df['Pred_LR'] = df['close'] * np.exp(preds_hist['Linear Regression'])
    df['Pred_RF'] = df['close'] * np.exp(preds_hist['Random Forest'])
    df['Pred_SVR'] = df['close'] * np.exp(preds_hist['SVR'])

    last_n = 200
    plt.figure(figsize=(12,5))
    plt.plot(df['timestamp'].tail(last_n), df['close'].tail(last_n), label='Actual', linewidth=2)
    plt.plot(df['timestamp'].tail(last_n), df['Pred_LR'].tail(last_n), '--', label='Linear Regression')
    plt.plot(df['timestamp'].tail(last_n), df['Pred_RF'].tail(last_n), label='Random Forest')
    plt.plot(df['timestamp'].tail(last_n), df['Pred_SVR'].tail(last_n), ':', label='SVR')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

except Exception as e:
    st.warning(f"Grafik tidak bisa ditampilkan: {e}")

# ===========================
# TABEL EVALUASI MODEL
# ===========================

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.subheader("📊 Evaluasi Model (RMSE, MAE, R²)")

eval_data = []

for name in ['Linear Regression', 'Random Forest', 'SVR']:
    col = (
        'Pred_LR' if name == "Linear Regression" else
        'Pred_RF' if name == "Random Forest" else
        'Pred_SVR'
    )

    preds = df[col]
    actual = df['close']

    mse = mean_squared_error(actual, preds)
    rmse = mse ** 0.5   # Perbaikan kompatibel
    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)

    eval_data.append([name, rmse, mae, r2])

eval_df = pd.DataFrame(eval_data, columns=["Model", "RMSE", "MAE", "R²"])
st.dataframe(eval_df)

