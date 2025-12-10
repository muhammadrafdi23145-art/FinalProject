# ================================
#     TRAINING MODEL BBRI
# ================================

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# ======================================
# 1. LOAD DATA
# ======================================
df = pd.read_csv("BBRI.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --------------------------------------
# FEATURE ENGINEERING
# --------------------------------------
df['Open_Log']   = np.log(df['open'])
df['High_Log']   = np.log(df['high'])
df['Low_Log']    = np.log(df['low'])
df['Close_Log']  = np.log(df['close'])
df['Volume_Log'] = np.log(df['volume'] + 1)
df['MA5_Log']    = df['Close_Log'].rolling(5).mean()

df['Target_Return'] = df['Close_Log'].shift(-1) - df['Close_Log']

df['Feat_Return_1d'] = df['Close_Log'] - df['Close_Log'].shift(1)
df['Feat_Close_Open'] = df['Close_Log'] - df['Open_Log']
df['Feat_High_Low'] = df['High_Log'] - df['Low_Log']
df['Feat_Dist_MA5'] = df['Close_Log'] - df['MA5_Log']
df['Feat_Vol_Change'] = df['Volume_Log'] - df['Volume_Log'].shift(1)

df.dropna(inplace=True)

features = [
    'Feat_Return_1d',
    'Feat_Close_Open',
    'Feat_High_Low',
    'Feat_Dist_MA5',
    'Feat_Vol_Change'
]

X = df[features]
y = df['Target_Return']

# ======================================
# 2. TRAIN-TEST SPLIT (Time Series)
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ======================================
# 3. SCALING (Untuk LR dan SVR)
# ======================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ======================================
# 4. TRAIN 3 MODELS
# ======================================

models = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr

# Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)  # RF tidak butuh scaling
models['Random Forest'] = rf

# SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1)
svr.fit(X_train_scaled, y_train)
models['SVR'] = svr

# ======================================
# 5. SAVE MODEL DAN SCALER
# ======================================
with open("models/models.pkl", "wb") as f:
    pickle.dump(models, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training selesai! Model & scaler berhasil disimpan.")

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


# ===========================
# TABEL EVALUASI MODEL
# ===========================


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


st.subheader("Evaluasi Model (RMSE, MAE, R²)")


eval_data = []
for name in ['Linear Regression','Random Forest','SVR']:
preds = df[f'Pred_{"LR" if name=="Linear Regression" else "RF" if name=="Random Forest" else "SVR"}']
actual = df['close']


rmse = mean_squared_error(actual, preds, squared=False)
mae = mean_absolute_error(actual, preds)
r2 = r2_score(actual, preds)


eval_data.append([name, rmse, mae, r2])


eval_df = pd.DataFrame(eval_data, columns=["Model", "RMSE", "MAE", "R²"])
st.dataframe(eval_df)


except Exception as e:
st.warning(f"Grafik & evaluasi tidak bisa ditampilkan: {e}")(f"Grafik tidak bisa ditampilkan: {e}")
