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
with open("models.pkl", "wb") as f:
    pickle.dump(models, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training selesai! Model & scaler berhasil disimpan.")
