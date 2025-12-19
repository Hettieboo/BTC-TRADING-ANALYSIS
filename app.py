import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="BTC Algorithmic Trading Dashboard", layout="wide")
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

st.title("📈 BTC Algorithmic Trading Strategy - Interactive Dashboard")
st.markdown("""
This interactive dashboard demonstrates applied statistics, time series analysis, hypothesis testing, 
machine learning, and trading strategy backtesting on BTC data.
""")

# =========================
# 1. Data Acquisition
# =========================
st.header("Step 1: Data Acquisition")
btc = yf.download('BTC-USD', start='2020-01-01', end='2024-12-19', progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

st.write("Data Snapshot:", btc.head())
st.write(f"Period: {btc.index[0].date()} to {btc.index[-1].date()}")
st.write(f"Missing values: {btc.isnull().sum().sum()}")

# =========================
# 2. Interactive Parameters
# =========================
st.sidebar.header("Interactive Parameters")
ma_short = st.sidebar.slider("Short MA Window", min_value=5, max_value=30, value=7, step=1)
ma_long = st.sidebar.slider("Long MA Window", min_value=30, max_value=120, value=30, step=1)
threshold = st.sidebar.slider("Trading Signal Threshold (%)", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001)

# =========================
# 3. Feature Engineering
# =========================
st.header("Step 2: Feature Engineering")
df = btc.copy()
df['Returns'] = df['Close'].pct_change()
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# Moving averages
df[f'MA_{ma_short}'] = df['Close'].rolling(ma_short).mean()
df[f'MA_{ma_long}'] = df['Close'].rolling(ma_long).mean()
df['Volatility_7'] = df['Returns'].rolling(7).std()
df['Volatility_30'] = df['Returns'].rolling(30).std()
df['Momentum_7'] = df['Close'] - df['Close'].shift(7)

# RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
df['RSI'] = calculate_rsi(df['Close'])

# Lag features and target
for i in [1, 2, 3]:
    df[f'Return_Lag_{i}'] = df['Returns'].shift(i)
df['Target'] = df['Returns'].shift(-1)
df = df.dropna()

st.write(f"Features created: {df.shape[1]} columns, {df.shape[0]} rows")

# =========================
# 4. Machine Learning
# =========================
st.header("Step 3: ML Model Training & Prediction")
feature_cols = [f'MA_{ma_short}', f'MA_{ma_long}', 'Volatility_7', 'Volatility_30', 
                'Momentum_7', 'RSI', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3']

X = df[feature_cols]
y = df['Target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

split_idx = int(len(df) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

st.write(f"Random Forest Test R²: {r2_score(y_test, y_pred):.4f}")

# =========================
# 5. Trading Strategy Backtest
# =========================
st.header("Step 4: Trading Strategy Backtest")
df_test = df.iloc[split_idx:].copy()
df_test['Predicted_Return'] = y_pred
df_test['Signal'] = 0
df_test.loc[df_test['Predicted_Return'] > threshold, 'Signal'] = 1
df_test.loc[df_test['Predicted_Return'] < -threshold, 'Signal'] = -1
df_test['Strategy_Returns'] = df_test['Signal'] * df_test['Returns']
df_test['Cumulative_Market'] = (1 + df_test['Returns']).cumprod()
df_test['Cumulative_Strategy'] = (1 + df_test['Strategy_Returns']).cumprod()

# Performance metrics
strategy_sharpe = (df_test['Strategy_Returns'].mean()/df_test['Strategy_Returns'].std())*np.sqrt(252)
st.write({
    "Buy & Hold Return (%)": round((df_test['Cumulative_Market'].iloc[-1]-1)*100, 2),
    "Strategy Return (%)": round((df_test['Cumulative_Strategy'].iloc[-1]-1)*100, 2),
    "Strategy Sharpe Ratio": round(strategy_sharpe, 2)
})

# =========================
# 6. Visualizations
# =========================
st.header("Step 5: Visualizations")

# Price with MA
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df.index, df['Close'], label='BTC Price', alpha=0.7)
ax.plot(df.index, df[f'MA_{ma_short}'], label=f'MA {ma_short}', alpha=0.7)
ax.plot(df.index, df[f'MA_{ma_long}'], label=f'MA {ma_long}', alpha=0.7)
ax.set_title("BTC Price with Moving Averages")
ax.legend()
st.pyplot(fig)

# Cumulative returns
st.line_chart(df_test[['Cumulative_Market', 'Cumulative_Strategy']])

# Feature importance
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.barh(importances['Feature'], importances['Importance'], color='steelblue')
ax2.set_title("Random Forest Feature Importance")
st.pyplot(fig2)
