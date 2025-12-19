import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="BTC Algorithmic Trading Dashboard", layout="wide")
sns.set_style('darkgrid')

st.title("📈 BTC Algorithmic Trading Strategy - Interactive Dashboard")
st.markdown("""
This interactive dashboard demonstrates applied statistics, time series analysis, hypothesis testing, 
machine learning, and trading strategy backtesting on BTC data.
""")

# =========================
# Interactive Parameters (moved to top)
# =========================
st.sidebar.header("Interactive Parameters")
ma_short = st.sidebar.slider("Short MA Window", min_value=5, max_value=30, value=7, step=1)
ma_long = st.sidebar.slider("Long MA Window", min_value=30, max_value=120, value=30, step=1)
threshold = st.sidebar.slider("Trading Signal Threshold (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

# Convert threshold to decimal
threshold = threshold / 100

# =========================
# 1. Data Acquisition
# =========================
st.header("Step 1: Data Acquisition")

# Add caching to improve performance
@st.cache_data
def load_data():
    btc = yf.download('BTC-USD', start='2020-01-01', end='2024-12-19', progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    return btc

try:
    btc = load_data()
    st.write("Data Snapshot:")
    st.dataframe(btc.head(), use_container_width=True)
    st.write(f"**Period:** {btc.index[0].date()} to {btc.index[-1].date()}")
    st.write(f"**Total observations:** {len(btc)}")
    st.write(f"**Missing values:** {btc.isnull().sum().sum()}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# =========================
# 2. Feature Engineering
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

st.write(f"**Features created:** {df.shape[1]} columns, {df.shape[0]} rows after cleaning")
st.write(f"**Date range after feature engineering:** {df.index[0].date()} to {df.index[-1].date()}")

# =========================
# 3. Machine Learning
# =========================
st.header("Step 3: ML Model Training & Prediction")

feature_cols = [f'MA_{ma_short}', f'MA_{ma_long}', 'Volatility_7', 'Volatility_30', 
                'Momentum_7', 'RSI', 'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3']

X = df[feature_cols]
y = df['Target']

# Check for any remaining NaN values
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    st.warning("Found NaN values in features or target. Dropping them...")
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    df = df[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

split_idx = int(len(df) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train model
with st.spinner("Training Random Forest model..."):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
st.metric("Random Forest Test R²", f"{r2:.4f}")

# =========================
# 4. Trading Strategy Backtest
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
strategy_sharpe = (df_test['Strategy_Returns'].mean() / df_test['Strategy_Returns'].std()) * np.sqrt(252) if df_test['Strategy_Returns'].std() > 0 else 0
market_return = (df_test['Cumulative_Market'].iloc[-1] - 1) * 100
strategy_return = (df_test['Cumulative_Strategy'].iloc[-1] - 1) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Buy & Hold Return", f"{market_return:.2f}%")
col2.metric("Strategy Return", f"{strategy_return:.2f}%", f"{strategy_return - market_return:.2f}%")
col3.metric("Strategy Sharpe Ratio", f"{strategy_sharpe:.2f}")

# Trade statistics
num_trades = (df_test['Signal'].diff() != 0).sum()
long_trades = (df_test['Signal'] == 1).sum()
short_trades = (df_test['Signal'] == -1).sum()

st.write("**Trading Statistics:**")
col1, col2, col3 = st.columns(3)
col1.metric("Total Signals", num_trades)
col2.metric("Long Positions", long_trades)
col3.metric("Short Positions", short_trades)

# =========================
# 5. Visualizations
# =========================
st.header("Step 5: Visualizations")

# Price with MA
st.subheader("BTC Price with Moving Averages")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['Close'], label='BTC Price', alpha=0.7, linewidth=1.5)
ax.plot(df.index, df[f'MA_{ma_short}'], label=f'MA {ma_short}', alpha=0.8, linewidth=1.2)
ax.plot(df.index, df[f'MA_{ma_long}'], label=f'MA {ma_long}', alpha=0.8, linewidth=1.2)
ax.set_title("BTC Price with Moving Averages", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Cumulative returns
st.subheader("Cumulative Returns Comparison")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df_test.index, df_test['Cumulative_Market'], label='Buy & Hold', linewidth=2)
ax2.plot(df_test.index, df_test['Cumulative_Strategy'], label='ML Strategy', linewidth=2)
ax2.set_title("Cumulative Returns: Buy & Hold vs ML Strategy", fontsize=14, fontweight='bold')
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Return")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# Feature importance
st.subheader("Feature Importance")
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
bars = ax3.barh(importances['Feature'], importances['Importance'], color='steelblue')
ax3.set_title("Random Forest Feature Importance", fontsize=14, fontweight='bold')
ax3.set_xlabel("Importance")
ax3.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# Returns distribution
st.subheader("Strategy Returns Distribution")
fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 4))

ax4.hist(df_test['Returns'], bins=50, alpha=0.7, label='Market Returns', edgecolor='black')
ax4.hist(df_test['Strategy_Returns'], bins=50, alpha=0.7, label='Strategy Returns', edgecolor='black')
ax4.set_title("Returns Distribution")
ax4.set_xlabel("Returns")
ax4.set_ylabel("Frequency")
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5.plot(df_test.index, df_test['Signal'], drawstyle='steps-post', linewidth=1.5)
ax5.set_title("Trading Signals Over Time")
ax5.set_xlabel("Date")
ax5.set_ylabel("Signal (-1: Short, 0: Hold, 1: Long)")
ax5.grid(True, alpha=0.3)
ax5.set_ylim(-1.5, 1.5)

plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.success("✅ Analysis complete! Adjust parameters in the sidebar to see how they affect the strategy.")
