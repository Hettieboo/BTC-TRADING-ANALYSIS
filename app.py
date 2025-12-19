import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config MUST be first
st.set_page_config(
    page_title="BTC AI Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# =========================
# Generate Synthetic Data
# =========================
@st.cache_data
def generate_btc_data(n_days=1500):
    """Generate realistic synthetic BTC data"""
    np.random.seed(42)
    date_range = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate price with trend and volatility
    returns = np.random.normal(0.0005, 0.025, n_days)
    trend = np.linspace(0, 0.3, n_days)
    seasonal = np.sin(np.linspace(0, 8*np.pi, n_days)) * 0.008
    returns += seasonal + trend * 0.0008
    
    prices = 30000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=date_range)
    df['Close'] = prices
    df['Open'] = df['Close'] * (1 + np.random.uniform(-0.008, 0.008, n_days))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    df['Volume'] = np.random.uniform(8e8, 2e9, n_days)
    
    return df

# =========================
# Header
# =========================
st.title("⚡ BTC AI Trading Dashboard")
st.markdown("### Advanced Algorithmic Trading with Machine Learning")
st.info("🎮 **DEMO MODE**: Using synthetic BTC data for demonstration")

# =========================
# Sidebar
# =========================
st.sidebar.header("🎛️ Strategy Parameters")
ma_short = st.sidebar.slider("Short MA", 5, 30, 7)
ma_long = st.sidebar.slider("Long MA", 30, 120, 30)
rsi_period = st.sidebar.slider("RSI Period", 7, 28, 14)
threshold = st.sidebar.slider("Threshold (%)", 0.0, 1.0, 0.05, 0.01) / 100

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("ML Model", ["Random Forest", "Gradient Boosting"])
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

if st.sidebar.button("🔄 Regenerate Data"):
    st.cache_data.clear()
    st.rerun()

# =========================
# Load and Process Data
# =========================
with st.spinner("📊 Generating data..."):
    btc = generate_btc_data()

# Feature engineering
@st.cache_data
def add_features(df, ma_short, ma_long, rsi_period):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    
    # Moving averages
    data[f'MA_{ma_short}'] = data['Close'].rolling(ma_short).mean()
    data[f'MA_{ma_long}'] = data['Close'].rolling(ma_long).mean()
    data['MA_Diff'] = data[f'MA_{ma_short}'] - data[f'MA_{ma_long}']
    
    # Volatility
    data['Volatility_7'] = data['Returns'].rolling(7).std()
    data['Volatility_30'] = data['Returns'].rolling(30).std()
    
    # Momentum
    data['Momentum_7'] = data['Close'] - data['Close'].shift(7)
    data['Momentum_14'] = data['Close'] - data['Close'].shift(14)
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Mid'] - (data['BB_Std'] * 2)
    
    # Lags
    for i in [1, 2, 3]:
        data[f'Lag_{i}'] = data['Returns'].shift(i)
    
    data['Target'] = data['Returns'].shift(-1)
    return data.dropna()

df = add_features(btc, ma_short, ma_long, rsi_period)

# =========================
# Metrics Row
# =========================
latest = df.iloc[-1]
prev = df.iloc[-2]

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 BTC Price", f"${latest['Close']:,.0f}", f"{(latest['Close']-prev['Close'])/prev['Close']*100:+.2f}%")
col2.metric("📈 RSI", f"{latest['RSI']:.1f}", "Overbought" if latest['RSI']>70 else "Oversold" if latest['RSI']<30 else "Neutral")
col3.metric("💹 Volatility", f"{latest['Volatility_7']*100:.2f}%")
col4.metric("🎯 Signal", "🟢 Bullish" if latest[f'MA_{ma_short}']>latest[f'MA_{ma_long}'] else "🔴 Bearish")

st.markdown("---")

# =========================
# ML Model
# =========================
feature_cols = [f'MA_{ma_short}', f'MA_{ma_long}', 'MA_Diff', 'Volatility_7', 'Volatility_30',
                'Momentum_7', 'Momentum_14', 'RSI', 'Lag_1', 'Lag_2', 'Lag_3']

X = df[feature_cols]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_idx = int(len(df) * (1 - test_size))
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Train model
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
else:
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

# Backtest
df_test = df.iloc[split_idx:].copy()
df_test['Predicted'] = y_pred
df_test['Signal'] = 0
df_test.loc[df_test['Predicted'] > threshold, 'Signal'] = 1
df_test.loc[df_test['Predicted'] < -threshold, 'Signal'] = -1
df_test['Strat_Returns'] = df_test['Signal'] * df_test['Returns']
df_test['Cum_Market'] = (1 + df_test['Returns']).cumprod()
df_test['Cum_Strategy'] = (1 + df_test['Strat_Returns']).cumprod()

market_ret = (df_test['Cum_Market'].iloc[-1] - 1) * 100
strategy_ret = (df_test['Cum_Strategy'].iloc[-1] - 1) * 100
sharpe = (df_test['Strat_Returns'].mean() / df_test['Strat_Returns'].std()) * np.sqrt(252) if df_test['Strat_Returns'].std() > 0 else 0

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Signals", "🤖 ML Model", "📉 Performance"])

with tab1:
    st.subheader("Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index[-300:], df['Close'][-300:], label='Price', linewidth=2, color='#8a5cf6')
    ax.plot(df.index[-300:], df[f'MA_{ma_short}'][-300:], label=f'MA{ma_short}', linewidth=2, color='#10b981')
    ax.plot(df.index[-300:], df[f'MA_{ma_long}'][-300:], label=f'MA{ma_long}', linewidth=2, color='#f59e0b')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volume")
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['#ef4444' if r < 0 else '#10b981' for r in df['Returns'][-300:]]
        ax.bar(df.index[-300:], df['Volume'][-300:], color=colors, alpha=0.6, width=1)
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Returns Distribution")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(df['Returns'].dropna()*100, bins=50, color='#8a5cf6', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Returns (%)')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RSI Indicator")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[-200:], df['RSI'][-200:], linewidth=2, color='#8a5cf6')
        ax.axhline(70, color='#ef4444', linestyle='--', linewidth=2, label='Overbought')
        ax.axhline(30, color='#10b981', linestyle='--', linewidth=2, label='Oversold')
        ax.fill_between(df.index[-200:], 70, 100, alpha=0.2, color='red')
        ax.fill_between(df.index[-200:], 0, 30, alpha=0.2, color='green')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Bollinger Bands")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[-200:], df['Close'][-200:], label='Price', linewidth=2)
        ax.plot(df.index[-200:], df['BB_Upper'][-200:], '--', color='#ef4444', label='Upper')
        ax.plot(df.index[-200:], df['BB_Lower'][-200:], '--', color='#10b981', label='Lower')
        ax.fill_between(df.index[-200:], df['BB_Lower'][-200:], df['BB_Upper'][-200:], alpha=0.2)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("Training Samples", f"{len(X_train):,}")
    col3.metric("Test Samples", f"{len(X_test):,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance['Feature'], importance['Importance'], color='#8a5cf6')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Predictions vs Actual")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, color='#8a5cf6', s=30)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab4:
    st.subheader("Strategy Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏦 Market Return", f"{market_ret:.2f}%")
    col2.metric("🚀 Strategy Return", f"{strategy_ret:.2f}%", f"{strategy_ret-market_ret:+.2f}%")
    col3.metric("📈 Sharpe Ratio", f"{sharpe:.2f}")
    
    trades = len(df_test[df_test['Signal'] != 0])
    col4.metric("🎯 Total Trades", trades)
    
    st.subheader("Cumulative Returns")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_test.index, df_test['Cum_Market'], label='Buy & Hold', linewidth=3, color='#ef4444')
    ax.plot(df_test.index, df_test['Cum_Strategy'], label='ML Strategy', linewidth=3, color='#10b981')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylabel('Cumulative Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading Signals")
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['#10b981' if s==1 else '#ef4444' if s==-1 else '#6b7280' for s in df_test['Signal']]
        ax.scatter(df_test.index, df_test['Signal'], c=colors, s=20, alpha=0.6)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Short', 'Hold', 'Long'])
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Returns Comparison")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(df_test['Returns']*100, bins=40, alpha=0.6, label='Market', color='#ef4444')
        ax.hist(df_test['Strat_Returns']*100, bins=40, alpha=0.6, label='Strategy', color='#10b981')
        ax.axvline(0, color='white', linestyle='--', linewidth=2)
        ax.legend()
        ax.set_xlabel('Returns (%)')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Footer
st.markdown("---")
st.markdown(f"🚀 ML-Powered Trading | {len(df):,} Data Points | Model: {model_choice} | Last Update: {df.index[-1].strftime('%Y-%m-%d')}")
