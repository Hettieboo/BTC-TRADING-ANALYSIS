import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="BTC AI Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, rgba(138, 92, 246, 0.2), rgba(118, 75, 162, 0.2));
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(138, 92, 246, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(138, 92, 246, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        border: 1px solid rgba(138, 92, 246, 0.3);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8a5cf6, #764ba2);
        border: 1px solid #8a5cf6;
    }
    h1, h2, h3 {
        color: #8a5cf6 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("⚡ BTC AI Trading Dashboard")
st.markdown("### Advanced Algorithmic Trading with Machine Learning")

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("🎛️ Strategy Parameters")
st.sidebar.markdown("---")

ma_short = st.sidebar.slider("Short MA Window", 5, 30, 7, help="Short-term moving average period")
ma_long = st.sidebar.slider("Long MA Window", 30, 120, 30, help="Long-term moving average period")
rsi_period = st.sidebar.slider("RSI Period", 7, 28, 14, help="RSI calculation period")
threshold = st.sidebar.slider("Signal Threshold (%)", 0.0, 1.0, 0.05, 0.01, help="Trading signal threshold") / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
model_choice = st.sidebar.selectbox("ML Model", ["Random Forest", "Gradient Boosting"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# =========================
# Data Loading
# =========================
@st.cache_data(ttl=3600)
def load_btc_data(start='2020-01-01', end='2024-12-19'):
    try:
        btc = yf.download('BTC-USD', start=start, end=end, progress=False)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)
        return btc
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

with st.spinner("📊 Loading BTC data..."):
    btc = load_btc_data()

if btc is None or btc.empty:
    st.error("❌ Failed to load data. Please check your internet connection.")
    st.stop()

# =========================
# Feature Engineering
# =========================
@st.cache_data
def engineer_features(df, ma_short, ma_long, rsi_period):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
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
    
    # Volume features
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Lag features
    for i in [1, 2, 3, 5]:
        data[f'Return_Lag_{i}'] = data['Returns'].shift(i)
        data[f'Volume_Lag_{i}'] = data['Volume'].shift(i)
    
    # Target
    data['Target'] = data['Returns'].shift(-1)
    
    return data.dropna()

df = engineer_features(btc, ma_short, ma_long, rsi_period)

# =========================
# Key Metrics
# =========================
latest = df.iloc[-1]
prev = df.iloc[-2]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
    st.metric(
        "💰 BTC Price",
        f"${latest['Close']:,.0f}",
        f"{price_change:+.2f}%"
    )

with col2:
    volume_change = (latest['Volume'] - prev['Volume']) / prev['Volume'] * 100
    st.metric(
        "📊 24h Volume",
        f"${latest['Volume']/1e9:.2f}B",
        f"{volume_change:+.2f}%"
    )

with col3:
    rsi_status = "🔴 Overbought" if latest['RSI'] > 70 else "🟢 Oversold" if latest['RSI'] < 30 else "🟡 Neutral"
    st.metric(
        "📈 RSI",
        f"{latest['RSI']:.1f}",
        rsi_status
    )

with col4:
    vol_change = ((latest['Volatility_7'] - prev['Volatility_7']) / prev['Volatility_7'] * 100) if prev['Volatility_7'] > 0 else 0
    st.metric(
        "💹 Volatility (7d)",
        f"{latest['Volatility_7']*100:.2f}%",
        f"{vol_change:+.1f}%"
    )

with col5:
    ma_signal = "🟢 Bullish" if latest[f'MA_{ma_short}'] > latest[f'MA_{ma_long}'] else "🔴 Bearish"
    st.metric(
        "🎯 MA Signal",
        ma_signal,
        f"{ma_short}/{ma_long}"
    )

st.markdown("---")

# =========================
# Machine Learning
# =========================
feature_cols = [f'MA_{ma_short}', f'MA_{ma_long}', 'MA_Diff', 'Volatility_7', 'Volatility_30',
                'Momentum_7', 'Momentum_14', 'RSI', 'BB_Position', 'Volume_Ratio',
                'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5']

X = df[feature_cols]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

split_idx = int(len(df) * (1 - test_size))
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train model
@st.cache_resource
def train_model(model_type, X_train, y_train):
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

with st.spinner(f"🤖 Training {model_choice} model..."):
    model = train_model(model_choice, X_train, y_train)
    y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Backtest
df_test = df.iloc[split_idx:].copy()
df_test['Predicted_Return'] = y_pred
df_test['Signal'] = 0
df_test.loc[df_test['Predicted_Return'] > threshold, 'Signal'] = 1
df_test.loc[df_test['Predicted_Return'] < -threshold, 'Signal'] = -1
df_test['Strategy_Returns'] = df_test['Signal'] * df_test['Returns']
df_test['Cumulative_Market'] = (1 + df_test['Returns']).cumprod()
df_test['Cumulative_Strategy'] = (1 + df_test['Strategy_Returns']).cumprod()

market_return = (df_test['Cumulative_Market'].iloc[-1] - 1) * 100
strategy_return = (df_test['Cumulative_Strategy'].iloc[-1] - 1) * 100

if df_test['Strategy_Returns'].std() > 0:
    sharpe = (df_test['Strategy_Returns'].mean() / df_test['Strategy_Returns'].std()) * np.sqrt(252)
else:
    sharpe = 0

cumulative = df_test['Cumulative_Strategy']
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max * 100
max_drawdown = drawdown.min()

winning_trades = len(df_test[(df_test['Signal'] != 0) & (df_test['Strategy_Returns'] > 0)])
total_trades = len(df_test[df_test['Signal'] != 0])
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Trading Signals", "🤖 ML Analysis", "📉 Risk & Performance"])

with tab1:
    st.subheader("Price Action with Moving Averages")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index[-300:], df['Close'][-300:], label='BTC Price', linewidth=2, alpha=0.8, color='#8a5cf6')
    ax.plot(df.index[-300:], df[f'MA_{ma_short}'][-300:], label=f'MA {ma_short}', linewidth=2, alpha=0.9, color='#10b981')
    ax.plot(df.index[-300:], df[f'MA_{ma_long}'][-300:], label=f'MA {ma_long}', linewidth=2, alpha=0.9, color='#f59e0b')
    ax.fill_between(df.index[-300:], df[f'MA_{ma_short}'][-300:], df[f'MA_{ma_long}'][-300:], alpha=0.2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading Volume")
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['#ef4444' if r < 0 else '#10b981' for r in df['Returns'][-300:]]
        ax.bar(df.index[-300:], df['Volume'][-300:], color=colors, alpha=0.6)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volume', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Daily Returns Distribution")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(df['Returns'].dropna() * 100, bins=50, color='#8a5cf6', alpha=0.7, edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Returns (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"RSI Indicator (Period: {rsi_period})")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[-200:], df['RSI'][-200:], linewidth=2, color='#8a5cf6')
        ax.axhline(70, color='#ef4444', linestyle='--', linewidth=2, alpha=0.7, label='Overbought')
        ax.axhline(30, color='#10b981', linestyle='--', linewidth=2, alpha=0.7, label='Oversold')
        ax.fill_between(df.index[-200:], 70, 100, alpha=0.2, color='red')
        ax.fill_between(df.index[-200:], 0, 30, alpha=0.2, color='green')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Bollinger Bands")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[-200:], df['Close'][-200:], label='Price', linewidth=2, color='white')
        ax.plot(df.index[-200:], df['BB_Upper'][-200:], label='Upper Band', linewidth=1, linestyle='--', color='#ef4444')
        ax.plot(df.index[-200:], df['BB_Middle'][-200:], label='Middle Band', linewidth=1, linestyle='--', color='#f59e0b')
        ax.plot(df.index[-200:], df['BB_Lower'][-200:], label='Lower Band', linewidth=1, linestyle='--', color='#10b981')
        ax.fill_between(df.index[-200:], df['BB_Lower'][-200:], df['BB_Upper'][-200:], alpha=0.2)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Price Volatility Over Time")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df.index[-300:], 0, df['Volatility_7'][-300:] * 100, alpha=0.7, color='#ef4444')
    ax.plot(df.index[-300:], df['Volatility_7'][-300:] * 100, linewidth=2, color='#ef4444')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    st.subheader("🤖 Machine Learning Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAE", f"{mae:.6f}")
    col3.metric("Training Samples", f"{len(X_train):,}")
    col4.metric("Test Samples", f"{len(X_test):,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        ax.set_xlabel('Importance', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Predictions vs Actual Returns")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_test.values, y_pred, alpha=0.5, s=50, color='#8a5cf6')
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Returns', fontsize=12)
        ax.set_ylabel('Predicted Returns', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Model Residuals Analysis")
    residuals = y_test.values - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    ax1.scatter(y_pred, residuals, alpha=0.5, s=30, color='#8a5cf6')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Returns', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(residuals, bins=50, color='#8a5cf6', alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.subheader("📊 Strategy Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏦 Market Return", f"{market_return:.2f}%")
    col2.metric("🚀 Strategy Return", f"{strategy_return:.2f}%", f"{strategy_return - market_return:+.2f}%")
    col3.metric("📈 Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("📉 Max Drawdown", f"{max_drawdown:.2f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Total Trades", total_trades)
    col2.metric("✅ Win Rate", f"{win_rate:.1f}%")
    col3.metric("🟢 Long Signals", len(df_test[df_test['Signal'] == 1]))
    col4.metric("🔴 Short Signals", len(df_test[df_test['Signal'] == -1]))
    
    st.subheader("Cumulative Returns: Strategy vs Buy & Hold")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_test.index, df_test['Cumulative_Market'], label='Buy & Hold', linewidth=3, color='#ef4444', alpha=0.8)
    ax.plot(df_test.index, df_test['Cumulative_Strategy'], label='ML Strategy', linewidth=3, color='#10b981', alpha=0.8)
    ax.fill_between(df_test.index, df_test['Cumulative_Market'], df_test['Cumulative_Strategy'], alpha=0.2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Drawdown")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(df_test.index, 0, drawdown, alpha=0.7, color='#f59e0b')
        ax.plot(df_test.index, drawdown, linewidth=2, color='#f59e0b')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Trading Signals Timeline")
        fig, ax = plt.subplots(figsize=(12, 4))
        signal_colors = ['#10b981' if s == 1 else '#ef4444' if s == -1 else '#6b7280' for s in df_test['Signal']]
        ax.scatter(df_test.index, df_test['Signal'], c=signal_colors, s=30, alpha=0.6)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Signal', fontsize=12)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Short', 'Hold', 'Long'])
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Returns Distribution Comparison")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.hist(df_test['Returns'] * 100, bins=50, alpha=0.6, label='Market Returns', color='#ef4444', edgecolor='white')
    ax.hist(df_test['Strategy_Returns'] * 100, bins=50, alpha=0.6, label='Strategy Returns', color='#10b981', edgecolor='white')
    ax.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Returns (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; opacity: 0.7;'>"
    f"🚀 Powered by Machine Learning | {len(df):,} Data Points | "
    f"Last Updated: {df.index[-1].strftime('%Y-%m-%d %H:%M')} | "
    f"Model: {model_choice} | Test Size: {int(test_size*100)}%"
    f"</div>",
    unsafe_allow_html=True
)
