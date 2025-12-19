import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
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
# Data Loading Functions
# =========================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_live_btc_data():
    """Load real BTC data from Yahoo Finance"""
    try:
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1500)
        
        btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)
        
        if btc.empty or len(btc) < 100:
            return None, "Insufficient data returned"
        
        return btc, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def generate_synthetic_btc_data(n_days=1500):
    """Generate realistic synthetic BTC data for demo"""
    np.random.seed(42)
    date_range = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
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
# AI Recommendation Engine
# =========================
def generate_recommendations(df, latest, strategy_ret, market_ret, sharpe, win_rate, signal):
    """Generate AI-powered trading recommendations for non-technical users"""
    recommendations = []
    risk_level = "Low"
    action = "HOLD"
    confidence = 0
    
    # Analyze current market conditions
    rsi = latest['RSI']
    volatility = latest['Volatility_7'] * 100
    momentum = (latest['Close'] - df['Close'].iloc[-30]) / df['Close'].iloc[-30] * 100
    
    # Determine action and confidence
    if signal == 1:  # Bullish
        action = "BUY"
        confidence = min(abs(latest['MA_Diff'] / latest['Close'] * 100) * 20, 95)
        
        if rsi < 30:
            recommendations.append("🟢 **Strong Buy Signal**: RSI indicates oversold conditions. Good entry point.")
            confidence += 5
        elif rsi < 50:
            recommendations.append("🟢 **Buy Signal**: Technical indicators suggest upward momentum.")
        else:
            recommendations.append("🟡 **Cautious Buy**: Price is rising but may be approaching overbought territory.")
            confidence -= 10
    
    elif signal == -1:  # Bearish
        action = "SELL"
        confidence = min(abs(latest['MA_Diff'] / latest['Close'] * 100) * 20, 95)
        
        if rsi > 70:
            recommendations.append("🔴 **Strong Sell Signal**: RSI indicates overbought conditions. Consider taking profits.")
            confidence += 5
        elif rsi > 50:
            recommendations.append("🔴 **Sell Signal**: Technical indicators suggest downward pressure.")
        else:
            recommendations.append("🟡 **Cautious Sell**: Downtrend detected but may be oversold.")
            confidence -= 10
    
    else:  # Neutral
        action = "HOLD"
        confidence = 50
        recommendations.append("🟡 **Hold Position**: Market conditions are unclear. Wait for stronger signals.")
    
    # Add momentum analysis
    if abs(momentum) > 10:
        if momentum > 0:
            recommendations.append(f"📈 **Strong Uptrend**: Price is up {momentum:.1f}% over the last 30 days.")
        else:
            recommendations.append(f"📉 **Strong Downtrend**: Price is down {abs(momentum):.1f}% over the last 30 days.")
    
    # Add volatility warning
    if volatility > 5:
        risk_level = "High"
        recommendations.append(f"⚠️ **High Volatility Warning**: Current volatility at {volatility:.1f}%. Expect large price swings.")
    elif volatility > 3:
        risk_level = "Medium"
        recommendations.append(f"⚡ **Moderate Volatility**: Volatility at {volatility:.1f}%. Normal market conditions.")
    else:
        risk_level = "Low"
        recommendations.append(f"✅ **Low Volatility**: Market is relatively stable at {volatility:.1f}% volatility.")
    
    # Add strategy performance context
    if strategy_ret > market_ret + 5:
        recommendations.append(f"🎯 **Strategy Outperforming**: ML strategy beat buy-and-hold by {strategy_ret - market_ret:.1f}%.")
    elif strategy_ret < market_ret - 5:
        recommendations.append(f"⚠️ **Strategy Underperforming**: Strategy trailing buy-and-hold by {market_ret - strategy_ret:.1f}%.")
    
    # Add win rate context
    if win_rate > 60:
        recommendations.append(f"✅ **High Win Rate**: {win_rate:.1f}% of trades are profitable. Strategy shows consistency.")
    elif win_rate < 45:
        recommendations.append(f"⚠️ **Low Win Rate**: Only {win_rate:.1f}% of trades profitable. Exercise caution.")
    
    # Add risk-adjusted performance
    if sharpe > 1.5:
        recommendations.append(f"🏆 **Excellent Risk-Adjusted Returns**: Sharpe ratio of {sharpe:.2f} indicates strong performance.")
    elif sharpe < 0.5:
        recommendations.append(f"⚠️ **Poor Risk-Adjusted Returns**: Low Sharpe ratio ({sharpe:.2f}). Returns don't justify the risk.")
    
    confidence = max(min(confidence, 95), 30)  # Clamp between 30-95%
    
    return {
        'action': action,
        'confidence': confidence,
        'risk_level': risk_level,
        'recommendations': recommendations
    }

# =========================
# Header
# =========================
st.title("⚡ BTC AI Trading Dashboard")
st.markdown("### Advanced Algorithmic Trading with Machine Learning")

# =========================
# Sidebar
# =========================
st.sidebar.header("🎛️ Strategy Parameters")
use_live_data = st.sidebar.checkbox("📡 Use Live Data (Yahoo Finance)", value=True)

ma_short = st.sidebar.slider("Short MA", 5, 30, 7)
ma_long = st.sidebar.slider("Long MA", 30, 120, 30)
rsi_period = st.sidebar.slider("RSI Period", 7, 28, 14)
threshold = st.sidebar.slider("Threshold (%)", 0.0, 1.0, 0.05, 0.01) / 100

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("ML Model", ["Random Forest", "Gradient Boosting"])
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**📖 How to Use:**
1. Toggle live data on/off
2. Adjust strategy parameters
3. Review AI recommendations
4. Monitor model performance
5. Data refreshes hourly
""")

# =========================
# Load Data
# =========================
demo_mode = False
if use_live_data:
    with st.spinner("📡 Fetching live BTC data from Yahoo Finance..."):
        btc, error = load_live_btc_data()
        
        if btc is None:
            st.warning(f"⚠️ Unable to fetch live data: {error}. Using demo data instead.")
            btc = generate_synthetic_btc_data()
            demo_mode = True
        else:
            st.success(f"✅ Live data loaded! Last updated: {btc.index[-1].strftime('%Y-%m-%d %H:%M')}")
else:
    with st.spinner("🎮 Generating demo data..."):
        btc = generate_synthetic_btc_data()
        demo_mode = True
        st.info("🎮 **DEMO MODE**: Using synthetic data for demonstration")

# Feature engineering
@st.cache_data
def add_features(df, ma_short, ma_long, rsi_period):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    
    data[f'MA_{ma_short}'] = data['Close'].rolling(ma_short).mean()
    data[f'MA_{ma_long}'] = data['Close'].rolling(ma_long).mean()
    data['MA_Diff'] = data[f'MA_{ma_short}'] - data[f'MA_{ma_long}']
    
    data['Volatility_7'] = data['Returns'].rolling(7).std()
    data['Volatility_30'] = data['Returns'].rolling(30).std()
    
    data['Momentum_7'] = data['Close'] - data['Close'].shift(7)
    data['Momentum_14'] = data['Close'] - data['Close'].shift(14)
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['BB_Mid'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Mid'] - (data['BB_Std'] * 2)
    
    for i in [1, 2, 3]:
        data[f'Lag_{i}'] = data['Returns'].shift(i)
    
    data['Target'] = data['Returns'].shift(-1)
    return data.dropna()

df = add_features(btc, ma_short, ma_long, rsi_period)

# =========================
# ML Model Training
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

winning_trades = len(df_test[(df_test['Signal'] != 0) & (df_test['Strat_Returns'] > 0)])
total_trades = len(df_test[df_test['Signal'] != 0])
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

# Get latest data and signal
latest = df.iloc[-1]
current_signal = 1 if latest[f'MA_{ma_short}'] > latest[f'MA_{ma_long}'] else -1

# =========================
# AI RECOMMENDATIONS SECTION
# =========================
st.markdown("---")
st.header("🤖 AI Trading Recommendations")
st.markdown("*Easy-to-understand insights based on advanced analysis*")

ai_insights = generate_recommendations(df, latest, strategy_ret, market_ret, sharpe, win_rate, current_signal)

# Action Banner - Compact Button Style
action_colors = {
    'BUY': ('background: linear-gradient(135deg, #10b981, #059669);', '🟢'),
    'SELL': ('background: linear-gradient(135deg, #ef4444, #dc2626);', '🔴'),
    'HOLD': ('background: linear-gradient(135deg, #f59e0b, #d97706);', '🟡')
}

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
    <div style='{action_colors[ai_insights["action"]][0]} color: white; padding: 20px 40px; border-radius: 12px; text-align: center; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); cursor: pointer; transition: transform 0.2s;'>
        <h2 style='margin: 0; font-size: 2em;'>{action_colors[ai_insights["action"]][1]} {ai_insights['action']}</h2>
        <p style='margin: 8px 0 0 0; font-size: 0.95em; opacity: 0.95;'>Confidence: {ai_insights['confidence']:.0f}% • Risk: {ai_insights['risk_level']}</p>
    </div>
    """, unsafe_allow_html=True)

# Key Insights
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Key Insights & Recommendations")
    for rec in ai_insights['recommendations']:
        st.markdown(f"- {rec}")

with col2:
    st.subheader("📊 Quick Stats")
    st.metric("Current Price", f"${latest['Close']:,.0f}")
    st.metric("30-Day Change", f"{((latest['Close'] - df['Close'].iloc[-30])/df['Close'].iloc[-30]*100):+.1f}%")
    st.metric("Strategy vs Market", f"{strategy_ret - market_ret:+.1f}%", 
              "Outperforming" if strategy_ret > market_ret else "Underperforming")

st.markdown("---")

# =========================
# Key Metrics Row
# =========================
col1, col2, col3, col4 = st.columns(4)
prev = df.iloc[-2]

col1.metric("💰 BTC Price", f"${latest['Close']:,.0f}", f"{(latest['Close']-prev['Close'])/prev['Close']*100:+.2f}%")
col2.metric("📈 RSI", f"{latest['RSI']:.1f}", "Overbought" if latest['RSI']>70 else "Oversold" if latest['RSI']<30 else "Neutral")
col3.metric("💹 Volatility", f"{latest['Volatility_7']*100:.2f}%")
col4.metric("🎯 Signal", "🟢 Bullish" if current_signal == 1 else "🔴 Bearish")

st.markdown("---")

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
    col4.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    
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
data_source = "Live (Yahoo Finance)" if not demo_mode else "Demo (Synthetic)"
st.markdown(f"🚀 ML-Powered Trading | Data Source: {data_source} | Model: {model_choice} | Last Update: {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
st.caption("⚠️ Disclaimer: This is for educational purposes only. Not financial advice. Always do your own research before trading.")
