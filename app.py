import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# Hide Streamlit branding
st.markdown("""
<style>
    /* Hide Deploy button */
    .stAppDeployButton {
        display: none !important;
    }
    /* Hide MainMenu (hamburger) */
    #MainMenu {
        visibility: hidden !important;
    }
    /* Hide footer */
    footer {
        visibility: hidden !important;
    }
    /* Hide "Made with Streamlit" badge */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    /* Hide the toolbar (Share, Star, etc.) */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Force sidebar to be visible */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
    /* Make sidebar collapse button visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
    }
    
    /* Make expander arrows more visible */
    [data-testid="stExpander"] summary {
        background-color: rgba(138, 92, 246, 0.1) !important;
        border: 1px solid rgba(138, 92, 246, 0.3) !important;
        border-radius: 4px !important;
        padding: 10px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: rgba(138, 92, 246, 0.2) !important;
        border-color: rgba(138, 92, 246, 0.5) !important;
    }
    /* Make the arrow icon more visible */
    [data-testid="stExpander"] summary svg {
        fill: #8a5cf6 !important;
        stroke: #8a5cf6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# =========================
# PASSWORD PROTECTION (ADD THIS SECTION)
# =========================
import hashlib

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == st.secrets.get("password", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 BTC AI Trading Dashboard - Demo Access")
    st.markdown("### Private Demo for Educational Review")
    st.info("👋 Welcome! This is a private demo. Please enter the access code provided.")
    
    st.text_input(
        "Access Code", 
        type="password", 
        on_change=password_entered, 
        key="password",
        help="Enter the demo access code shared with you"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Incorrect access code. Please check your email for the correct code.")
    
    st.markdown("---")
    st.caption("For demo access, contact: henrietta.atsenokhai@gmail.com")
    
    return False

# Uncomment this line to enable password protection:
# if not check_password():
#     st.stop()

# =========================
# Data Loading Functions
# =========================
@st.cache_data(ttl=3600)
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
    """Generate AI-powered trading recommendations"""
    recommendations = []
    risk_level = "Low"
    action = "HOLD"
    confidence = 0
    
    rsi = latest['RSI']
    volatility = latest['Volatility_7'] * 100
    momentum = (latest['Close'] - df['Close'].iloc[-30]) / df['Close'].iloc[-30] * 100
    
    if signal == 1:
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
    
    elif signal == -1:
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
    
    else:
        action = "HOLD"
        confidence = 50
        recommendations.append("🟡 **Hold Position**: Market conditions are unclear. Wait for stronger signals.")
    
    if abs(momentum) > 10:
        if momentum > 0:
            recommendations.append(f"📈 **Strong Uptrend**: Price is up {momentum:.1f}% over the last 30 days.")
        else:
            recommendations.append(f"📉 **Strong Downtrend**: Price is down {abs(momentum):.1f}% over the last 30 days.")
    
    if volatility > 5:
        risk_level = "High"
        recommendations.append(f"⚠️ **High Volatility Warning**: Current volatility at {volatility:.1f}%. Expect large price swings.")
    elif volatility > 3:
        risk_level = "Medium"
        recommendations.append(f"⚡ **Moderate Volatility**: Volatility at {volatility:.1f}%. Normal market conditions.")
    else:
        risk_level = "Low"
        recommendations.append(f"✅ **Low Volatility**: Market is relatively stable at {volatility:.1f}% volatility.")
    
    if strategy_ret > market_ret + 5:
        recommendations.append(f"🎯 **Strategy Outperforming**: ML strategy beat buy-and-hold by {strategy_ret - market_ret:.1f}%.")
    elif strategy_ret < market_ret - 5:
        recommendations.append(f"⚠️ **Strategy Underperforming**: Strategy trailing buy-and-hold by {market_ret - strategy_ret:.1f}%.")
    
    if win_rate > 60:
        recommendations.append(f"✅ **High Win Rate**: {win_rate:.1f}% of trades are profitable. Strategy shows consistency.")
    elif win_rate < 45:
        recommendations.append(f"⚠️ **Low Win Rate**: Only {win_rate:.1f}% of trades profitable. Exercise caution.")
    
    if sharpe > 1.5:
        recommendations.append(f"🏆 **Excellent Risk-Adjusted Returns**: Sharpe ratio of {sharpe:.2f} indicates strong performance.")
    elif sharpe < 0.5:
        recommendations.append(f"⚠️ **Poor Risk-Adjusted Returns**: Low Sharpe ratio ({sharpe:.2f}). Returns don't justify the risk.")
    
    confidence = max(min(confidence, 95), 30)
    
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
use_live_data = st.sidebar.checkbox("📡 Use Live Data (Yahoo Finance)", value=True, 
                                     help="Fetch real-time Bitcoin data from Yahoo Finance")

ma_short = st.sidebar.slider("Short MA", 5, 30, 7, 
                              help="Short-term moving average period in days")
ma_long = st.sidebar.slider("Long MA", 30, 120, 30, 
                             help="Long-term moving average period in days")
rsi_period = st.sidebar.slider("RSI Period", 7, 28, 14, 
                                help="Period for calculating Relative Strength Index")
threshold = st.sidebar.slider("Threshold (%)", 0.0, 1.0, 0.05, 0.01, 
                               help="Minimum predicted return to trigger buy/sell signal") / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Model Configuration")
model_choice = st.sidebar.selectbox("ML Model", ["Random Forest", "Gradient Boosting", "XGBoost"],
                                     help="Machine learning algorithm for predictions")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20, 
                               help="Percentage of data reserved for testing") / 100
retrain_window = st.sidebar.slider("Retrain Window (days)", 100, 500, 252, 
                                     help="Number of recent days used for model training")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**📖 Quick Guide:**
1. Toggle live/demo data
2. Adjust MA periods & RSI
3. Select ML model
4. Review AI recommendations
5. Explore visualizations
""")

# =========================
# Load Data
# =========================
demo_mode = False
if use_live_data:
    with st.spinner("📡 Fetching live BTC data..."):
        btc, error = load_live_btc_data()
        
        if btc is None:
            st.warning(f"⚠️ Unable to fetch live data. Using demo data.")
            btc = generate_synthetic_btc_data()
            demo_mode = True
        else:
            st.success(f"✅ Live data loaded! Last updated: {btc.index[-1].strftime('%Y-%m-%d')}")
else:
    btc = generate_synthetic_btc_data()
    demo_mode = True
    st.info("🎮 **DEMO MODE**: Using synthetic data")

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

train_start_idx = max(0, len(df) - retrain_window - int(len(df) * test_size))
split_idx = len(df) - int(len(df) * test_size)

X_train = X_scaled[train_start_idx:split_idx]
X_test = X_scaled[split_idx:]
y_train = y.iloc[train_start_idx:split_idx]
y_test = y.iloc[split_idx:]

# Model selection
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, min_samples_split=10)
elif model_choice == "Gradient Boosting":
    model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5, min_samples_split=10)
else:  # XGBoost
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=50, random_state=42, max_depth=5, learning_rate=0.1)
    except ImportError:
        st.warning("⚠️ XGBoost not installed. Using Gradient Boosting instead.")
        model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Rolling performance
rolling_window = 30
rolling_r2 = []
for i in range(rolling_window, len(y_test)):
    window_r2 = r2_score(y_test.iloc[i-rolling_window:i], y_pred[i-rolling_window:i])
    rolling_r2.append(window_r2)

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

latest = df.iloc[-1]
current_signal = 1 if latest[f'MA_{ma_short}'] > latest[f'MA_{ma_long}'] else -1

# =========================
# AI RECOMMENDATIONS & METRICS
# =========================
ai_insights = generate_recommendations(df, latest, strategy_ret, market_ret, sharpe, win_rate, current_signal)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: rgba(138, 92, 246, 0.1);
        border: 1px solid rgba(138, 92, 246, 0.2);
        padding: 8px 12px;
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.2, 1, 1, 1.2, 1])

action_colors = {
    'BUY': ('linear-gradient(135deg, #10b981, #059669)', '🟢'),
    'SELL': ('linear-gradient(135deg, #ef4444, #dc2626)', '🔴'),
    'HOLD': ('linear-gradient(135deg, #f59e0b, #d97706)', '🟡')
}

with col1:
    st.markdown(f"""
    <div style='background: {action_colors[ai_insights["action"]][0]}; color: white; padding: 12px; border-radius: 6px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-bottom: 16px;'>
        <h3 style='margin: 0; font-size: 1.1em;'>{action_colors[ai_insights["action"]][1]} {ai_insights['action']}</h3>
        <p style='margin: 2px 0 0 0; font-size: 0.68em;'>Confidence: {ai_insights['confidence']:.0f}% • {ai_insights['risk_level']} Risk</p>
    </div>
    """, unsafe_allow_html=True)

prev = df.iloc[-2]
with col2:
    st.metric("💰 Price", f"${latest['Close']:,.0f}", f"{(latest['Close']-prev['Close'])/prev['Close']*100:+.1f}%")
with col3:
    st.metric("📈 RSI", f"{latest['RSI']:.0f}", "OB" if latest['RSI']>70 else "OS" if latest['RSI']<30 else "OK")
with col4:
    st.metric("💹 Vol", f"{latest['Volatility_7']*100:.1f}%")
with col5:
    st.metric("🚀 Strat", f"{strategy_ret:.1f}%", f"{strategy_ret-market_ret:+.1f}%")
with col6:
    st.metric("✅ Win", f"{win_rate:.0f}%")

with st.expander("📋 View AI Insights & Recommendations"):
    for rec in ai_insights['recommendations']:
        st.markdown(f"- {rec}")

st.markdown("---")

# =========================
# Tabs with Collapsible Descriptions
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
    
    # AI-powered analysis
    ma_cross = "bullish crossover" if latest[f'MA_{ma_short}'] > latest[f'MA_{ma_long}'] else "bearish crossover"
    recent_trend = "uptrend" if df['Close'].iloc[-1] > df['Close'].iloc[-30] else "downtrend"
    price_vs_ma = "above" if latest['Close'] > latest[f'MA_{ma_long}'] else "below"
    
    with st.expander("📊 Chart Analysis (AI-Powered)"):
        st.info(f"""
        **Current Market Condition:** The chart shows a **{ma_cross}** pattern. Price is **{price_vs_ma}** the long-term average, 
        indicating a **{recent_trend}**. When the short MA (green) crosses above the long MA (orange), it's typically a buy signal. 
        When it crosses below, it suggests selling. The purple line represents actual Bitcoin price movements over the last 300 days.
        """)
    
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
        
        avg_volume = df['Volume'][-30:].mean()
        recent_volume = df['Volume'][-5:].mean()
        volume_trend = "higher" if recent_volume > avg_volume else "lower"
        
        with st.expander("📊 Volume Analysis (AI-Powered)"):
            st.info(f"""
            **Volume Insight:** Recent trading volume is **{volume_trend}** than the 30-day average. 
            Green bars indicate days when price increased; red bars show price decreases. 
            High volume during price increases suggests strong buying pressure and conviction. 
            Low volume during moves may indicate weak momentum that could reverse.
            """)
    
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
        
        positive_days = (df['Returns'] > 0).sum()
        total_days = len(df['Returns'].dropna())
        positive_pct = (positive_days / total_days * 100)
        avg_return = df['Returns'].mean() * 100
        
        with st.expander("📊 Returns Analysis (AI-Powered)"):
            st.info(f"""
            **Distribution Insight:** Bitcoin had positive returns on **{positive_pct:.1f}%** of trading days. 
            Average daily return is **{avg_return:.3f}%**. The distribution shows most days cluster near zero (red line) 
            with occasional large moves. A wider spread indicates higher volatility and risk. 
            The shape reveals the probability of different return outcomes.
            """)

with tab2:
    st.subheader("RSI Indicator")
    fig, ax = plt.subplots(figsize=(14, 4))
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
    
    rsi_status = "overbought (>70)" if latest['RSI'] > 70 else "oversold (<30)" if latest['RSI'] < 30 else "neutral (30-70)"
    
    with st.expander("📊 RSI Analysis (AI-Powered)"):
        st.info(f"""
        **RSI Reading:** Currently at **{latest['RSI']:.1f}** ({rsi_status}). 
        RSI measures momentum on a scale of 0-100. Above 70 (red zone) suggests overbought conditions—potential reversal down. 
        Below 30 (green zone) indicates oversold—potential bounce up. 
        RSI helps identify when price has moved too far too fast and may correct.
        """)
    
    st.subheader("Bollinger Bands")
    fig, ax = plt.subplots(figsize=(14, 4))
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
    
    bb_position = ((latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100)
    bb_status = "near upper band" if bb_position > 80 else "near lower band" if bb_position < 20 else "in the middle"
    
    with st.expander("📊 Bollinger Bands Analysis (AI-Powered)"):
        st.info(f"""
        **Band Position:** Price is currently **{bb_status}** ({bb_position:.0f}% position within bands). 
        Bollinger Bands expand during high volatility and contract during calm periods. 
        Price touching upper band (red) may signal overbought; touching lower band (green) suggests oversold. 
        Price tends to bounce between bands—touching one band often leads to movement toward the other.
        """)
    
    st.subheader("Volatility Over Time")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df.index[-200:], 0, df['Volatility_7'][-200:] * 100, alpha=0.7, color='#ef4444')
    ax.plot(df.index[-200:], df['Volatility_7'][-200:] * 100, linewidth=2, color='#ef4444')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    avg_vol = df['Volatility_7'].mean() * 100
    current_vol_status = "high" if latest['Volatility_7'] * 100 > avg_vol * 1.5 else "low" if latest['Volatility_7'] * 100 < avg_vol * 0.5 else "normal"
    
    with st.expander("📊 Volatility Analysis (AI-Powered)"):
        st.info(f"""
        **Volatility Status:** Currently at **{latest['Volatility_7']*100:.2f}%** ({current_vol_status} compared to {avg_vol:.2f}% average). 
        Higher volatility means larger price swings—more risk but also more trading opportunities. 
        Volatility spikes often precede major price moves. Low volatility suggests stable, predictable prices. 
        Traders adjust position sizes based on volatility to manage risk.
        """)

with tab3:
    st.subheader(f"Model Performance: {model_choice}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAE", f"{mae:.6f}")
    col3.metric("Training Samples", f"{len(X_train):,}")
    col4.metric("Test Samples", f"{len(X_test):,}")
    
    st.subheader("Model Adaptation Over Time (Rolling R²)")
    fig_rolling, ax_rolling = plt.subplots(figsize=(14, 4))
    test_dates = df.index[split_idx + rolling_window:]
    ax_rolling.plot(test_dates, rolling_r2, linewidth=2, color='#8a5cf6', label='Rolling R² (30-day window)')
    ax_rolling.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax_rolling.set_ylabel('R² Score')
    ax_rolling.legend()
    ax_rolling.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_rolling)
    plt.close()
    
    with st.expander("📊 Model Adaptation Analysis (AI-Powered)"):
        avg_rolling_r2 = np.mean(rolling_r2)
        st.info(f"""
        **Model Stability:** The rolling R² shows how well the model predicts over time. 
        Average rolling R² is **{avg_rolling_r2:.4f}**. Positive values indicate the model adds predictive value. 
        Fluctuations are normal as market conditions change. Consistent positive R² suggests robust predictions. 
        Drops below zero indicate periods where the model struggled with market regime changes.
        """)
    
    st.subheader("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#8a5cf6')
        ax.set_xlabel('Importance')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        top_feature = feature_importance_df.iloc[-1]['Feature']
        top_importance = feature_importance_df.iloc[-1]['Importance']
        
        with st.expander("📊 Feature Importance Analysis (AI-Powered)"):
            st.info(f"""
            **Most Important Feature:** **{top_feature}** (importance: {top_importance:.4f}). 
            Feature importance shows which indicators the model relies on most for predictions. 
            Higher bars mean the model considers that feature more critical for making accurate predictions. 
            This helps understand what drives the model's trading decisions.
            """)
    else:
        st.info("Feature importance not available for this model type.")
    
    st.subheader("Predictions vs Actual Returns")
    fig, ax = plt.subplots(figsize=(14, 5))
    sample_size = min(200, len(y_test))
    ax.scatter(y_test[-sample_size:], y_pred[-sample_size:], alpha=0.5, color='#8a5cf6')
    
    # Perfect prediction line
    min_val = min(y_test[-sample_size:].min(), y_pred[-sample_size:].min())
    max_val = max(y_test[-sample_size:].max(), y_pred[-sample_size:].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Returns')
    ax.set_ylabel('Predicted Returns')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    with st.expander("📊 Prediction Accuracy Analysis (AI-Powered)"):
        st.info(f"""
        **Prediction Quality:** Points closer to the red diagonal line indicate accurate predictions. 
        R² score of **{r2:.4f}** measures overall fit. Scatter around the line shows prediction variance. 
        The model aims to predict whether returns will be positive or negative, not exact values. 
        Clustering near the line suggests the model captures market direction well.
        """)

with tab4:
    st.subheader("Cumulative Returns: Strategy vs Market")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_test.index, (df_test['Cum_Market'] - 1) * 100, 
            label='Buy & Hold', linewidth=2.5, color='#f59e0b')
    ax.plot(df_test.index, (df_test['Cum_Strategy'] - 1) * 100, 
            label='ML Strategy', linewidth=2.5, color='#10b981')
    ax.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    outperformance = strategy_ret - market_ret
    better = "outperformed" if outperformance > 0 else "underperformed"
    
    with st.expander("📊 Performance Comparison (AI-Powered)"):
        st.info(f"""
        **Strategy Performance:** The ML strategy **{better}** buy-and-hold by **{abs(outperformance):.2f}%**. 
        The green line shows returns from following ML signals; orange shows simple buy-and-hold. 
        Outperformance suggests the model successfully times entries and exits. 
        Underperformance indicates transaction costs or poor market timing may be hurting results.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drawdown Analysis")
        
        # Calculate drawdowns
        cumulative = df_test['Cum_Strategy']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(df_test.index, drawdown, 0, alpha=0.7, color='#ef4444')
        ax.plot(df_test.index, drawdown, linewidth=2, color='#ef4444')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        max_dd = drawdown.min()
        
        with st.expander("📊 Drawdown Analysis (AI-Powered)"):
            st.info(f"""
            **Maximum Drawdown:** **{max_dd:.2f}%** - the largest peak-to-trough decline. 
            Drawdowns show how much capital was lost from the highest point before recovering. 
            Smaller drawdowns indicate better risk management and capital preservation. 
            Large drawdowns can test investor patience and increase emotional trading decisions.
            """)
    
    with col2:
        st.subheader("Trade Distribution")
        
        trade_signals = df_test[df_test['Signal'] != 0]['Signal']
        signal_counts = trade_signals.value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = ['Buy Signals', 'Sell Signals']
        colors = ['#10b981', '#ef4444']
        sizes = [signal_counts.get(1, 0), signal_counts.get(-1, 0)]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 12})
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        buy_pct = (signal_counts.get(1, 0) / total_trades * 100) if total_trades > 0 else 0
        
        with st.expander("📊 Trading Pattern Analysis (AI-Powered)"):
            st.info(f"""
            **Signal Distribution:** **{buy_pct:.1f}%** buy signals vs **{100-buy_pct:.1f}%** sell signals. 
            Balanced distribution suggests the model responds to both bullish and bearish conditions. 
            Heavy bias toward one direction may indicate trend-following behavior. 
            Total of **{total_trades}** trades executed during the test period.
            """)
    
    st.subheader("Performance Metrics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate additional metrics
    total_return_strat = strategy_ret
    total_return_market = market_ret
    volatility_strat = df_test['Strat_Returns'].std() * np.sqrt(252) * 100
    volatility_market = df_test['Returns'].std() * np.sqrt(252) * 100
    
    col1.metric("📊 Total Return (Strategy)", f"{total_return_strat:.2f}%")
    col2.metric("📊 Total Return (Market)", f"{total_return_market:.2f}%")
    col3.metric("📉 Volatility (Strategy)", f"{volatility_strat:.2f}%")
    col4.metric("📉 Volatility (Market)", f"{volatility_market:.2f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    col3.metric("📈 Total Trades", f"{total_trades}")
    col4.metric("✅ Winning Trades", f"{winning_trades}")
    
    with st.expander("📊 Metrics Explanation"):
        st.markdown("""
        **Key Metrics Explained:**
        
        - **Total Return**: Overall percentage gain/loss from start to end of test period
        - **Volatility**: Annualized standard deviation of returns (higher = more risk)
        - **Sharpe Ratio**: Risk-adjusted return measure (>1 is good, >2 is excellent)
        - **Win Rate**: Percentage of profitable trades out of total trades
        - **Max Drawdown**: Largest peak-to-trough decline (measures worst-case loss)
        """)

# =========================
# Footer
# =========================
st.markdown("---")
col1, col2, col3 = st.columns([2, 3, 2])

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #888; font-size: 0.9em;'>
            ⚡ <strong>BTC AI Trading Dashboard</strong> • Built by Henrietta Atsenokhai with Streamlit & Scikit-learn<br>
            Educational purposes only • Not financial advice<br>
            © 2025 Henrietta Atsenokhai. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)

if demo_mode:
    st.info("🎮 **Demo Mode Active**: This dashboard is using synthetic data. Enable 'Use Live Data' in the sidebar to fetch real Bitcoin prices from Yahoo Finance.")
