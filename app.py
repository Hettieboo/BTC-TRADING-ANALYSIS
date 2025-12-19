import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
ma_short = st.sidebar.slider("Short MA Window", 5, 30, 7)
ma_long = st.sidebar.slider("Long MA Window", 30, 120, 30)
rsi_period = st.sidebar.slider("RSI Period", 7, 28, 14)
threshold = st.sidebar.slider("Signal Threshold (%)", 0.0, 1.0, 0.05, 0.01) / 100
model_choice = st.sidebar.selectbox("ML Model", ["Random Forest", "Gradient Boosting"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100

st.sidebar.markdown("---")
refresh_data = st.sidebar.button("🔄 Refresh Data", use_container_width=True)

# =========================
# Data Loading with Cache
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

# Load data
with st.spinner("Loading BTC data..."):
    btc = load_btc_data()

if btc is None or btc.empty:
    st.error("Failed to load data. Please check your internet connection.")
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
# Key Metrics at Top
# =========================
latest = df.iloc[-1]
prev = df.iloc[-2]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "BTC Price",
        f"${latest['Close']:,.0f}",
        f"{(latest['Close'] - prev['Close']) / prev['Close'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "24h Volume",
        f"${latest['Volume']/1e9:.2f}B",
        f"{(latest['Volume'] - prev['Volume']) / prev['Volume'] * 100:.2f}%"
    )

with col3:
    st.metric(
        "RSI",
        f"{latest['RSI']:.1f}",
        "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"
    )

with col4:
    st.metric(
        "Volatility (7d)",
        f"{latest['Volatility_7']*100:.2f}%",
        f"{((latest['Volatility_7'] - prev['Volatility_7']) / prev['Volatility_7'] * 100):.1f}%"
    )

with col5:
    ma_signal = "Bullish" if latest[f'MA_{ma_short}'] > latest[f'MA_{ma_long}'] else "Bearish"
    st.metric(
        "MA Signal",
        ma_signal,
        f"{ma_short}/{ma_long}"
    )

st.markdown("---")

# =========================
# Tabs for Different Views
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🎯 Trading Signals", 
    "🤖 ML Analysis", 
    "📉 Risk Metrics",
    "📈 Advanced Charts"
])

# =========================
# TAB 1: Overview
# =========================
with tab1:
    # Price Chart with MAs
    fig_price = go.Figure()
    
    fig_price.add_trace(go.Candlestick(
        x=df.index[-200:],
        open=df['Open'][-200:],
        high=df['High'][-200:],
        low=df['Low'][-200:],
        close=df['Close'][-200:],
        name='BTC Price'
    ))
    
    fig_price.add_trace(go.Scatter(
        x=df.index[-200:],
        y=df[f'MA_{ma_short}'][-200:],
        name=f'MA {ma_short}',
        line=dict(color='#10b981', width=2)
    ))
    
    fig_price.add_trace(go.Scatter(
        x=df.index[-200:],
        y=df[f'MA_{ma_long}'][-200:],
        name=f'MA {ma_long}',
        line=dict(color='#f59e0b', width=2)
    ))
    
    fig_price.update_layout(
        title='BTC Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Volume Chart
    fig_volume = go.Figure()
    colors = ['red' if row['Returns'] < 0 else 'green' for idx, row in df[-200:].iterrows()]
    
    fig_volume.add_trace(go.Bar(
        x=df.index[-200:],
        y=df['Volume'][-200:],
        name='Volume',
        marker_color=colors
    ))
    
    fig_volume.update_layout(
        title='Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_dark',
        height=300
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

# =========================
# TAB 2: Trading Signals
# =========================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['RSI'][-200:],
            name='RSI',
            line=dict(color='#8b5cf6', width=2)
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig_rsi.update_layout(
            title=f'RSI Indicator (Period: {rsi_period})',
            xaxis_title='Date',
            yaxis_title='RSI',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # Bollinger Bands
        fig_bb = go.Figure()
        
        fig_bb.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['BB_Upper'][-200:],
            name='Upper Band',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig_bb.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['Close'][-200:],
            name='Price',
            line=dict(color='white', width=2)
        ))
        
        fig_bb.add_trace(go.Scatter(
            x=df.index[-200:],
            y=df['BB_Lower'][-200:],
            name='Lower Band',
            line=dict(color='green', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        fig_bb.update_layout(
            title='Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_bb, use_container_width=True)
    
    # Volatility
    fig_vol = go.Figure()
    
    fig_vol.add_trace(go.Scatter(
        x=df.index[-200:],
        y=df['Volatility_7'][-200:] * 100,
        name='7-day Volatility',
        fill='tozeroy',
        line=dict(color='#ef4444')
    ))
    
    fig_vol.update_layout(
        title='Price Volatility',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        template='plotly_dark',
        height=300
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)

# =========================
# TAB 3: ML Analysis
# =========================
with tab3:
    st.subheader("Machine Learning Model Training")
    
    # Feature selection
    feature_cols = [f'MA_{ma_short}', f'MA_{ma_long}', 'MA_Diff', 'Volatility_7', 'Volatility_30',
                    'Momentum_7', 'Momentum_14', 'RSI', 'BB_Position', 'Volume_Ratio',
                    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5']
    
    X = df[feature_cols]
    y = df['Target']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    # Split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    with st.spinner(f"Training {model_choice} model..."):
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAE", f"{mae:.6f}")
    col3.metric("Training Samples", f"{len(X_train):,}")
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig_importance = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(color=importance_df['Importance'], colorscale='Viridis')
    ))
    
    fig_importance.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction vs Actual
    fig_pred = go.Figure()
    
    fig_pred.add_trace(go.Scatter(
        x=y_test.values,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='#8b5cf6', size=8, opacity=0.6)
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig_pred.update_layout(
        title='Predicted vs Actual Returns',
        xaxis_title='Actual Returns',
        yaxis_title='Predicted Returns',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)

# =========================
# TAB 4: Risk Metrics
# =========================
with tab4:
    # Backtest the strategy
    df_test = df.iloc[split_idx:].copy()
    df_test['Predicted_Return'] = y_pred
    df_test['Signal'] = 0
    df_test.loc[df_test['Predicted_Return'] > threshold, 'Signal'] = 1
    df_test.loc[df_test['Predicted_Return'] < -threshold, 'Signal'] = -1
    df_test['Strategy_Returns'] = df_test['Signal'] * df_test['Returns']
    df_test['Cumulative_Market'] = (1 + df_test['Returns']).cumprod()
    df_test['Cumulative_Strategy'] = (1 + df_test['Strategy_Returns']).cumprod()
    
    # Calculate metrics
    market_return = (df_test['Cumulative_Market'].iloc[-1] - 1) * 100
    strategy_return = (df_test['Cumulative_Strategy'].iloc[-1] - 1) * 100
    
    if df_test['Strategy_Returns'].std() > 0:
        sharpe = (df_test['Strategy_Returns'].mean() / df_test['Strategy_Returns'].std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Calculate max drawdown
    cumulative = df_test['Cumulative_Strategy']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_trades = len(df_test[(df_test['Signal'] != 0) & (df_test['Strategy_Returns'] > 0)])
    total_trades = len(df_test[df_test['Signal'] != 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Return", f"{market_return:.2f}%")
    col2.metric("Strategy Return", f"{strategy_return:.2f}%", f"{strategy_return - market_return:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", total_trades)
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Long Signals", len(df_test[df_test['Signal'] == 1]))
    col4.metric("Short Signals", len(df_test[df_test['Signal'] == -1]))
    
    # Cumulative Returns
    fig_cumulative = go.Figure()
    
    fig_cumulative.add_trace(go.Scatter(
        x=df_test.index,
        y=df_test['Cumulative_Market'],
        name='Buy & Hold',
        line=dict(color='#ef4444', width=3)
    ))
    
    fig_cumulative.add_trace(go.Scatter(
        x=df_test.index,
        y=df_test['Cumulative_Strategy'],
        name='ML Strategy',
        line=dict(color='#10b981', width=3)
    ))
    
    fig_cumulative.update_layout(
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Drawdown Chart
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=df_test.index,
        y=drawdown,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#f59e0b')
    ))
    
    fig_dd.update_layout(
        title='Strategy Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        height=300
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)

# =========================
# TAB 5: Advanced Charts
# =========================
with tab5:
    # Returns Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df_test['Returns'] * 100,
            name='Market Returns',
            opacity=0.7,
            marker_color='#ef4444'
        ))
        fig_dist.add_trace(go.Histogram(
            x=df_test['Strategy_Returns'] * 100,
            name='Strategy Returns',
            opacity=0.7,
            marker_color='#10b981'
        ))
        
        fig_dist.update_layout(
            title='Returns Distribution',
            xaxis_title='Returns (%)',
            yaxis_title='Frequency',
            template='plotly_dark',
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Signal Timeline
        fig_signals = go.Figure()
        
        fig_signals.add_trace(go.Scatter(
            x=df_test.index,
            y=df_test['Signal'],
            mode='lines',
            name='Trading Signal',
            line=dict(color='#8b5cf6', width=2),
            fill='tozeroy'
        ))
        
        fig_signals.update_layout(
            title='Trading Signals Over Time',
            xaxis_title='Date',
            yaxis_title='Signal',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_signals, use_container_width=True)
    
    # Correlation Heatmap
    corr_features = ['Returns', 'RSI', 'Volatility_7', 'MA_Diff', 'Volume_Ratio']
    corr_matrix = df[corr_features].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig_corr.update_layout(
        title='Feature Correlation Matrix',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: white;'>"
    f"🚀 Powered by AI | {len(df)} Data Points | Last Updated: {df.index[-1].strftime('%Y-%m-%d')}"
    f"</div>",
    unsafe_allow_html=True
)
