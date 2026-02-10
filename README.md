# ⚡ BTC AI Trading Dashboard

An advanced algorithmic trading dashboard for Bitcoin using Machine Learning and technical analysis. Built with Streamlit, this interactive application provides real-time trading signals, performance analytics, and AI-powered insights.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 📊 Real-Time Data
- **Live Bitcoin prices** from Yahoo Finance (BTC-USD)
- **Demo mode** with synthetic data for testing
- Automatic data refresh capability
- Historical data spanning 1500+ days

### 🤖 Machine Learning Models
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost** (optional, requires installation)
- Customizable hyperparameters and training windows
- Rolling performance evaluation

### 📈 Technical Indicators
- **Moving Averages** (MA7, MA30 - customizable)
- **RSI** (Relative Strength Index)
- **Bollinger Bands**
- **Volatility Analysis** (7-day & 30-day)
- **Momentum Indicators**

### 🎯 AI-Powered Recommendations
- **Smart Trading Signals**: BUY, SELL, or HOLD with confidence levels
- **Risk Assessment**: Low, Medium, or High risk ratings
- **Contextual Insights**: Explains market conditions in plain English
- **Performance Comparison**: Strategy vs Buy-and-Hold

### 📉 Comprehensive Analytics
- **4 Interactive Tabs**:
  1. **Overview**: Price charts, volume, returns distribution
  2. **Signals**: RSI, Bollinger Bands, volatility analysis
  3. **ML Model**: Feature importance, predictions accuracy, rolling R²
  4. **Performance**: Cumulative returns, drawdown analysis, win rate metrics

### 🎨 User Experience
- **Dark mode interface** with custom styling
- **Responsive layout** optimized for all screen sizes
- **Interactive tooltips** explaining each control
- **Collapsible sections** with AI-powered explanations
- **Real-time metrics** with visual indicators

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/btc-ai-dashboard.git
   cd btc-ai-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

---

## 📦 Requirements

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
yfinance>=0.2.28
xgboost>=2.0.0
```

---

## 🎛️ Configuration

### Sidebar Controls

#### Strategy Parameters
- **Use Live Data**: Toggle between real Yahoo Finance data and synthetic demo data
- **Short MA** (5-30 days): Short-term moving average period
- **Long MA** (30-120 days): Long-term moving average period
- **RSI Period** (7-28 days): Period for calculating momentum
- **Threshold** (0-1%): Minimum predicted return to trigger signals

#### Model Configuration
- **ML Model**: Choose between Random Forest, Gradient Boosting, or XGBoost
- **Test Size** (10-40%): Percentage of data reserved for testing
- **Retrain Window** (100-500 days): Number of recent days used for training

---

## 📊 Understanding the Metrics

### Top-Level Metrics
- **Action Signal**: BUY/SELL/HOLD recommendation with confidence percentage
- **Price**: Current Bitcoin price with daily change
- **RSI**: Momentum indicator (OB=Overbought, OS=Oversold, OK=Neutral)
- **Vol**: 7-day volatility percentage
- **Strat**: Strategy return vs market benchmark
- **Win**: Percentage of profitable trades

### Performance Metrics
- **Total Return**: Cumulative percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return measure (>1 is good, >2 is excellent)
- **Win Rate**: Percentage of winning trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns

---

## 🧠 How It Works

### 1. Data Collection
- Fetches historical Bitcoin price data (OHLCV)
- Calculates technical indicators and features
- Splits data into training and testing sets

### 2. Feature Engineering
The model uses 11 features:
- Moving averages (short & long term)
- MA difference (trend strength)
- Volatility (7-day & 30-day)
- Momentum (7-day & 14-day)
- RSI (momentum oscillator)
- Lagged returns (1, 2, 3 days)

### 3. Model Training
- Trains on recent data (configurable window)
- Predicts next-day returns
- Generates buy/sell signals based on threshold
- Evaluates performance on test set

### 4. Signal Generation
- **BUY**: When predicted return > threshold
- **SELL**: When predicted return < -threshold
- **HOLD**: When predicted return is within threshold range

### 5. Backtesting
- Simulates trading the strategy on historical data
- Compares against buy-and-hold benchmark
- Calculates risk metrics and trade statistics

---

## 🎯 Trading Strategy

### Current Strategy: **Momentum-Based**
The dashboard uses a **momentum/trend-following** approach:
- Waits for confirmation of trend direction
- Uses MA crossovers and RSI for timing
- Aims to ride trends while avoiding false signals

### Characteristics
✅ **Strengths**:
- Works well in trending markets
- Reduces whipsaw trades
- Clear entry/exit rules

❌ **Limitations**:
- Misses exact bottoms/tops (waits for confirmation)
- Struggles in sideways/choppy markets
- Doesn't account for your personal entry price

### Alternative Strategies
The models can be adapted for:
- **Mean Reversion**: Buy dips, sell rallies
- **Breakout Trading**: Enter on volatility expansion
- **Support/Resistance**: Trade at key price levels

---

## 📚 AI Insights Explained

Each chart includes AI-powered analysis that explains:
- **What the pattern means**: Interpretation of the visual data
- **Current market conditions**: Bullish, bearish, or neutral
- **What to watch for**: Key indicators of potential changes
- **Trading implications**: How this affects decision-making

Example insights:
- "RSI at 72 (overbought) suggests potential reversal"
- "Price near lower Bollinger Band may indicate oversold conditions"
- "High volatility (6.2%) means expect large price swings"

---

## 🔒 Security & Privacy

### Password Protection (Optional)
The dashboard includes optional password protection:

1. **Uncomment** in code:
   ```python
   if not check_password():
       st.stop()
   ```

2. **Create** `.streamlit/secrets.toml`:
   ```toml
   password = "your_sha256_hash_here"
   ```

3. **Generate hash**:
   ```python
   import hashlib
   password = "your_password"
   hash_value = hashlib.sha256(password.encode()).hexdigest()
   print(hash_value)
   ```

---

## 🚀 Deployment

### Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the main file (`app.py`)
   - Click "Deploy"

3. **Add Secrets** (if using password protection)
   - In Streamlit Cloud dashboard
   - Go to App Settings → Secrets
   - Add your password hash

---

## ⚠️ Important Disclaimers

### Educational Purpose Only
This dashboard is designed for **educational and research purposes**. It is NOT:
- ❌ Financial advice
- ❌ Investment recommendation
- ❌ Guaranteed to be profitable
- ❌ A substitute for professional guidance

### Risk Warning
**Cryptocurrency trading involves substantial risk of loss.**
- Past performance does not guarantee future results
- ML models can be wrong
- Market conditions change rapidly
- Never invest more than you can afford to lose

### No Guarantees
- The models are probabilistic, not deterministic
- Backtested performance may not reflect live trading
- Technical analysis has limitations
- Always do your own research (DYOR)

---

## 🤔 FAQ

### Q: Why does the model say "HOLD" when price is low?
**A:** The model uses a momentum strategy that waits for trend confirmation. It prioritizes avoiding "catching a falling knife" over buying at the absolute bottom. Low price doesn't automatically mean good entry - it looks for reversal signals.

### Q: What's the difference between the ML models?
**A:** All three are ensemble tree models with similar logic but different internal math:
- **Random Forest**: Averages many independent decision trees
- **Gradient Boosting**: Builds trees sequentially, learning from errors
- **XGBoost**: Optimized gradient boosting with better performance

### Q: Can I track my personal portfolio with this?
**A:** No, this is a market timing tool, not a portfolio tracker. It shows strategy performance, not your personal P&L. You'd need to separately track your entry prices and position sizes.

### Q: Why doesn't it use more advanced features?
**A:** The dashboard balances sophistication with performance and interpretability. More complex features (sentiment analysis, on-chain metrics) would require additional data sources and processing time.

### Q: How often should I retrain the model?
**A:** The "Retrain Window" controls how much recent data to use. Shorter windows (100-200 days) adapt faster to new conditions but may overfit. Longer windows (300-500 days) are more stable but slower to adapt.

---

## 🛠️ Customization

### Adding New Features
Edit the `add_features()` function:
```python
def add_features(df, ma_short, ma_long, rsi_period):
    # Add your custom technical indicators here
    data['Your_Feature'] = calculate_something(df)
    return data
```

### Changing Color Scheme
Modify the CSS in the `st.markdown()` sections:
```python
st.markdown("""
<style>
    /* Your custom styles here */
</style>
""", unsafe_allow_html=True)
```

### Adding New Models
Add to the model selection logic:
```python
if model_choice == "Your Model":
    model = YourModelClass(params)
```

---

## 🐛 Troubleshooting

### "XGBoost not installed" warning
```bash
pip install xgboost
```

### Charts not displaying
- Clear Streamlit cache: `st.cache_data.clear()`
- Restart the app
- Check matplotlib backend

### Live data not loading
- Check internet connection
- Verify Yahoo Finance is accessible
- Try demo mode as fallback

### CSS changes not appearing
- Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
- Clear browser cache
- Try incognito/private mode

---

## 📈 Future Enhancements

Potential features for future versions:
- [ ] Multiple cryptocurrency support
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Sentiment analysis integration
- [ ] Real-time alerts and notifications
- [ ] Portfolio tracking module
- [ ] Multi-timeframe analysis
- [ ] Options pricing models
- [ ] Risk management calculator
- [ ] Export reports to PDF
- [ ] API integration for automated trading

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Henrietta Atsenokhai**
- Email: henrietta.atsenokhai@gmail.com


---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data from [Yahoo Finance](https://finance.yahoo.com/)
- ML models from [Scikit-learn](https://scikit-learn.org/)
- Visualization with [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)


---

**© 2025 Henrietta Atsenokhai. All rights reserved.**

*Educational purposes only • Not financial advice*
