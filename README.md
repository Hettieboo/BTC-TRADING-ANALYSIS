# BTC-TRADING-ANALYSIS
# Bitcoin Algorithmic Trading Strategy
## Advanced Statistical Analysis & Predictive Modeling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project demonstrates advanced statistical analysis and machine learning techniques applied to cryptocurrency trading. Using Bitcoin (BTC-USD) historical price data from 2020-2024, I develop and backtest a predictive trading strategy that leverages time series analysis, hypothesis testing, and multiple machine learning models.

**Key Objectives:**
- Apply rigorous statistical methods to financial time series data
- Develop and compare multiple predictive models
- Implement a backtested trading strategy with proper evaluation metrics
- Demonstrate proficiency in statistical modeling, hypothesis testing, and experimental design

---

## 🎯 Skills Demonstrated

### Statistical Methods
- **Descriptive Statistics**: Mean, variance, skewness, kurtosis, Sharpe ratio
- **Hypothesis Testing**: Shapiro-Wilk normality test, one-sample t-test, Ljung-Box autocorrelation test
- **Time Series Analysis**: Augmented Dickey-Fuller stationarity test, ARIMA modeling, ACF/PACF analysis
- **Experimental Design**: Proper train-test splitting for temporal data, model comparison framework

### Machine Learning
- **Models**: Linear Regression, Ridge Regression, Lasso Regression, Random Forest
- **Validation**: Time series cross-validation, proper temporal splitting
- **Evaluation**: R², RMSE, MAE, feature importance analysis
- **Feature Engineering**: Technical indicators (RSI, moving averages, volatility, momentum)

### Tools & Libraries
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy, Statsmodels
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn

---

## 📊 Methodology

### 1. Data Acquisition
- **Source**: Yahoo Finance (yfinance API)
- **Asset**: Bitcoin (BTC-USD)
- **Period**: January 2020 - December 2024
- **Frequency**: Daily OHLCV (Open, High, Low, Close, Volume)

### 2. Exploratory Data Analysis
- Summary statistics and distribution analysis
- Returns calculation (simple and logarithmic)
- Volatility assessment
- Visual inspection of price trends and patterns

### 3. Statistical Hypothesis Testing

#### Test 1: Normality of Returns
- **Method**: Shapiro-Wilk test
- **Hypothesis**: H₀: Returns follow a normal distribution
- **Result**: Rejected (p < 0.05)
- **Interpretation**: Returns exhibit fat tails and are not normally distributed, consistent with financial asset behavior

#### Test 2: Mean Return Significance
- **Method**: One-sample t-test
- **Hypothesis**: H₀: Mean daily return = 0
- **Result**: Statistical significance indicates positive drift in BTC prices

#### Test 3: Autocorrelation
- **Method**: Ljung-Box test
- **Hypothesis**: H₀: No autocorrelation in returns
- **Interpretation**: Tests for predictable patterns in return series

### 4. Time Series Analysis

#### Stationarity Testing
- **Price Series**: Non-stationary (p-value > 0.05)
- **Returns Series**: Stationary (p-value < 0.05)
- **Implication**: Returns are suitable for modeling; differencing transforms non-stationary price to stationary returns

#### ARIMA Modeling
- Fitted ARIMA(1,0,1) model to returns
- Evaluated using AIC and BIC information criteria
- ACF/PACF analysis for order selection

### 5. Feature Engineering

Created technical indicators commonly used in quantitative trading:

**Trend Indicators:**
- Moving Averages (7-day, 30-day, 90-day)
- Momentum (7-day, 30-day price changes)

**Volatility Indicators:**
- Rolling standard deviation (7-day, 30-day)
- Volatility clustering analysis

**Momentum Oscillators:**
- Relative Strength Index (RSI)
- Volume ratios

**Lag Features:**
- Historical returns (1, 2, 3, 5, 7 days)

### 6. Predictive Modeling

#### Model Comparison
Trained and evaluated four regression models:

1. **Linear Regression** (baseline)
2. **Ridge Regression** (L2 regularization)
3. **Lasso Regression** (L1 regularization, feature selection)
4. **Random Forest** (non-linear, ensemble method)

#### Validation Strategy
- **80/20 temporal split** (no data leakage from future to past)
- Standardized features using training set parameters
- Evaluated on held-out test set

#### Performance Metrics
- R² Score (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Feature importance rankings

**Key Finding**: Random Forest achieved the best out-of-sample performance, capturing non-linear relationships between technical indicators and future returns.

### 7. Trading Strategy Backtesting

#### Strategy Design
- **Signal Generation**: Predicted returns from best-performing model (Random Forest)
- **Position**: Long when predicted return > threshold, Short when < -threshold, Neutral otherwise
- **Threshold**: 0.05% to filter noise and reduce transaction frequency

#### Performance Evaluation
Compared strategy against buy-and-hold benchmark:

**Metrics Calculated:**
- Total return
- Annualized volatility
- Sharpe ratio (risk-adjusted return)
- Maximum drawdown
- Win rate
- Number of trades

**Risk Considerations:**
- No transaction costs included (conservative estimate)
- No slippage modeling
- Assumes perfect execution at close prices

---

## 📈 Key Results

### Statistical Findings
1. **Non-Normal Distribution**: Bitcoin returns exhibit significant fat tails and are not normally distributed (reject normality hypothesis)
2. **Stationarity**: Price series is non-stationary; returns series is stationary (suitable for modeling)
3. **Positive Drift**: Mean daily return is statistically significant and positive over the analysis period
4. **Autocorrelation**: Evidence of weak autocorrelation in returns, suggesting potential predictability

### Model Performance
- **Best Model**: Random Forest Regressor
- **Test R²**: ~0.02-0.05 (typical for daily return prediction in efficient markets)
- **Important Features**: Recent returns, volatility measures, and RSI most predictive

### Trading Strategy
- **Strategy Return**: Outperformed buy-and-hold in backtesting period
- **Sharpe Ratio**: Improved risk-adjusted returns
- **Practical Considerations**: Results are theoretical; real-world implementation would face transaction costs, slippage, and execution risk

---

## 🔍 Assumptions & Limitations

### Assumptions
1. Markets are not perfectly efficient; historical patterns contain predictive information
2. Technical indicators capture relevant market dynamics
3. Training period patterns persist into test period
4. Price data is accurate and free from survivorship bias

### Limitations
1. **Look-Ahead Bias**: Carefully avoided by using only historical data for predictions
2. **Transaction Costs**: Not included in backtest (reduces real returns)
3. **Sample Period**: Limited to 2020-2024; may not generalize to different market regimes
4. **Model Risk**: Past performance does not guarantee future results
5. **Overfitting**: Mitigated through cross-validation but remains a risk
6. **Market Impact**: Assumes small position sizes with no price impact

---

## 🚀 Future Improvements

### Model Enhancements
- **Deep Learning**: LSTM networks for sequence modeling
- **Ensemble Methods**: Combine multiple models with different strengths
- **Alternative Data**: Incorporate sentiment analysis, on-chain metrics, macroeconomic indicators

### Strategy Refinements
- **Position Sizing**: Kelly criterion or risk parity approach
- **Risk Management**: Dynamic stop-loss, trailing stops, portfolio diversification
- **Transaction Cost Model**: Realistic fee structure and slippage assumptions
- **Regime Detection**: Separate models for bull/bear/sideways markets

### Statistical Rigor
- **Bayesian Methods**: Posterior distributions for parameter uncertainty
- **Monte Carlo Simulation**: Stress testing under various scenarios
- **Walk-Forward Analysis**: Rolling window optimization and out-of-sample testing

---

## 📁 Project Structure

```
btc-trading-analysis/
│
├── README.md                          # This file
├── btc_trading_analysis.ipynb         # Main analysis notebook
├── requirements.txt                   # Python dependencies
│
├── outputs/
│   ├── btc_eda_analysis.png          # EDA visualizations
│   └── btc_ml_strategy.png           # Model and strategy results
│
└── data/
    └── BTC-USD.csv                    # Historical price data (if using CSV)
```

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone or download the project**
```bash
git clone <repository-url>
cd btc-trading-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analysis**
```bash
jupyter notebook btc_trading_analysis.ipynb
```

The notebook will automatically:
- Download BTC price data via yfinance
- Perform all statistical analyses
- Train and evaluate models
- Generate visualizations
- Save output charts

---

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
statsmodels>=0.14.0
scipy>=1.9.0
scikit-learn>=1.2.0
yfinance>=0.2.0
```

---

## 📝 Interpretation & Insights

### Statistical Perspective
The analysis confirms several well-documented characteristics of cryptocurrency returns:
- **Fat tails**: Extreme events occur more frequently than normal distribution predicts
- **Volatility clustering**: High volatility periods tend to persist
- **Weak form efficiency**: Some predictability exists, though exploiting it is challenging

### Practical Trading Implications
1. **Risk Management is Critical**: High volatility and fat tails make position sizing crucial
2. **Transaction Costs Matter**: Small edge can be eliminated by fees and slippage
3. **Regime Changes**: Models trained on one period may fail in different market conditions
4. **Model Uncertainty**: Low R² is expected; financial markets are inherently noisy

### Academic Rigor
This project demonstrates:
- Proper experimental design with temporal data
- Multiple hypothesis testing with appropriate corrections
- Model validation without data leakage
- Honest assessment of limitations
- Statistical reasoning throughout analysis

---

## Value

This project showcases proficiency in:
- **Applied Statistics**: Hypothesis testing, time series analysis, probability theory
- **Machine Learning**: Supervised learning, model evaluation, feature engineering
- **Financial Econometrics**: Return calculations, risk metrics, backtesting methodology
- **Python Programming**: Pandas, NumPy, Scikit-learn, Statsmodels
- **Scientific Communication**: Clear documentation, visualization, interpretation

---

## 📧 Contact

**Author**: [Henrietta Atsenokhai]  
**Email**: [henrietta.atsenokhai@gmail.com]  


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Data source: Yahoo Finance via yfinance library
- Statistical methods: Statsmodels documentation
- Inspiration: Quantitative finance literature and academic papers on market efficiency

---

**Disclaimer**: This project is for educational and demonstration purposes only. It does not constitute financial advice. Cryptocurrency trading carries substantial risk of loss. Past performance does not guarantee future results.
