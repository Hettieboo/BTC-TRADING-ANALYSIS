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

**1. Distribution Analysis**
- **Mean Daily Return**: 0.203% (51.11% annualized)
- **Volatility**: 3.36% daily (53.37% annualized) 
- **Skewness**: -0.52 (negative skew indicates more extreme downside moves)
- **Kurtosis**: 11.13 (fat tails - extreme events occur more frequently than normal distribution)
- **Shapiro-Wilk Test**: p-value < 0.001 → **Reject normality** (returns are NOT normally distributed)

**2. Stationarity Tests (Augmented Dickey-Fuller)**
- **Price Series**: ADF = -0.24, p-value = 0.93 → **Non-stationary**
- **Returns Series**: ADF = -13.86, p-value < 0.001 → **Stationary** ✓
- **Interpretation**: Returns are suitable for time series modeling; differencing successfully removes trend

**3. Hypothesis Tests**
- **Mean Return vs Zero**: t-statistic = 2.57, p-value = 0.010 → Mean return is **statistically significant**
- **Ljung-Box Autocorrelation**: Test statistic = 27.45, p-value = 0.002 → Evidence of **autocorrelation** in returns
- **Sharpe Ratio**: 0.96 (annualized) - positive risk-adjusted returns during period

**4. Time Series Model**
- **ARIMA(1,0,1)**: AIC = -7161.00, BIC = -7138.99
- Model captures some temporal dependencies in return series

### Model Performance

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|-------|----------|---------|-----------|----------|
| Linear Regression | 0.0077 | -0.0450 | 0.0287 | 0.0207 |
| Ridge Regression | 0.0077 | -0.0449 | 0.0287 | 0.0207 |
| Lasso Regression | 0.0033 | -0.0069 | 0.0281 | 0.0204 |
| Random Forest | 0.8509 | -0.3614 | 0.0327 | 0.0247 |

**Key Findings:**
- **Negative Test R²**: All models show negative out-of-sample R², indicating predictions are worse than simply predicting the mean
- **Overfitting in Random Forest**: High train R² (0.85) but very poor test performance (-0.36) indicates severe overfitting
- **Best Performer**: Lasso Regression (least negative test R²), suggesting simpler models generalize better
- **Market Efficiency**: Results support the semi-strong form of market efficiency - technical indicators alone provide limited predictive power for daily returns

### Trading Strategy Performance

**Backtest Results (Test Period: 2024)**
- **Buy & Hold Return**: +141.54%
- **Strategy Return**: -76.77%
- **Underperformance**: -218.31%
- **Win Rate**: 43.95% (149 wins / 339 trades)
- **Strategy Sharpe Ratio**: -2.20 (negative risk-adjusted returns)
- **Maximum Drawdown**: -79.17%

**Critical Analysis:**
The strategy significantly underperformed buy-and-hold, which provides important lessons:

1. **Model Limitations**: Negative test R² translated directly to poor trading performance
2. **Transaction Costs Not Included**: Real performance would be even worse with fees (typically 0.1-0.5% per trade × 339 trades)
3. **Overfitting Risk**: Models that look good on paper often fail in live trading
4. **Market Context**: 2024 saw strong BTC performance; simple buy-and-hold was hard to beat
5. **Threshold Issues**: 0.05% prediction threshold may have been too aggressive given model accuracy

**What This Demonstrates:**
- Honest evaluation without cherry-picking results
- Understanding that low R² is common in financial prediction
- Recognition of the challenges in beating market returns
- Proper interpretation of statistical vs. practical significance
- Realistic assessment of model limitations

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
- **Fat tails**: Kurtosis of 11.13 shows extreme events occur far more frequently than normal distribution predicts
- **Negative skewness**: -0.52 indicates downside moves tend to be more extreme than upside moves
- **Volatility clustering**: High volatility periods tend to persist (confirmed by significant autocorrelation)
- **Market efficiency**: Negative test R² across all models supports the difficulty of predicting returns using only historical price data

### Why The Models Failed
This project demonstrates a crucial lesson in quantitative finance:

**Statistical Significance ≠ Practical Profitability**
- While we found statistically significant autocorrelation (p = 0.002), this wasn't strong enough to generate profitable trading signals
- The mean return was significantly positive (p = 0.01), but predicting *which days* would be positive proved extremely difficult
- Technical indicators capture some market dynamics but lack predictive power for next-day returns

**Overfitting vs Generalization**
- Random Forest achieved 85% train R² by memorizing training patterns
- This completely failed on new data (test R² = -0.36)
- Simpler models (Linear, Ridge, Lasso) performed better out-of-sample, though still poorly
- This is why proper validation and honest evaluation are critical

### What Success Would Look Like
For context, in quantitative finance:
- **Test R² > 0.01** for daily returns is considered meaningful
- **Sharpe Ratio > 1.0** indicates decent risk-adjusted returns
- **Win Rate > 50%** with proper position sizing can be profitable
- Our results fell short on all metrics, which is honest and expected for a simple approach

### Practical Trading Implications
1. **Transaction Costs Matter**: 339 trades × 0.2% fees = 68% lost to fees alone
2. **Market Timing is Hard**: Even with perfect 2020-2024 hindsight, beating buy-and-hold is challenging
3. **Regime Sensitivity**: Models trained on one period often fail when market dynamics change
4. **Feature Limitations**: Price-based features may need augmentation with sentiment, on-chain metrics, or macro data

### Academic Rigor
This project demonstrates:
- **Honest reporting**: Publishing negative results shows integrity
- **Proper methodology**: Time series splits, no data leakage, multiple model comparison
- **Critical thinking**: Understanding *why* results are poor is as valuable as good results
- **Real-world awareness**: Acknowledging that 141% buy-and-hold return would be hard to beat
- **Statistical literacy**: Distinguishing between statistical significance and practical importance

---

## 🎓 Educational Value

This project showcases proficiency in:
- **Applied Statistics**: Hypothesis testing, time series analysis, probability theory
- **Machine Learning**: Supervised learning, model evaluation, feature engineering
- **Financial Econometrics**: Return calculations, risk metrics, backtesting methodology
- **Python Programming**: Pandas, NumPy, Scikit-learn, Statsmodels
- **Scientific Communication**: Clear documentation, visualization, interpretation

---

## 📧 Contact

**Author**: Henrietta Atsenokhai  
**Email**: henrietta.atsenokhai@gmail.com  
**Phone**: +33 7 58 75 06 82  
**LinkedIn**: [www.linkedin.com/in/henrietta-a-19810b280](https://www.linkedin.com/in/henrietta-a-19810b280)  
**GitHub**: [https://github.com/Hettieb](https://github.com/Hettieb)  
**Location**: Neuilly Sur-Marne, Île-de-France, France

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Data source: Yahoo Finance via yfinance library
- Statistical methods: Statsmodels documentation
- Inspiration: Quantitative finance literature and academic papers on market efficiency

---

**Disclaimer**: This project is for educational and demonstration purposes only. It does not constitute financial advice. Cryptocurrency trading carries substantial risk of loss. Past performance does not guarantee future results.
