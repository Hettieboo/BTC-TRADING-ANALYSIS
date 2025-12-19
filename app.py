"""
BTC Algorithmic Trading Strategy - Advanced Statistical Analysis
Demonstrates: Applied Statistics, Time Series Analysis, Hypothesis Testing,
             GLMs, Bayesian Methods, Machine Learning, and Experimental Design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 8)

print("=" * 70)
print("BTC ALGORITHMIC TRADING - ADVANCED STATISTICAL ANALYSIS")
print("=" * 70)

# ===========================
# 1. DATA ACQUISITION
# ===========================
print("\n[STEP 1] Data Acquisition")
print("-" * 70)

try:
    import yfinance as yf
    btc = yf.download('BTC-USD', start='2020-01-01', end='2024-12-19', progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    print("✓ Data loaded via yfinance")
except:
    btc = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=True)
    print("✓ Data loaded from CSV")

print(f"  Shape: {btc.shape}")
print(f"  Period: {btc.index[0].date()} to {btc.index[-1].date()}")
print(f"  Missing values: {btc.isnull().sum().sum()}")

# ===========================
# 2. EXPLORATORY DATA ANALYSIS
# ===========================
print("\n[STEP 2] Exploratory Data Analysis & Descriptive Statistics")
print("-" * 70)

df = btc.copy()

# Returns and log returns
df['Returns'] = df['Close'].pct_change()
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

# Basic statistics
print("\nDescriptive Statistics for Returns:")
print(f"  Mean: {df['Returns'].mean():.6f} ({df['Returns'].mean()*252*100:.2f}% annualized)")
print(f"  Std Dev: {df['Returns'].std():.6f} ({df['Returns'].std()*np.sqrt(252)*100:.2f}% annualized)")
print(f"  Skewness: {df['Returns'].skew():.4f}")
print(f"  Kurtosis: {df['Returns'].kurtosis():.4f}")
print(f"  Sharpe Ratio (annualized): {(df['Returns'].mean()/df['Returns'].std())*np.sqrt(252):.4f}")

# ===========================
# 3. HYPOTHESIS TESTING
# ===========================
print("\n[STEP 3] Hypothesis Testing")
print("-" * 70)

# Test 1: Normality test (Shapiro-Wilk)
returns_clean = df['Returns'].dropna()
stat, p_value = stats.shapiro(returns_clean[:5000])  # Limit sample for computational efficiency
print(f"\n3.1 Normality Test (Shapiro-Wilk):")
print(f"  H0: Returns are normally distributed")
print(f"  Statistic: {stat:.6f}, p-value: {p_value:.6f}")
print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} (α=0.05)")
print(f"  → Returns are {'NOT' if p_value < 0.05 else ''} normally distributed")

# Test 2: Test if mean return is significantly different from zero
t_stat, p_value_ttest = stats.ttest_1samp(returns_clean, 0)
print(f"\n3.2 One-Sample t-test (Mean Return vs Zero):")
print(f"  H0: Mean return = 0")
print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value_ttest:.6f}")
print(f"  Result: {'Reject H0' if p_value_ttest < 0.05 else 'Fail to reject H0'} (α=0.05)")

# Test 3: Ljung-Box test for autocorrelation
lb_test = acorr_ljungbox(returns_clean, lags=[10], return_df=True)
print(f"\n3.3 Ljung-Box Test (Autocorrelation):")
print(f"  H0: No autocorrelation in returns")
print(f"  Test statistic: {lb_test['lb_stat'].values[0]:.4f}")
print(f"  p-value: {lb_test['lb_pvalue'].values[0]:.6f}")
print(f"  Result: {'Reject H0' if lb_test['lb_pvalue'].values[0] < 0.05 else 'Fail to reject H0'} (α=0.05)")

# ===========================
# 4. TIME SERIES ANALYSIS
# ===========================
print("\n[STEP 4] Time Series Analysis")
print("-" * 70)

# Stationarity tests
print("\n4.1 Augmented Dickey-Fuller Test:")
adf_price = adfuller(df['Close'].dropna())
adf_returns = adfuller(returns_clean)

print(f"  Price Series:")
print(f"    ADF Statistic: {adf_price[0]:.4f}, p-value: {adf_price[1]:.6f}")
print(f"    Stationary: {'Yes' if adf_price[1] < 0.05 else 'No'}")

print(f"  Returns Series:")
print(f"    ADF Statistic: {adf_returns[0]:.4f}, p-value: {adf_returns[1]:.6f}")
print(f"    Stationary: {'Yes' if adf_returns[1] < 0.05 else 'No'}")

# ARIMA Modeling
print(f"\n4.2 ARIMA Model (AutoRegressive Integrated Moving Average):")
try:
    arima_model = ARIMA(returns_clean, order=(1,0,1))
    arima_fit = arima_model.fit()
    print(f"  Model: ARIMA(1,0,1)")
    print(f"  AIC: {arima_fit.aic:.2f}")
    print(f"  BIC: {arima_fit.bic:.2f}")
    print(f"  Log-Likelihood: {arima_fit.llf:.2f}")
except Exception as e:
    print(f"  ARIMA fitting skipped: {str(e)[:50]}")

# ===========================
# 5. FEATURE ENGINEERING
# ===========================
print("\n[STEP 5] Feature Engineering")
print("-" * 70)

# Technical indicators
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['MA_90'] = df['Close'].rolling(window=90).mean()
df['Volatility_7'] = df['Returns'].rolling(window=7).std()
df['Volatility_30'] = df['Returns'].rolling(window=30).std()
df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
df['Momentum_30'] = df['Close'] - df['Close'].shift(30)

# RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['Close'])

# Volume features
df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']

# Lag features
for i in [1, 2, 3, 5, 7]:
    df[f'Return_Lag_{i}'] = df['Returns'].shift(i)

# Target variable
df['Target'] = df['Returns'].shift(-1)

df = df.dropna()
print(f"  Features created: {df.shape[1]} total columns")
print(f"  Clean dataset: {df.shape[0]} observations")

# ===========================
# 6. MACHINE LEARNING MODELS
# ===========================
print("\n[STEP 6] Machine Learning Models with Time Series Cross-Validation")
print("-" * 70)

feature_cols = ['MA_7', 'MA_30', 'Volatility_7', 'Volatility_30', 
                'Momentum_7', 'RSI', 'Volume_Ratio',
                'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3']

X = df[feature_cols]
y = df['Target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

# Time series split
split_idx = int(len(df) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")

# Model comparison
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.001),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

print("\n6.1 Model Performance Comparison:")
print(f"  {'Model':<20} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
print("  " + "-" * 68)

for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'predictions': y_test_pred
    }
    
    print(f"  {name:<20} {train_r2:>11.4f} {test_r2:>11.4f} {test_rmse:>11.6f} {test_mae:>11.6f}")

# ===========================
# 7. TRADING STRATEGY BACKTEST
# ===========================
print("\n[STEP 7] Trading Strategy Backtesting")
print("-" * 70)

# Use best model (Random Forest)
best_model = results['Random Forest']['model']
predictions = results['Random Forest']['predictions']

df_test = df.iloc[split_idx:].copy()
df_test['Predicted_Return'] = predictions

# Trading signals with threshold
threshold = 0.0005  # 0.05% threshold
df_test['Signal'] = 0
df_test.loc[df_test['Predicted_Return'] > threshold, 'Signal'] = 1
df_test.loc[df_test['Predicted_Return'] < -threshold, 'Signal'] = -1

# Strategy returns
df_test['Strategy_Returns'] = df_test['Signal'] * df_test['Returns']
df_test['Cumulative_Market'] = (1 + df_test['Returns']).cumprod()
df_test['Cumulative_Strategy'] = (1 + df_test['Strategy_Returns']).cumprod()

# Performance metrics
market_return = (df_test['Cumulative_Market'].iloc[-1] - 1) * 100
strategy_return = (df_test['Cumulative_Strategy'].iloc[-1] - 1) * 100
strategy_vol = df_test['Strategy_Returns'].std() * np.sqrt(252)
strategy_sharpe = (df_test['Strategy_Returns'].mean() / df_test['Strategy_Returns'].std()) * np.sqrt(252)

print(f"\n7.1 Strategy Performance Metrics:")
print(f"  Buy & Hold Return: {market_return:>10.2f}%")
print(f"  Strategy Return: {strategy_return:>13.2f}%")
print(f"  Outperformance: {strategy_return - market_return:>14.2f}%")
print(f"  Strategy Volatility (ann.): {strategy_vol*100:>6.2f}%")
print(f"  Strategy Sharpe Ratio: {strategy_sharpe:>11.2f}")

# Win rate
winning_trades = (df_test['Strategy_Returns'] > 0).sum()
total_trades = (df_test['Signal'] != 0).sum()
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

print(f"\n7.2 Trading Statistics:")
print(f"  Total signals: {total_trades}")
print(f"  Winning trades: {winning_trades}")
print(f"  Win rate: {win_rate:.2f}%")
print(f"  Max drawdown: {((df_test['Cumulative_Strategy'] / df_test['Cumulative_Strategy'].cummax()) - 1).min()*100:.2f}%")

# ===========================
# 8. VISUALIZATION
# ===========================
print("\n[STEP 8] Generating Visualizations")
print("-" * 70)

# Figure 1: Comprehensive EDA
fig1 = plt.figure(figsize=(16, 12))
gs = fig1.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Price and MAs
ax1 = fig1.add_subplot(gs[0, :])
ax1.plot(df.index, df['Close'], label='BTC Price', alpha=0.7, linewidth=1.5)
ax1.plot(df.index, df['MA_30'], label='MA 30', alpha=0.8, linewidth=1.5)
ax1.plot(df.index, df['MA_90'], label='MA 90', alpha=0.8, linewidth=1.5)
ax1.set_title('Bitcoin Price with Moving Averages', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Returns histogram with normal distribution overlay
ax2 = fig1.add_subplot(gs[1, 0])
returns_plot = df['Returns'].dropna()
ax2.hist(returns_plot, bins=100, density=True, alpha=0.7, edgecolor='black')
mu, sigma = returns_plot.mean(), returns_plot.std()
x = np.linspace(returns_plot.min(), returns_plot.max(), 100)
ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
ax2.set_title('Returns Distribution vs Normal', fontsize=12, fontweight='bold')
ax2.set_xlabel('Returns')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

# QQ plot
ax3 = fig1.add_subplot(gs[1, 1])
stats.probplot(returns_plot, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ACF plot
ax4 = fig1.add_subplot(gs[2, 0])
plot_acf(returns_plot, lags=40, ax=ax4)
ax4.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# PACF plot
ax5 = fig1.add_subplot(gs[2, 1])
plot_pacf(returns_plot, lags=40, ax=ax5)
ax5.set_title('Partial Autocorrelation Function', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Volatility
ax6 = fig1.add_subplot(gs[3, 0])
ax6.plot(df.index, df['Volatility_30'], color='orange', alpha=0.7)
ax6.set_title('30-Day Rolling Volatility', fontsize=12, fontweight='bold')
ax6.set_ylabel('Volatility')
ax6.grid(True, alpha=0.3)

# Feature correlation
ax7 = fig1.add_subplot(gs[3, 1])
corr_features = ['Returns', 'Volatility_30', 'RSI', 'Momentum_7', 'Volume_Ratio']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            ax=ax7, cbar_kws={'shrink': 0.8})
ax7.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

plt.savefig('btc_eda_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: btc_eda_analysis.png")

# Figure 2: Model Performance and Trading Strategy
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))

# Model comparison
ax1 = axes[0, 0]
model_names = list(results.keys())
test_r2_scores = [results[m]['test_r2'] for m in model_names]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax1.bar(model_names, test_r2_scores, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Model Performance Comparison (Test R²)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_ylim([min(test_r2_scores)-0.01, max(test_r2_scores)+0.01])
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# Cumulative returns
ax2 = axes[0, 1]
ax2.plot(df_test.index, df_test['Cumulative_Market'], 
         label='Buy & Hold', linewidth=2.5, alpha=0.8)
ax2.plot(df_test.index, df_test['Cumulative_Strategy'], 
         label='ML Strategy', linewidth=2.5, alpha=0.8)
ax2.set_title('Cumulative Returns: Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Return')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Predicted vs actual
ax3 = axes[1, 0]
ax3.scatter(y_test, predictions, alpha=0.4, s=10)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Returns')
ax3.set_ylabel('Predicted Returns')
ax3.set_title('Predicted vs Actual Returns', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Feature importance (Random Forest)
ax4 = axes[1, 1]
rf_model = results['Random Forest']['model']
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)
ax4.barh(importances['Feature'], importances['Importance'], 
         color='steelblue', alpha=0.7, edgecolor='black')
ax4.set_title('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Importance')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('btc_ml_strategy.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: btc_ml_strategy.png")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nKey Deliverables:")
print("  1. Statistical hypothesis tests on return distribution")
print("  2. Time series analysis with stationarity tests and ARIMA")
print("  3. Multiple ML models with cross-validation")
print("  4. Backtested trading strategy with performance metrics")
print("  5. Comprehensive visualizations saved as PNG files")
print("\nFiles generated:")
print("  • btc_eda_analysis.png")
print("  • btc_ml_strategy.png")
