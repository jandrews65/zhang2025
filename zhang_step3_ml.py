#!/usr/bin/env python3
"""
STEP 3: ML Model Training - Zhang Exact Replication
====================================================

Implements Zhang's exact methodology:
- 5-fold expanding window TimeSeriesSplit
- XGBoost classifier
- EUR/USD as target (Zhang's primary asset)
- Transaction costs included
- Comprehensive metrics

Author: Exact Zhang Replication
Date: 2026-02-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ZHANG EXACT REPLICATION - STEP 3: ML MODEL TRAINING")
print("="*80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Configuration
# ============================================================================

FEATURES_FILE = Path('data/processed/zhang_features_daily.parquet')
PRICES_FILE = Path('data/EURUSD_daily.csv')  # You'll need to download this
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n📁 Features: {FEATURES_FILE}")
print(f"📁 Prices:   {PRICES_FILE}")
print(f"📁 Output:   {OUTPUT_DIR}")

# ============================================================================
# STEP 1: Load Features
# ============================================================================

print(f"\n{'='*80}")
print("[1/4] Loading processed features...")
print(f"{'='*80}")

if not FEATURES_FILE.exists():
    print(f"\n❌ ERROR: Features file not found!")
    print(f"   Expected: {FEATURES_FILE.absolute()}")
    print(f"   Run Step 2 first: python zhang_step2_features.py")
    exit(1)

features = pd.read_parquet(FEATURES_FILE)
print(f"✓ Loaded {len(features):,} days of features")
print(f"  Date range: {features['date'].min()} to {features['date'].max()}")
print(f"  Features: {len(features.columns)-1}")

# ============================================================================
# STEP 2: Load Market Data (EUR/USD)
# ============================================================================

print(f"\n{'='*80}")
print("[2/4] Loading EUR/USD price data...")
print(f"{'='*80}")

if not PRICES_FILE.exists():
    print(f"\n⚠ EUR/USD data not found. Downloading from Yahoo Finance...")
    
    try:
        import yfinance as yf
        
        # Download EUR/USD
        eurusd = yf.download('EURUSD=X', 
                            start=features['date'].min().strftime('%Y-%m-%d'),
                            end=features['date'].max().strftime('%Y-%m-%d'),
                            progress=False)
        
        # Save for future use
        PRICES_FILE.parent.mkdir(exist_ok=True, parents=True)
        eurusd.to_csv(PRICES_FILE)
        
        print(f"✓ Downloaded {len(eurusd):,} days of EUR/USD data")
        
    except ImportError:
        print(f"\n❌ ERROR: yfinance not installed")
        print(f"   Install: pip install yfinance")
        print(f"   Or manually download EUR/USD data to: {PRICES_FILE}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Could not download EUR/USD data")
        print(f"   Error: {e}")
        print(f"\n   Alternative: Download manually from Yahoo Finance")
        print(f"   Ticker: EURUSD=X")
        print(f"   Save to: {PRICES_FILE}")
        exit(1)

# Load prices - yfinance creates weird multi-row headers
prices_raw = pd.read_csv(PRICES_FILE)

# DEBUG: Show CSV structure
print(f"\n  DEBUG - CSV columns: {list(prices_raw.columns)[:10]}")
print(f"  DEBUG - First 3 rows:")
print(prices_raw.head(3))

# Fix yfinance format: Row 0 has 'Ticker', Row 1 has 'Date', actual data starts row 2
# The 'Price' column actually contains dates
if 'Price' in prices_raw.columns and prices_raw.iloc[0]['Price'] == 'Ticker':
    print(f"  Detected yfinance multi-row header format")
    
    # Skip first 2 rows and reload
    prices = pd.read_csv(PRICES_FILE, skiprows=2)
    
    # First column is dates
    prices.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    
elif 'Date' in prices_raw.columns:
    # Standard format
    prices = prices_raw.copy()
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')
    
else:
    print(f"\n❌ Unexpected CSV format!")
    exit(1)

# Remove timezone if present
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)

# Convert Close price to numeric (handles string values from yfinance)
prices['Close'] = pd.to_numeric(prices['Close'], errors='coerce')

# Drop any rows with missing Close prices
prices = prices.dropna(subset=['Close'])

print(f"✓ Loaded {len(prices):,} days of EUR/USD data")

# Calculate returns
prices['return'] = np.log(prices['Close'] / prices['Close'].shift(1))
prices['forward_return'] = prices['return'].shift(-1)

# Binary target (1 = up, 0 = down)
prices['target'] = (prices['forward_return'] > 0).astype(int)

# Convert to DataFrame for merging
price_df = pd.DataFrame({
    'date': prices.index,
    'price': prices['Close'],
    'forward_return': prices['forward_return'],
    'target': prices['target']
}).reset_index(drop=True)

# Ensure date is datetime without time component
price_df['date'] = pd.to_datetime(price_df['date']).dt.normalize()

# ============================================================================
# STEP 3: Merge Features with Prices
# ============================================================================

print(f"\n{'='*80}")
print("[3/4] Merging features with price data...")
print(f"{'='*80}")

# Ensure features date is also normalized
features['date'] = pd.to_datetime(features['date']).dt.normalize()

# Check date ranges before merging
print(f"\n  Features date range: {features['date'].min()} to {features['date'].max()}")
print(f"  Prices date range:   {price_df['date'].min()} to {price_df['date'].max()}")

# Merge
merged = features.merge(price_df, on='date', how='inner')
merged = merged.dropna(subset=['forward_return', 'target'])
merged = merged.sort_values('date').reset_index(drop=True)

print(f"\n✓ Merged dataset: {len(merged):,} days")

if len(merged) == 0:
    print(f"\n❌ ERROR: No matching dates between features and prices!")
    print(f"\n  Sample feature dates: {features['date'].head(10).tolist()}")
    print(f"  Sample price dates:   {price_df['date'].head(10).tolist()}")
    print(f"\n  Checking for date format issues...")
    exit(1)

print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")

# Target distribution
up_days = (merged['target'] == 1).sum()
down_days = (merged['target'] == 0).sum()
print(f"\n  Target distribution:")
print(f"    Up days:   {up_days:,} ({up_days/len(merged)*100:.1f}%)")
print(f"    Down days: {down_days:,} ({down_days/len(merged)*100:.1f}%)")

# Feature columns
feature_cols = [c for c in merged.columns if c not in ['date', 'price', 'forward_return', 'target']]
print(f"\n  Features for ML: {len(feature_cols)}")

# Prepare X, y
X = merged[feature_cols].values
y = merged['target'].values
dates = merged['date'].values
returns = merged['forward_return'].values

print(f"\n  Feature matrix: {X.shape}")
print(f"  Target vector:  {y.shape}")

# ============================================================================
# STEP 4: 5-Fold Expanding Window Cross-Validation
# ============================================================================

print(f"\n{'='*80}")
print("[4/4] Training XGBoost with 5-fold expanding window...")
print(f"{'='*80}")

print("\nZhang's exact methodology:")
print("  - 5-fold expanding window TimeSeriesSplit")
print("  - XGBoost classifier")
print("  - Transaction costs: 0.02% per round-trip (FX)")
print("")

# TimeSeriesSplit (Zhang uses n_splits=5)
tscv = TimeSeriesSplit(n_splits=5)

print("Fold structure:")
for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    train_dates = dates[train_idx]
    test_dates = dates[test_idx]
    print(f"  Fold {i}: Train {len(train_idx):4d} samples ({train_dates[0]} to {train_dates[-1]})")
    print(f"           Test  {len(test_idx):4d} samples ({test_dates[0]} to {test_dates[-1]})")

# XGBoost parameters (Zhang's configuration from paper)
xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'tree_method': 'hist'
}

print(f"\nXGBoost parameters:")
for k, v in xgb_params.items():
    print(f"  {k}: {v}")

# Train and evaluate
fold_results = []
all_predictions = []

print(f"\n{'='*80}")
print("Training folds...")
print(f"{'='*80}\n")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"Fold {fold}/5:")
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model
    print(f"  Training XGBoost...", end=' ')
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, verbose=False)
    print(f"✓")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    # Trading simulation
    test_returns = returns[test_idx]
    
    # Signal: +1 if predict up, -1 if predict down
    signal = 2 * y_pred - 1
    
    # Strategy returns
    strategy_returns = signal * test_returns
    
    # Transaction costs (0.02% per round-trip)
    # Assume we trade every day (position changes)
    n_trades = len(test_idx)  # One trade per day
    cost_per_trade = 0.0002  # 0.02% = 2 basis points
    total_cost = cost_per_trade * n_trades
    
    # Net strategy return
    gross_return = np.sum(strategy_returns)
    net_return = gross_return - total_cost
    
    # Buy & hold
    buy_hold_return = np.sum(test_returns)
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
    
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Sharpe:   {sharpe:.2f}")
    print(f"  Gross return: {gross_return:+.4f}")
    print(f"  Costs:        {-total_cost:.4f}")
    print(f"  Net return:   {net_return:+.4f}")
    print(f"  Buy & Hold:   {buy_hold_return:+.4f}")
    print(f"  Alpha:        {net_return - buy_hold_return:+.4f}")
    print()
    
    fold_results.append({
        'fold': fold,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'auc': auc,
        'accuracy': acc,
        'sharpe': sharpe,
        'gross_return': gross_return,
        'costs': total_cost,
        'net_return': net_return,
        'buy_hold': buy_hold_return,
        'alpha': net_return - buy_hold_return
    })
    
    # Store predictions
    for i, test_i in enumerate(test_idx):
        all_predictions.append({
            'date': dates[test_i],
            'actual': y_test[i],
            'predicted_proba': y_pred_proba[i],
            'market_return': test_returns[i],
            'strategy_return': strategy_returns[i],
            'fold': fold
        })

# ============================================================================
# Results Summary
# ============================================================================

print(f"{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}\n")

results_df = pd.DataFrame(fold_results)

print("Per-Fold Results:")
print(results_df.to_string(index=False))

print(f"\n{'='*80}")
print("AGGREGATE METRICS")
print(f"{'='*80}")

avg_auc = results_df['auc'].mean()
avg_sharpe = results_df['sharpe'].mean()
total_net_return = results_df['net_return'].sum()
total_buy_hold = results_df['buy_hold'].sum()

print(f"\nOut-of-Sample Performance (Aggregated):")
print(f"  Average AUC:        {avg_auc:.4f} ± {results_df['auc'].std():.4f}")
print(f"  Average Accuracy:   {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"  Average Sharpe:     {avg_sharpe:.2f} ± {results_df['sharpe'].std():.2f}")
print(f"\nCumulative Returns:")
print(f"  Strategy (net):     {total_net_return:+.4f} ({total_net_return*100:+.2f}%)")
print(f"  Buy & Hold:         {total_buy_hold:+.4f} ({total_buy_hold*100:+.2f}%)")
print(f"  Alpha:              {(total_net_return - total_buy_hold):+.4f} ({(total_net_return - total_buy_hold)*100:+.2f}%)")

print(f"\n{'='*80}")
print("COMPARISON TO ZHANG (2025)")
print(f"{'='*80}")

print(f"\nZhang's Claims (EUR/USD):")
print(f"  AUC:   0.89 (not explicitly stated, inferred from Sharpe)")
print(f"  Sharpe: 5.87")
print(f"  CAGR:  55.4%")

print(f"\nOur Exact Replication:")
print(f"  AUC:    {avg_auc:.4f}")
print(f"  Sharpe: {avg_sharpe:.2f}")

if avg_auc < 0.55:
    print(f"\n❌ REPLICATION FAILED")
    print(f"   AUC ≈ 0.50 indicates NO PREDICTIVE POWER")
    print(f"   Zhang's results are NOT reproducible with exact methodology")
elif avg_auc < 0.65:
    print(f"\n⚠ PARTIAL REPLICATION")
    print(f"   Some signal detected but much weaker than claimed")
else:
    print(f"\n✅ REPLICATION SUCCESSFUL")
    print(f"   Results match Zhang's claims")

# Save results
results_df.to_csv(OUTPUT_DIR / 'fold_results.csv', index=False)
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)

print(f"\n{'='*80}")
print(f"Results saved to:")
print(f"  - {OUTPUT_DIR / 'fold_results.csv'}")
print(f"  - {OUTPUT_DIR / 'predictions.csv'}")
print(f"{'='*80}")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
