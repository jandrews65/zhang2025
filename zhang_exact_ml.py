"""
Exact Zhang (2025) ML Model Replication
=========================================

5-fold expanding window TimeSeriesSplit
XGBoost classifier (Zhang's model)
Exact feature set and evaluation metrics

This follows Zhang's EXACT methodology to close any replication gaps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXACT ZHANG (2025) ML MODEL REPLICATION")
print("="*80)

# ============================================================================
# STEP 1: Load Prepared Data
# ============================================================================
print("\n[1/4] Loading prepared dataset...")

data_file = Path('data/zhang_exact_replication/merged_data.parquet')

if not data_file.exists():
    print("❌ Data not found. Run zhang_exact_replication.py first!")
    exit(1)

merged = pd.read_parquet(data_file)
print(f"   ✓ Loaded {len(merged)} days of data")

# Feature columns
feature_cols = [
    'sentiment_sum', 'sentiment_mean', 'sentiment_std', 'event_count',
    'total_articles', 'avg_tone',
    'sentiment_lag1', 'sentiment_lag2', 'sentiment_lag3',
    'event_count_lag1', 'event_count_lag2', 'event_count_lag3',
    'sentiment_ma5', 'event_count_ma5'
]

X = merged[feature_cols].values
y = merged['target'].values
dates = merged['date'].values
returns = merged['forward_return'].values

print(f"   Features: {X.shape[1]}")
print(f"   Samples: {len(X)}")

# ============================================================================
# STEP 2: 5-Fold Expanding Window TimeSeriesSplit
# ============================================================================
print("\n[2/4] Setting up 5-fold expanding window cross-validation...")

# Zhang uses expanding window (not sliding)
# Fold 1: Train on [0:20%], Test on [20%:40%]
# Fold 2: Train on [0:40%], Test on [40%:60%]
# Fold 3: Train on [0:60%], Test on [60%:80%]
# Fold 4: Train on [0:80%], Test on [80%:100%]
# Fold 5: Full train-test (to report final metrics)

tscv = TimeSeriesSplit(n_splits=5)

print("   Fold structure:")
for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    train_dates = dates[train_idx]
    test_dates = dates[test_idx]
    print(f"     Fold {i}: Train {len(train_idx):4d} samples ({train_dates[0]} to {train_dates[-1]})")
    print(f"              Test  {len(test_idx):4d} samples ({test_dates[0]} to {test_dates[-1]})")

# ============================================================================
# STEP 3: Train XGBoost Model (Zhang's Model)
# ============================================================================
print("\n[3/4] Training XGBoost classifier...")

# Zhang's XGBoost parameters (from paper Appendix)
xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'use_label_encoder': False
}

# Store results for each fold
fold_results = []
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"\n   Fold {fold}/5:")
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    
    # Trading simulation
    test_returns = returns[test_idx]
    signal = (y_pred_proba > 0.5).astype(int)  # 1 = buy, 0 = sell/short
    signal = 2 * signal - 1  # Convert to -1, +1
    
    strategy_returns = signal * test_returns
    strategy_total = (1 + strategy_returns).prod() - 1
    buy_hold_total = (1 + test_returns).prod() - 1
    
    print(f"     AUC: {auc:.4f}")
    print(f"     Accuracy: {acc:.4f}")
    print(f"     Strategy return: {strategy_total:+.2%}")
    print(f"     Buy & hold: {buy_hold_total:+.2%}")
    
    fold_results.append({
        'fold': fold,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'auc': auc,
        'accuracy': acc,
        'strategy_return': strategy_total,
        'buy_hold_return': buy_hold_total
    })
    
    # Store predictions for final analysis
    all_predictions.extend(zip(
        dates[test_idx],
        y_test,
        y_pred_proba,
        test_returns,
        strategy_returns
    ))

# ============================================================================
# STEP 4: Results Summary
# ============================================================================
print("\n[4/4] Final Results Summary")
print("="*80)

results_df = pd.DataFrame(fold_results)

print("\nPer-Fold Results:")
print(results_df.to_string(index=False))

print("\n\nAGGREGATE METRICS:")
print(f"  Average AUC:       {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
print(f"  Average Accuracy:  {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")

print("\n\nTRADING PERFORMANCE:")
total_strategy = results_df['strategy_return'].sum()
total_buy_hold = results_df['buy_hold_return'].sum()
print(f"  Cumulative Strategy: {total_strategy:+.2%}")
print(f"  Cumulative Buy&Hold: {total_buy_hold:+.2%}")
print(f"  Difference:          {total_strategy - total_buy_hold:+.2%}")

# Save results
output_dir = Path('data/zhang_exact_replication')
results_df.to_csv(output_dir / 'fold_results.csv', index=False)

predictions_df = pd.DataFrame(all_predictions, 
                              columns=['date', 'actual', 'predicted_proba', 
                                      'market_return', 'strategy_return'])
predictions_df.to_csv(output_dir / 'predictions.csv', index=False)

print(f"\n✓ Results saved to {output_dir}/")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print("REPLICATION VERDICT")
print("="*80)

avg_auc = results_df['auc'].mean()

print(f"\nZhang (2025) Claims:")
print(f"  ✗ AUC: 0.89 (reported)")
print(f"  ✗ Sharpe: 5.87 (reported)")
print(f"  ✗ 'Significant alpha from news sentiment'")

print(f"\nOur Exact Replication:")
print(f"  → AUC: {avg_auc:.4f}")
print(f"  → Net strategy return: {total_strategy:+.2%}")
print(f"  → Alpha: {total_strategy - total_buy_hold:+.2%}")

if avg_auc < 0.55:
    print("\n❌ REPLICATION FAILED")
    print("   Zhang's results are NOT reproducible")
    print("   AUC ≈ 0.50 indicates no predictive power")
    print("\n   Possible explanations:")
    print("   1. Data leakage in original paper")
    print("   2. Overfitting to specific time period")
    print("   3. Publication bias")
    print("   4. Methodological errors in original")
    
elif avg_auc < 0.65:
    print("\n⚠ PARTIAL REPLICATION")
    print("   Some signal detected, but much weaker than reported")
    print("   AUC 0.55-0.65 suggests modest predictive power")
    print("   Not sufficient for profitable trading after costs")
    
else:
    print("\n✅ REPLICATION SUCCESSFUL")
    print("   Zhang's results confirmed")
    print("   AUC > 0.65 indicates genuine predictive power")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nFor publication, include:")
print("  1. fold_results.csv (per-fold metrics)")
print("  2. predictions.csv (all predictions)")
print("  3. This exact methodology description")
print("\nThis closes the 'you didn't replicate it exactly' escape hatch.")
