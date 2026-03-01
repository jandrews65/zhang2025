"""
Momentum Strategy - Full Historical Validation (2015-2023)
Run on Windows (network works here)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("MOMENTUM STRATEGY - FULL HISTORICAL VALIDATION")
print("="*80)
print("\nDownloading 9 years of FREE data from Yahoo Finance...")
print("(This may take 2-3 minutes)\n")

# Install yfinance if needed
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '-q'])
    import yfinance as yf

# ============================================================================
# STEP 1: Download Historical Data
# ============================================================================
print("\n1. Downloading historical data...")

SYMBOLS = {
    'USO': 'Oil ETF',
    'GLD': 'Gold ETF', 
    'SLV': 'Silver ETF',
    'SPY': 'S&P 500 ETF'
}

data = {}

for symbol, name in SYMBOLS.items():
    try:
        print(f"   {symbol} ({name})...", end=' ')
        df = yf.download(symbol, start='2015-01-01', end='2024-12-31', 
                        progress=False)
        
        if len(df) > 0:
            data[symbol] = df['Close']
            print(f"✓ {len(df)} days")
        else:
            print(f"✗ No data")
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")

if not data:
    print("\n❌ Failed to download data!")
    exit(1)

print(f"\n✓ Downloaded {len(data)} assets")

# ============================================================================
# STEP 2: Calculate Momentum Returns
# ============================================================================
print("\n2. Testing momentum strategy (2015-2023)...")

LOOKBACK = 20  # Winner from 60-day test
HOLDING = 5

results_summary = []

for symbol, prices in data.items():
    print(f"\n{'='*80}")
    print(f"{symbol} - FULL 9-YEAR BACKTEST")
    print(f"{'='*80}")
    
    # Calculate momentum
    df = pd.DataFrame({'price': prices})
    df['return'] = df['price'].pct_change()
    df['momentum'] = df['price'].pct_change(LOOKBACK)
    df['signal'] = (df['momentum'] > 0).astype(int)
    df['fwd_ret'] = df['price'].pct_change(HOLDING).shift(-HOLDING)
    df['strategy_ret'] = df['signal'].shift(1) * df['fwd_ret']
    
    df = df.dropna()
    
    # Overall performance
    total_strategy = (1 + df['strategy_ret']).prod() - 1
    total_buy_hold = (1 + df['fwd_ret']).prod() - 1
    
    # Annualized metrics
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr_strategy = (1 + total_strategy) ** (1/years) - 1
    cagr_buy_hold = (1 + total_buy_hold) ** (1/years) - 1
    
    # Sharpe ratio
    sharpe = (df['strategy_ret'].mean() / df['strategy_ret'].std()) * np.sqrt(252/HOLDING) if df['strategy_ret'].std() > 0 else 0
    
    # Win rate
    win_rate = (df['strategy_ret'] > 0).mean()
    
    # Max drawdown
    cumulative = (1 + df['strategy_ret']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    print(f"\nOVERALL PERFORMANCE ({years:.1f} years):")
    print(f"  Total Return:  {total_strategy:>8.1%} (strategy) vs {total_buy_hold:>8.1%} (B&H)")
    print(f"  CAGR:          {cagr_strategy:>8.1%} (strategy) vs {cagr_buy_hold:>8.1%} (B&H)")
    print(f"  Sharpe Ratio:  {sharpe:>8.2f}")
    print(f"  Win Rate:      {win_rate:>8.1%}")
    print(f"  Max Drawdown:  {max_dd:>8.1%}")
    print(f"  Trades:        {len(df):>8,}")
    
    # Year-by-year
    print(f"\nYEAR-BY-YEAR RETURNS:")
    print(f"  {'Year':<6} {'Strategy':>10} {'Buy&Hold':>10} {'Difference':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    
    for year in range(2015, 2024):
        year_data = df[df.index.year == year]
        if len(year_data) > 0:
            strat_ret = (1 + year_data['strategy_ret']).prod() - 1
            bh_ret = (1 + year_data['fwd_ret']).prod() - 1
            diff = strat_ret - bh_ret
            print(f"  {year:<6} {strat_ret:>9.1%} {bh_ret:>9.1%} {diff:>+9.1%}")
    
    results_summary.append({
        'symbol': symbol,
        'total_return': total_strategy,
        'cagr': cagr_strategy,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'trades': len(df)
    })

# ============================================================================
# STEP 3: Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - ALL ASSETS")
print("="*80)

summary_df = pd.DataFrame(results_summary)
print("\n" + summary_df.to_string(index=False))

best = summary_df.loc[summary_df['sharpe'].idxmax()]

print(f"\nBEST PERFORMER:")
print(f"  Asset: {best['symbol']}")
print(f"  Sharpe: {best['sharpe']:.2f}")
print(f"  CAGR: {best['cagr']:.1%}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print("FINAL VERDICT - THE MOMENT OF TRUTH")
print("="*80)

avg_sharpe = summary_df['sharpe'].mean()
positive_sharpe = (summary_df['sharpe'] > 0).sum()

print(f"\nAverage Sharpe across all assets: {avg_sharpe:.2f}")
print(f"Assets with positive Sharpe: {positive_sharpe}/{len(summary_df)}")

if avg_sharpe > 1.0:
    print("\n✅✅✅ MOMENTUM EDGE IS REAL! ✅✅✅")
    print(f"\n   Average Sharpe {avg_sharpe:.2f} over 9 years")
    print("\n   THIS IS TRADEABLE ALPHA!")
    
elif avg_sharpe > 0.5:
    print("\n⚠ MODERATE EDGE")
    print(f"\n   Sharpe {avg_sharpe:.2f} - some edge exists")
    print("   Lower than 60-day test suggested")
    
else:
    print("\n❌ 60-DAY RESULT WAS LUCK")
    print(f"\n   Sharpe {avg_sharpe:.2f} over full period")
    print("   No real edge after proper testing")

# Save
Path('data/momentum_results').mkdir(exist_ok=True, parents=True)
summary_df.to_csv('data/momentum_results/full_validation_summary.csv', index=False)

print(f"\n✓ Results saved to data/momentum_results/")
print("\n✓ ANALYSIS COMPLETE!")

