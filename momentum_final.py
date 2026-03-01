"""
Momentum Strategy - Full Historical Validation
FINAL FIX
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("MOMENTUM STRATEGY - FULL HISTORICAL VALIDATION")
print("="*80)

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '-q'])
    import yfinance as yf

# Download
print("\n1. Downloading historical data...")

SYMBOLS = ['USO', 'GLD', 'SLV', 'SPY']
data = {}

for symbol in SYMBOLS:
    print(f"   {symbol}...", end=' ')
    df = yf.download(symbol, start='2015-01-01', end='2024-12-31', progress=False)
    
    if len(df) > 0:
        # Extract Close column
        if isinstance(df, pd.DataFrame):
            if 'Close' in df.columns:
                data[symbol] = df['Close']
            else:
                data[symbol] = df.iloc[:, 3]  # Usually Close
        else:
            data[symbol] = df
        
        print(f"✓ {len(data[symbol])} days")

print(f"\n✓ Downloaded {len(data)} assets\n")

# Test momentum
LOOKBACK = 20
HOLDING = 5
results = []

for symbol, prices in data.items():
    print("="*80)
    print(f"{symbol} - FULL 9-YEAR BACKTEST")
    print("="*80)
    
    # Ensure it's a Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    
    df = pd.DataFrame({'price': prices.values}, index=prices.index)
    df['momentum'] = df['price'].pct_change(LOOKBACK)
    df['signal'] = (df['momentum'] > 0).astype(int)
    df['fwd_ret'] = df['price'].pct_change(HOLDING).shift(-HOLDING)
    df['strategy_ret'] = df['signal'].shift(1) * df['fwd_ret']
    df = df.dropna()
    
    # Metrics
    total_strat = (1 + df['strategy_ret']).prod() - 1
    total_bh = (1 + df['fwd_ret']).prod() - 1
    
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr_strat = (1 + total_strat) ** (1/years) - 1
    cagr_bh = (1 + total_bh) ** (1/years) - 1
    
    sharpe = (df['strategy_ret'].mean() / df['strategy_ret'].std()) * np.sqrt(252/HOLDING)
    win_rate = (df['strategy_ret'] > 0).mean()
    
    cumulative = (1 + df['strategy_ret']).cumprod()
    max_dd = ((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min()
    
    print(f"\nOVERALL ({years:.1f} years, {len(df):,} trades):")
    print(f"  Total:    {total_strat:>8.1%} vs {total_bh:>8.1%} (B&H)")
    print(f"  CAGR:     {cagr_strat:>8.1%} vs {cagr_bh:>8.1%}")
    print(f"  Sharpe:   {sharpe:>8.2f}")
    print(f"  WinRate:  {win_rate:>8.1%}")
    print(f"  MaxDD:    {max_dd:>8.1%}")
    
    print(f"\nYEAR-BY-YEAR:")
    for year in range(2015, 2025):
        year_data = df[df.index.year == year]
        if len(year_data) > 10:
            strat = (1 + year_data['strategy_ret']).prod() - 1
            bh = (1 + year_data['fwd_ret']).prod() - 1
            print(f"  {year}: {strat:>7.1%} vs {bh:>7.1%} ({strat-bh:>+6.1%})")
    
    results.append({
        'symbol': symbol,
        'sharpe': sharpe,
        'cagr': cagr_strat,
        'win_rate': win_rate,
        'max_dd': max_dd
    })
    print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)

df_summary = pd.DataFrame(results)
print(f"\n{df_summary.to_string(index=False)}\n")

avg_sharpe = df_summary['sharpe'].mean()

print("="*80)
print("FINAL VERDICT")
print("="*80)

print(f"\nAverage Sharpe: {avg_sharpe:.2f}")

if avg_sharpe > 1.0:
    print(f"\n✅ MOMENTUM IS REAL! Sharpe {avg_sharpe:.2f}")
elif avg_sharpe > 0.5:
    print(f"\n⚠ MODERATE EDGE. Sharpe {avg_sharpe:.2f}")
else:
    print(f"\n❌ 60-DAY WAS LUCK. Sharpe {avg_sharpe:.2f}")

Path('data/momentum_results').mkdir(exist_ok=True, parents=True)
df_summary.to_csv('data/momentum_results/summary.csv', index=False)
print("\n✓ Done!\n")

