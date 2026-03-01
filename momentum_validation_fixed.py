"""
Momentum Strategy - Full Historical Validation (2015-2023)
FIXED VERSION
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

# ============================================================================
# STEP 1: Download Historical Data
# ============================================================================
print("\n1. Downloading historical data...")

SYMBOLS = ['USO', 'GLD', 'SLV', 'SPY']

data = {}

for symbol in SYMBOLS:
    print(f"   {symbol}...", end=' ')
    df = yf.download(symbol, start='2015-01-01', end='2024-12-31', progress=False)
    
    if len(df) > 0:
        # Handle both single and multi-column formats
        if 'Close' in df.columns:
            data[symbol] = df['Close']
        elif isinstance(df.columns, pd.MultiIndex):
            data[symbol] = df[('Close', symbol)]
        else:
            data[symbol] = df.iloc[:, 3]  # Close is usually 4th column
        
        print(f"✓ {len(data[symbol])} days")

print(f"\n✓ Downloaded {len(data)} assets\n")

# ============================================================================
# STEP 2: Test Momentum
# ============================================================================

LOOKBACK = 20
HOLDING = 5

results = []

for symbol, prices in data.items():
    print(f"{'='*80}")
    print(f"{symbol} - FULL 9-YEAR BACKTEST")
    print(f"{'='*80}")
    
    # Create DataFrame from Series
    df = prices.to_frame('price').copy()
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
    print(f"  Total Return:  {total_strat:>8.1%} vs {total_bh:>8.1%} (B&H)")
    print(f"  CAGR:          {cagr_strat:>8.1%} vs {cagr_bh:>8.1%}")
    print(f"  Sharpe Ratio:  {sharpe:>8.2f}")
    print(f"  Win Rate:      {win_rate:>8.1%}")
    print(f"  Max Drawdown:  {max_dd:>8.1%}")
    
    print(f"\nYEAR-BY-YEAR:")
    print(f"  Year   Strategy  Buy&Hold   Diff")
    print(f"  ----   --------  --------   ----")
    
    for year in range(2015, 2025):
        year_data = df[df.index.year == year]
        if len(year_data) > 10:
            strat = (1 + year_data['strategy_ret']).prod() - 1
            bh = (1 + year_data['fwd_ret']).prod() - 1
            print(f"  {year}   {strat:>7.1%}   {bh:>7.1%}   {strat-bh:>+6.1%}")
    
    results.append({
        'symbol': symbol,
        'sharpe': sharpe,
        'cagr': cagr_strat,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'total': total_strat
    })
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)

df_summary = pd.DataFrame(results)
print(f"\n{df_summary.to_string(index=False)}\n")

avg_sharpe = df_summary['sharpe'].mean()
best = df_summary.loc[df_summary['sharpe'].idxmax()]

print(f"Average Sharpe: {avg_sharpe:.2f}")
print(f"Best: {best['symbol']} (Sharpe {best['sharpe']:.2f})")

# ============================================================================
# VERDICT
# ============================================================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if avg_sharpe > 1.0:
    print(f"\n✅✅✅ MOMENTUM IS REAL! ✅✅✅")
    print(f"\n   Average Sharpe {avg_sharpe:.2f} over 9 years")
    print(f"   This is TRADEABLE ALPHA!")
    print("\n   Comparison:")
    print(f"   • 60-day test: Sharpe 2.97")
    print(f"   • 9-year test: Sharpe {avg_sharpe:.2f}")
    print(f"   • Still strong! Edge is real.")
    
elif avg_sharpe > 0.5:
    print(f"\n⚠ MODERATE EDGE")
    print(f"\n   Sharpe {avg_sharpe:.2f} shows some edge")
    print(f"   Lower than 60-day (2.97) but still positive")
    print(f"   After costs, may be marginal")
    
elif avg_sharpe > 0:
    print(f"\n❌ WEAK - 60-DAY WAS MOSTLY LUCK")
    print(f"\n   Sharpe {avg_sharpe:.2f} barely positive")
    print(f"   Transaction costs will eliminate edge")
    
else:
    print(f"\n❌ 60-DAY WAS PURE LUCK")
    print(f"\n   Sharpe {avg_sharpe:.2f} is NEGATIVE")
    print(f"   No edge exists")

Path('data/momentum_results').mkdir(exist_ok=True, parents=True)
df_summary.to_csv('data/momentum_results/summary.csv', index=False)

print(f"\n✓ Complete! Results saved.\n")

