"""
Multi-Factor Portfolio Strategy - FIXED
"""

import yfinance as yf
import pandas as pd
import numpy as np

print("="*80)
print("MULTI-FACTOR PORTFOLIO STRATEGY")
print("="*80)

print("\n1. Downloading data (2015-2024)...")

SYMBOLS = ['MTUM', 'VLUE', 'QUAL', 'USMV', 'SPY']
data = {}

for symbol in SYMBOLS:
    print(f"   {symbol}...", end=' ')
    df = yf.download(symbol, start='2015-01-01', end='2024-12-31', progress=False)
    if len(df) > 0:
        if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
            data[symbol] = df['Close']
        else:
            data[symbol] = df.iloc[:, 3]
        print(f"✓ {len(data[symbol])} days")

print("\n2. Backtesting...")

# Combine into DataFrame
prices = pd.concat(data, axis=1)
returns = prices.pct_change().dropna()

# Portfolio (equal weight, no rebalancing for simplicity)
portfolio_returns = returns[['MTUM', 'VLUE', 'QUAL', 'USMV']].mean(axis=1)
benchmark_returns = returns['SPY']

# Metrics
years = (returns.index[-1] - returns.index[0]).days / 365.25

total_port = (1 + portfolio_returns).prod() - 1
total_bench = (1 + benchmark_returns).prod() - 1

cagr_port = (1 + total_port) ** (1/years) - 1
cagr_bench = (1 + total_bench) ** (1/years) - 1

vol_port = portfolio_returns.std() * np.sqrt(252)
vol_bench = benchmark_returns.std() * np.sqrt(252)

sharpe_port = (cagr_port - 0.02) / vol_port
sharpe_bench = (cagr_bench - 0.02) / vol_bench

cumulative = (1 + portfolio_returns).cumprod()
max_dd = ((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min()

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nPERFORMANCE ({years:.1f} years, {len(returns):,} days):")
print(f"  Total:      {total_port:>8.1%} vs {total_bench:>8.1%} (SPY)")
print(f"  CAGR:       {cagr_port:>8.1%} vs {cagr_bench:>8.1%}")
print(f"  Volatility: {vol_port:>8.1%} vs {vol_bench:>8.1%}")
print(f"  Sharpe:     {sharpe_port:>8.2f} vs {sharpe_bench:>8.2f}")
print(f"  Max DD:     {max_dd:>8.1%}")

print("\nYEAR-BY-YEAR:")
for year in range(2015, 2025):
    year_port = portfolio_returns[portfolio_returns.index.year == year]
    year_bench = benchmark_returns[benchmark_returns.index.year == year]
    if len(year_port) > 10:
        p = (1 + year_port).prod() - 1
        b = (1 + year_bench).prod() - 1
        print(f"  {year}: {p:>7.1%} vs {b:>7.1%} ({p-b:>+6.1%})")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if sharpe_port > 1.0:
    print(f"\n✅ FACTOR INVESTING WORKS!")
    print(f"   Sharpe {sharpe_port:.2f} is excellent")
elif sharpe_port > 0.7:
    print(f"\n✅ SOLID STRATEGY")
    print(f"   Sharpe {sharpe_port:.2f} beats most managers")
else:
    print(f"\n⚠ MODERATE PERFORMANCE")
    print(f"   Sharpe {sharpe_port:.2f}")

print(f"\n📊 IMPLEMENTATION:")
print(f"   Portfolio: 25% MTUM + 25% VLUE + 25% QUAL + 25% USMV")
print(f"   Expected:  {cagr_port:.1%} CAGR, {sharpe_port:.2f} Sharpe, {max_dd:.1%} max drawdown")
print(f"   vs SPY:    {cagr_bench:.1%} CAGR, {sharpe_bench:.2f} Sharpe")
print(f"   Difference: {cagr_port - cagr_bench:+.1%} CAGR, {sharpe_port - sharpe_bench:+.2f} Sharpe")

if sharpe_port > sharpe_bench:
    print(f"\n   ⭐ Factor portfolio BEATS buy-and-hold SPY!")
else:
    print(f"\n   ❌ Just buy SPY instead")

print("\n✓ Done!\n")

