"""
Multi-Factor Portfolio Strategy - FINAL WORKING VERSION
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

# Combine
prices = pd.concat(data, axis=1)
returns = prices.pct_change().dropna()

# Portfolio (equal weight)
portfolio_returns = returns[['MTUM', 'VLUE', 'QUAL', 'USMV']].mean(axis=1)
benchmark_returns = returns['SPY']

# Ensure Series
if isinstance(portfolio_returns, pd.DataFrame):
    portfolio_returns = portfolio_returns.iloc[:, 0]
if isinstance(benchmark_returns, pd.DataFrame):
    benchmark_returns = benchmark_returns.iloc[:, 0]

# Metrics
years = (returns.index[-1] - returns.index[0]).days / 365.25

total_port = float((1 + portfolio_returns).prod() - 1)
total_bench = float((1 + benchmark_returns).prod() - 1)

cagr_port = (1 + total_port) ** (1/years) - 1
cagr_bench = (1 + total_bench) ** (1/years) - 1

vol_port = float(portfolio_returns.std() * np.sqrt(252))
vol_bench = float(benchmark_returns.std() * np.sqrt(252))

sharpe_port = (cagr_port - 0.02) / vol_port
sharpe_bench = (cagr_bench - 0.02) / vol_bench

cumulative = (1 + portfolio_returns).cumprod()
max_dd = float(((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min())

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nPERFORMANCE ({years:.1f} years):")
print(f"  Total:      {total_port:>8.1%} vs {total_bench:>8.1%} (SPY)")
print(f"  CAGR:       {cagr_port:>8.1%} vs {cagr_bench:>8.1%}")
print(f"  Volatility: {vol_port:>8.1%} vs {vol_bench:>8.1%}")
print(f"  Sharpe:     {sharpe_port:>8.2f} vs {sharpe_bench:>8.2f}")
print(f"  Max DD:     {max_dd:>8.1%}")

print("\nYEAR-BY-YEAR:")
for year in range(2015, 2025):
    yp = portfolio_returns[portfolio_returns.index.year == year]
    yb = benchmark_returns[benchmark_returns.index.year == year]
    if len(yp) > 10:
        p = float((1 + yp).prod() - 1)
        b = float((1 + yb).prod() - 1)
        print(f"  {year}: {p:>7.1%} vs {b:>7.1%} ({p-b:>+6.1%})")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if sharpe_port > 1.0:
    print(f"\n✅ FACTOR INVESTING WORKS!")
    print(f"   Sharpe {sharpe_port:.2f} is excellent")
elif sharpe_port > 0.7:
    print(f"\n✅ SOLID STRATEGY")
    print(f"   Sharpe {sharpe_port:.2f} is good")
else:
    print(f"\n⚠ MODERATE")
    print(f"   Sharpe {sharpe_port:.2f}")

print(f"\n📊 SUMMARY:")
print(f"   Multi-Factor: {cagr_port:.1%} CAGR, {sharpe_port:.2f} Sharpe")
print(f"   SPY Benchmark: {cagr_bench:.1%} CAGR, {sharpe_bench:.2f} Sharpe")
print(f"   Improvement: {cagr_port - cagr_bench:+.1%} CAGR, {sharpe_port - sharpe_bench:+.2f} Sharpe")

if sharpe_port > sharpe_bench:
    print(f"\n   ⭐ Factor portfolio BEATS SPY!")
else:
    print(f"\n   ❌ Just buy SPY instead")

print("\n✓ Done!\n")

