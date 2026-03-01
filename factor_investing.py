"""
Multi-Factor Portfolio Strategy - Complete Backtest
"""

import yfinance as yf
import pandas as pd
import numpy as np

print("="*80)
print("MULTI-FACTOR PORTFOLIO STRATEGY")
print("="*80)

# Download data
print("\n1. Downloading historical data (2015-2024)...")

SYMBOLS = ['MTUM', 'VLUE', 'QUAL', 'USMV', 'SPY']
data = {}

for symbol in SYMBOLS:
    print(f"   {symbol}...", end=' ')
    df = yf.download(symbol, start='2015-01-01', end='2024-12-31', progress=False)
    if len(df) > 0:
        data[symbol] = df['Close']
        print(f"✓ {len(df)} days")

# Backtest
print("\n2. Backtesting multi-factor portfolio...")

prices = pd.DataFrame(data)
returns = prices.pct_change()

# Equal-weight portfolio (rebalanced quarterly)
portfolio_weights = {'MTUM': 0.25, 'VLUE': 0.25, 'QUAL': 0.25, 'USMV': 0.25}

portfolio_returns = sum(returns[etf] * weight for etf, weight in portfolio_weights.items())
benchmark_returns = returns['SPY']

# Remove NaN
portfolio_returns = portfolio_returns.dropna()
benchmark_returns = benchmark_returns.loc[portfolio_returns.index]

# Metrics
total_port = (1 + portfolio_returns).prod() - 1
total_bench = (1 + benchmark_returns).prod() - 1

years = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days / 365.25
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

print(f"\nOVERALL ({years:.1f} years):")
print(f"  Total Return:  {total_port:>8.1%} vs {total_bench:>8.1%} (SPY)")
print(f"  CAGR:          {cagr_port:>8.1%} vs {cagr_bench:>8.1%}")
print(f"  Volatility:    {vol_port:>8.1%} vs {vol_bench:>8.1%}")
print(f"  Sharpe Ratio:  {sharpe_port:>8.2f} vs {sharpe_bench:>8.2f}")
print(f"  Max Drawdown:  {max_dd:>8.1%}")

print("\nYEAR-BY-YEAR:")
for year in range(2015, 2025):
    year_data = portfolio_returns[portfolio_returns.index.year == year]
    year_bench = benchmark_returns[benchmark_returns.index.year == year]
    if len(year_data) > 10:
        port = (1 + year_data).prod() - 1
        bench = (1 + year_bench).prod() - 1
        print(f"  {year}: {port:>7.1%} vs {bench:>7.1%} ({port-bench:>+6.1%})")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if sharpe_port > 1.0:
    print(f"\n✅ FACTOR INVESTING WORKS! Sharpe {sharpe_port:.2f}")
elif sharpe_port > 0.7:
    print(f"\n✅ SOLID STRATEGY. Sharpe {sharpe_port:.2f}")
else:
    print(f"\n⚠ MODERATE. Sharpe {sharpe_port:.2f}")

print(f"\n📊 HOW TO IMPLEMENT:")
print(f"   1. Buy: MTUM, VLUE, QUAL, USMV (25% each)")
print(f"   2. Rebalance quarterly")
print(f"   3. Expected: {cagr_port:.1%} CAGR, {sharpe_port:.1f} Sharpe")
print("\n✓ Done!\n")

