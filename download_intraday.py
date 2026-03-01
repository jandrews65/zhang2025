import yfinance as yf
import pandas as pd
from pathlib import Path

print("Downloading 15-minute intraday data...")
print("="*70)

# Create output directory
Path("data/intraday").mkdir(parents=True, exist_ok=True)

# Tickers to download
tickers = {
    'GLD': 'Gold ETF',
    'USO': 'Oil ETF',
    'GC=F': 'Gold Futures',
    'CL=F': 'Oil Futures',
}

# Download maximum available intraday data
for ticker, name in tickers.items():
    print(f"\nDownloading {name} ({ticker})...")
    
    try:
        # Try different periods to get maximum data
        for period in ['60d', '730d']:  # 60 days, 2 years
            print(f"  Trying period={period}...")
            data = yf.download(ticker, interval='15m', period=period, progress=False)
            
            if len(data) > 0:
                # Save to parquet
                filepath = f"data/intraday/{ticker.replace('=', '_')}_15min.parquet"
                data.to_parquet(filepath)
                
                print(f"  ✓ Downloaded {len(data):,} bars")
                print(f"    Range: {data.index.min()} to {data.index.max()}")
                print(f"    Saved: {filepath}")
                break
        else:
            print(f"  ✗ No data available")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*70)
print("Download complete!")
print("\nFiles saved to: data/intraday/")
