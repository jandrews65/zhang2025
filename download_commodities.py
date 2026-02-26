import yfinance as yf
import pandas as pd

tickers = {
    'Oil': 'USO',    # Oil ETF
    'Gold': 'GLD',   # Gold ETF  
    'Silver': 'SLV'  # Silver ETF
}

for name, ticker in tickers.items():
    print(f'Downloading {name} ({ticker})...')
    try:
        df = yf.download(ticker, start='2015-02-18', end='2023-12-31', progress=False)
        if len(df) > 100:
            df['forward_return'] = df['Close'].pct_change().shift(-1)
            df.to_parquet(f'{ticker}.parquet')
            print(f'  ✓ Saved {ticker}.parquet ({len(df)} rows)')
        else:
            print(f'  ✗ Insufficient data')
    except Exception as e:
        print(f'  ✗ Error: {e}')
