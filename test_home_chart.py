import sys
sys.path.insert(0, 'd:/projects/FSD/stock_market_ml')

from app.utils.market_data import get_multi_ticker
import pandas as pd

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
data = get_multi_ticker(tickers, period='1mo', interval='1d')

print(f"Data shape: {data.shape if not data.empty else 'EMPTY'}")
print(f"Data empty: {data.empty}")
print(f"Has MultiIndex: {isinstance(data.columns, pd.MultiIndex) if not data.empty else False}")

if not data.empty:
    print(f"\nColumn structure:")
    if isinstance(data.columns, pd.MultiIndex):
        print(f"  MultiIndex levels: {data.columns.nlevels}")
        print(f"  Level 0 (tickers): {data.columns.get_level_values(0).unique().tolist()}")
        print(f"  Level 1 (columns): {data.columns.get_level_values(1).unique().tolist()}")
        
        print(f"\n Checking each ticker:")
        for t in tickers:
            if t in data.columns.get_level_values(0):
                cols = data[t].columns.tolist()
                print(f"  {t}: {cols}")
                print(f"    Has 'Close': {'Close' in cols}")
                print(f"    Has 'Adj Close': {'Adj Close' in cols}")
                print(f"    Sample data: {data[t]['Close'].head(2).tolist() if 'Close' in cols else 'N/A'}")
    else:
        print(f"  Simple columns: {data.columns.tolist()}")
