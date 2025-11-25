import yfinance as yf
import pandas as pd
df = yf.download('TSLA', period='5d', interval='1d', progress=False, auto_adjust=True)
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print(df)
