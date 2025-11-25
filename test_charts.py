from app.utils.market_data import get_multi_ticker

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
data = get_multi_ticker(tickers, period='1mo', interval='1d')

print('Data shape:', data.shape)
print('Empty:', data.empty)
print('Has MultiIndex:', hasattr(data.columns, 'get_level_values'))

if not data.empty and hasattr(data.columns, 'get_level_values'):
    print('Tickers:', data.columns.get_level_values(0).unique().tolist())
    if 'AAPL' in data.columns.get_level_values(0):
        print('Columns for AAPL:', data['AAPL'].columns.tolist())
        print('Sample AAPL data:', data['AAPL'].head(2))
