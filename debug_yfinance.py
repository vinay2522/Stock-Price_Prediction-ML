import yfinance as yf
import pandas as pd

def debug_yfinance_fetch(ticker):
    """
    Attempts to fetch data for a single ticker and prints the result or error.
    """
    print(f"Attempting to fetch data for ticker: {ticker}")
    try:
        # Attempt to download the last 5 days of data for the ticker
        data = yf.download(ticker, period="5d")
        
        if data.empty:
            print(f"No data returned for {ticker}. The ticker may be invalid or delisted.")
        else:
            print(f"Successfully fetched data for {ticker}:")
            print(data.head())
            
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    # Ticker to test
    test_ticker = "AAPL"
    
    # Print yfinance and pandas version for debugging context
    print(f"yfinance version: {yf.__version__}")
    print(f"pandas version: {pd.__version__}")
    print("-" * 30)
    
    debug_yfinance_fetch(test_ticker)
