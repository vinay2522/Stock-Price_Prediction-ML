import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Simple in-memory cache (per process). For production replace with Redis/memcached.
CACHE = {}
TTL_SECONDS = 300  # 5 minutes default TTL
MAX_RETRIES = 3 # Increased retries for more resilience
TIMEOUT = 10  # seconds per download attempt


def _cache_get(key):
    entry = CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry['time'] > TTL_SECONDS:
        # expired
        CACHE.pop(key, None)
        return None
    return entry['value']


def _cache_set(key, value):
    CACHE[key] = {'value': value, 'time': time.time()}


def get_multi_ticker(tickers, period='1mo', interval='1d', retries=MAX_RETRIES):
    key = ('multi', tuple(tickers), period, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    
    data = pd.DataFrame()
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers=tickers,
                group_by='ticker',
                threads=True,
                period=period,
                interval=interval,
                timeout=TIMEOUT,
                progress=False,
                auto_adjust=True
            )
            if data is not None and not data.empty:
                # Accept partial success - as long as we got some data
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for get_multi_ticker: {e}")
            if attempt < retries - 1:
                time.sleep(1) # Wait a bit longer before retrying
            else:
                data = pd.DataFrame() # Ensure data is empty on total failure
    
    if data is not None and not data.empty:
        _cache_set(key, data)
        
    return data


def get_intraday(ticker, period='1d', interval='1m', retries=MAX_RETRIES):
    key = ('intraday', ticker.upper(), period, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    
    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(tickers=ticker.upper(), period=period, interval=interval, progress=False, timeout=TIMEOUT, auto_adjust=True)
            if not df.empty:
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for get_intraday '{ticker}': {e}")
            if attempt < retries - 1:
                time.sleep(1)
    
    if not df.empty:
        _cache_set(key, df)
    return df


def get_daily(ticker, period='1d', interval='1d', retries=MAX_RETRIES):
    key = ('daily', ticker.upper(), period, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(tickers=ticker.upper(), period=period, interval=interval, progress=False, timeout=TIMEOUT, auto_adjust=True)
            if not df.empty:
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for get_daily '{ticker}': {e}")
            if attempt < retries - 1:
                time.sleep(1)

    if not df.empty:
        _cache_set(key, df)
    return df


def get_history_for_forecast(ticker, retries=MAX_RETRIES):
    """Return historical data for forecasting. Try hourly first then daily fallback."""
    key = ('forecast_hist', ticker.upper())
    cached = _cache_get(key)
    if cached is not None:
        return cached

    df = pd.DataFrame()
    # Try 3 months hourly (faster than minute for training)
    for attempt in range(retries):
        try:
            df = yf.download(tickers=ticker.upper(), period='3mo', interval='1h', progress=False, timeout=TIMEOUT, auto_adjust=True)
            if not df.empty:
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for forecast history (hourly) '{ticker}': {e}")
            if attempt < retries - 1:
                time.sleep(1)

    if df.empty:
        # Fallback daily
        for attempt in range(retries):
            try:
                df = yf.download(tickers=ticker.upper(), period='6mo', interval='1d', progress=False, timeout=TIMEOUT, auto_adjust=True)
                if not df.empty:
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for forecast history (daily) '{ticker}': {e}")
                if attempt < retries - 1:
                    time.sleep(1)
    
    if not df.empty:
        _cache_set(key, df)
    return df


def clear_cache():
    CACHE.clear()

