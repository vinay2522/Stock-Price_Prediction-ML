from urllib import request
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import RequestContext
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegistrationForm, LoginForm

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
import qrcode

from .models import Project
from .utils.market_data import (
    get_multi_ticker,
    get_intraday,
    get_daily,
    get_history_for_forecast,
)

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm





# The Home page when Server loads up
def index(request):
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    data = get_multi_ticker(tickers, period='1mo', interval='1d')

    dates = data.index if not data.empty else []
    available_top = set()
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        available_top = set(data.columns.get_level_values(0))

    fig_left = go.Figure()
    for t in tickers:
        if t in available_top:
            try:
                cols = data[t].columns.tolist()
                # With auto_adjust=True, use 'Close' instead of 'Adj Close'
                if 'Close' in cols:
                    fig_left.add_trace(go.Scatter(x=dates, y=data[t]['Close'], name=t, mode='lines'))
                elif 'Adj Close' in cols:
                    fig_left.add_trace(go.Scatter(x=dates, y=data[t]['Adj Close'], name=t, mode='lines'))
            except (KeyError, Exception) as e:
                print(f"Error adding trace for {t}: {e}")
                continue
    
    if len(fig_left.data) == 0:
        fig_left.add_annotation(text="No ticker data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="white", size=16))
    else:
        # Apply vibrant colors to each trace
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']  # Blue, Green, Amber, Red
        for i, trace in enumerate(fig_left.data):
            trace.line.color = colors[i % len(colors)]
            trace.line.width = 3
    
    fig_left.update_layout(
        title=dict(text='Active Stocks - 1 Month Trend', font=dict(size=18, color='#FFFFFF', family='Arial Black')),
        paper_bgcolor="rgba(15, 23, 42, 0.95)", 
        plot_bgcolor="rgba(30, 41, 59, 0.95)",
        font=dict(color="#F8FAFC", size=13, family='Arial'),
        height=500,
        margin=dict(l=70, r=50, t=90, b=70),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#475569',
            gridwidth=1.5,
            showline=True,
            linecolor='#64748B',
            linewidth=2,
            title=dict(text='Date', font=dict(size=14, color='#F8FAFC', family='Arial Bold')),
            tickfont=dict(size=12, color='#E2E8F0')
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#475569',
            gridwidth=1.5,
            showline=True,
            linecolor='#64748B',
            linewidth=2,
            title=dict(text='Price (USD)', font=dict(size=14, color='#F8FAFC', family='Arial Bold')),
            tickfont=dict(size=12, color='#E2E8F0')
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#60A5FA",
            font=dict(size=14, color="#FFFFFF", family='Arial', weight='bold')
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.9)",
            bordercolor="#60A5FA",
            borderwidth=2,
            font=dict(color="#FFFFFF", size=13, family='Arial')
        )
    )
    plot_config = {
        'displayModeBar': True,
        'responsive': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare'],
        'modeBarButtonsToRemove': [],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'stock_chart',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
    plot_div_left = plot(fig_left, output_type='div', include_plotlyjs=False, config=plot_config)

    recent_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
    data_recent = get_multi_ticker(recent_symbols, period='1d', interval='1d')

    recent_stocks = []
    if data_recent is not None and not data_recent.empty:
        try:
            # The data comes with a MultiIndex columns ('AAPL', 'Open'), ('AAPL', 'Close'), etc.
            # We need to stack it to get Ticker as a column.
            df_stacked = data_recent.stack(level=0, future_stack=True).reset_index()
            df_stacked.rename(columns={'level_1': 'Ticker'}, inplace=True)
            
            # Get the available columns dynamically
            available_cols = df_stacked.columns.tolist()
            
            # Build the list of columns we need, handling both 'Adj Close' and 'Close' variations
            cols_to_use = ['Ticker']
            col_mapping = {}
            
            # Map standard names to what's actually in the dataframe
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in available_cols:
                    cols_to_use.append(col)
            
            # Handle Adj Close variations
            if 'Adj Close' in available_cols:
                cols_to_use.append('Adj Close')
                col_mapping['Adj Close'] = 'Adj_Close'
            elif 'Close' in available_cols and 'Adj_Close' not in available_cols:
                # If only Close exists, use it as Adj_Close
                if 'Close' not in cols_to_use:
                    cols_to_use.append('Close')
                col_mapping['Close'] = 'Adj_Close'
            
            # Filter to only the columns we have
            df_filtered = df_stacked[cols_to_use].copy()
            
            # Rename columns to match template expectations
            if col_mapping:
                df_filtered.rename(columns=col_mapping, inplace=True)
            
            # Ensure we have Adj_Close column
            if 'Adj_Close' not in df_filtered.columns and 'Close' in df_filtered.columns:
                df_filtered['Adj_Close'] = df_filtered['Close']

            json_records = df_filtered.to_json(orient='records')
            recent_stocks = json.loads(json_records)
        except Exception as e:
            print(f"Error processing recent stocks data: {e}")
            recent_stocks = []
    else:
        recent_stocks = []

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # Read CSV and ensure we only have Symbol and Name columns
    ticker_df = pd.read_csv('app/Data/new_tickers.csv')
    # Only keep Symbol and Name columns if they exist
    if 'Symbol' in ticker_df.columns and 'Name' in ticker_df.columns:
        ticker_df = ticker_df[['Symbol', 'Name']]
    json_ticker = ticker_df.to_json(orient='records')
    ticker_list = json.loads(json_ticker)
    return render(request, 'ticker.html', { 'ticker_list': ticker_list })


def predict(request, ticker_value, number_of_days):
    ticker_value = ticker_value.upper()
    
    # Try to get recent price data - prefer daily for reliability
    df = get_daily(ticker_value, period='1mo', interval='1d')
    price_warning = "Showing recent daily stock prices."
    
    # Validate we got data
    if df is None or df.empty:
        # Try intraday as backup
        df = get_intraday(ticker_value, period='5d', interval='15m')
        price_warning = "Showing intraday price data."
        
    if df is None or df.empty:
        return render(request, 'Invalid_Ticker.html', {'ticker_value': ticker_value})
    
    # Handle MultiIndex columns - yfinance returns ('Close', 'TSLA') format
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten to just the price type names
        df.columns = df.columns.get_level_values(0)
    
    # Clean up the data
    df = df.dropna()
    
    if df.empty:
        return render(request, 'Invalid_Ticker.html', {'ticker_value': ticker_value})

    # Drop timezone information (Plotly struggles with tz-aware indexes) and
    # make sure data is sorted chronologically.
    if hasattr(df.index, 'tz_localize'):
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)

    if price_warning is None and len(df) < 20:
        price_warning = "Market activity is low; chart may look flat due to limited data points."

    try:
        number_of_days = int(number_of_days)
    except Exception:
        return render(request, 'Invalid_Days_Format.html', {})

    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})

    # Create subplot figure with price and volume
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker_value} Live Share Price Evolution', 'Volume')
    )
    
    # Check what columns we have
    has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in df.columns
    
    # Add candlestick chart (matches reference image style better)
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing=dict(line=dict(color='#00FF00', width=1), fillcolor='#00CC00'),
            decreasing=dict(line=dict(color='#FF0000', width=1), fillcolor='#CC0000'),
            showlegend=True
        ), row=1, col=1)
    
    # Add volume bars if available
    if has_volume:
        colors = ['#00CC00' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#CC0000' 
                  for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker=dict(color=colors),
            showlegend=False
        ), row=2, col=1)
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#1a1f2e",
        font=dict(color="#FFFFFF", size=12, family='Arial'),
        height=600,
        margin=dict(l=60, r=40, t=80, b=50),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            bordercolor="#FFFFFF",
            font=dict(size=13, color="#FFFFFF", family='Arial')
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(26, 31, 46, 0.9)",
            bordercolor="#FFFFFF",
            borderwidth=1,
            font=dict(color="#FFFFFF", size=11)
        )
    )
    
    # Update price chart axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#2d3748',
        gridwidth=0.5,
        showline=True,
        linecolor='#4a5568',
        linewidth=1,
        tickfont=dict(size=11, color='#FFFFFF'),
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Price (USD)',
        showgrid=True,
        gridcolor='#2d3748',
        gridwidth=0.5,
        showline=True,
        linecolor='#4a5568',
        linewidth=1,
        title_font=dict(size=12, color='#FFFFFF'),
        tickfont=dict(size=11, color='#FFFFFF'),
        row=1, col=1
    )
    
    # Update volume chart axes
    fig.update_xaxes(
        title_text='Date',
        showgrid=True,
        gridcolor='#2d3748',
        gridwidth=0.5,
        tickfont=dict(size=11, color='#FFFFFF'),
        row=2, col=1
    )
    fig.update_yaxes(
        title_text='Volume',
        showgrid=False,
        tickfont=dict(size=10, color='#FFFFFF'),
        row=2, col=1
    )
    # Add range selector for time periods
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='#1a1f2e',
            activecolor='#3B82F6',
            font=dict(color='#FFFFFF', size=10),
            x=0,
            y=1.05
        ),
        row=1, col=1
    )
    plot_config = {
        'displayModeBar': True,
        'responsive': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{ticker_value}_live_chart',
            'height': 800,
            'width': 1400,
            'scale': 2
        }
    }
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config=plot_config)

    # Historical data for ML forecast (use hourly granularity)
    df_ml = get_history_for_forecast(ticker_value)

    forecast_out = number_of_days
    confidence = 0
    forecast = []
    
    # Determine which price column to use (Close with auto_adjust=True or Adj Close)
    price_col = None
    if df_ml is not None and not df_ml.empty:
        if 'Close' in df_ml.columns:
            price_col = 'Close'
        elif 'Adj Close' in df_ml.columns:
            price_col = 'Adj Close'
    
    if price_col and df_ml.shape[0] > forecast_out and forecast_out > 0:
        core_df = df_ml[[price_col]].copy()
        core_df['Prediction'] = core_df[[price_col]].shift(-forecast_out)
        X_all = np.array(core_df.drop(['Prediction'], axis=1))
        # Guard before scaling to avoid ValueError on empty arrays
        if X_all.shape[0] > forecast_out:
            try:
                X_all = preprocessing.scale(X_all)
            except Exception:
                X_all = X_all  # leave unscaled if failure
            X_forecast = X_all[-forecast_out:]
            X = X_all[:-forecast_out]
            y = np.array(core_df['Prediction'])[:-forecast_out]
            if X.shape[0] > 0 and y.shape[0] > 0:
                try:
                    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
                    clf = LinearRegression()
                    clf.fit(X_train, y_train)
                    confidence = clf.score(X_test, y_test)
                    forecast_prediction = clf.predict(X_forecast)
                    forecast = forecast_prediction.tolist()
                except Exception:
                    confidence = 0
                    forecast = []

    pred_dict = {"Date": [], "Prediction": []}
    for i, val in enumerate(forecast):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(val)
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(
        x=pred_df['Date'], 
        y=pred_df['Prediction'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#60A5FA', width=3),
        marker=dict(size=8, color='#3B82F6', line=dict(color='white', width=1))
    )])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(
        title=dict(text=f'Predicted Prices for Next {number_of_days} Days', font=dict(size=18, color='#FFFFFF', family='Arial Black')),
        yaxis_title='Predicted Stock Price (USD)',
        xaxis_title='Date',
        paper_bgcolor="rgba(15, 23, 42, 0.95)", 
        plot_bgcolor="rgba(30, 41, 59, 0.95)",
        font=dict(color="#FFFFFF", size=13, family='Arial'),
        height=500,
        margin=dict(l=70, r=50, t=90, b=70),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#475569',
            gridwidth=1.5,
            showline=True,
            linecolor='#64748B',
            linewidth=2,
            title_font=dict(size=14, color='#FFFFFF', family='Arial Bold'),
            tickfont=dict(size=12, color='#FFFFFF')
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#475569',
            gridwidth=1.5,
            showline=True,
            linecolor='#64748B',
            linewidth=2,
            title_font=dict(size=14, color='#FFFFFF', family='Arial Bold'),
            tickfont=dict(size=12, color='#FFFFFF')
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#60A5FA",
            font=dict(size=14, color="#FFFFFF", family='Arial', weight='bold')
        )
    )
    plot_config_pred = {
        'displayModeBar': True,
        'responsive': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare'],
        'modeBarButtonsToRemove': [],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{ticker_value}_forecast',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
    plot_div_pred = plot(pred_fig, output_type='div', include_plotlyjs=False, config=plot_config_pred)

    info_df = pd.read_csv('app/Data/Tickers.csv')
    info_df.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                       'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    row = info_df[info_df.Symbol == ticker_value]
    if row.empty:
        Symbol = ticker_value
        Name = 'N/A'
        Last_Sale = Net_Change = Percent_Change = Market_Cap = Country = IPO_Year = Volume = Sector = Industry = 'N/A'
    else:
        r = row.iloc[0]
        Symbol = r.Symbol
        Name = r.Name
        Last_Sale = r.Last_Sale
        Net_Change = r.Net_Change
        Percent_Change = r.Percent_Change
        Market_Cap = r.Market_Cap
        Country = r.Country
        IPO_Year = r.IPO_Year
        Volume = r.Volume
        Sector = r.Sector
        Industry = r.Industry

    return render(request, 'result.html', {
        'plot_div': plot_div,
        'confidence': confidence,
        'forecast': forecast,
        'ticker_value': ticker_value,
        'number_of_days': number_of_days,
        'plot_div_pred': plot_div_pred,
        'Symbol': Symbol,
        'Name': Name,
        'Last_Sale': Last_Sale,
        'Net_Change': Net_Change,
        'Percent_Change': Percent_Change,
        'Market_Cap': Market_Cap,
        'Country': Country,
        'IPO_Year': IPO_Year,
        'Volume': Volume,
        'Sector': Sector,
        'Industry': Industry,
        'price_warning': price_warning,
    })


# Authentication Views
def register_view(request):
    if request.user.is_authenticated:
        return redirect('/')
    
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Account created successfully! Welcome, {user.username}!')
            return redirect('/')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = RegistrationForm()
    
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('/')
    
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                return redirect('/')
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('/login/')
