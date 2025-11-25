[//]: # (Hello welcome to my project 
  This project is already uploaded to my GitHub https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django.git
)


![iStock-1349355855](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/c8d3869e-363c-4b95-bf4c-2462f8172519)


## Introduction
<p>
  Welcome to Stock Price Prediction with Machine Learning! My website, powered by linear regression and a Django App, provides real-time data of stock prices on the home page. To predict stock prices, simply navigate to the prediction page, enter a valid ticker value and the number of days you want to predict, and click the predict button. This page displays the predicted stock price along with the details of the searched ticker. On the prediction page, you'll find two graphs: the left graph shows the real-time stock price of the searched ticker for the past day, while the right graph displays the predicted stock price for the specified number of days. Additionally, our Ticker Info page provides comprehensive details about all the valid tickers accepted by the application.
</p>

## Aim
<p>
   Title: Stock Price Prediction with Machine Learning
</p>
<p> 
Aim: To predict stock prices according to real-time data values fetched from API.
</p>

## Objective
<p>
  Develop a web application for stock price prediction based on real-time data.
  
</p>

## Scope
<p>
  The project is applicable to any business organization, providing users with stock price prediction capabilities and comprehensive summary data.
</p>

## Technology Used

- Languages: HTML, CSS, JavaScript, Python
- Framework: Bootstrap, Django
- Machine Learning Algorithms: Multiple Linear Regression
- ML/DL Libraries: NumPy, Pandas, scikit-learn
- Database: SQLite
- APIs: Yahoo Finance API, REST API
- IDE: VS Code, Jupyter Notebook




## Project Installation:
**STEP 1:** Clone the repository from GitHub.
```bash
  git clone https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django.git
```

**STEP 2:** Change the directory to the repository.
```bash
  cd FolderName
```

**STEP 3:** Create a virtual environment
(For Windows)
```bash
  python -m venv virtualenv
```

**STEP 4:** Activate the virtual environment.
(For Windows)
```bash
  virtualenv\Scripts\activate
```

**STEP 5:** Install the dependencies.
```bash
  pip install -r requirements.txt          (already text attached in the project)
```

**STEP 6:** Migrate the Django project.
(For Windows)
```bash
  python manage.py migrate
```

**STEP 7:** Run the application.
(For Windows)
```bash
  python manage.py runserver
```

## Quick Start (Current Workspace Setup)
If you already have a `.venv` created in this workspace (as in our current setup):
```powershell
& .\.venv\Scripts\Activate.ps1
cd stock_market_ml
python manage.py migrate
python manage.py createsuperuser  # optional
python manage.py runserver
```

Visit: `http://127.0.0.1:8000/`

## Application Endpoints
| Path | Purpose |
|------|---------|
| `/` | Homepage: multi-ticker (AAPL, AMZN, QCOM, META, NVDA, JPM) adjusted-close chart + recent snapshot list |
| `/search/` | Simple form to input ticker + forecast days |
| `/ticker/` | Lists all available tickers from `app/Data/new_tickers.csv` |
| `/predict/<ticker>/<days>/` | Forecast view: candlestick + future price regression + fundamentals |

## Prediction Input Rules
| Parameter | Rule | Example |
|-----------|------|---------|
| `ticker` | Public stock symbol (case-insensitive). Falls back to error page if not found or empty data. | `AAPL` |
| `days` | Integer, 1–365. Non-int values, negatives, or >365 show dedicated error pages. | `5` |

Examples:
```text
/predict/AAPL/5       -> 5‑day forecast + confidence score
/predict/NVDA/30      -> 30‑day forecast (may truncate if insufficient history)
/predict/XXXX/5       -> Invalid_Ticker.html (unknown symbol)
/predict/AAPL/-3      -> Negative_Days.html
/predict/AAPL/400     -> Overflow_days.html
/predict/AAPL/abc     -> Invalid_Days_Format.html
```

## What You Should Expect (Outputs)
Forecast Page (`/predict/<ticker>/<days>/`) returns:
- Candlestick chart: intraday movement for current day (1m interval)
- Forecast line chart: Predicted adjusted close for N future days (only if enough historical hourly data exists)
- Confidence (R² score) of regression model (0 if not enough data)
- Fundamentals table sourced from `Tickers.csv` (Symbol, Name, Sector, Industry, etc.)
- Empty forecast gracefully handled (no crash, confidence=0)

## Error Pages & Conditions
| Page | Trigger |
|------|---------|
| `Invalid_Ticker.html` | yfinance returns empty data (symbol unknown or delisted) |
| `Invalid_Days_Format.html` | `days` is not an integer |
| `Negative_Days.html` | `days < 0` |
| `Overflow_days.html` | `days > 365` |
| `API_Down.html` | Network/API exception during download |

## Internal Logic Summary
1. Homepage aggregates multiple tickers; missing series are skipped (no KeyErrors).
2. Forecast routine downloads ~3 months of hourly data. If rows > forecast horizon, model trains; else it skips.
3. Feature: single input (Adj Close) scaled then shifted by `days` to create supervised labels.
4. Model: `LinearRegression` trained on 80/20 split; confidence is R² on test split.
5. Predicted future dates start at today + 0..N-1 days (simple forward shift, not trading-day aware).

## Troubleshooting
| Issue | Cause | Resolution |
|-------|-------|------------|
| Empty forecast | Insufficient history vs requested days | Reduce `days` (e.g., try 5 or 10) |
| Confidence = 0 | Skip path (not enough data) or poor fit | Check symbol liquidity / adjust horizon |
| Invalid_Ticker.html | Typo / unsupported symbol | Verify symbol on Yahoo Finance first |
| API_Down.html | Network / rate limit | Wait and retry after a short interval |
| Slow page load | Repeated API calls | Built-in 5‑min in-memory cache reduces redundant downloads |

## Clean Re-run (from fresh clone)
```powershell
git clone https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django.git
cd -Stock-market-Prediction-with-Machine-Learning-Django
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Data & Extensibility Notes
- Adjust ticker universe in `app/views.py` (list `tickers` in `index`) and `new_tickers.csv` for listing.
- Extend model by adding technical indicators (e.g., rolling averages) to the feature set before scaling.
- Move repeated yfinance calls into a caching layer (Redis or in-memory) to reduce API load.
- Current caching: See `app/utils/market_data.py` (TTL = 300s). Tune `TTL_SECONDS` for freshness vs speed.

## Disclaimer
Predictions are illustrative and not investment advice. Model uses only price history (Adj Close) and does not account for corporate events, macro factors, or volatility clustering.



## Output Screen-shots:
Home page displaying real-time data of stock prices

![Screenshot (342)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/754e23b7-1d8b-47df-92c2-50d8abdb0f5b)

![Screenshot (343)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/a76afff4-c812-4db1-8e1d-64097a1cd178)


Prediction page where users enter valid ticker value and number of days
![Screenshot (345)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/b1c4e87c-9d94-4da9-986f-e471f7129d4f)

Prediction page displaying predicted stock price and ticker details.
![Screenshot (346)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/6814ac70-6079-4d8b-aef8-3c2959a82a1d)


 Left graph shows real-time stock price for past day, right graph shows predicted stock price for specified days
 ![Screenshot (347)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/bdfb6e97-7fca-45e5-afc0-42c861305f9a)


Ticker Info page displaying details of valid tickers
![Screenshot (348)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/faa1429f-81a3-4988-8638-c3b5d28bca9c)

Overview of code section

![Screenshot (349)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/e490bab6-e758-4cef-a09c-4c2eb99a773a)
![Screenshot (350)](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/assets/83393190/a45cf9c2-6a48-4e02-aab1-d04ff53ac5d2)

## Conclusion:
Our Stock Price Prediction with Machine Learning website, utilizing linear regression and Django, enables users to predict stock prices based on real-time data.
With easy-to-use interfaces and insightful graphs, users can make informed investment decisions.
We provide comprehensive ticker information and ensure accurate predictions through our machine learning algorithms.

## Report PDF  of Project.
[Final Report.pdf](https://github.com/vishal815/-Stock-market-Prediction-with-Machine-Learning-Django/files/11960207/Final.Report.KUKBIT.pdf)



## Thank you!

