# stocksight-ml

A Random Forest model that forecasts the 5-day forward log-return of US
equities from technical indicators and macro features, with a Streamlit
front-end for browsing the predictions.

The default ticker is AAPL. The Streamlit app also pre-trains models for the
top 20 S&P 500 names by market cap and lets you compare them.

## What it does

- Pulls daily OHLCV from yfinance for the chosen ticker plus six macro series
  (VIX, S&P 500, DXY, 10y yield, oil, gold).
- Builds about 40 features: lag returns, rolling stats, RSI, MACD, Bollinger,
  microstructure, calendar dummies, and macro lags.
- Trains a fixed RandomForestRegressor (300 trees, depth 3, leaf size 50) on
  data from 2015 onward, with the target winsorized at +/- 2 sigma.
- Predicts the next 5 business days using both a direct h=5 model and an
  anchor-shift vintage forecast. A 90% confidence band comes from RF tree
  variance.
- Evaluates against three baselines (NaiveZero, AlwaysLong, LinearRegression)
  on RMSE, MAE, R^2, DirAcc, IC, and a long-only Sharpe.
- Runs a walk-forward backtest (expanding train, 21-day refit, non-overlapping
  bets, 5 bps/leg cost) for AAPL and compares the equity curve to LR and SPY
  buy-and-hold.

## Repo layout

```
src/
  data.py          yfinance loaders, dataset assembly, train/val/test split
  features.py      technical + macro + calendar feature engineering
  models.py        baselines, RF tuner, evaluation metrics
  forecast.py      walk-forward, direct h=5, anchor-shift, backtest
  model_io.py      joblib save/load for per-ticker artifacts
  pretrain.py      TOP20 list and training pipeline
scripts/
  pretrain_models.py   CLI to pre-train models outside of Streamlit
notebooks/
  01_main_pipeline.ipynb   end-to-end walkthrough
streamlit_app.py   the demo app
requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

First launch trains the 20 RF models and caches them under `./models/`. That
takes a few minutes. Every launch after that is instant. Click "Retrain all"
in the sidebar to start over.

To pre-train without opening Streamlit:

```bash
python scripts/pretrain_models.py            # train the missing ones
python scripts/pretrain_models.py --force    # retrain everything
```

## Caveats

This is a demo, not a trading signal. IC on the test split sits around +0.05
for AAPL, which is real but small, and the backtest does not model slippage,
financing, or shorting costs.
