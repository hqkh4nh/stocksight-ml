import pandas as pd
import yfinance as yf

TICKER_DEFAULT = "AAPL"
START_DATE = "2012-01-01"
MACRO_TICKERS = {
    "VIX":  "^VIX",
    "SPX":  "^GSPC",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",
    "OIL":  "CL=F",
    "GOLD": "GC=F",
}

def fetch_stock(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    """Download daily OHLCV for one ticker from yfinance."""
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    # flatten multiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = ticker
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    """Download all macro indicators and merge into one DataFrame."""
    parts = []
    for name, symbol in MACRO_TICKERS.items():
        s = yf.download(symbol, start=start, progress=False, auto_adjust=True)["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = name
        parts.append(s)
    macro = pd.concat(parts, axis=1)
    # forward-fill small gaps (cross-market holidays), limit 2 days
    macro = macro.ffill(limit=2)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    return macro

def build_dataset(ticker: str, stocks: dict, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge one stock with all macros on date index"""
    stock_df = stocks[ticker]
    df = stock_df.join(macro, how="inner")
    df = df.dropna()
    return df

# raw / non-stationary columns we don't feed to the model
RAW_DROP = [
    "Open", "High", "Low", "Close", "Volume", "LogVolume", "Ticker",
    "VIX", "SPX", "DXY", "TNX", "OIL", "GOLD",   # macro levels; we keep _ret5 and _lag1
    "ret",                                       # daily return helper; ret_lag5 covers momentum
    "target",
    # --- pruned (importance < 0.5% or |corr| > 0.9 with a stronger feature) ---
    "dow", "is_month_end", "month",              # calendar group: ~1% total importance
    "GOLD_ret1", "OIL_ret1", "VIX_ret1",         # macro 1-day returns: all < 0.5%
    "DXY_ret1", "SPX_ret1", "TNX_ret1",
    "ret_lag1", "ret_lag10",                     # keep ret_lag5 as momentum representative
    "gap",                                       # microstructure noise
    "VIX_lag1",                                  # corr 0.95 with VIX_log_lag1
    "ret_ma10",                                  # corr 0.92 with price_ma20_ratio
    "bb_pos",                                    # rsi14 covers mean-reversion
    "ret_ma5",                                   # ret_ma20 is the stronger rolling mean
    "VIX_ret5",                                  # vix_z + VIX_log_lag1 cover VIX info
]


def split_xy(df: pd.DataFrame):
    """Drop non-stationary columns and the target. Returns (X, y, feature_cols)."""
    feature_cols = [c for c in df.columns if c not in RAW_DROP]
    X = df[feature_cols].copy()
    y = df["target"].copy()
    return X, y, feature_cols


def time_split(X: pd.DataFrame, y: pd.Series,
               train_ratio: float = 0.70, val_ratio: float = 0.15,
               horizon: int = 5):
    """Sequential split with embargo: purge `horizon` rows at each boundary so
    train labels don't reference val prices (and val labels don't reference test).
    target[t] = log(Close[t+horizon]/Close[t]) leaks `horizon` rows across each cut.
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    X_train, y_train = X.iloc[:n_train - horizon],                y.iloc[:n_train - horizon]
    X_val,   y_val   = X.iloc[n_train:n_train + n_val - horizon], y.iloc[n_train:n_train + n_val - horizon]
    X_test,  y_test  = X.iloc[n_train + n_val:],                  y.iloc[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test