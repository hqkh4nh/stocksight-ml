import numpy as np
import pandas as pd

from src.data import build_dataset

def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Lag returns, rolling stats, price ratios, RSI, MACD, Bollinger, microstructure."""
    df = df.copy()

    # lag returns - short-term momentum
    for lag in [1, 5, 10]:
        df["ret_lag" + str(lag)] = df["ret"].shift(lag)

    # rolling mean/std of return
    for window in [5, 10, 20]:
        df["ret_ma"  + str(window)] = df["ret"].rolling(window).mean()
        df["ret_std" + str(window)] = df["ret"].rolling(window).std()

    # price/MA ratio - stationary version of price levels
    for window in [20, 50]:
        df["price_ma" + str(window) + "_ratio"] = df["Close"] / df["Close"].rolling(window).mean()

    # microstructure (no lookahead - uses today's OHLC)
    prev_close = df["Close"].shift(1)
    df["gap"]            = (df["Open"] - prev_close) / prev_close
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    range_hl             = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_loc"]      = (df["Close"] - df["Low"]) / range_hl

    # Bollinger band position (20-day z-score)
    ma_20  = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    df["bb_pos"] = (df["Close"] - ma_20) / (2 * std_20)

    # RSI(14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    macd_signal = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - macd_signal

    # log-volume z-score (20-day)
    df["vol_z20"] = (df["LogVolume"] - df["LogVolume"].rolling(20).mean()) \
                    / df["LogVolume"].rolling(20).std()
    return df

def add_macro(df: pd.DataFrame) -> pd.DataFrame:
    """1-day and 5-day log-returns + lag-1 level for each macro series.
    Inputs are .shift(1) first so today's macro close (released after equity close) doesn't leak.
    """
    df = df.copy()
    for col in ["VIX", "SPX", "DXY", "TNX", "OIL", "GOLD"]:
        df[col + "_ret1"] = np.log(df[col].shift(1) / df[col].shift(2))
        df[col + "_ret5"] = np.log(df[col].shift(1) / df[col].shift(6))
        df[col + "_lag1"] = df[col].shift(1)

    # VIX is right-skewed -> log + lag
    df["VIX_log_lag1"] = np.log(df["VIX"].shift(1))

    # VIX z-score on 60-day window: "is VIX abnormally high/low?"
    vix_lag = df["VIX"].shift(1)
    df["vix_z"] = (vix_lag - vix_lag.rolling(60).mean()) / vix_lag.rolling(60).std()
    return df

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month, month-end flag."""
    df = df.copy()
    df["dow"]          = df.index.dayofweek
    df["month"]        = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(int)
    return df

def build_features_only(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build all features but do noy add target."""
    df = raw_df.copy()
    df["LogVolume"] = np.log1p(df["Volume"])
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = add_technical(df)
    df = add_macro(df)
    df = add_calendar(df)
    return df


def build_training_set(ticker: str, stocks: dict, macro: pd.DataFrame,
                       horizon: int = 5) -> pd.DataFrame:
    """Build features + h-day forward log-return target, drop NaN rows."""
    raw_df   = build_dataset(ticker, stocks, macro)
    features = build_features_only(raw_df)
    features["target"] = np.log(features["Close"].shift(-horizon) / features["Close"])
    return features.dropna()

def fit_winsorize(train_series: pd.Series, n_std: float = 2.0):
    """Return mean, std, lower_bound, upper_bound. Use on train only.

    Default +/- 2 sigma. Tighter than the conventional 3 sigma because the model
    learns the everyday signal better when extreme returns (earnings, COVID-style
    shocks) are clipped harder.
    """
    mean = train_series.mean()
    std  = train_series.std()
    lower = mean - n_std * std
    upper = mean + n_std * std
    return mean, std, lower, upper