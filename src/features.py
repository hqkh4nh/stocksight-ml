import numpy as np
import pandas as pd

from src.data import build_dataset

def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum, volatility, trend, RSI, MACD, microstructure, volume."""
    df = df.copy()

    # 5-day lag return - main momentum signal
    df["ret_lag5"] = df["ret"].shift(5)

    # realized volatility at 3 horizons + 20-day return mean (medium-term momentum)
    df["ret_std5"]  = df["ret"].rolling(5).std()
    df["ret_std10"] = df["ret"].rolling(10).std()
    df["ret_ma20"]  = df["ret"].rolling(20).mean()
    df["ret_std20"] = df["ret"].rolling(20).std()

    # price stretch vs N-day MA - trend strength (price_ma20_ratio is a top-6 feature)
    for w in [20, 50]:
        df["price_ma" + str(w) + "_ratio"] = df["Close"] / df["Close"].rolling(w).mean()

    # microstructure: intraday range (vol proxy) + close location in day's range
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    range_hl             = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_loc"]      = (df["Close"] - df["Low"]) / range_hl

    # RSI(14) - overbought (>70) / oversold (<30) oscillator
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # MACD - trend-following (EMA12 - EMA26) + its acceleration histogram
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]      = ema_12 - ema_26
    df["macd_hist"] = df["macd"] - df["macd"].ewm(span=9, adjust=False).mean()

    # abnormal volume (20-day z-score of log-volume)
    df["vol_z20"] = (df["LogVolume"] - df["LogVolume"].rolling(20).mean()) \
                    / df["LogVolume"].rolling(20).std()
    return df

def add_macro(df: pd.DataFrame) -> pd.DataFrame:
    """5-day log-return + lag-1 level for each macro series.

    .shift(1) first so today's macro close (released after equity close) doesn't leak.
    Macro levels (lag1) dominate feature importance — GOLD_lag1, TNX_lag1, SPX_lag1,
    DXY_lag1 occupy the top-5 alongside GOLD_ret5.
    """
    df = df.copy()
    for col in ["SPX", "DXY", "TNX", "OIL", "GOLD"]:
        df[col + "_ret5"] = np.log(df[col].shift(1) / df[col].shift(6))
        df[col + "_lag1"] = df[col].shift(1)

    # VIX gets a dedicated treatment: log-level (right-skew fix) + 60-day z-score
    df["VIX_log_lag1"] = np.log(df["VIX"].shift(1))
    vix_lag = df["VIX"].shift(1)
    df["vix_z"] = (vix_lag - vix_lag.rolling(60).mean()) / vix_lag.rolling(60).std()
    return df

def build_features_only(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build all features but do not add target."""
    df = raw_df.copy()
    df["LogVolume"] = np.log1p(df["Volume"])               # compress skewed volume to log scale
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1)) # 1-day log-return, stationary
    df = add_technical(df)
    df = add_macro(df)
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
