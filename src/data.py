import pandas as pd
import yfinance as yf

TICKER_DEFAULT = "AAPL"
START_DATE = "2015-01-01"
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
