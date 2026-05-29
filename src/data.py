import pandas as pd
import yfinance as yf

# File này phụ trách tầng dữ liệu thô của project:
# - tải giá cổ phiếu
# - tải dữ liệu vĩ mô
# - ghép dữ liệu theo ngày
# - tách X/y và chia train/val/test theo thời gian
#
# Đây là nơi quyết định "mẫu nào được giữ lại" trước khi bước sang feature engineering.

TICKER_DEFAULT = "AAPL"
START_DATE = "2012-01-01"
MACRO_TICKERS = {
    # VIX   : biến động kỳ vọng của thị trường Mỹ
    # SPX   : S&P 500, đại diện cho thị trường chung
    # DXY   : Dollar Index
    # TNX   : lợi suất trái phiếu Mỹ 10 năm
    # OIL   : giá dầu
    # GOLD  : giá vàng
    "VIX":  "^VIX",
    "SPX":  "^GSPC",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",
    "OIL":  "CL=F",
    "GOLD": "GC=F",
}


def fetch_stock(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    """Download daily OHLCV for one ticker from yfinance."""
    # auto_adjust=True giúp giá đã được điều chỉnh cho split/dividend,
    # phù hợp hơn cho bài toán dự đoán return.
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)

    # yfinance đôi khi trả về MultiIndex; ép về 1 tầng để phần code sau
    # luôn có thể truy cập đơn giản bằng "Close", "Volume", ...
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Chỉ giữ các cột OHLCV mà project thực sự dùng.
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Thêm ticker để giữ ngữ cảnh nếu cần debug hoặc kiểm tra dữ liệu.
    df["Ticker"] = ticker
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    """Download all macro indicators and merge into one DataFrame."""
    parts = []
    for name, symbol in MACRO_TICKERS.items():
        # Mỗi biến vĩ mô chỉ lấy chuỗi giá đóng cửa.
        s = yf.download(symbol, start=start, progress=False, auto_adjust=True)["Close"]
        if isinstance(s, pd.DataFrame):
            # Có trường hợp yfinance trả về DataFrame 1 cột thay vì Series.
            s = s.iloc[:, 0]
        s.name = name
        parts.append(s)

    macro = pd.concat(parts, axis=1)

    # Bù các khoảng trống ngắn do lịch nghỉ khác nhau giữa các thị trường.
    # limit=2 để tránh kéo dữ liệu cũ quá xa.
    macro = macro.ffill(limit=2)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    return macro


def build_dataset(ticker: str, stocks: dict, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge one stock with all macros on date index"""
    # stocks là dict {ticker: dataframe}. Ở đây ta chỉ lấy đúng ticker đang xét.
    stock_df = stocks[ticker]

    # inner join chỉ giữ các ngày có đủ cả stock và macro.
    df = stock_df.join(macro, how="inner")

    # Dọn sạch mọi hàng còn NaN trước khi bước sang feature engineering.
    df = df.dropna()
    return df


# Các cột thô / không dừng sẽ không đưa trực tiếp vào mô hình.
RAW_DROP = [
    "Open", "High", "Low", "Close", "Volume", "LogVolume", "Ticker",
    "VIX", "SPX", "DXY", "TNX", "OIL", "GOLD",   # bỏ mức gốc, chỉ giữ feature macro đã chuẩn hóa
    "ret",                                       # biến phụ trợ để tạo feature kỹ thuật
    "target",
]

# Lý do loại các cột này:
# - giá gốc và mức gốc macro còn xu hướng dài hạn
# - mô hình tài chính thường học tốt hơn trên return / ratio / z-score
# - target là nhãn nên tuyệt đối không được nằm trong X


def split_xy(df: pd.DataFrame):
    """Drop non-stationary columns and the target. Returns (X, y, feature_cols)."""
    # feature_cols được trả về để:
    # - lưu vào artifact
    # - đảm bảo inference dùng đúng thứ tự cột như lúc train
    feature_cols = [c for c in df.columns if c not in RAW_DROP]
    X = df[feature_cols].copy()
    y = df["target"].copy()
    return X, y, feature_cols


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    horizon: int = 5,
):
    """Sequential split with embargo: purge `horizon` rows at each boundary so
    train labels don't reference val prices (and val labels don't reference test).
    target[t] = log(Close[t+horizon]/Close[t]) leaks `horizon` rows across each cut.
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Đây là time series nên không được shuffle.
    # Ta chia tuần tự:
    # - train: phần đầu
    # - val  : phần giữa
    # - test : phần cuối
    #
    # Embargo `horizon` dòng ở mỗi biên là bước chống leakage quan trọng:
    # target[t] nhìn tới Close[t+horizon], nên các hàng sát biên có thể vô tình
    # dùng giá của split kế tiếp để tạo nhãn.
    X_train, y_train = X.iloc[: n_train - horizon], y.iloc[: n_train - horizon]
    X_val, y_val = X.iloc[n_train : n_train + n_val - horizon], y.iloc[n_train : n_train + n_val - horizon]
    X_test, y_test = X.iloc[n_train + n_val :], y.iloc[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test
