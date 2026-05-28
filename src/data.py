import pandas as pd
import yfinance as yf

# File này chịu trách nhiệm cho tầng dữ liệu "thô":
# 1. Tải dữ liệu cổ phiếu từ yfinance
# 2. Tải các chuỗi vĩ mô bên ngoài
# 3. Ghép dữ liệu theo ngày
# 4. Tách X/y và chia train/val/test theo đúng thứ tự thời gian
#
# Đây là tầng nền của toàn bộ project. Nếu hiểu rõ file này thì sẽ hiểu:
# - dữ liệu nào đi vào mô hình
# - dữ liệu nào bị loại bỏ
# - project chống rò rỉ dữ liệu thời gian ra sao

# Mã cổ phiếu mặc định dùng trong notebook demo.
TICKER_DEFAULT = "AAPL"
# Lấy lịch sử đủ dài để tính rolling indicators và chia tập theo thời gian.
START_DATE = "2012-01-01"
# Các biến thị trường/vĩ mô dùng làm ngữ cảnh cho mô hình dự đoán cổ phiếu.
MACRO_TICKERS = {
    # ^VIX   : biến động kỳ vọng của thị trường Mỹ
    # ^GSPC  : chỉ số S&P 500, đại diện cho thị trường chung
    # DX-Y.NYB: Dollar Index
    # ^TNX   : lợi suất trái phiếu Mỹ 10 năm
    # CL=F   : hợp đồng tương lai dầu
    # GC=F   : hợp đồng tương lai vàng
    "VIX":  "^VIX",
    "SPX":  "^GSPC",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",
    "OIL":  "CL=F",
    "GOLD": "GC=F",
}

def fetch_stock(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    """Download daily OHLCV for one ticker from yfinance."""
    # Với một ticker duy nhất, hàm này trả về bảng dữ liệu theo ngày gồm:
    # Open, High, Low, Close, Volume.
    #
    # Đây là dữ liệu cơ sở để tạo các feature kỹ thuật như:
    # - return trễ
    # - rolling mean/std
    # - RSI, MACD, Bollinger
    # - gap và intraday range

    # auto_adjust=True giúp giá đã được điều chỉnh theo split/dividend.
    # Điều này an toàn hơn cho bài toán dự đoán return so với dùng giá thô.
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    # yfinance đôi khi trả về cột dạng MultiIndex; ép về 1 tầng để phần code
    # phía sau luôn truy cập được bằng tên cột đơn giản như "Close", "Volume".
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Chỉ giữ các cột mà project thực sự sử dụng.
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # Thêm mã ticker vào dữ liệu để giữ ngữ cảnh nếu cần kiểm tra/debug về sau.
    df["Ticker"] = ticker
    # Chuẩn hóa index về kiểu thời gian để các phép shift/rolling/join hoạt động ổn định.
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    """Download all macro indicators and merge into one DataFrame."""
    # Ý tưởng: ngoài dữ liệu riêng của cổ phiếu, mô hình còn cần "bối cảnh thị trường".
    # Ví dụ:
    # - VIX phản ánh nỗi sợ / biến động thị trường
    # - SPX phản ánh xu hướng thị trường chung
    # - DXY phản ánh sức mạnh đồng USD
    # - TNX phản ánh lãi suất dài hạn
    # - OIL, GOLD phản ánh nhóm hàng hóa lớn
    parts = []
    for name, symbol in MACRO_TICKERS.items():
        # Mỗi biến vĩ mô chỉ cần chuỗi giá đóng cửa.
        s = yf.download(symbol, start=start, progress=False, auto_adjust=True)["Close"]
        if isinstance(s, pd.DataFrame):
            # Có trường hợp yfinance trả về DataFrame 1 cột thay vì Series.
            s = s.iloc[:, 0]
        # Đặt tên cột ngay tại đây để khi concat xong không cần rename lại.
        s.name = name
        parts.append(s)
    macro = pd.concat(parts, axis=1)
    # Điền tiếp các khoảng trống ngắn do lịch nghỉ khác nhau giữa các thị trường.
    # Giới hạn 2 ngày để tránh kéo dài dữ liệu cũ quá xa.
    macro = macro.ffill(limit=2)
    # Sau bước này, mỗi cột trong macro là một chuỗi thời gian có tên riêng:
    # VIX, SPX, DXY, TNX, OIL, GOLD
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    return macro

def build_dataset(ticker: str, stocks: dict, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge one stock with all macros on date index"""
    # stocks là dict dạng {"AAPL": df_AAPL, "MSFT": df_MSFT, ...}
    # Ở hàm này ta chỉ lấy đúng ticker đang quan tâm.
    stock_df = stocks[ticker]
    # Inner join chỉ giữ các ngày xuất hiện đồng thời ở cả dữ liệu cổ phiếu
    # và dữ liệu vĩ mô. Cách này chặt hơn nhưng đảm bảo các hàng được căn thẳng.
    df = stock_df.join(macro, how="inner")
    # dropna() bảo đảm dữ liệu sau khi ghép không còn lỗ hổng trước khi đi vào
    # bước tạo feature. Đây là bước làm sạch tối thiểu nhưng quan trọng.
    df = df.dropna()
    return df

# Các cột thô / không dừng sẽ không đưa trực tiếp vào mô hình.
RAW_DROP = [
    "Open", "High", "Low", "Close", "Volume", "LogVolume", "Ticker",
    "VIX", "SPX", "DXY", "TNX", "OIL", "GOLD",   # bỏ mức gốc; chỉ giữ bản return/lag
    "ret",                                       # biến phụ trợ daily return; đã có ret_lag1
    "target",
]

# Lý do loại các cột trên:
# - Giá mức gốc như Close, SPX, VIX... thường không dừng và mang xu hướng dài hạn.
# - Mô hình tài chính thường học tốt hơn trên biến đổi tương đối như return, ratio, z-score.
# - "target" là nhãn cần dự đoán nên không được xuất hiện trong X.
# - "ret" là biến phụ trợ để tạo feature; sau đó ta giữ lại các phiên bản trễ/rolling của nó.


def split_xy(df: pd.DataFrame):
    """Drop non-stationary columns and the target. Returns (X, y, feature_cols)."""
    # Chỉ đưa các đặc trưng đã engineering vào mô hình. Các mức giá thô bị loại
    # vì chúng không dừng và có thể làm mô hình học lệch theo mức giá tuyệt đối.
    feature_cols = [c for c in df.columns if c not in RAW_DROP]
    # X là ma trận đặc trưng đầu vào của mô hình.
    X = df[feature_cols].copy()
    # y là nhãn: log-return 5 ngày tới.
    y = df["target"].copy()
    # feature_cols được trả về để:
    # - lưu vào artifact
    # - đảm bảo inference dùng đúng thứ tự cột như lúc train
    return X, y, feature_cols


def time_split(X: pd.DataFrame, y: pd.Series,
               train_ratio: float = 0.70, val_ratio: float = 0.15,
               horizon: int = 5):
    """Sequential split with embargo: purge `horizon` rows at each boundary so
    train labels don't reference val prices (and val labels don't reference test).
    target[t] = log(Close[t+horizon]/Close[t]) leaks `horizon` rows across each cut.
    """
    n = len(X)
    # Quy mô dữ liệu thực tế sau khi đã drop NaN và tạo feature/target.
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    # Cách chia:
    # - train: phần đầu chuỗi
    # - val: phần ở giữa
    # - test: phần cuối chuỗi
    #
    # Không được shuffle vì đây là chuỗi thời gian. Nếu shuffle thì mô hình sẽ
    # vô tình "nhìn thấy tương lai" trong tập train.

    # "Embargo" loại bỏ `horizon` dòng cuối ở train và val.
    # Nếu không làm vậy, nhãn ở gần biên có thể nhìn sang giá tương lai thuộc
    # split kế tiếp, tức là gây rò rỉ dữ liệu.
    #
    # Ví dụ:
    # target[t] = log(Close[t+5] / Close[t])
    # Nếu t nằm sát cuối train thì Close[t+5] có thể nằm sang vùng validation.
    # Khi đó train đã dùng thông tin của validation để tạo nhãn -> leakage.
    X_train, y_train = X.iloc[:n_train - horizon],                y.iloc[:n_train - horizon]
    X_val,   y_val   = X.iloc[n_train:n_train + n_val - horizon], y.iloc[n_train:n_train + n_val - horizon]
    X_test,  y_test  = X.iloc[n_train + n_val:],                  y.iloc[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test
