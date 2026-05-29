import numpy as np
import pandas as pd

from src.data import build_dataset

# File này là phần feature engineering thực tế của project sau khi đã prune feature.
# Hiện tại feature set gọn hơn trước:
# - bỏ calendar
# - bỏ gap, Bollinger position
# - bỏ nhiều lag/rolling ngắn không mang thêm nhiều giá trị
# - giữ lại các feature có tín hiệu tốt hơn theo importance / validation IC


def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum, volatility, trend, RSI, MACD, microstructure, volume."""
    df = df.copy()

    # 1. Momentum + volatility
    # ret_lag5  : tín hiệu động lượng ở mốc 1 tuần giao dịch trước
    # ret_std*  : mức biến động gần đây của return
    # ret_ma20  : xu hướng return trong khoảng 1 tháng giao dịch
    df["ret_lag5"] = df["ret"].shift(5)
    df["ret_std5"] = df["ret"].rolling(5).std()
    df["ret_std10"] = df["ret"].rolling(10).std()
    df["ret_ma20"] = df["ret"].rolling(20).mean()
    df["ret_std20"] = df["ret"].rolling(20).std()

    # 2. Trend theo tỷ lệ giá / trung bình động
    # Dùng ratio thay vì đưa raw price level vào model để giảm tính không dừng.
    for w in [20, 50]:
        df["price_ma" + str(w) + "_ratio"] = df["Close"] / df["Close"].rolling(w).mean()

    # 3. Microstructure đơn giản từ OHLC
    # intraday_range : biên độ dao động trong ngày
    # close_loc      : close nằm ở đâu trong khoảng high-low, phản ánh áp lực mua/bán cuối phiên
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    range_hl = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_loc"] = (df["Close"] - df["Low"]) / range_hl

    # 4. RSI(14): chỉ báo quá mua / quá bán
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # 5. MACD và MACD histogram: đo chênh lệch giữa EMA nhanh và EMA chậm
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_hist"] = df["macd"] - df["macd"].ewm(span=9, adjust=False).mean()

    # 6. Volume bất thường theo z-score 20 phiên
    df["vol_z20"] = (
        (df["LogVolume"] - df["LogVolume"].rolling(20).mean())
        / df["LogVolume"].rolling(20).std()
    )
    return df


def add_macro(df: pd.DataFrame) -> pd.DataFrame:
    """5-day log-return + lag-1 level per macro series. .shift(1) avoids same-day leak.

    Macro levels dominate feature importance — GOLD/TNX/SPX/DXY _lag1 occupy the top-5.
    """
    df = df.copy()

    # Bản feature hiện tại chỉ giữ:
    # - _ret5 : xu hướng macro trong 5 phiên
    # - _lag1 : trạng thái mức gốc đã biết ở phiên trước
    #
    # So với bản cũ, các macro _ret1 đã bị bỏ sau khi prune feature.
    for col in ["SPX", "DXY", "TNX", "OIL", "GOLD"]:
        # shift(1) là bước chống same-day leak:
        # dùng macro đã biết tới hết hôm trước để dự đoán tương lai.
        df[col + "_ret5"] = np.log(df[col].shift(1) / df[col].shift(6))
        df[col + "_lag1"] = df[col].shift(1)

    # VIX được xử lý riêng:
    # - log-level vì VIX lệch phải
    # - z-score 60 ngày để đo "regime" biến động hiện tại cao hay thấp bất thường
    df["VIX_log_lag1"] = np.log(df["VIX"].shift(1))
    vix_lag = df["VIX"].shift(1)
    df["vix_z"] = (vix_lag - vix_lag.rolling(60).mean()) / vix_lag.rolling(60).std()
    return df


def build_features_only(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build all features but do not add target."""
    df = raw_df.copy()

    # LogVolume dùng để nén volume lệch phải.
    df["LogVolume"] = np.log1p(df["Volume"])

    # ret là 1-day log-return, chuỗi gốc để xây các feature kỹ thuật.
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))

    df = add_technical(df)
    df = add_macro(df)
    return df


def build_training_set(
    ticker: str,
    stocks: dict,
    macro: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Build features + h-day forward log-return target, drop NaN rows."""
    # 1. Ghép stock + macro
    raw_df = build_dataset(ticker, stocks, macro)

    # 2. Tạo feature
    features = build_features_only(raw_df)

    # 3. Tạo target là log-return 5 ngày tới
    features["target"] = np.log(features["Close"].shift(-horizon) / features["Close"])

    # 4. dropna() vì rolling/shift sẽ làm đầu và cuối chuỗi bị thiếu
    return features.dropna()


def fit_winsorize(train_series: pd.Series, n_std: float = 2.0):
    """Return mean, std, lower_bound, upper_bound. Use on train only.

    Default +/- 2 sigma (tighter than conventional 3) so the model learns the
    everyday signal better when extreme returns are clipped harder.
    """
    # Chỉ fit clipping range trên train để tránh leakage.
    mean = train_series.mean()
    std = train_series.std()
    lower = mean - n_std * std
    upper = mean + n_std * std
    return mean, std, lower, upper
