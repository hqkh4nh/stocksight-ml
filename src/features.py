"""
Feature engineering cho mô hình Random Forest dự đoán log-return 5 phiên tới.

Module này nhận DataFrame giá OHLCV + macro (SPX, DXY, TNX, OIL, GOLD, VIX) và sinh ra
các feature kỹ thuật (momentum, volatility, trend, RSI, MACD, microstructure, volume)
cùng feature macro (return 5 ngày + level lag-1) để RF học pattern.

Lưu ý quan trọng:
- Mọi feature macro đều .shift(1) để tránh look-ahead bias (xem chi tiết trong add_macro).
- Target là log-return tương lai 5 phiên: log(Close[T+5] / Close[T]).
- build_features_only dùng cho inference (không có target).
- build_training_set dùng cho train/val/test (có target + dropna).
"""

import numpy as np
import pandas as pd

from src.data import build_dataset

def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum, volatility, trend, RSI, MACD, microstructure, volume."""
    df = df.copy()

    # === Momentum + Volatility ===
    # ret_lag5: log-return của 5 phiên TRƯỚC -> momentum mid-term.
    # RF dùng để nhận diện "tuần qua tăng/giảm bao nhiêu" -> trend gần đây.
    df["ret_lag5"]  = df["ret"].shift(5)
    # ret_stdN: độ lệch chuẩn return trong cửa sổ N ngày -> đo volatility.
    # Giúp model phân biệt phiên "yên ả" (std thấp) vs "hỗn loạn" (std cao).
    df["ret_std5"]  = df["ret"].rolling(5).std()
    df["ret_std10"] = df["ret"].rolling(10).std()
    # ret_ma20: trung bình return 20 ngày -> drift gần đây (xu hướng trung bình tháng qua).
    df["ret_ma20"]  = df["ret"].rolling(20).mean()
    df["ret_std20"] = df["ret"].rolling(20).std()

    # === Trend: giá so với MA dài hạn ===
    # price_maW_ratio = Close / MA(W) -> đo "khoảng cách giá so với trend".
    # >1: giá đang TRÊN trend (bullish), <1: giá đang DƯỚI trend (bearish).
    # Dùng ratio thay vì hiệu để chuẩn hoá theo mức giá (so sánh được giữa các thời kỳ).
    for w in [20, 50]:
        df["price_ma" + str(w) + "_ratio"] = df["Close"] / df["Close"].rolling(w).mean()

    # === Microstructure ===
    # intraday_range = (High-Low)/Close -> biên độ giao dịch trong ngày, đo "căng thẳng" của phiên.
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    # close_loc: vị trí giá đóng cửa trong range của ngày.
    # = 1: đóng cửa ngay ở high (bull, mua mạnh cuối phiên).
    # = 0: đóng cửa ngay ở low (bear, bán tháo cuối phiên).
    # replace(0, nan) để tránh chia 0 ở những ngày market đứng giá (High == Low).
    range_hl             = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_loc"]      = (df["Close"] - df["Low"]) / range_hl

    # === RSI(14) - Relative Strength Index ===
    # Công thức cổ điển: chia tách gain và loss, lấy rolling mean 14 ngày.
    # rs = avg_gain / avg_loss, rsi = 100 - 100/(1+rs).
    # RSI > 70: quá mua (overbought, có thể sắp điều chỉnh).
    # RSI < 30: quá bán (oversold, có thể sắp hồi).
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)  # tránh chia 0 khi không có ngày giảm
    df["rsi14"] = 100 - 100 / (1 + rs)

    # === MACD - Moving Average Convergence Divergence ===
    # MACD line = EMA(12) - EMA(26) -> chênh lệch xu hướng ngắn vs trung hạn.
    # MACD histogram = MACD - EMA(9 of MACD) -> đo GIA TỐC momentum (đạo hàm bậc 2).
    # Histogram dương và tăng: momentum bull đang mạnh lên.
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]      = ema_12 - ema_26
    df["macd_hist"] = df["macd"] - df["macd"].ewm(span=9, adjust=False).mean()

    # === Volume bất thường ===
    # vol_z20: z-score của LogVolume so với rolling mean/std 20 ngày.
    # > 2: volume cực cao bất thường so với tháng qua -> thường đi kèm tin tức / breakout.
    # < -2: volume cực thấp -> thị trường thờ ơ.
    df["vol_z20"] = (df["LogVolume"] - df["LogVolume"].rolling(20).mean()) \
                    / df["LogVolume"].rolling(20).std()
    return df

def add_macro(df: pd.DataFrame) -> pd.DataFrame:
    """5-day log-return + lag-1 level per macro series. .shift(1) avoids same-day leak.

    Macro levels dominate feature importance — GOLD/TNX/SPX/DXY _lag1 occupy the top-5.
    """
    # === CHỐNG LOOK-AHEAD BIAS (rất quan trọng) ===
    # Tất cả feature macro phải .shift(1). Lý do:
    # Khi ta dự đoán giá AAPL phiên T+5 dựa trên dữ liệu chốt ngày T, ta KHÔNG được
    # dùng macro của ngày T -> vì lúc AAPL đóng cửa ngày T, các thị trường macro
    # (SPX, DXY, VIX...) cũng đang giao dịch hoặc đóng cửa cùng/sau giờ -> dữ liệu
    # macro ngày T thực chất chỉ có sau khi AAPL đã đóng cửa, dùng nó là cheat.
    # An toàn nhất: dùng macro của T-1 (ngày hôm trước, chắc chắn đã chốt).
    df = df.copy()
    for col in ["SPX", "DXY", "TNX", "OIL", "GOLD"]:
        # _ret5 = log return 5 phiên của macro (đã shift trước 1 ngày).
        # Stationary, mô tả "macro tuần qua đã đi như thế nào" -> mô hình thích vì
        # phân phối ổn định qua thời gian, không bị trend hoá.
        df[col + "_ret5"] = np.log(df[col].shift(1) / df[col].shift(6))
        # _lag1 = level tuyệt đối của macro tại T-1 (đã ffill từ fetch_macro nên không thiếu).
        # Tại sao giữ cả level (không stationary) ngoài return? Vì RF chia tree theo ngưỡng
        # giá trị -> phát hiện được REGIME: "khi GOLD > 1800 thì AAPL hành xử khác
        # khi GOLD < 1500". Đây là regime split, không cần stationary.
        # Thực tế: GOLD_lag1, TNX_lag1, SPX_lag1, DXY_lag1 luôn nằm trong top-5 feature
        # importance của RF -> giữ lại là quyết định đúng.
        df[col + "_lag1"] = df[col].shift(1)

    # === VIX - chỉ số sợ hãi của thị trường ===
    # VIX_log_lag1: lấy log vì VIX có phân phối right-skew nặng (thường ~15-20, đôi khi
    # spike lên 80+ khi crash). Log nén các spike, giúp model học ổn định hơn.
    df["VIX_log_lag1"] = np.log(df["VIX"].shift(1))
    # vix_z: z-score VIX so với rolling 60 ngày (~3 tháng).
    # > 2: VIX cao bất thường so với 3 tháng qua -> market đang panic so với gần đây.
    # Detect regime switch chứ không phải mức tuyệt đối.
    vix_lag = df["VIX"].shift(1)
    df["vix_z"] = (vix_lag - vix_lag.rolling(60).mean()) / vix_lag.rolling(60).std()
    return df

def build_features_only(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build all features but do not add target."""
    # Dùng cho INFERENCE: lúc cần dự đoán giá T+5, ta chỉ có dữ liệu đến ngày T,
    # không có target (target ở tương lai chưa biết). Hàm này build mọi feature
    # giống y hệt lúc train nhưng KHÔNG cần target -> tránh dropna ăn mất ngày cuối.
    df = raw_df.copy()
    # LogVolume = log(1 + Volume). Dùng log1p thay vì log để xử lý an toàn khi Volume=0
    # hoặc rất nhỏ: log(0) = -inf gây NaN, log1p(0) = 0 ok.
    df["LogVolume"] = np.log1p(df["Volume"])                # log-scale skewed volume
    # ret = daily log-return, là base cho mọi feature momentum/volatility ở add_technical.
    # Dùng log-return (không phải simple return) vì cộng được qua thời gian và đối xứng.
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))  # 1-day log-return
    df = add_technical(df)
    df = add_macro(df)
    return df


def build_training_set(ticker: str, stocks: dict, macro: pd.DataFrame,
                       horizon: int = 5) -> pd.DataFrame:
    """Build features + h-day forward log-return target, drop NaN rows."""
    raw_df   = build_dataset(ticker, stocks, macro)
    features = build_features_only(raw_df)
    # target = log-return tương lai HORIZON phiên: log(Close[T+5] / Close[T]).
    # shift(-5) = nhìn TƯƠNG LAI 5 ngày để lấy label.
    # Đây là supervised learning: label được tạo từ future hoàn toàn HỢP PHÁP khi TRAIN,
    # vì lúc dự đoán thực tế ta sẽ không có nó (chỉ có khi đã trôi qua 5 ngày để chấm điểm).
    features["target"] = np.log(features["Close"].shift(-horizon) / features["Close"])
    # dropna() cắt bỏ những row có NaN:
    # - Đầu series: rolling window 20/50/60 ngày chưa đủ data -> feature NaN.
    # - Cuối series: shift(-5) chưa có future Close -> target NaN.
    return features.dropna()

def fit_winsorize(train_series: pd.Series, n_std: float = 2.0):
    """Return mean, std, lower_bound, upper_bound. Use on train only.

    Default +/- 2 sigma (tighter than conventional 3) so the model learns the
    everyday signal better when extreme returns are clipped harder.
    """
    # Winsorize = clip outlier của target về một khoảng [lower, upper].
    # CHỈ fit (tính mean/std) trên tập TRAIN -> tránh leakage. Bound này sau đó
    # được áp dụng nguyên xi cho val/test (không tính lại).
    # n_std mặc định 2.0 chặt hơn quy ước 3.0 -> vì 5-day return ít cực đoan hơn
    # daily return, clip mạnh hơn giúp RF tập trung học signal "everyday" thay vì
    # bị các phiên crash kéo lệch.
    mean = train_series.mean()
    std  = train_series.std()
    lower = mean - n_std * std
    upper = mean + n_std * std
    return mean, std, lower, upper
