import numpy as np
import pandas as pd

from src.data import build_dataset

# File này là "trái tim" của feature engineering.
# Nhiệm vụ chính:
# 1. Nhận dữ liệu cổ phiếu + macro đã ghép theo ngày
# 2. Tạo ra các đặc trưng có ý nghĩa tài chính
# 3. Tạo target là log-return h ngày tới
#
# Nếu data.py trả lời câu hỏi "dữ liệu đến từ đâu?" thì features.py trả lời:
# "từ dữ liệu thô, ta biến thành tín hiệu học máy như thế nào?"

def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Lag returns, rolling stats, price ratios, RSI, MACD, Bollinger, microstructure."""
    # copy() để tránh vô tình sửa trực tiếp DataFrame gốc được truyền vào.
    df = df.copy()

    # ---------------------------
    # 1. Nhóm động lượng (momentum / reversal)
    # ---------------------------
    # Ý tưởng:
    # - Nếu các return gần đây dương liên tiếp, cổ phiếu có thể đang có động lượng tăng.
    # - Nếu return gần đây âm mạnh, có thể tồn tại hiệu ứng hồi lại hoặc tiếp tục giảm.
    # Return trễ cho mô hình biết cổ phiếu đang có động lượng hay có xu hướng
    # đảo chiều trong ngắn và trung hạn.
    for lag in [1, 5, 10]:
        # ret_lag1  = return hôm qua
        # ret_lag5  = return cách đây 5 phiên
        # ret_lag10 = return cách đây 10 phiên
        df["ret_lag" + str(lag)] = df["ret"].shift(lag)

    # ---------------------------
    # 2. Nhóm xu hướng và biến động cục bộ
    # ---------------------------
    # rolling mean của return ~ xu hướng gần đây
    # rolling std của return ~ mức nhiễu / mức rủi ro gần đây
    # Rolling mean/std tóm tắt xu hướng cục bộ và mức biến động gần đây.
    for window in [5, 10, 20]:
        # rolling(window).mean()/std() chỉ dùng dữ liệu quá khứ và hiện tại,
        # không dùng tương lai, nên an toàn cho bài toán time series.
        df["ret_ma"  + str(window)] = df["ret"].rolling(window).mean()
        df["ret_std" + str(window)] = df["ret"].rolling(window).std()

    # ---------------------------
    # 3. Giá tương đối so với đường trung bình
    # ---------------------------
    # Thay vì đưa trực tiếp Close vào mô hình, project dùng tỷ lệ:
    # Close / MA(window)
    # Cách này giữ lại ý nghĩa "giá đang cao hay thấp tương đối" mà vẫn ổn định hơn.
    # Tỷ lệ giá so với MA ổn định hơn việc đưa trực tiếp mức giá tuyệt đối.
    for window in [20, 50]:
        # Nếu ratio > 1: giá đang cao hơn MA
        # Nếu ratio < 1: giá đang thấp hơn MA
        df["price_ma" + str(window) + "_ratio"] = df["Close"] / df["Close"].rolling(window).mean()

    # ---------------------------
    # 4. Microstructure đơn giản từ OHLC
    # ---------------------------
    # Đây là các feature mô tả hành vi phiên giao dịch hiện tại.
    # Nhóm feature này mô tả "hình dạng" phiên giao dịch hôm nay:
    # - gap: mở cửa lệch bao nhiêu so với đóng cửa hôm trước
    # - intraday_range: biên độ dao động trong ngày
    # - close_loc: giá đóng cửa nằm ở vị trí nào trong biên độ ngày
    prev_close = df["Close"].shift(1)
    # prev_close là giá đóng cửa của hôm trước, dùng để đo "gap".
    df["gap"]            = (df["Open"] - prev_close) / prev_close
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    range_hl             = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_loc"]      = (df["Close"] - df["Low"]) / range_hl

    # ---------------------------
    # 5. Bollinger position
    # ---------------------------
    # Nếu giá nằm xa khỏi MA20 theo đơn vị std, có thể thị trường đang:
    # - quá nóng
    # - quá yếu
    # - hoặc đang breakout
    # Vị trí Bollinger gần giống z-score của giá quanh trung bình 20 ngày.
    ma_20  = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    # Chia cho 2*std vì Bollinger band thường dùng MA ± 2*std.
    df["bb_pos"] = (df["Close"] - ma_20) / (2 * std_20)

    # ---------------------------
    # 6. RSI
    # ---------------------------
    # RSI dùng trung bình mức tăng và mức giảm để đo xung lực giá.
    # Giá trị cao thường gợi ý quá mua, thấp thường gợi ý quá bán.
    # RSI(14) là chỉ báo kinh điển để đo trạng thái quá mua / quá bán.
    delta = df["Close"].diff()
    # gain chỉ giữ phần tăng, loss chỉ giữ phần giảm (đổi dấu về dương).
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)

    # ---------------------------
    # 7. MACD
    # ---------------------------
    # EMA12 phản ứng nhanh hơn EMA26.
    # Hiệu của chúng thể hiện đà ngắn hạn so với xu hướng dài hơn.
    # MACD đo chênh lệch giữa xu hướng EMA ngắn hạn và dài hạn.
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    # Histogram = MACD - tín hiệu của MACD, thường dùng để đo gia tốc xu hướng.
    macd_signal = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - macd_signal

    # ---------------------------
    # 8. Volume anomaly
    # ---------------------------
    # Câu hỏi: volume hôm nay có cao bất thường so với 20 ngày gần nhất không?
    # Nhiều chuyển động giá có ý nghĩa hơn khi đi kèm volume bất thường.
    # Volume z-score trả lời câu hỏi: hôm nay khối lượng giao dịch có bất thường không?
    df["vol_z20"] = (df["LogVolume"] - df["LogVolume"].rolling(20).mean()) \
                    / df["LogVolume"].rolling(20).std()
    return df

def add_macro(df: pd.DataFrame) -> pd.DataFrame:
    """1-day and 5-day log-returns + lag-1 level for each macro series.
    Inputs are .shift(1) first so today's macro close (released after equity close) doesn't leak.
    """
    # Macro feature thường giúp mô hình hiểu "môi trường thị trường" chứ không
    # chỉ nhìn hành vi riêng của cổ phiếu.
    df = df.copy()

    # Với mỗi chuỗi vĩ mô, project tạo ba kiểu thông tin:
    # - return 1 ngày: phản ứng rất ngắn hạn
    # - return 5 ngày: xu hướng ngắn hạn
    # - lag 1 mức gốc: trạng thái tuyệt đối gần nhất đã biết
    for col in ["VIX", "SPX", "DXY", "TNX", "OIL", "GOLD"]:
        # shift(1) là bước chống leakage quan trọng nhất ở nhóm macro:
        # dùng dữ liệu vĩ mô của hôm trước để dự đoán, không dùng thông tin có
        # thể chỉ hoàn chỉnh sau khi phiên cổ phiếu hôm nay kết thúc.
        df[col + "_ret1"] = np.log(df[col].shift(1) / df[col].shift(2))
        df[col + "_ret5"] = np.log(df[col].shift(1) / df[col].shift(6))
        df[col + "_lag1"] = df[col].shift(1)

    # VIX thường lệch phải nên log-transform sẽ dễ học hơn.
    df["VIX_log_lag1"] = np.log(df["VIX"].shift(1))

    # VIX z-score cho biết biến động hiện tại có cao bất thường so với lịch sử
    # gần đây hay không, thay vì chỉ nhìn mức tuyệt đối.
    # Đây là một feature quan trọng vì "mức sợ hãi bất thường" của thị trường
    # thường ảnh hưởng mạnh đến hành vi của cổ phiếu riêng lẻ.
    vix_lag = df["VIX"].shift(1)
    df["vix_z"] = (vix_lag - vix_lag.rolling(60).mean()) / vix_lag.rolling(60).std()
    return df

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, month, month-end flag."""
    df = df.copy()
    # Hiệu ứng lịch là nhóm feature đơn giản nhưng thường hữu ích trong tài chính.
    # Một số cổ phiếu/chiến lược có thể phản ứng khác nhau theo thứ trong tuần
    # hoặc ở gần cuối tháng do tái cân bằng danh mục/quỹ.
    df["dow"]          = df.index.dayofweek
    df["month"]        = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(int)
    return df

def build_features_only(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build all features but do noy add target."""
    df = raw_df.copy()
    # Log-volume giúp giảm độ lệch do những phiên volume tăng đột biến.
    df["LogVolume"] = np.log1p(df["Volume"])
    # Daily log-return là chuỗi gốc để xây nhiều chỉ báo kỹ thuật.
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # Xây feature theo từng bước để pipeline dễ đọc, dễ kiểm tra và tái sử dụng.
    # Thứ tự này cũng phản ánh logic phân nhóm feature của project.
    df = add_technical(df)
    df = add_macro(df)
    df = add_calendar(df)
    # Kết quả lúc này vẫn còn chứa một số cột thô như Close, Volume...
    # Việc loại bỏ cột nào sẽ được xử lý ở data.split_xy().
    return df


def build_training_set(ticker: str, stocks: dict, macro: pd.DataFrame,
                       horizon: int = 5) -> pd.DataFrame:
    """Build features + h-day forward log-return target, drop NaN rows."""
    # Bước 1: ghép dữ liệu cổ phiếu với macro.
    raw_df   = build_dataset(ticker, stocks, macro)
    # Bước 2: tạo toàn bộ feature đầu vào.
    features = build_features_only(raw_df)
    # target[t] là log-return 5 ngày trong tương lai, tính từ ngày t đến t+horizon.
    # Nhờ đó bài toán được chuyển thành hồi quy có giám sát.
    features["target"] = np.log(features["Close"].shift(-horizon) / features["Close"])
    # Rolling window và shift sẽ sinh ra NaN ở đầu/cuối chuỗi; chỉ drop sau khi
    # đã tạo xong toàn bộ feature và target để tránh bỏ sót logic.
    return features.dropna()

def fit_winsorize(train_series: pd.Series, n_std: float = 2.0):
    """Return mean, std, lower_bound, upper_bound. Use on train only.

    Default +/- 2 sigma. Tighter than the conventional 3 sigma because the model
    learns the everyday signal better when extreme returns (earnings, COVID-style
    shocks) are clipped harder.
    """
    # Chỉ fit ngưỡng clipping trên train. Nếu dùng val/test ở đây thì mô hình sẽ
    # gián tiếp biết trước mức biến động của tương lai.
    #
    # Ý nghĩa của winsorization:
    # - Không xóa hẳn điểm dữ liệu cực đoan
    # - Chỉ cắt chúng về một ngưỡng hợp lý hơn
    # - Giúp mô hình đỡ bị một vài cú sốc hiếm làm lệch trọng tâm học
    mean = train_series.mean()
    std  = train_series.std()
    # lower/upper là 2 ngưỡng cắt trên dưới của target train.
    lower = mean - n_std * std
    upper = mean + n_std * std
    return mean, std, lower, upper
