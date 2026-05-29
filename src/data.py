"""
Module data.py — Tầng dữ liệu (data layer) của pipeline StockSight-RF.

Tóm tắt vai trò trong pipeline:
----------------------------------------------------------------------
Module này phụ trách TẢI và GỘP dữ liệu thô từ yfinance, gồm hai nguồn:
  1) Dữ liệu cổ phiếu (OHLCV) của 1 mã ticker (ví dụ AAPL).
  2) Dữ liệu vĩ mô (macro indicators) như VIX, S&P500, USD index, lãi suất
     trái phiếu 10 năm, dầu thô, vàng. Các chỉ số này ảnh hưởng đến giá cổ
     phiếu nên được dùng làm feature bổ sung.

Sau khi tải xong, module cung cấp các hàm:
  - build_dataset(): gộp 1 stock với toàn bộ macro theo trục thời gian.
  - split_xy():      tách feature (X) khỏi target (y) và LOẠI BỎ các cột
                     thô (non-stationary) không nên đưa vào Random Forest.
  - time_split():    chia tập theo THỜI GIAN (không shuffle) thành
                     train/val/test, đồng thời "embargo" (cắt bỏ) một số
                     dòng ở mỗi mép để chống data leakage do nhãn target
                     được tạo bằng cách nhìn về tương lai `horizon` ngày.

Lưu ý chung:
  - Pipeline này hướng đến bài toán dự đoán log-return 5 ngày tới
    (target[t] = log(Close[t+5] / Close[t])).
  - Random Forest không học tốt với feature dạng "level" (giá tuyệt đối)
    vì chúng phi dừng (non-stationary) — giá AAPL năm 2012 và 2024 ở 2
    khoảng giá hoàn toàn khác nhau, một rule kiểu "Close > 150" sẽ vô
    nghĩa khi áp sang giai đoạn khác. Vì vậy chúng ta thay thế bằng các
    biến dẫn xuất (return, lag, log-volume, ...) trong feature engineering.
"""

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# HẰNG SỐ CẤU HÌNH (configuration constants)
# ---------------------------------------------------------------------------
# Mã ticker mặc định nếu caller không truyền ticker cụ thể.
TICKER_DEFAULT = "AAPL"
# Ngày bắt đầu tải dữ liệu — lấy đủ dài để có sample cho train/val/test.
START_DATE = "2012-01-01"

# Bảng ánh xạ TÊN NGẮN GỌN (dùng làm tên cột trong dataset) -> SYMBOL THẬT
# trên yfinance. Các chỉ số macro này phản ánh môi trường thị trường rộng
# hơn, có khả năng tác động đến giá cổ phiếu cá nhân:
#   - VIX  (^VIX)      : "fear index" — chỉ số đo độ biến động ngụ ý của
#                        S&P500. VIX cao = nhà đầu tư sợ hãi, thị trường
#                        biến động mạnh.
#   - SPX  (^GSPC)     : chỉ số S&P500 — đại diện cho toàn bộ thị trường
#                        chứng khoán Mỹ. Cổ phiếu lớn thường đi cùng SPX.
#   - DXY  (DX-Y.NYB)  : US Dollar Index — sức mạnh đồng USD so với rổ tiền
#                        tệ. USD mạnh thường khiến doanh thu quốc tế của
#                        công ty Mỹ (vd AAPL) bị ảnh hưởng tiêu cực.
#   - TNX  (^TNX)      : lợi suất trái phiếu chính phủ Mỹ kỳ hạn 10 năm.
#                        Lãi suất tăng -> chiết khấu dòng tiền tương lai
#                        cao hơn -> cổ phiếu tăng trưởng (tech) thường giảm.
#   - OIL  (CL=F)      : hợp đồng tương lai dầu thô WTI — proxy cho lạm
#                        phát chi phí, ảnh hưởng đến hàng không, vận tải.
#   - GOLD (GC=F)      : hợp đồng tương lai vàng — tài sản trú ẩn an toàn;
#                        thường tăng khi nhà đầu tư lo ngại rủi ro.
MACRO_TICKERS = {
    "VIX":  "^VIX",
    "SPX":  "^GSPC",
    "DXY":  "DX-Y.NYB",
    "TNX":  "^TNX",
    "OIL":  "CL=F",
    "GOLD": "GC=F",
}

def fetch_stock(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    """Download daily OHLCV for one ticker from yfinance.

    --- Tiếng Việt ---
    Tải dữ liệu giá cổ phiếu hàng ngày (OHLCV) cho MỘT mã ticker.

    Tham số:
        ticker : mã cổ phiếu (vd "AAPL", "MSFT").
        start  : ngày bắt đầu (chuỗi "YYYY-MM-DD"); mặc định 2012-01-01.

    Trả về:
        pd.DataFrame với index là Date (datetime) và các cột:
        Open, High, Low, Close, Volume, Ticker.

    Gotcha (điểm cần lưu ý):
        - Khi yfinance tải dữ liệu, đôi khi nó trả về cột dạng MultiIndex
          (khi gọi nhiều ticker hoặc do thay đổi của lib). Ta phải flatten
          để dùng tiếp được như bảng phẳng.
        - auto_adjust=True quan trọng: giá Close đã được điều chỉnh sẵn
          cho các sự kiện chia tách cổ phiếu (split) và cổ tức (dividend).
          Không có nó, mô hình sẽ nhìn thấy "cú nhảy giá ảo" mỗi lần split.
    """
    # auto_adjust=True: yfinance tự điều chỉnh OHLC theo split & dividend
    # -> giá liền mạch theo thời gian, tránh các "đứt gãy" giả tạo mà
    # Random Forest có thể nhầm thành tín hiệu thật.
    # progress=False: tắt thanh tiến trình cho output gọn gàng.
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    # flatten multiIndex columns
    # --- VN ---
    # yfinance đôi khi trả về DataFrame có cột dạng MultiIndex
    # (vd: ("Close", "AAPL")). Khi đó ta lấy level 0 — tên trường OHLCV —
    # và bỏ level 1 (tên ticker), để bảng có cột phẳng dễ thao tác.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Chỉ giữ 5 cột OHLCV chuẩn. Các cột khác (Adj Close, Dividends,
    # Stock Splits, ...) bị loại vì:
    #  - Adj Close: với auto_adjust=True thì Close đã chính là adjusted.
    #  - Dividends / Splits: không dùng làm feature trong pipeline này.
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # Thêm cột Ticker để khi nối nhiều stock sau này còn biết dòng nào
    # thuộc về mã nào (phòng trường hợp mở rộng multi-ticker).
    df["Ticker"] = ticker
    # Đảm bảo index là kiểu datetime để các phép join/resample theo thời
    # gian hoạt động đúng. Đặt tên index = "Date" cho rõ ràng.
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    """Download all macro indicators and merge into one DataFrame.

    --- Tiếng Việt ---
    Tải toàn bộ chỉ số vĩ mô trong MACRO_TICKERS và gộp thành 1 DataFrame
    duy nhất, mỗi cột là một chỉ số (VIX, SPX, DXY, TNX, OIL, GOLD).

    Tham số:
        start : ngày bắt đầu (mặc định 2012-01-01).

    Trả về:
        pd.DataFrame với index Date, các cột là tên macro (không phải symbol).

    Gotcha:
        - Mỗi thị trường (CK Mỹ, futures, FX) có lịch nghỉ lễ khác nhau,
          nên khi gộp lại sẽ có các ngày NaN rải rác. Ta dùng ffill với
          giới hạn 2 ngày để vá những lỗ nhỏ này, KHÔNG forward-fill vô
          tận để tránh "kéo dài" giá quá hạn (vd nghỉ dài thực sự).
    """
    # Tải từng macro riêng rồi nối lại — đơn giản hơn là cố gắng tải
    # tất cả 1 lần (vì các symbol khác sàn / khác loại sản phẩm).
    parts = []
    for name, symbol in MACRO_TICKERS.items():
        # Chỉ lấy cột "Close" — với index/futures ta không cần OHLCV đầy đủ.
        s = yf.download(symbol, start=start, progress=False, auto_adjust=True)["Close"]
        # Một vài trường hợp yfinance trả về DataFrame 1 cột thay vì Series
        # (do thay đổi version). Chuyển về Series cho đồng nhất.
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        # Đặt tên Series = tên ngắn gọn (VIX, SPX, ...) thay vì symbol thật.
        s.name = name
        parts.append(s)
    # Nối các Series theo trục cột (axis=1) -> DataFrame có 6 cột macro.
    # pandas tự align theo index thời gian, ngày thiếu sẽ thành NaN.
    macro = pd.concat(parts, axis=1)
    # forward-fill small gaps (cross-market holidays), limit 2 days
    # --- VN ---
    # Vá các lỗ NaN do lệch lịch nghỉ giữa các thị trường (vd ngày Mỹ
    # nghỉ lễ Thanksgiving nhưng futures dầu vẫn giao dịch, hoặc ngược
    # lại). limit=2 -> chỉ kéo giá tối đa 2 phiên; nếu nghỉ lâu hơn vẫn
    # để NaN, tránh "fake data" kéo dài làm méo mô hình.
    macro = macro.ffill(limit=2)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    return macro

def build_dataset(ticker: str, stocks: dict, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge one stock with all macros on date index

    --- Tiếng Việt ---
    Gộp DataFrame của một stock với toàn bộ macro theo trục Date.

    Tham số:
        ticker : mã muốn build (key trong dict stocks).
        stocks : dict {ticker: DataFrame OHLCV} đã fetch trước.
        macro  : DataFrame macro đã fetch.

    Trả về:
        DataFrame đã gộp, không còn NaN, sẵn sàng cho feature engineering.

    Gotcha:
        - how="inner" -> chỉ giữ ngày có ĐỦ data của cả stock và macro.
          Nếu ngày X có stock nhưng macro NaN (chưa vá được) thì bỏ luôn
          ngày X. Như vậy tránh để model học trên hàng có NaN macro.
    """
    stock_df = stocks[ticker]
    # inner join trên index (Date) — drop những ngày chỉ có 1 trong 2 nguồn.
    df = stock_df.join(macro, how="inner")
    # dropna() lần nữa cho chắc: phòng trường hợp inner join vẫn còn NaN
    # rải rác (vd cột macro nào đó vẫn còn lỗ sau ffill limit=2).
    df = df.dropna()
    return df

# raw / non-stationary columns we don't feed to the model
# --- VN ---
# Danh sách các cột THÔ sẽ bị loại bỏ trước khi đưa vào Random Forest.
# Vì sao phải loại?
#
#   1) Open/High/Low/Close/Volume (giá và khối lượng tuyệt đối):
#      Đây là các biến NON-STATIONARY — giá AAPL năm 2012 (~$15) và năm
#      2024 (~$220) khác xa nhau. Nếu RF học rule "Close > 150 thì mua",
#      rule này chỉ đúng cho giai đoạn gần, vô nghĩa cho giai đoạn cũ.
#      Mô hình sẽ overfit theo "level" thay vì hiểu PATTERN.
#      => Thay bằng return (% thay đổi) và các biến dẫn xuất khác trong
#         feature engineering, vốn có phân phối ổn định theo thời gian.
#
#   2) LogVolume: đã được tạo trong feature engineering làm phiên bản
#      "đỡ phi dừng" của Volume; cột Volume gốc không còn cần thiết.
#
#   3) Ticker: chuỗi text, không phải feature số.
#
#   4) VIX/SPX/DXY/TNX/OIL/GOLD (levels): tương tự giá cổ phiếu, các
#      chỉ số macro ở dạng "level" cũng phi dừng (vd SPX leo từ 1300 lên
#      4500 trong giai đoạn này). Ta thay bằng:
#         - {name}_ret5 : log-return 5 ngày của macro (đo "đà thay đổi").
#         - {name}_lag1 : giá trị macro của ngày hôm qua (đo "trạng thái
#                         gần đây" mà không leak thông tin hôm nay).
#      Hai biến này stationary hơn và có ý nghĩa kinh tế rõ ràng.
#
#   5) ret: daily return (% thay đổi 1 ngày) — chỉ là biến trung gian
#      khi tính các feature khác. ret_lag5 (return trễ 5 ngày) đã đại
#      diện cho yếu tố momentum nên ret thô bị loại.
#
#   6) target: đây là NHÃN (y), tuyệt đối không được nằm trong X — nếu
#      lọt vào sẽ thành "data leakage hoàn hảo" và model đạt R²=1.0 giả.
RAW_DROP = [
    "Open", "High", "Low", "Close", "Volume", "LogVolume", "Ticker",
    "VIX", "SPX", "DXY", "TNX", "OIL", "GOLD",   # macro levels; we keep _ret5 and _lag1
    "ret",                                       # daily return helper; ret_lag5 covers momentum
    "target",
]


def split_xy(df: pd.DataFrame):
    """Drop non-stationary columns and the target. Returns (X, y, feature_cols).

    --- Tiếng Việt ---
    Tách dataset đã gộp thành:
      - X            : ma trận feature (đã LOẠI các cột thô trong RAW_DROP).
      - y            : Series target (log-return 5 ngày tương lai).
      - feature_cols : danh sách tên cột feature, để log/inspect sau này.

    Lưu ý:
        Hàm này GIẢ ĐỊNH cột "target" đã tồn tại trong df. Cột target được
        tạo ở bước feature engineering / build_training_set TRƯỚC khi gọi
        split_xy. Công thức: target[t] = log(Close[t+horizon] / Close[t]).
    """
    # Lấy mọi cột KHÔNG nằm trong RAW_DROP -> đó là feature hợp lệ.
    feature_cols = [c for c in df.columns if c not in RAW_DROP]
    X = df[feature_cols].copy()
    # y lấy riêng cột target (đã được tạo trước ở bước feature engineering).
    y = df["target"].copy()
    return X, y, feature_cols


def time_split(X: pd.DataFrame, y: pd.Series,
               train_ratio: float = 0.70, val_ratio: float = 0.15,
               horizon: int = 5):
    """Sequential split with embargo: purge `horizon` rows at each boundary so
    train labels don't reference val prices (and val labels don't reference test).
    target[t] = log(Close[t+horizon]/Close[t]) leaks `horizon` rows across each cut.

    --- Tiếng Việt ---
    Chia dữ liệu theo THỜI GIAN (không shuffle!) thành 3 tập:
        train : 70% đầu  (mặc định)
        val   : 15% giữa
        test  : 15% cuối
    Đồng thời "embargo" — cắt bỏ `horizon` dòng cuối ở mỗi tập train/val
    để CHỐNG DATA LEAKAGE.

    -------------------------------------------------------------------
    VÌ SAO PHẢI EMBARGO? (đây là điểm CỰC KỲ QUAN TRỌNG)
    -------------------------------------------------------------------
    Target được định nghĩa là:
            target[t] = log( Close[t + horizon] / Close[t] )
    Tức là để biết nhãn của ngày t, ta phải nhìn vào giá đóng cửa của
    ngày t + 5 (với horizon=5).

    Giả sử ta chia train = [0 .. n_train-1], val = [n_train .. n_train+n_val-1]
    mà KHÔNG embargo. Khi đó:
      - Hàng cuối cùng của train (t = n_train - 1) có nhãn:
            target[n_train-1] = log(Close[n_train + 4] / Close[n_train-1])
        => nhãn này CHỨA THÔNG TIN giá của 5 ngày đầu tập VAL!
      - Mô hình học trên hàng đó, vô tình "thấy trước" tương lai val.
      - Kết quả: metric trên val (và test) bị THỔI PHỒNG ảo. Khi deploy
        thật, model sẽ không bao giờ có được thông tin "rò rỉ" này nên
        hiệu năng thực tế sụp đổ.

    Cách fix: BỎ `horizon` dòng cuối cùng của train (và của val).
    Cụ thể trong code dưới:
        X_train kết thúc ở vị trí n_train - horizon (chứ không phải n_train).
        X_val   kết thúc ở vị trí n_train + n_val - horizon.
        X_test  bắt đầu ở n_train + n_val (không cắt đầu vì biên test-future
                không tồn tại trong dataset — các nhãn vượt ngoài đã được
                build_training_set bỏ NaN trước đó rồi).

    Như vậy giữa cuối train và đầu val có "vùng đệm" `horizon` ngày
    không thuộc tập nào — đủ để nhãn train không chạm tới giá val.

    Tham số:
        X, y        : feature/target đã align theo Date.
        train_ratio : tỉ lệ train (mặc định 0.70).
        val_ratio   : tỉ lệ val   (mặc định 0.15). Phần còn lại = test.
        horizon     : số ngày nhìn tương lai khi build target (mặc định 5).

    Trả về:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Tổng số mẫu sau khi đã dropna trong build_training_set.
    n = len(X)
    # Số dòng dành cho train và val tính theo tỉ lệ.
    # Lưu ý dùng int() => làm tròn xuống, có thể "lệch" 1-2 dòng so với
    # tỉ lệ chính xác — chấp nhận được vì dataset đủ lớn.
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    # ----------------------------------------------------------------
    # Cắt 3 tập theo thứ tự thời gian (KHÔNG shuffle — dữ liệu time-series).
    # Embargo: trừ `horizon` ở mép phải của train và val để xoá phần
    # nhãn bị leak (xem giải thích kỹ trong docstring ở trên).
    # ----------------------------------------------------------------
    X_train, y_train = X.iloc[:n_train - horizon],                y.iloc[:n_train - horizon]
    X_val,   y_val   = X.iloc[n_train:n_train + n_val - horizon], y.iloc[n_train:n_train + n_val - horizon]
    X_test,  y_test  = X.iloc[n_train + n_val:],                  y.iloc[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test
