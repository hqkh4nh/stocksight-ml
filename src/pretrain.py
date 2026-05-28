"""Train and save RF artifacts for the curated TOP20 ticker list."""
from typing import Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data import (fetch_stock, fetch_macro, split_xy, time_split)
from src.features import build_training_set, fit_winsorize
from src.model_io import has_artifact, save_artifact

SEED = 42
# horizon=5 nghĩa là mô hình học cách dự báo lợi suất sau 5 ngày giao dịch.
HORIZON = 5
# App triển khai cố ý dùng lịch sử ngắn hơn notebook demo để thời gian train
# thực tế hơn và tập trung nhiều hơn vào các giai đoạn thị trường gần đây.
START_DATE = "2015-01-01"

# Bộ tham số RF cố định dùng cho bản triển khai.
RF_PARAMS = dict(
    # 300 cây đủ để ổn định hơn so với số lượng quá ít.
    n_estimators=300,
    # depth=3 giữ cây nông để giảm overfitting.
    max_depth=3,
    # leaf lớn buộc mô hình học tín hiệu tổng quát hơn thay vì nhiễu riêng lẻ.
    min_samples_leaf=50,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

# Top 20 cổ phiếu S&P 500 theo vốn hóa thị trường (xấp xỉ đầu năm 2026).
TOP20 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "V",    "LLY",  "UNH",   "JNJ",   "XOM",
    "WMT",  "MA",   "PG",    "ORCL",  "HD",
]

# File này đóng vai trò "pipeline huấn luyện deploy":
# - notebook dùng để nghiên cứu / thử nghiệm
# - còn file này dùng để train mô hình thật cho app Streamlit sử dụng
#
# Vì vậy nó ưu tiên:
# - ít tham số hơn
# - train ổn định hơn
# - tốc độ phù hợp để pretrain nhiều ticker


def train_one(ticker: str, start: str = START_DATE, horizon: int = HORIZON,
              macro=None) -> dict:
    """Fetch data, build features, fit the RF, save the artifact, return it.

    Pass `macro` (a pre-fetched fetch_macro() result) to avoid refetching macros
    when training many tickers in a row.
    """
    # stocks được giữ ở dạng dict để tái sử dụng interface giống notebook/pipeline chung.
    stocks = {ticker: fetch_stock(ticker, start=start)}
    if macro is None:
        # Nếu không truyền sẵn macro vào, tự tải macro mới cho ticker này.
        macro = fetch_macro(start=start)

    # Pipeline huấn luyện từng ticker gồm 4 lớp:
    # 1. tạo feature + target
    # 2. chia train/val/test theo thời gian
    # 3. xử lý target và scale feature
    # 4. train RF rồi lưu artifact
    feat_df = build_training_set(ticker, stocks, macro, horizon=horizon)
    # split_xy đồng thời trả về feature_cols để lưu vào artifact sau này.
    X, y, feature_cols = split_xy(feat_df)
    X_tr, y_tr, X_va, y_va, X_te, y_te = time_split(X, y, horizon=horizon)

    # Chỉ clip target ở train để giảm ảnh hưởng của các cú sốc cực đoan.
    # Validation/test vẫn giữ nguyên để việc đánh giá phản ánh dữ liệu thật hơn.
    mean, std, lo, hi = fit_winsorize(y_tr)
    y_tr_clip = y_tr.clip(lo, hi)

    # Lưu scaler và thứ tự cột feature để lúc dự đoán về sau dữ liệu đầu vào
    # được biểu diễn đúng y hệt như lúc train.
    scaler = StandardScaler().fit(X_tr)
    # Ở bản deploy, mô hình chỉ train trên train split đã clip, không tune lại ở đây.
    model = RandomForestRegressor(**RF_PARAMS).fit(scaler.transform(X_tr), y_tr_clip)

    # Artifact lưu đủ thông tin để app không phải train lại mỗi lần mở lên.
    save_artifact(
        ticker=ticker,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        win_params=(mean, std, lo, hi),
        horizon=horizon,
    )
    return {
        # Trả thêm model/scaler/feature_cols để hàm này có thể tái sử dụng ở nơi khác
        # ngoài việc chỉ lưu xuống đĩa.
        "ticker": ticker,
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "win_params": (mean, std, lo, hi),
        "horizon": horizon,
    }


def train_missing(
    tickers: list[str] | None = None,
    force: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    """Train every ticker that has no artifact yet (or all of them when force=True).

    progress_callback(done_count, total, current_ticker) fires after each ticker
    so a Streamlit progress bar can update.
    """
    tickers = list(tickers) if tickers is not None else TOP20
    # Ép sang list để tránh trường hợp caller truyền tuple/generator.
    # Nếu force=False: chỉ train ticker chưa có artifact
    # Nếu force=True : train lại toàn bộ danh sách
    targets = [t for t in tickers if force or not has_artifact(t)]
    if not targets:
        # Trả list rỗng để caller biết không có việc phải làm.
        return []

    # Tải macro một lần rồi dùng lại cho mọi ticker vì tất cả cùng chia sẻ
    # một bảng dữ liệu vĩ mô. Cách này tránh tải lặp lại 20 lần.
    macro = fetch_macro(start=START_DATE)

    trained = []
    for i, ticker in enumerate(targets, start=1):
        # Mỗi vòng lặp train xong một ticker rồi mới cập nhật tiến độ.
        train_one(ticker, macro=macro)
        trained.append(ticker)
        if progress_callback is not None:
            progress_callback(i, len(targets), ticker)
    return trained
