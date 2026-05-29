"""Train and save RF artifacts for the curated TOP20 ticker list."""
from typing import Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data import fetch_macro, fetch_stock, split_xy, time_split
from src.features import build_training_set, fit_winsorize
from src.model_io import has_artifact, save_artifact

SEED = 42
HORIZON = 5
START_DATE = "2015-01-01"

# File này là pipeline huấn luyện dùng cho deploy / Streamlit app.
# Khác với notebook phân tích:
# - notebook có thể tune RF và thử nhiều biến thể
# - file này ưu tiên train ổn định, gọn và có thể pretrain nhiều ticker

# Fixed RF params (same as scripts/quick_eval.py and the old RF_QUICK in streamlit).
RF_PARAMS = dict(
    # Lưu ý: đây là tham số deploy hiện tại.
    # Nó có thể khác với bộ tham số thắng cuộc trong notebook phân tích mới nhất.
    n_estimators=300,
    max_depth=3,
    min_samples_leaf=50,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

# Top 20 S&P 500 by market cap (~early 2026).
TOP20 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "V", "LLY", "UNH", "JNJ", "XOM",
    "WMT", "MA", "PG", "ORCL", "HD",
]


def train_one(ticker: str, start: str = START_DATE, horizon: int = HORIZON, macro=None) -> dict:
    """Fetch data, build features, fit the RF, save the artifact, return it.

    Pass `macro` (a pre-fetched fetch_macro() result) to avoid refetching macros
    when training many tickers in a row.
    """
    # stocks được giữ ở dạng dict để tái sử dụng interface chung của pipeline.
    stocks = {ticker: fetch_stock(ticker, start=start)}
    if macro is None:
        macro = fetch_macro(start=start)

    # Pipeline train 1 ticker:
    # 1. tạo feature + target
    # 2. split theo thời gian
    # 3. clip target train
    # 4. scale X_train
    # 5. train RF và lưu artifact
    feat_df = build_training_set(ticker, stocks, macro, horizon=horizon)
    X, y, feature_cols = split_xy(feat_df)
    X_tr, y_tr, X_va, y_va, X_te, y_te = time_split(X, y, horizon=horizon)

    # Chỉ fit clipping range trên train để tránh leakage.
    mean, std, lo, hi = fit_winsorize(y_tr)
    y_tr_clip = y_tr.clip(lo, hi)

    # Lưu scaler và thứ tự feature để inference về sau khớp hoàn toàn với lúc train.
    scaler = StandardScaler().fit(X_tr)
    model = RandomForestRegressor(**RF_PARAMS).fit(scaler.transform(X_tr), y_tr_clip)

    # Artifact chứa đủ thứ cần thiết để app load model mà không phải train lại.
    save_artifact(
        ticker=ticker,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        win_params=(mean, std, lo, hi),
        horizon=horizon,
    )
    return {
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

    # force=False: chỉ train ticker chưa có artifact
    # force=True : train lại toàn bộ
    targets = [t for t in tickers if force or not has_artifact(t)]
    if not targets:
        return []

    # Macro được tải một lần rồi tái sử dụng cho mọi ticker để tiết kiệm thời gian.
    macro = fetch_macro(start=START_DATE)

    trained = []
    for i, ticker in enumerate(targets, start=1):
        # Train tuần tự từng ticker rồi mới cập nhật tiến độ.
        train_one(ticker, macro=macro)
        trained.append(ticker)
        if progress_callback is not None:
            progress_callback(i, len(targets), ticker)
    return trained
