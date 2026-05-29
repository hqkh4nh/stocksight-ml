"""Train and save RF artifacts for the curated TOP20 ticker list."""
# Module này pre-train Random Forest cho 20 ticker S&P500 lớn nhất rồi save xuống
# thư mục models/ để streamlit có thể load nhanh, không phải tune lại mỗi lần user
# mở app.
from typing import Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data import (fetch_stock, fetch_macro, split_xy, time_split)
from src.features import build_training_set, fit_winsorize
from src.model_io import has_artifact, save_artifact

SEED = 42
HORIZON = 5
# Dữ liệu lấy từ 2012 — đúng range với notebook để kết quả backtest ở 2 nơi (notebook
# và streamlit) match nhau.
START_DATE = "2012-01-01"

# Mirrors best_params from notebook cell 38 tune_random_forest grid search,
# so streamlit's pretrained models reproduce the notebook's walk-forward results.
# Best params chốt từ bước tune ở notebook cell 38:
#  - n_estimators=500: số cây sau khi tune.
#  - max_depth=3: cây nông để regularize, tránh overfit.
#  - min_samples_leaf=50: lá lớn → giảm nhiễu, ổn định prediction.
#  - max_features='sqrt': mỗi split chọn ~sqrt(n_features) ≈ 5 feature → tăng đa dạng cây.
RF_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=50,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

# Top 20 S&P 500 by market cap (~early 2026).
# Danh sách 20 mã vốn hoá lớn nhất S&P500 đầu 2026 — gom đủ các nhóm ngành chính:
#  - Tech: AAPL, MSFT, NVDA, GOOGL, AMZN, META, AVGO, ORCL
#  - Finance: JPM, V, MA, BRK-B
#  - Healthcare: LLY, UNH, JNJ
#  - Energy: XOM
#  - Retail/Consumer: WMT, HD, PG
#  - EV: TSLA
TOP20 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "V",    "LLY",  "UNH",   "JNJ",   "XOM",
    "WMT",  "MA",   "PG",    "ORCL",  "HD",
]


def train_one(ticker: str, start: str = START_DATE, horizon: int = HORIZON,
              macro=None) -> dict:
    """Fetch data, build features, fit the RF, save the artifact, return it.

    Pass `macro` (a pre-fetched fetch_macro() result) to avoid refetching macros
    when training many tickers in a row.
    """
    # Fetch giá cổ phiếu. Macro có thể được truyền sẵn từ ngoài để tránh refetch
    # khi train nhiều ticker liên tiếp — tiết kiệm thời gian + tránh bị rate-limit
    # của yfinance.
    stocks = {ticker: fetch_stock(ticker, start=start)}
    if macro is None:
        macro = fetch_macro(start=start)

    # build_training_set: tạo feature engineering + target (forward return) + dropna.
    feat_df = build_training_set(ticker, stocks, macro, horizon=horizon)
    # split_xy tách X / y / tên cột feature; time_split chia train/val/test theo thứ tự
    # thời gian kèm embargo để tránh leakage giữa các tập.
    X, y, feature_cols = split_xy(feat_df)
    X_tr, y_tr, X_va, y_va, X_te, y_te = time_split(X, y, horizon=horizon)

    # fit_winsorize chỉ tính bound clip TRÊN y_train (mean/std/lo/hi), tránh leak
    # phân phối của val/test. Sau đó áp bound lên y_train để clip outlier.
    mean, std, lo, hi = fit_winsorize(y_tr)
    y_tr_clip = y_tr.clip(lo, hi)

    # StandardScaler fit trên X_train rồi transform — scaler được save lại để
    # inference ở streamlit dùng đúng mean/std đã học.
    scaler = StandardScaler().fit(X_tr)
    # Fit RF trên train đã scale + y đã clip winsorize.
    model = RandomForestRegressor(**RF_PARAMS).fit(scaler.transform(X_tr), y_tr_clip)

    # save_artifact: pickle model + scaler + danh sách feature_cols + winsorize
    # bounds + horizon. Streamlit load lại sẽ tái dùng đúng scaler/bounds để
    # inference consistent với lúc train.
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
    # Mặc định lấy TOP20. force=True → train lại tất cả; ngược lại chỉ train
    # những ticker chưa có file .pkl trên disk.
    tickers = list(tickers) if tickers is not None else TOP20
    targets = [t for t in tickers if force or not has_artifact(t)]
    if not targets:
        return []

    # fetch macros once and reuse for every ticker
    # Macro (lãi suất, VIX, ...) chung cho mọi cổ phiếu → fetch 1 lần rồi truyền
    # vào train_one để tiết kiệm thời gian + tránh gọi API trùng lặp.
    macro = fetch_macro(start=START_DATE)

    trained = []
    for i, ticker in enumerate(targets, start=1):
        train_one(ticker, macro=macro)
        trained.append(ticker)
        # progress_callback: streamlit dùng để cập nhật progress bar lúc first-run
        # (khi user mới clone repo và chưa có model nào).
        if progress_callback is not None:
            progress_callback(i, len(targets), ticker)
    return trained
