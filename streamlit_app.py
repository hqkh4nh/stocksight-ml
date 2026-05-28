"""Streamlit demo. Two sidebar views: AAPL with backtest, and a TOP20 explorer.

Models are pre-trained on first launch and cached under ./models/.
"""
import sys
import warnings
from pathlib import Path

from scipy.stats import ConstantInputWarning

# Ẩn các cảnh báo nhiễu từ scipy/numpy do baseline hằng số và các giá trị NaN đầu chuỗi.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r"invalid value encountered in log")
warnings.filterwarnings("ignore", category=ConstantInputWarning)

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
# Đảm bảo khi chạy bằng Streamlit vẫn import được package nội bộ trong thư mục src/.
sys.path.insert(0, str(ROOT))

from src.data     import fetch_stock, fetch_macro, build_dataset, split_xy, time_split
from src.features import build_training_set, build_features_only, fit_winsorize
from src.models   import train_all_models, evaluate
from src.forecast import (walk_forward, backtest_strategy, perf_metrics,
                          direct_h5, anchor_shift_forecast)
from src.model_io import (load_artifact, list_trained, manifest_status,
                          delete_all)
from src.pretrain import TOP20, HORIZON, START_DATE, train_missing

SEED = 42
# Dùng cùng seed với phần train để hành vi model/backtest nhất quán.

st.set_page_config(page_title="StockSight RF", layout="wide")


# Các hàm chuẩn bị dữ liệu

@st.cache_data(persist="disk", show_spinner=False)
def _fetch_stock(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    # Cache dữ liệu tải về để thao tác UI không gọi API lặp lại nhiều lần.
    return fetch_stock(ticker, start=start)


@st.cache_data(persist="disk", show_spinner=False)
def _fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    # Cache riêng cho macro vì nhiều ticker sẽ cùng dùng chung một bảng macro.
    return fetch_macro(start=start)


@st.cache_resource(show_spinner=False)
def _prep_eval_data(ticker: str) -> dict:
    """Rebuild train/val/test splits using the saved scaler and column order."""
    # Phải load artifact trước vì trong đó có thứ tự feature và scaler đã dùng
    # lúc train. Dự đoán về sau phải khớp chính xác với pipeline đã train.
    art = load_artifact(ticker)
    feat_cols = art["feature_cols"]
    scaler = art["scaler"]
    lo, hi = art["win_params"][2], art["win_params"][3]
    # lo/hi là ngưỡng winsorize đã fit từ train ở thời điểm huấn luyện.

    stocks = {ticker: _fetch_stock(ticker)}
    macro = _fetch_macro()
    feat_df = build_training_set(ticker, stocks, macro, horizon=HORIZON)
    # build_training_set ở đây được chạy lại từ dữ liệu mới tải về, còn model
    # vẫn là model đã được lưu sẵn trong artifact.

    X, y, _ = split_xy(feat_df)
    X = X[feat_cols]                              # ép đúng thứ tự cột như lúc train
    X_tr, y_tr, X_va, y_va, X_te, y_te = time_split(X, y, horizon=HORIZON)
    # Clip lại y theo đúng ngưỡng train cũ để phép so sánh trong app bám sát pipeline train.
    y_tr_c = y_tr.clip(lo, hi)
    y_va_c = y_va.clip(lo, hi)
    y_te_c = y_te.clip(lo, hi)

    # feat_full dùng cho phần dự báo mới nhất. Ta bỏ các dòng thiếu feature để
    # dòng cuối cùng luôn sẵn sàng đưa vào model.
    dataset = build_dataset(ticker, stocks, macro)
    feat_full = build_features_only(dataset).dropna(subset=feat_cols)

    return {
        "artifact": art,
        "dataset": dataset, "feat_full": feat_full,
        "X": X, "y": y, "feat_cols": feat_cols,
        # Các ma trận X_*_s là phiên bản đã scale, dùng cho baseline và đánh giá trực tiếp.
        "X_tr_s": scaler.transform(X_tr),
        "X_va_s": scaler.transform(X_va),
        "X_te_s": scaler.transform(X_te),
        "y_tr": y_tr_c, "y_va": y_va_c, "y_te": y_te_c,
    }


@st.cache_resource(show_spinner=False)
def _forecast_payload(ticker: str) -> dict:
    """Direct T+5 + 5-anchor vintage forecast off the saved model."""
    # Hàm này gom toàn bộ dữ liệu mà phần Forecast cần dùng:
    # - dự báo trực tiếp T+5
    # - đường forecast nhiều anchor
    # - giá hiện tại và ngày hiện tại
    d = _prep_eval_data(ticker)
    art = d["artifact"]
    rf = art["model"]
    scaler = art["scaler"]
    feat_cols = art["feature_cols"]

    # Lấy hàng feature mới nhất hiện có trong dữ liệu lịch sử.
    latest = d["feat_full"][feat_cols].iloc[-1:]
    Xl_s = scaler.transform(latest)
    cur_close = float(d["feat_full"]["Close"].loc[latest.index[0]])
    cur_date = latest.index[0]

    # Dự đoán của Random Forest là trung bình dự đoán của các cây. Ta lấy riêng
    # output từng cây để ước lượng một dải bất định đơn giản.
    tree_preds = np.array([t.predict(Xl_s)[0] for t in rf.estimators_])
    direct = direct_h5(rf, Xl_s, cur_close, tree_predictions=tree_preds)
    vintage = anchor_shift_forecast(rf, scaler, d["feat_full"], feat_cols,
                                    horizon=HORIZON)

    return {
        "ticker": ticker,
        "direct": direct, "vintage": vintage,
        "current_close": cur_close, "current_date": cur_date,
        "dataset": d["dataset"],
        # dataset được giữ lại để vẽ phần lịch sử giá ở chart.
    }


def _eval_one(y_true, y_pred) -> dict:
    """RMSE, MAE, R2, DirAcc, IC, plus a long-only Sharpe."""
    # Đây là hàm "nâng cấp" từ evaluate() trong src.models:
    # ngoài RMSE/DirAcc/IC còn tính thêm MAE, R2 và Sharpe đơn giản.
    y_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    p_arr = np.asarray(y_pred)

    base = evaluate(y_true, y_pred)                       # các metric lõi: RMSE / DirAcc / IC
    mae = float(np.mean(np.abs(y_arr - p_arr)))
    # ss_res: tổng bình phương sai số; ss_tot: tổng bình phương độ lệch khỏi mean.
    ss_res = float(np.sum((y_arr - p_arr) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    # Nếu y hầu như không biến động thì R2 không còn ý nghĩa -> trả NaN.
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Biến dự đoán thành chiến lược long-only đơn giản để xem dấu của mô hình có
    # mang giá trị giao dịch hay không, chứ không chỉ là sai số hồi quy thấp.
    pos = (p_arr > 0).astype(float)
    strat = pos * y_arr
    ann = np.sqrt(252 / HORIZON)
    sd = float(strat.std())
    sharpe = float(strat.mean() / sd * ann) if sd > 0 else float("nan")

    return {
        "RMSE":   base["RMSE"],
        "MAE":    mae,
        "R2":     r2,
        "DirAcc": base["DirAcc"],
        "IC":     base["IC"],
        "Sharpe": sharpe,
        # Sharpe ở đây là Sharpe của chiến lược minh họa, không phải backtest đầy đủ.
    }


@st.cache_resource(show_spinner=False)
def _eval_payload(ticker: str) -> dict:
    """Score baselines + saved RF on val and test."""
    # Hàm này chuẩn bị toàn bộ bảng so sánh mô hình cho 1 ticker.
    d = _prep_eval_data(ticker)
    rf = d["artifact"]["model"]

    # Fit lại các baseline trên train split để phép so sánh công bằng.
    baselines = train_all_models(d["X_tr_s"], d["y_tr"])
    models = {**baselines, "RandomForest": rf}

    rows_val, rows_test = [], []
    for name, m in models.items():
        # Chấm từng mô hình trên validation và test để app hiển thị song song.
        v = _eval_one(d["y_va"], m.predict(d["X_va_s"]))
        t = _eval_one(d["y_te"], m.predict(d["X_te_s"]))
        rows_val.append({"model": name, **v})
        rows_test.append({"model": name, **t})
    val_table  = pd.DataFrame(rows_val).set_index("model")
    test_table = pd.DataFrame(rows_test).set_index("model")
    # RF có sẵn feature_importances_, khá tiện để giải thích trong lớp dù đây
    # không phải phương pháp diễn giải hoàn hảo.
    fi = pd.Series(rf.feature_importances_, index=d["feat_cols"]) \
           .sort_values(ascending=False)
    return {"val_table": val_table, "test_table": test_table,
            "feat_importance": fi}


@st.cache_data(show_spinner=False)
def _backtest_payload(ticker: str) -> dict:
    """Walk-forward RF vs LR vs SPY buy-and-hold."""
    # Hàm này tốn thời gian hơn nên chỉ cache dữ liệu kết quả, không cache resource.
    d = _prep_eval_data(ticker)
    # Sao chép đúng hyperparameter của RF đã lưu để backtest dùng cùng một họ mô hình
    # như ở phần forecast và evaluation.
    rf_kw = {
        "n_estimators":     d["artifact"]["model"].n_estimators,
        "max_depth":        d["artifact"]["model"].max_depth,
        "min_samples_leaf": d["artifact"]["model"].min_samples_leaf,
        "max_features":     d["artifact"]["model"].max_features,
        "random_state":     SEED,
        "n_jobs":           -1,
    }
    wf_rf = walk_forward(d["X"], d["y"], initial_train=1000, refit_every=21,
                         horizon=HORIZON, rf_kwargs=rf_kw)
    # LR được dùng như baseline có cấu trúc đơn giản hơn RF.
    wf_lr = walk_forward(d["X"], d["y"], initial_train=1000, refit_every=21,
                         horizon=HORIZON, model_factory=lambda: LinearRegression())
    bt_rf = backtest_strategy(wf_rf, horizon=HORIZON, cost_bps=5.0)
    bt_lr = backtest_strategy(wf_lr, horizon=HORIZON, cost_bps=5.0)

    # SPY được dùng như benchmark thụ động để có ngữ cảnh so sánh.
    spy = _fetch_stock("SPY")
    spy_fwd = np.log(spy["Close"].shift(-HORIZON) / spy["Close"])
    spy_ret = spy_fwd.reindex(bt_rf["bh_ret"].index).dropna()
    bt_spy = perf_metrics(spy_ret, horizon=HORIZON)
    return {"bt_rf": bt_rf, "bt_lr": bt_lr, "bt_spy": bt_spy}


# Khởi tạo model nếu máy chưa có artifact

def _ensure_models_trained():
    status = manifest_status(TOP20)
    missing = [t for t, ts in status.items() if ts is None]
    if not missing:
        # Nếu tất cả model đã có sẵn trên đĩa thì không cần làm gì.
        return

    # Lần chạy đầu có thể chưa có artifact lưu sẵn. Khi đó app sẽ tự train danh
    # sách ticker được chọn trước, sau đó chạy lại giao diện.
    st.warning(
        f"First run: training {len(missing)} model(s). Takes about 5-10 minutes "
        f"and is cached after."
    )
    progress = st.progress(0.0, text="Starting...")
    log = st.empty()

    def cb(done, total, ticker):
        # Callback này giúp Streamlit cập nhật thanh tiến độ trực tiếp.
        progress.progress(done / total, text=f"Trained {ticker} ({done}/{total})")
        log.write(f"OK {ticker}")

    train_missing(progress_callback=cb)
    progress.empty()
    log.empty()
    # Sau khi train xong, rerun để toàn bộ cache/UI đọc trạng thái mới nhất.
    st.success(f"Done. Trained {len(missing)} model(s).")
    st.rerun()


# Các hàm render giao diện

def _render_forecast_section(ticker: str):
    # Hàm render phần trực quan nhất của app: người dùng nhìn thấy giá hiện tại,
    # giá dự báo T+5 và khoảng bất định.
    fc = _forecast_payload(ticker)
    d = fc["direct"]
    cur = fc["current_close"]
    delta_pct = (d["pred_price"] / cur - 1) * 100
    band_pct = (d["band_high_price"] - d["band_low_price"]) / cur * 100
    # delta_pct giúp người dùng hiểu forecast theo % thay vì chỉ nhìn giá tuyệt đối.

    c1, c2, c3 = st.columns(3)
    c1.metric("Current close", f"${cur:.2f}",
              help=f"as of {fc['current_date'].date()}")
    c2.metric(f"Predicted T+{HORIZON}", f"${d['pred_price']:.2f}",
              delta=f"{delta_pct:+.2f}%")
    c3.metric("90% band width", f"{band_pct:.2f}%",
              help=f"[{d['band_low_price']:.2f}, {d['band_high_price']:.2f}]")

    # Ghép các lớp biểu đồ:
    # - đường giá lịch sử gần đây
    # - đường forecast từ nhiều anchor
    # - điểm dự báo trực tiếp T+5
    # - dải bất định quanh điểm T+5
    hist = fc["dataset"]["Close"].tail(60).rename("close").to_frame()
    hist["date"] = hist.index
    hist["kind"] = "History"

    vint = fc["vintage"]
    vint_df = pd.DataFrame({
        "date":  [fc["current_date"]] + list(vint["target_date"]),
        "close": [cur] + list(vint["pred_price"]),
        "kind":  ["Forecast"] * (len(vint) + 1),
    })
    direct_df = pd.DataFrame({
        "date":  [vint["target_date"].iloc[-1]],
        "close": [d["pred_price"]],
        "kind":  ["Direct T+5"],
    })
    band_df = pd.DataFrame({
        "date": [vint["target_date"].iloc[-1]],
        "low":  [d["band_low_price"]],
        "high": [d["band_high_price"]],
    })

    base_x = alt.X("date:T", title="Date")
    # Dùng chung trục x cho toàn bộ layer để các thành phần chồng khít nhau.

    hist_line = alt.Chart(hist).mark_line(color="#1f77b4", strokeWidth=2).encode(
        x=base_x, y=alt.Y("close:Q", title="Price ($)",
                          scale=alt.Scale(zero=False)))
    vint_line = alt.Chart(vint_df).mark_line(
        color="#d62728", strokeWidth=2, point=alt.OverlayMarkDef(color="#d62728")
    ).encode(x=base_x, y="close:Q")
    band = alt.Chart(band_df).mark_errorbar(color="#ff7f0e", thickness=2).encode(
        x=base_x, y="low:Q", y2="high:Q")
    star = alt.Chart(direct_df).mark_point(
        shape="diamond", size=350, color="#ff7f0e", filled=True
    ).encode(x=base_x, y="close:Q")

    chart = (hist_line + vint_line + band + star).properties(
        height=380, title=f"{ticker}: last 60 days + {HORIZON}-day forecast"
    )
    # width="stretch" giúp biểu đồ co giãn theo chiều ngang của layout.
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Anchor {fc['current_date'].date()}, horizon {HORIZON} business days, "
        f"orange band is +/- 1.645 sigma over RF tree predictions (90%)."
    )


def _render_performance_section(ticker: str):
    # Phần này trả lời câu hỏi: model RF có thực sự tốt hơn baseline không?
    payload = _eval_payload(ticker)

    fmt = {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:+.4f}",
           "DirAcc": "{:.4f}", "IC": "{:+.4f}", "Sharpe": "{:+.2f}"}

    st.subheader(f"Model comparison: {ticker}")
    st.markdown("**Test set (most recent ~15%).** Out-of-sample, what matters most.")
    st.dataframe(
        payload["test_table"].style.format(fmt).background_gradient(
            subset=["IC", "Sharpe", "DirAcc"], cmap="RdYlGn", axis=0
        ).background_gradient(
            subset=["RMSE", "MAE"], cmap="RdYlGn_r", axis=0
        ),
        width="stretch",
    )
    with st.expander("Validation set"):
        # Validation được ẩn trong expander vì test set quan trọng hơn cho người xem cuối.
        st.dataframe(payload["val_table"].style.format(fmt), width="stretch")

    st.subheader("Top 15 features (RF importance)")
    top = payload["feat_importance"].head(15)
    # Chỉ lấy top 15 để biểu đồ gọn và dễ đọc hơn với người xem.
    st.bar_chart(top, horizontal=True, height=380)


def _render_backtest_section(ticker: str):
    # Đây là phần nối dự đoán ML với góc nhìn chiến lược giao dịch.
    st.caption(
        f"Walk-forward, expanding train, refit every 21 days. Non-overlapping "
        f"{HORIZON}-day bets, long-only, 5 bps/leg cost. First load ~30s."
    )
    with st.spinner("Running walk-forward backtest..."):
        # Backtest nặng hơn các phần khác nên hiển thị spinner để người dùng biết app vẫn đang chạy.
        bt = _backtest_payload(ticker)
    rf, lr, spy = bt["bt_rf"], bt["bt_lr"], bt["bt_spy"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total return", f"{rf['total_return'] * 100:.1f}%")
    c2.metric("CAGR",         f"{rf['cagr'] * 100:.2f}%")
    c3.metric("Sharpe",       f"{rf['sharpe']:.2f}")
    c4.metric("Max DD",       f"{rf['max_dd'] * 100:.1f}%")
    c5.metric("Trades",       f"{rf['num_trades']}")
    c6.metric("Win rate",     f"{rf['win_rate'] * 100:.1f}%")

    eq = pd.DataFrame({"RF": rf["equity"], "LR": lr["equity"], "SPY": spy["equity"]})
    # So sánh 3 đường vốn trên cùng một chart để thấy bức tranh tổng thể.
    st.line_chart(eq, height=380)


# Sidebar điều khiển

def _render_sidebar() -> str:
    # Sidebar đóng vai trò điều hướng chính của app.
    st.sidebar.title("StockSight RF")
    st.sidebar.caption(
        f"5-day forward log-return forecast (Random Forest, technical + macro). "
        f"Training data starts {START_DATE}."
    )

    tab = st.sidebar.radio(
        "View",
        ["AAPL Demo", "Stock Explorer"],
        index=0,
        key="sidebar_tab",
    )
    # AAPL Demo: trang cố định đầy đủ forecast + performance + backtest
    # Stock Explorer: trang linh hoạt để chọn ticker trong TOP20

    st.sidebar.markdown("---")
    trained = list_trained()
    st.sidebar.metric("Models ready", f"{len(trained & set(TOP20))} / {len(TOP20)}")

    if st.sidebar.button("Retrain all", help="Delete every saved model and retrain."):
        # Xóa artifact đã lưu và cache của Streamlit để lần chạy sau build lại từ đầu.
        delete_all()
        _prep_eval_data.clear()
        _forecast_payload.clear()
        _eval_payload.clear()
        _backtest_payload.clear()
        st.rerun()

    st.sidebar.caption(f"SEED={SEED} | horizon={HORIZON}")
    return tab


# Luồng chính của app

tab = _render_sidebar()
_ensure_models_trained()

# Router đơn giản: hoặc hiển thị trang demo AAPL cố định, hoặc trang chọn ticker.
if tab == "AAPL Demo":
    st.title(f"AAPL: {HORIZON}-day forecast, performance, backtest")
    st.header("Forecast")
    _render_forecast_section("AAPL")
    st.markdown("---")
    st.header("Performance")
    _render_performance_section("AAPL")
    st.markdown("---")
    st.header("Backtest")
    _render_backtest_section("AAPL")

else:
    st.title("Stock Explorer")
    col1, _ = st.columns([1, 3])
    # Explorer cho phép chọn một ticker bất kỳ trong TOP20 để xem forecast/performance.
    ticker = col1.selectbox("Pick a ticker", TOP20, index=0, key="explorer_ticker")

    st.header(f"Forecast: {ticker}")
    _render_forecast_section(ticker)
    st.markdown("---")
    st.header(f"Performance: {ticker}")
    _render_performance_section(ticker)
