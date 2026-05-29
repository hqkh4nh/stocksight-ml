"""Streamlit demo. Two sidebar views: AAPL with backtest, and a TOP20 explorer.

Models are pre-trained on first launch and cached under ./models/.
"""
# =============================================================================
# ENTRY POINT CỦA STREAMLIT APP - StockSight RF
# =============================================================================
# File này là entry point chính. Chạy: `streamlit run streamlit_app.py`.
#
# App có 2 view chọn từ sidebar:
#   1. AAPL Demo      : Forecast + Performance + Backtest cho AAPL (đầy đủ).
#   2. Stock Explorer : Chọn 1 ticker trong TOP20, xem forecast + performance.
#
# Tất cả model đều đã pre-train trên 20 ticker và lưu artifact dưới ./models/.
# Lần đầu chạy app: nếu chưa có artifact thì tự động train (~5-10 phút).
# Lần sau: load artifact từ disk -> render UI nhanh (<5s).
# =============================================================================
import sys
import warnings
from pathlib import Path

from scipy.stats import ConstantInputWarning

# scipy/numpy noise from constant baselines and leading-NaN macro shifts
# Nén log cảnh báo cho UI sạch:
#   - "invalid value encountered in log": xảy ra khi macro shift đầu chuỗi
#     có NaN và numpy thử log(NaN). Không ảnh hưởng kết quả.
#   - ConstantInputWarning: khi 1 cột feature có std=0 trong rolling window
#     (vd: volume không đổi trong vài phiên), scipy.pearsonr warn. Vô hại.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r"invalid value encountered in log")
warnings.filterwarnings("ignore", category=ConstantInputWarning)

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
# Thêm ROOT vào sys.path để `from src....` chạy được khi user launch streamlit
# từ thư mục bất kỳ (vd: `cd /tmp && streamlit run /path/to/streamlit_app.py`).
sys.path.insert(0, str(ROOT))

from src.data     import fetch_stock, fetch_macro, build_dataset, split_xy, time_split
from src.features import build_training_set, build_features_only, fit_winsorize
from src.models   import train_all_models, evaluate
from src.forecast import (walk_forward, backtest_strategy, perf_metrics,
                          direct_h5, anchor_shift_forecast)
from src.model_io import (load_artifact, list_trained, manifest_status,
                          delete_all)
from src.pretrain import TOP20, HORIZON, START_DATE, train_missing

# SEED chung cho reproducibility (RF train, walk-forward refit, ...).
SEED = 42

st.set_page_config(page_title="StockSight RF", layout="wide")


# data

@st.cache_data(persist="disk", show_spinner=False)
def _fetch_stock(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    # Cache OHLCV ra DISK (~/.streamlit/cache/). Lần sau gọi cùng (ticker, start)
    # = HIT cache, không call yfinance nữa -> nhanh + tránh rate limit.
    # WARNING: cache key chỉ hash explicit args. Nếu pretrain.py đổi START_DATE
    # mà không xoá ~/.streamlit/cache/ thì có thể HIT cache cũ với data khác.
    return fetch_stock(ticker, start=start)


@st.cache_data(persist="disk", show_spinner=False)
def _fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    # Cùng cơ chế cache disk cho macro (VIX, DXY, US10Y, US2Y, ...).
    return fetch_macro(start=start)


@st.cache_resource(show_spinner=False)
def _prep_eval_data(ticker: str) -> dict:
    """Rebuild train/val/test splits using the saved scaler and column order."""
    # @st.cache_resource: cache IN-MEMORY, không persist disk.
    # Mỗi session streamlit chỉ chạy 1 lần cho mỗi ticker. Reload trang = cache mới.
    art = load_artifact(ticker)
    feat_cols = art["feature_cols"]
    scaler = art["scaler"]
    # win_params = (col_indices, q_low, q_high_bound, lo_bound, hi_bound).
    # Lấy lo/hi bound (giá trị clip) đã fit lúc train -> dùng lại cho consistency.
    lo, hi = art["win_params"][2], art["win_params"][3]

    # Re-fetch OHLCV + macro (cache disk = nhanh) và build features y hệt lúc train.
    stocks = {ticker: _fetch_stock(ticker)}
    macro = _fetch_macro()
    feat_df = build_training_set(ticker, stocks, macro, horizon=HORIZON)

    X, y, _ = split_xy(feat_df)
    # ÉP đúng thứ tự cột của lúc train. RF không quan tâm thứ tự nhưng
    # StandardScaler thì có -> sai thứ tự = scale sai cột.
    X = X[feat_cols]                              # same column order as training
    X_tr, y_tr, X_va, y_va, X_te, y_te = time_split(X, y, horizon=HORIZON)
    # Clip y bằng bound đã fit lúc train (giống pipeline pretrain) -> số liệu
    # eval khớp với metric trong artifact.
    y_tr_c = y_tr.clip(lo, hi)
    y_va_c = y_va.clip(lo, hi)
    y_te_c = y_te.clip(lo, hi)

    # dataset = raw OHLCV+macro đã merge (chưa drop NaN của target),
    # feat_full = bảng features không drop target row -> dùng cho forecast
    # section (cần row mới nhất kể cả khi chưa biết y T+5).
    dataset = build_dataset(ticker, stocks, macro)
    feat_full = build_features_only(dataset).dropna(subset=feat_cols)

    return {
        "artifact": art,
        "dataset": dataset, "feat_full": feat_full,
        "X": X, "y": y, "feat_cols": feat_cols,
        # Scale sẵn 3 split để các renderer không phải scale lại.
        "X_tr_s": scaler.transform(X_tr),
        "X_va_s": scaler.transform(X_va),
        "X_te_s": scaler.transform(X_te),
        "y_tr": y_tr_c, "y_va": y_va_c, "y_te": y_te_c,
    }


@st.cache_resource(show_spinner=False)
def _forecast_payload(ticker: str) -> dict:
    """Direct T+5 + 5-anchor vintage forecast off the saved model."""
    # Cache in-memory: forecast cùng ticker trong session chỉ tính 1 lần.
    d = _prep_eval_data(ticker)
    art = d["artifact"]
    rf = art["model"]
    scaler = art["scaler"]
    feat_cols = art["feature_cols"]

    # latest = row features cuối cùng (anchor T). Scale rồi predict.
    latest = d["feat_full"][feat_cols].iloc[-1:]
    Xl_s = scaler.transform(latest)
    # cur_close, cur_date = giá đóng cửa và ngày anchor T.
    cur_close = float(d["feat_full"]["Close"].loc[latest.index[0]])
    cur_date = latest.index[0]

    # tree_preds: predict riêng của từng cây trong RF.
    # Dùng để tính confidence band 90% (mean +/- 1.645 * std across trees).
    tree_preds = np.array([t.predict(Xl_s)[0] for t in rf.estimators_])
    # Direct = single-shot forecast tại T+5 (giá + band low/high).
    direct = direct_h5(rf, Xl_s, cur_close, tree_predictions=tree_preds)
    # Vintage = 5 anchor predictions tại T+1, T+2, ..., T+5 (vintage forecast).
    vintage = anchor_shift_forecast(rf, scaler, d["feat_full"], feat_cols,
                                    horizon=HORIZON)

    return {
        "ticker": ticker,
        "direct": direct, "vintage": vintage,
        "current_close": cur_close, "current_date": cur_date,
        "dataset": d["dataset"],
    }


def _eval_one(y_true, y_pred) -> dict:
    """RMSE, MAE, R2, DirAcc, IC, plus a long-only Sharpe."""
    # Tính 6 metric chuẩn cho 1 cặp (y_true, y_pred).
    y_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    p_arr = np.asarray(y_pred)

    # evaluate() trả RMSE, DirAcc, IC. MAE và R2 tính tay ở đây.
    base = evaluate(y_true, y_pred)                       # RMSE / DirAcc / IC
    mae = float(np.mean(np.abs(y_arr - p_arr)))
    ss_res = float(np.sum((y_arr - p_arr) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    # R2 có thể âm khi model tệ hơn baseline mean.
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # toy strategy: long whenever pred > 0
    # Toy strategy đơn giản: long khi pred>0, flat khi pred<=0 (không short).
    # Sharpe annualize bằng sqrt(252/HORIZON) vì return là 5-day forward.
    pos = (p_arr > 0).astype(float)
    strat = pos * y_arr
    ann = np.sqrt(252 / HORIZON)
    sd = float(strat.std())
    # Check sd>0 để tránh chia 0 (xảy ra với baseline NaiveZero -> luôn flat).
    sharpe = float(strat.mean() / sd * ann) if sd > 0 else float("nan")

    return {
        "RMSE":   base["RMSE"],
        "MAE":    mae,
        "R2":     r2,
        "DirAcc": base["DirAcc"],
        "IC":     base["IC"],
        "Sharpe": sharpe,
    }


@st.cache_resource(show_spinner=False)
def _eval_payload(ticker: str) -> dict:
    """Score baselines + saved RF on val and test."""
    # Train baselines (NaiveZero, AlwaysLong, LinearRegression) trên train,
    # eval mọi model (baselines + RF đã load) trên val và test.
    d = _prep_eval_data(ticker)
    rf = d["artifact"]["model"]

    baselines = train_all_models(d["X_tr_s"], d["y_tr"])
    models = {**baselines, "RandomForest": rf}

    # Loop qua từng model, tính metric cho val và test riêng.
    rows_val, rows_test = [], []
    for name, m in models.items():
        v = _eval_one(d["y_va"], m.predict(d["X_va_s"]))
        t = _eval_one(d["y_te"], m.predict(d["X_te_s"]))
        rows_val.append({"model": name, **v})
        rows_test.append({"model": name, **t})
    val_table  = pd.DataFrame(rows_val).set_index("model")
    test_table = pd.DataFrame(rows_test).set_index("model")
    # Feature importance của RF (sorted desc) -> dùng để vẽ bar chart top 15.
    fi = pd.Series(rf.feature_importances_, index=d["feat_cols"]) \
           .sort_values(ascending=False)
    return {"val_table": val_table, "test_table": test_table,
            "feat_importance": fi}


@st.cache_data(show_spinner=False)
def _backtest_payload(ticker: str) -> dict:
    """Walk-forward RF vs LR vs SPY buy-and-hold."""
    d = _prep_eval_data(ticker)
    # Pull RF hyperparams từ artifact -> đảm bảo backtest dùng CÙNG config
    # với model pretrained. random_state=SEED cho deterministic giữa các run.
    rf_kw = {
        "n_estimators":     d["artifact"]["model"].n_estimators,
        "max_depth":        d["artifact"]["model"].max_depth,
        "min_samples_leaf": d["artifact"]["model"].min_samples_leaf,
        "max_features":     d["artifact"]["model"].max_features,
        "random_state":     SEED,
        "n_jobs":           -1,
    }
    # Walk-forward expanding train, refit mỗi 21 phiên (~ 1 tháng).
    # initial_train=1000 phiên (~ 4 năm) cho seed đủ data.
    wf_rf = walk_forward(d["X"], d["y"], initial_train=1000, refit_every=21,
                         horizon=HORIZON, rf_kwargs=rf_kw)
    # Baseline LR cùng setup walk-forward -> so sánh fair.
    wf_lr = walk_forward(d["X"], d["y"], initial_train=1000, refit_every=21,
                         horizon=HORIZON, model_factory=lambda: LinearRegression())
    # backtest_strategy: long-only khi pred>0, non-overlapping 5-day bets,
    # cost 5bps mỗi leg (= 10bps round-trip).
    bt_rf = backtest_strategy(wf_rf, horizon=HORIZON, cost_bps=5.0)
    bt_lr = backtest_strategy(wf_lr, horizon=HORIZON, cost_bps=5.0)

    # SPY benchmark: passive buy-and-hold cùng tần suất 5-day forward return.
    spy = _fetch_stock("SPY")
    spy_fwd = np.log(spy["Close"].shift(-HORIZON) / spy["Close"])
    spy_ret = spy_fwd.reindex(bt_rf["bh_ret"].index).dropna()
    bt_spy = perf_metrics(spy_ret, horizon=HORIZON)
    return {"bt_rf": bt_rf, "bt_lr": bt_lr, "bt_spy": bt_spy}


# bootstrap

def _ensure_models_trained():
    # Lần đầu chạy: check manifest, tìm ticker nào CHƯA có artifact -> train.
    status = manifest_status(TOP20)
    missing = [t for t, ts in status.items() if ts is None]
    if not missing:
        return

    # Hiện UI progress để user biết app đang train, không phải đứng hình.
    st.warning(
        f"First run: training {len(missing)} model(s). Takes about 5-10 minutes "
        f"and is cached after."
    )
    progress = st.progress(0.0, text="Starting...")
    log = st.empty()

    # Callback được pretrain.train_missing() gọi sau khi train xong mỗi ticker.
    def cb(done, total, ticker):
        progress.progress(done / total, text=f"Trained {ticker} ({done}/{total})")
        log.write(f"OK {ticker}")

    train_missing(progress_callback=cb)
    progress.empty()
    log.empty()
    st.success(f"Done. Trained {len(missing)} model(s).")
    # Rerun để refresh layout với model vừa train xong.
    st.rerun()


# renderers

def _render_forecast_section(ticker: str):
    # 3 metric card: giá hiện tại, giá dự đoán T+5, độ rộng band 90%.
    fc = _forecast_payload(ticker)
    d = fc["direct"]
    cur = fc["current_close"]
    delta_pct = (d["pred_price"] / cur - 1) * 100
    band_pct = (d["band_high_price"] - d["band_low_price"]) / cur * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Current close", f"${cur:.2f}",
              help=f"as of {fc['current_date'].date()}")
    c2.metric(f"Predicted T+{HORIZON}", f"${d['pred_price']:.2f}",
              delta=f"{delta_pct:+.2f}%")
    c3.metric("90% band width", f"{band_pct:.2f}%",
              help=f"[{d['band_low_price']:.2f}, {d['band_high_price']:.2f}]")

    # 60d history + vintage line + diamond at T+5 + 90% band
    # Truncate hist at cur_date so its endpoint coincides with vint's first
    # point — otherwise a 1-bar gap appears whenever dropna(feat_cols) drops
    # the latest dataset row (e.g. stale yfinance macros).
    # Chuẩn bị 4 dataframe cho chart layered:
    # 1) hist : 60 phiên cuối lịch sử (line xanh).
    #    CẮT TẠI cur_date để endpoint khớp với điểm đầu của vint_line.
    #    Nếu không cắt, dataset có row mới hơn feat_full (do dropna(feat_cols)
    #    drop row cuối khi feature có NaN) -> xuất hiện gap 1 bar tại "hôm nay".
    hist = (fc["dataset"]["Close"]
            .loc[:fc["current_date"]]
            .tail(60)
            .rename("close")
            .to_frame())
    hist["date"] = hist.index
    hist["kind"] = "History"

    # 2) vint_df : anchor + 5 forecast points T+1..T+5 (line đỏ + marker).
    vint = fc["vintage"]
    vint_df = pd.DataFrame({
        "date":  [fc["current_date"]] + list(vint["target_date"]),
        "close": [cur] + list(vint["pred_price"]),
        "kind":  ["Forecast"] * (len(vint) + 1),
    })
    # 3) direct_df : 1 điểm tại T+5 (diamond cam).
    direct_df = pd.DataFrame({
        "date":  [vint["target_date"].iloc[-1]],
        "close": [d["pred_price"]],
        "kind":  ["Direct T+5"],
    })
    # 4) band_df : errorbar tại T+5 (band 90% mean +/- 1.645 sigma).
    band_df = pd.DataFrame({
        "date": [vint["target_date"].iloc[-1]],
        "low":  [d["band_low_price"]],
        "high": [d["band_high_price"]],
    })

    base_x = alt.X("date:T", title="Date")

    # Layered Altair chart: hist (xanh) + vint (đỏ) + band (cam) + diamond (cam).
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

    # Cộng (+) các layer lại = chart Altair layered duy nhất.
    chart = (hist_line + vint_line + band + star).properties(
        height=380, title=f"{ticker}: last 60 days + {HORIZON}-day forecast"
    )
    st.altair_chart(chart, width="stretch")
    # Caption ghi rõ anchor date, horizon, ý nghĩa band.
    st.caption(
        f"Anchor {fc['current_date'].date()}, horizon {HORIZON} business days, "
        f"orange band is +/- 1.645 sigma over RF tree predictions (90%)."
    )


def _render_performance_section(ticker: str):
    payload = _eval_payload(ticker)

    # Format cho từng cột metric (RMSE/MAE 4 chữ số thập phân, Sharpe 2 chữ số, ...).
    fmt = {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:+.4f}",
           "DirAcc": "{:.4f}", "IC": "{:+.4f}", "Sharpe": "{:+.2f}"}

    st.subheader(f"Model comparison: {ticker}")
    st.markdown("**Test set (most recent ~15%).** Out-of-sample, what matters most.")
    # Test table với background gradient:
    #   - IC, Sharpe, DirAcc dùng RdYlGn (xanh = lớn = tốt).
    #   - RMSE, MAE dùng RdYlGn_r (đảo) vì càng nhỏ càng tốt -> nhỏ = xanh.
    st.dataframe(
        payload["test_table"].style.format(fmt).background_gradient(
            subset=["IC", "Sharpe", "DirAcc"], cmap="RdYlGn", axis=0
        ).background_gradient(
            subset=["RMSE", "MAE"], cmap="RdYlGn_r", axis=0
        ),
        width="stretch",
    )
    # Validation table để trong expander (chi tiết bổ sung, ít quan trọng hơn test).
    with st.expander("Validation set"):
        st.dataframe(payload["val_table"].style.format(fmt), width="stretch")

    # Top 15 feature quan trọng nhất theo RF feature_importances_.
    st.subheader("Top 15 features (RF importance)")
    top = payload["feat_importance"].head(15)
    st.bar_chart(top, horizontal=True, height=380)


def _render_backtest_section(ticker: str):
    st.caption(
        f"Walk-forward, expanding train, refit every 21 days. Non-overlapping "
        f"{HORIZON}-day bets, long-only, 5 bps/leg cost. First load ~30s."
    )
    # Spinner ~30s lần đầu (walk-forward chạy hàng nghìn refit). Cached sau đó.
    with st.spinner("Running walk-forward backtest..."):
        bt = _backtest_payload(ticker)
    rf, lr, spy = bt["bt_rf"], bt["bt_lr"], bt["bt_spy"]

    # 6 metric card cho RF strategy (chiến lược chính được highlight).
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total return", f"{rf['total_return'] * 100:.1f}%")
    c2.metric("CAGR",         f"{rf['cagr'] * 100:.2f}%")
    c3.metric("Sharpe",       f"{rf['sharpe']:.2f}")
    c4.metric("Max DD",       f"{rf['max_dd'] * 100:.1f}%")
    c5.metric("Trades",       f"{rf['num_trades']}")
    c6.metric("Win rate",     f"{rf['win_rate'] * 100:.1f}%")

    # Line chart equity curve 3 đường: RF (chiến lược chính), LR (baseline),
    # SPY (passive benchmark) -> so sánh trực quan.
    eq = pd.DataFrame({"RF": rf["equity"], "LR": lr["equity"], "SPY": spy["equity"]})
    st.line_chart(eq, height=380)


# sidebar

def _render_sidebar() -> str:
    st.sidebar.title("StockSight RF")
    # Caption giải thích horizon (5 ngày) + ngày bắt đầu training data.
    st.sidebar.caption(
        f"5-day forward log-return forecast (Random Forest, technical + macro). "
        f"Training data starts {START_DATE}."
    )

    # Radio chọn view chính.
    tab = st.sidebar.radio(
        "View",
        ["AAPL Demo", "Stock Explorer"],
        index=0,
        key="sidebar_tab",
    )

    st.sidebar.markdown("---")
    # Hiển thị số model đã train sẵn / tổng số (vd: "20 / 20").
    trained = list_trained()
    st.sidebar.metric("Models ready", f"{len(trained & set(TOP20))} / {len(TOP20)}")

    # Nút Retrain all: xoá MỌI artifact + clear 4 cache in-memory + rerun.
    # CẢNH BÁO: KHÔNG clear disk cache `~/.streamlit/cache/` (yfinance fetch).
    # Nếu đổi START_DATE trong pretrain.py thì cần xoá tay folder đó.
    if st.sidebar.button("Retrain all", help="Delete every saved model and retrain."):
        delete_all()
        _prep_eval_data.clear()
        _forecast_payload.clear()
        _eval_payload.clear()
        _backtest_payload.clear()
        st.rerun()

    st.sidebar.caption(f"SEED={SEED} | horizon={HORIZON}")
    return tab


# main

# Flow chính (chạy mỗi lần streamlit rerun):
#   1. Render sidebar -> trả tab name user chọn.
#   2. Ensure tất cả 20 model đã train (nếu thiếu thì bootstrap train).
#   3. Render section tương ứng với tab.
tab = _render_sidebar()
_ensure_models_trained()

if tab == "AAPL Demo":
    # View AAPL: đầy đủ 3 section forecast + performance + backtest.
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
    # View Stock Explorer: chọn 1 ticker trong TOP20, chỉ forecast + performance
    # (bỏ backtest vì chậm và đã có ở AAPL Demo).
    st.title("Stock Explorer")
    col1, _ = st.columns([1, 3])
    ticker = col1.selectbox("Pick a ticker", TOP20, index=0, key="explorer_ticker")

    st.header(f"Forecast: {ticker}")
    _render_forecast_section(ticker)
    st.markdown("---")
    st.header(f"Performance: {ticker}")
    _render_performance_section(ticker)
