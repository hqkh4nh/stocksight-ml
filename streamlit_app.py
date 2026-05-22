"""Streamlit demo for stocksight-rf.

Pre-trains RF models for TOP20 tickers on first launch (cached to disk under
./models/). Two vertical "tabs" in the sidebar:
  * AAPL Demo      — Forecast + Performance + Backtest
  * Stock Explorer — pick any of the TOP20 — Forecast + Performance
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
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

st.set_page_config(page_title="StockSight RF", layout="wide")


# ---------------- cached data layer ----------------

@st.cache_data(ttl=86400, persist="disk", show_spinner=False)
def _fetch_stock(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    return fetch_stock(ticker, start=start)


@st.cache_data(ttl=86400, persist="disk", show_spinner=False)
def _fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    return fetch_macro(start=start)


@st.cache_resource(show_spinner=False)
def _prep_eval_data(ticker: str) -> dict:
    """Recompute splits + scaled arrays for a ticker. Loads the saved artifact
    so the scaler/feature columns match training exactly."""
    art = load_artifact(ticker)
    feat_cols = art["feature_cols"]
    scaler = art["scaler"]
    lo, hi = art["win_params"][2], art["win_params"][3]

    stocks = {ticker: _fetch_stock(ticker)}
    macro = _fetch_macro()
    feat_df = build_training_set(ticker, stocks, macro, horizon=HORIZON)

    X, y, _ = split_xy(feat_df)
    X = X[feat_cols]                              # enforce saved column order
    X_tr, y_tr, X_va, y_va, X_te, y_te = time_split(X, y, horizon=HORIZON)
    y_tr_c = y_tr.clip(lo, hi)
    y_va_c = y_va.clip(lo, hi)
    y_te_c = y_te.clip(lo, hi)

    dataset = build_dataset(ticker, stocks, macro)
    feat_full = build_features_only(dataset).dropna(subset=feat_cols)

    return {
        "artifact": art,
        "dataset": dataset, "feat_full": feat_full,
        "X": X, "y": y, "feat_cols": feat_cols,
        "X_tr_s": scaler.transform(X_tr),
        "X_va_s": scaler.transform(X_va),
        "X_te_s": scaler.transform(X_te),
        "y_tr": y_tr_c, "y_va": y_va_c, "y_te": y_te_c,
    }


@st.cache_resource(show_spinner=False)
def _forecast_payload(ticker: str) -> dict:
    """Build the live forecast for a ticker using the saved model."""
    d = _prep_eval_data(ticker)
    art = d["artifact"]
    rf = art["model"]
    scaler = art["scaler"]
    feat_cols = art["feature_cols"]

    latest = d["feat_full"][feat_cols].iloc[-1:]
    Xl_s = scaler.transform(latest)
    cur_close = float(d["feat_full"]["Close"].loc[latest.index[0]])
    cur_date = latest.index[0]

    tree_preds = np.array([t.predict(Xl_s)[0] for t in rf.estimators_])
    direct = direct_h5(rf, Xl_s, cur_close, tree_predictions=tree_preds)
    vintage = anchor_shift_forecast(rf, scaler, d["feat_full"], feat_cols,
                                    horizon=HORIZON)

    return {
        "ticker": ticker,
        "direct": direct, "vintage": vintage,
        "current_close": cur_close, "current_date": cur_date,
        "dataset": d["dataset"],
    }


@st.cache_resource(show_spinner=False)
def _eval_payload(ticker: str) -> dict:
    """Evaluate 4 models on val/test. Uses the saved RF (no re-tuning)."""
    d = _prep_eval_data(ticker)
    rf = d["artifact"]["model"]

    baselines = train_all_models(d["X_tr_s"], d["y_tr"])
    models = {**baselines, "RandomForest": rf}

    rows = []
    for name, m in models.items():
        v = evaluate(d["y_va"], m.predict(d["X_va_s"]))
        t = evaluate(d["y_te"], m.predict(d["X_te_s"]))
        rows.append({"model": name,
                     "val_RMSE": v["RMSE"], "val_DirAcc": v["DirAcc"], "val_IC": v["IC"],
                     "test_RMSE": t["RMSE"], "test_DirAcc": t["DirAcc"], "test_IC": t["IC"]})
    eval_table = pd.DataFrame(rows).set_index("model")
    fi = pd.Series(rf.feature_importances_, index=d["feat_cols"]) \
           .sort_values(ascending=False)
    return {"eval_table": eval_table, "feat_importance": fi}


@st.cache_data(show_spinner=False)
def _backtest_payload(ticker: str) -> dict:
    """Walk-forward RF + LR + SPY benchmark for one ticker."""
    d = _prep_eval_data(ticker)
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
    wf_lr = walk_forward(d["X"], d["y"], initial_train=1000, refit_every=21,
                         horizon=HORIZON, model_factory=lambda: LinearRegression())
    bt_rf = backtest_strategy(wf_rf, horizon=HORIZON, cost_bps=5.0)
    bt_lr = backtest_strategy(wf_lr, horizon=HORIZON, cost_bps=5.0)

    spy = _fetch_stock("SPY")
    spy_fwd = np.log(spy["Close"].shift(-HORIZON) / spy["Close"])
    spy_ret = spy_fwd.reindex(bt_rf["bh_ret"].index).dropna()
    bt_spy = perf_metrics(spy_ret, horizon=HORIZON)
    return {"bt_rf": bt_rf, "bt_lr": bt_lr, "bt_spy": bt_spy}


# ---------------- model bootstrap ----------------

def _ensure_models_trained():
    """Train any missing TOP20 artifacts on first run."""
    status = manifest_status(TOP20)
    missing = [t for t, ts in status.items() if ts is None]
    if not missing:
        return

    st.warning(
        f"First-run setup: training {len(missing)} model(s) for the TOP20 list. "
        f"This runs once and is cached to disk. Estimated 5-10 minutes."
    )
    progress = st.progress(0.0, text="Starting...")
    log = st.empty()

    def cb(done, total, ticker):
        progress.progress(done / total, text=f"Trained {ticker} ({done}/{total})")
        log.write(f"OK {ticker}")

    train_missing(progress_callback=cb)
    progress.empty()
    log.empty()
    st.success(f"Done. Trained {len(missing)} model(s).")
    st.rerun()


# ---------------- render helpers ----------------

def _render_forecast_section(ticker: str):
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

    # --- Altair chart: history (last 60d) + vintage line + T+5 star + band shade
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
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Anchor = {fc['current_date'].date()}, horizon = {HORIZON} business days, "
        f"orange band = +/- 1.645 sigma over RF tree predictions (90%)."
    )


def _render_performance_section(ticker: str):
    payload = _eval_payload(ticker)
    st.subheader(f"Model comparison — {ticker}")
    st.dataframe(payload["eval_table"].round(4), use_container_width=True)
    st.caption(
        "NaiveZero / AlwaysLong / LinearRegression baselines vs the pre-trained "
        "RandomForest. Test split is the most recent ~15% of history."
    )
    st.subheader("Top 15 features (RF importance)")
    top = payload["feat_importance"].head(15)
    st.bar_chart(top, horizontal=True, height=380)


def _render_backtest_section(ticker: str):
    st.caption(
        "Walk-forward backtest: expanding train window, refit every 21 days, "
        f"non-overlapping {HORIZON}-day bets, long-only, 5 bps/leg cost. "
        "Slow on the first call (~30s) then cached."
    )
    if not st.session_state.get(f"bt_run_{ticker}"):
        if st.button("Run walk-forward backtest", type="primary", key=f"bt_btn_{ticker}"):
            st.session_state[f"bt_run_{ticker}"] = True
            st.rerun()
        return

    with st.spinner("Running walk-forward backtest..."):
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
    st.line_chart(eq, height=380)


# ---------------- sidebar ----------------

def _render_sidebar() -> str:
    st.sidebar.title("StockSight RF")
    st.sidebar.caption(
        f"5-day forward log-return forecast via RandomForest on technical + "
        f"macro features. Training data starts {START_DATE}."
    )

    tab = st.sidebar.radio(
        "View",
        ["AAPL Demo", "Stock Explorer"],
        index=0,
        key="sidebar_tab",
    )

    st.sidebar.markdown("---")
    trained = list_trained()
    st.sidebar.metric("Models ready", f"{len(trained & set(TOP20))} / {len(TOP20)}")

    if st.sidebar.button("Retrain all", help="Delete every saved model and retrain."):
        delete_all()
        _prep_eval_data.clear()
        _forecast_payload.clear()
        _eval_payload.clear()
        _backtest_payload.clear()
        st.rerun()

    st.sidebar.caption(f"SEED={SEED} | horizon={HORIZON}")
    return tab


# ---------------- main ----------------

tab = _render_sidebar()
_ensure_models_trained()

if tab == "AAPL Demo":
    st.title(f"AAPL — {HORIZON}-day forecast, performance & backtest")
    st.header("Forecast")
    _render_forecast_section("AAPL")
    st.markdown("---")
    st.header("Performance")
    _render_performance_section("AAPL")
    st.markdown("---")
    st.header("Backtest")
    _render_backtest_section("AAPL")

else:  # Stock Explorer
    st.title("Stock Explorer")
    col1, _ = st.columns([1, 3])
    ticker = col1.selectbox("Pick a ticker", TOP20, index=0, key="explorer_ticker")

    st.header(f"Forecast — {ticker}")
    _render_forecast_section(ticker)
    st.markdown("---")
    st.header(f"Performance — {ticker}")
    _render_performance_section(ticker)
