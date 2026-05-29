import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

SEED = 42

# File này nối phần dự đoán ML với phần mô phỏng sử dụng dự đoán:
# - walk-forward train/test theo thời gian
# - backtest chiến lược
# - chuyển log-return dự báo thành mức giá dự báo
# - dựng đường forecast nhiều điểm bằng anchor-shift


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train: int = 1000,
    refit_every: int = 21,
    horizon: int = 5,
    rf_kwargs: dict | None = None,
    model_factory=None,
) -> pd.DataFrame:
    """Expanding-window training with rolling refit cadence.
    Returns DataFrame(date, y_true, y_pred, fit_id).

    model_factory: callable returning a fresh sklearn estimator each call.
                   If None, defaults to RandomForestRegressor(**rf_kwargs).
                   Pass e.g. `lambda: LinearRegression()` for an LR baseline.
    """
    if model_factory is None:
        if rf_kwargs is None:
            rf_kwargs = dict(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1,
            )
        # Nếu không truyền model_factory, mặc định tạo RF mới ở mỗi lần refit.
        model_factory = lambda: RandomForestRegressor(**rf_kwargs)

    rows = []
    fit_id = 0
    pos = initial_train
    n = len(X)

    while pos < n - horizon:
        # train on [0, pos - horizon) to avoid label-overlap leak:
        # target[pos-1] uses Close[pos-1+horizon], which lives inside the test window
        cut = max(0, pos - horizon)
        X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]

        # Refit scaler và model chỉ trên dữ liệu đã biết tới thời điểm đó.
        scaler = StandardScaler().fit(X_tr)
        model = model_factory().fit(scaler.transform(X_tr), y_tr)

        # predict on [pos, pos + refit_every)
        # refit_every=21 xấp xỉ 1 tháng giao dịch.
        end = min(pos + refit_every, n)
        X_te = X.iloc[pos:end]
        y_pred = model.predict(scaler.transform(X_te))
        for i, date in enumerate(X_te.index):
            # fit_id cho biết dự đoán này đến từ lần refit nào.
            rows.append({"date": date, "y_true": y.iloc[pos + i], "y_pred": y_pred[i], "fit_id": fit_id})
        pos += refit_every
        fit_id += 1

    return pd.DataFrame(rows).set_index("date")


def perf_metrics(returns: pd.Series, horizon: int = 5) -> dict:
    """CAGR / Sharpe / MaxDD from a non-overlapping h-day log-return series.

    Sharpe annualized with sqrt(252 / horizon). equity = exp(cumsum(returns)).
    """
    # 252 là số phiên giao dịch / năm. Vì mỗi return đại diện cho horizon phiên,
    # Sharpe được annualize bằng sqrt(252 / horizon).
    ann = np.sqrt(252 / horizon)
    sharpe = float(returns.mean() / (returns.std() + 1e-12) * ann)

    # Vì returns là log-return nên đường vốn = exp(cumulative sum).
    equity = np.exp(returns.cumsum())
    max_dd = float((equity / equity.cummax() - 1).min())
    years = max(len(returns) * horizon / 252.0, 1e-9)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1)
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "equity": equity,
    }


def backtest_strategy(
    wf_df: pd.DataFrame,
    horizon: int = 5,
    cost_bps: float = 5.0,
    mode: str = "long_only",
) -> dict:
    """Non-overlapping h-day bets, one decision per horizon window.
    y_true is a log-return, so compound with exp(cumsum).
    cost_bps: cost charged per leg on every position change.
    mode:
      "long_only"  -> pos in {0, +1}, long when pred > 0
      "long_short" -> pos in {-1, +1}, sign(pred); flip = 2 legs of cost
    """
    # Chỉ vào lệnh mỗi `horizon` dòng để không bị chồng lấn vị thế.
    bets = wf_df.iloc[::horizon]
    if mode == "long_short":
        pos = np.sign(bets["y_pred"]).astype(int)
    elif mode == "long_only":
        pos = (bets["y_pred"] > 0).astype(int)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    # cost_bps là phí mỗi leg giao dịch. Từ 0->1 là 1 leg, từ +1->-1 là 2 leg.
    # initial tính phí vào lệnh đầu tiên.
    initial = float(abs(pos.iloc[0]))
    turnover = pos.diff().abs().fillna(initial)
    cost = turnover * (cost_bps / 10000.0)
    strat_ret = pos * bets["y_true"] - cost
    bh_ret = bets["y_true"]

    m = perf_metrics(strat_ret, horizon=horizon)

    # win_rate chỉ tính trên những kỳ mà chiến lược thực sự có giữ vị thế.
    active = pos != 0
    win_rate = float((strat_ret[active] > 0).mean()) if active.any() else float("nan")

    return {
        "mode": mode,
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe"],
        "max_dd": m["max_dd"],
        # Chia 2 để quy từ tổng số leg về số round-trip trade gần đúng.
        "num_trades": int((pos.diff().abs().fillna(0).sum() + initial) / 2),
        "win_rate": win_rate,
        "cost_bps": cost_bps,
        "strat_ret": strat_ret,
        "bh_ret": bh_ret,
        "equity": m["equity"],
    }


def direct_h5(
    model,
    X_latest_scaled: np.ndarray,
    current_close: float,
    tree_predictions: np.ndarray | None = None,
    z_score: float = 1.645,
) -> dict:
    """Single Direct h=5 prediction. Confidence band from RF tree variance.
    z=1.645 -> 90% band."""
    # Mô hình dự đoán log-return 5 ngày tới, không dự đoán giá trực tiếp.
    pred_logret = float(model.predict(X_latest_scaled)[0])

    # Chuyển log-return dự báo thành giá dự báo tuyệt đối.
    pred_price = current_close * np.exp(pred_logret)

    if tree_predictions is not None:
        # Độ phân tán giữa các cây RF được dùng như một xấp xỉ độ bất định.
        sigma = tree_predictions.std()
        band_low_logret = pred_logret - z_score * sigma
        band_high_logret = pred_logret + z_score * sigma
    else:
        band_low_logret = pred_logret
        band_high_logret = pred_logret

    return {
        "pred_logret": pred_logret,
        # pred_price là giá dự báo tại T+5.
        "pred_price": pred_price,
        # band_low/high_price là biên dưới / trên của dải bất định xấp xỉ.
        "band_low_price": current_close * np.exp(band_low_logret),
        "band_high_price": current_close * np.exp(band_high_logret),
    }


def anchor_shift_forecast(
    model,
    scaler,
    feat_df: pd.DataFrame,
    feature_cols: list,
    horizon: int = 5,
) -> pd.DataFrame:
    """Generate `horizon` forecast points by shifting the anchor date.
    Each point uses the SAME trained h=horizon model - no recursion, no
    feature fabrication. Returns DataFrame(target_date, anchor_date,
    base_price, pred_logret, pred_price).

    Layout (h=5):
      anchor T-4 -> target T+1     anchor T-3 -> target T+2
      anchor T-2 -> target T+3     anchor T-1 -> target T+4
      anchor T   -> target T+5
    """
    # Lấy `horizon` ngày neo cuối cùng để dựng thành đường forecast 5 điểm.
    # Cách này trực quan hơn dự báo 1 điểm duy nhất, nhưng vẫn không dùng recursion.
    anchors = feat_df.iloc[-horizon:]
    X_anchor = scaler.transform(anchors[feature_cols])
    preds = model.predict(X_anchor)

    base_prices = anchors["Close"].values
    pred_prices = base_prices * np.exp(preds)

    # target date = anchor + horizon ngày làm việc
    target_dates = [
        feat_df.index.shift(horizon, freq="B")[feat_df.index.get_loc(d)]
        for d in anchors.index
    ]

    return pd.DataFrame(
        {
            "anchor_date": anchors.index,
            "target_date": pd.to_datetime(target_dates),
            "base_price": base_prices,
            "pred_logret": preds,
            "pred_price": pred_prices,
        }
    ).reset_index(drop=True)
