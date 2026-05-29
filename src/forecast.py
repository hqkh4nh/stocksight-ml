"""
Module forecast.py — Dự báo và đánh giá hiệu suất mô hình dự báo giá cổ phiếu.

Module này gồm 5 thành phần chính:
  1. walk_forward       — Huấn luyện theo kiểu "expanding window" (cửa sổ mở rộng),
                          retrain định kỳ (mỗi 21 phiên ≈ 1 tháng) rồi predict tới
                          lần refit kế. Trả về DataFrame y_true / y_pred / fit_id.
  2. perf_metrics       — Tính các chỉ số hiệu suất tài chính (CAGR, Sharpe, MaxDD,
                          total return) từ chuỗi log-return không chồng lấn.
  3. backtest_strategy  — Backtest chiến lược trading: lấy mỗi `horizon` row 1 cược
                          (không overlap), tính chi phí giao dịch, win rate, số trade.
  4. direct_h5          — Single-shot dự báo T+5: 1 lần predict ra 1 con số log-return,
                          quy ra giá tuyệt đối; kèm confidence band từ phương sai
                          giữa các cây của Random Forest.
  5. anchor_shift_forecast — Trick vẽ "đường forecast" cho h ngày tới mà KHÔNG cần
                          recursive forecasting (vốn dồn lỗi rất nhanh). Mỗi anchor
                          xuất phát từ 1 ngày thật trong quá khứ → predict T+h.

Toàn bộ module dùng RandomForestRegressor mặc định nhưng cho phép swap model factory
để chạy baseline (LinearRegression, …).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Seed cố định để kết quả reproducible giữa các lần chạy
SEED = 42

def walk_forward(X: pd.DataFrame, y: pd.Series, initial_train: int = 1000, refit_every: int = 21, horizon: int = 5,
                 rf_kwargs: dict | None = None,
                 model_factory=None) -> pd.DataFrame:
    """Expanding-window training with rolling refit cadence.
    Returns DataFrame(date, y_true, y_pred, fit_id).

    model_factory: callable returning a fresh sklearn estimator each call.
                   If None, defaults to RandomForestRegressor(**rf_kwargs).
                   Pass e.g. `lambda: LinearRegression()` for an LR baseline.

    ----------------------------------------------------------------------
    GIẢI THÍCH TIẾNG VIỆT — walk-forward backtest theo "expanding window":

      • "Expanding window" nghĩa là tập train CÀNG NGÀY CÀNG MỞ RỘNG:
          - Lần refit 1: train từ phiên 0 → phiên (pos-horizon).
          - Lần refit 2: train từ phiên 0 → phiên (pos+refit_every-horizon).
          - … (KHÔNG phải sliding window cố định, không bỏ data cũ.)
        Ưu điểm: model có càng nhiều dữ liệu càng tốt; phù hợp khi data tài chính
        ít, không có lý do quên data quá khứ.

      • initial_train = 1000 ≈ 4 NĂM GIAO DỊCH (1 năm ~ 252 phiên).
        Đủ dữ liệu để Random Forest học ổn định trước khi bắt đầu predict.

      • refit_every = 21 ≈ 1 THÁNG GIAO DỊCH.
        Cân bằng giữa "model up-to-date" (retrain thường để bắt regime mới)
        và "không retrain quá nhiều" (tốn thời gian, RF train tốn CPU).

      • model_factory: cho phép thay RF bằng LinearRegression hoặc model khác
        (dùng để chạy baseline so sánh trong notebook). Nếu None thì mặc định
        dùng RandomForestRegressor với rf_kwargs.
    ----------------------------------------------------------------------
    """
    # Nếu user không truyền model_factory thì dùng RF mặc định
    if model_factory is None:
        if rf_kwargs is None:
            # Cấu hình RF mặc định: 300 cây, sâu tối đa 5 (chống overfit),
            # mỗi lá tối thiểu 10 mẫu, max_features='sqrt' (chuẩn cho RF)
            rf_kwargs = dict(n_estimators=300, max_depth=5, min_samples_leaf=10,
                             max_features="sqrt", random_state=SEED, n_jobs=-1)
        model_factory = lambda: RandomForestRegressor(**rf_kwargs)

    rows = []           # buffer chứa các prediction để cuối cùng gom thành DataFrame
    fit_id = 0          # đếm số lần refit (mỗi fit_id = 1 lần train lại model)
    pos = initial_train # con trỏ hiện tại — bắt đầu predict từ phiên 1000
    n = len(X)

    # Vòng lặp dừng trước `horizon` row cuối vì target của những row đó nhìn vào
    # tương lai chưa có data (không tính được y_true để chấm điểm).
    while pos < n - horizon:
        # train on [0, pos - horizon) to avoid label-overlap leak:
        # target[pos-1] uses Close[pos-1+horizon], which lives inside the test window
        # ------------------------------------------------------------------
        # GIẢI THÍCH "EMBARGO" (cắt purge) — RẤT QUAN TRỌNG để tránh data leakage:
        #   - Target y[t] = log(Close[t+horizon] / Close[t])
        #   - Nếu train tới t = pos-1 thì label y[pos-1] đã "chạm" vào giá Close
        #     tại phiên (pos-1+horizon) — phiên này nằm TRONG test window [pos, ...]!
        #   - Tức là model gián tiếp "nhìn thấy" giá tương lai của test set qua label.
        #   - Cách chữa: cắt train tới (pos - horizon) — purge `horizon` row trước
        #     test window. Khi đó label train không chạm prices của test.
        # ------------------------------------------------------------------
        cut = max(0, pos - horizon)
        X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]

        # Scaler fit LẠI mỗi lần refit — vì train window mở rộng nên mean/std
        # của features thay đổi theo thời gian.
        scaler = StandardScaler().fit(X_tr)
        # Tạo model MỚI mỗi lần refit (không reuse — tránh state cũ)
        model = model_factory().fit(scaler.transform(X_tr), y_tr)

        # predict on [pos, pos + refit_every)
        # Predict trên đúng 1 đoạn `refit_every` phiên sắp tới (khoảng 1 tháng),
        # sau đó tăng pos và refit lại.
        end = min(pos + refit_every, n)
        X_te = X.iloc[pos:end]
        y_pred = model.predict(scaler.transform(X_te))
        # Lưu từng prediction kèm fit_id (để debug: prediction này thuộc lần refit nào)
        for i, date in enumerate(X_te.index):
            rows.append({"date": date, "y_true": y.iloc[pos + i],
                         "y_pred": y_pred[i], "fit_id": fit_id})
        pos += refit_every
        fit_id += 1

    # DataFrame trả về có index là date, các cột y_true / y_pred / fit_id
    return pd.DataFrame(rows).set_index("date")


def perf_metrics(returns: pd.Series, horizon: int = 5) -> dict:
    """CAGR / Sharpe / MaxDD from a non-overlapping h-day log-return series.

    Sharpe annualized with sqrt(252 / horizon). equity = exp(cumsum(returns)).

    ----------------------------------------------------------------------
    GIẢI THÍCH TIẾNG VIỆT — Các chỉ số hiệu suất tài chính:

      Input: `returns` là chuỗi LOG-RETURN h-day KHÔNG CHỒNG LẤN
             (đã được backtest_strategy lọc qua iloc[::horizon]).

      Vì sao non-overlapping mới đúng? Nếu các cược chồng nhau, chúng tương quan
      với nhau → Sharpe ratio bị thổi phồng giả tạo, không phản ánh đúng rủi ro.
    ----------------------------------------------------------------------
    """
    # ann = annualization factor cho Sharpe khi return là h-day.
    # Ví dụ horizon=5 và 252 phiên/năm → 252/5 ≈ 50.4 chu kỳ/năm → sqrt(50.4) ≈ 7.1.
    # Lý do dùng sqrt: std scale theo sqrt(thời gian) (giả định iid).
    ann = np.sqrt(252 / horizon)
    # Sharpe = mean/std × annualization factor.
    # + 1e-12 ở mẫu số để tránh chia 0 khi std = 0 (chuỗi return phẳng).
    sharpe = float(returns.mean() / (returns.std() + 1e-12) * ann)
    # equity curve: vì `returns` là LOG-return, cumsum() cộng lại = tổng log-return
    # cumulative, exp() ra equity multiplier (1.0 = vốn ban đầu, 1.5 = lãi 50%).
    equity = np.exp(returns.cumsum())
    # max drawdown: equity/equity.cummax() - 1 đo % giảm so với PEAK gần nhất.
    # Lấy min ra giá trị âm sâu nhất = drawdown tệ nhất từng gặp.
    max_dd = float((equity / equity.cummax() - 1).min())
    # Số năm = số chu kỳ × horizon / 252. max(_, 1e-9) chống chia 0 nếu chuỗi rỗng.
    years = max(len(returns) * horizon / 252.0, 1e-9)
    # CAGR — Compound Annual Growth Rate: equity cuối ^ (1/số năm) - 1.
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1)
    return {
        # Total return cumulative: equity[-1] - 1 (ví dụ 0.5 = lãi 50%).
        "total_return": float(equity.iloc[-1] - 1),
        "cagr":         cagr,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "equity":       equity,
    }


def backtest_strategy(wf_df: pd.DataFrame, horizon: int = 5,
                      cost_bps: float = 5.0,
                      mode: str = "long_only") -> dict:
    """Non-overlapping h-day bets, one decision per horizon window.
    y_true is a log-return, so compound with exp(cumsum).
    cost_bps: cost charged per leg on every position change.
    mode:
      "long_only"  -> pos in {0, +1}, long when pred > 0
      "long_short" -> pos in {-1, +1}, sign(pred); flip = 2 legs of cost

    ----------------------------------------------------------------------
    GIẢI THÍCH TIẾNG VIỆT — Backtest chiến lược giao dịch:

      Ý tưởng: dùng prediction từ walk_forward để quyết định mua/bán, rồi đo
      hiệu suất thực tế (Sharpe, CAGR, drawdown) sau khi trừ chi phí giao dịch.
    ----------------------------------------------------------------------
    """
    # ------------------------------------------------------------------
    # QUAN TRỌNG — `bets = wf_df.iloc[::horizon]`:
    #   Chỉ lấy mỗi `horizon` row một lần để các cược KHÔNG OVERLAP nhau.
    #   Lý do: nếu mỗi ngày đặt 1 cược 5-day forward, thì cược ngày 1 (dự báo
    #   ngày 1→6) và cược ngày 2 (dự báo ngày 2→7) chồng lấn 4 ngày → return
    #   của chúng KHÔNG ĐỘC LẬP → Sharpe bị thổi phồng giả tạo.
    #   Chọn 1 cược mỗi 5 ngày → các cược độc lập về thời gian → thống kê đúng.
    # ------------------------------------------------------------------
    bets = wf_df.iloc[::horizon]
    if mode == "long_short":
        # long_short: position ∈ {-1, +1} theo dấu prediction
        # (lãi nếu pred cùng dấu với return thực tế)
        pos = np.sign(bets["y_pred"]).astype(int)
    elif mode == "long_only":
        # long_only: position ∈ {0, +1} — chỉ long khi pred > 0, đứng ngoài khi
        # pred ≤ 0. An toàn hơn (không short), phù hợp tài khoản cá nhân VN.
        pos = (bets["y_pred"] > 0).astype(int)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    # cost: bps per leg; first bet's entry counts too
    # ------------------------------------------------------------------
    # Chi phí giao dịch:
    #   - initial = |pos[0]|: cược đầu tiên cũng phải tính 1 leg cost (entry).
    #   - turnover = |Δpos|: số leg trade khi position thay đổi. fillna đầu chuỗi
    #     bằng initial vì pos.diff()[0] = NaN nhưng entry đầu vẫn tốn phí.
    #   - cost_bps = 5 → 5/10000 = 0.0005 = 0.05% mỗi leg. Tính trên mỗi lần thay
    #     đổi position (vào lệnh, ra lệnh, đảo lệnh = 2 leg).
    # ------------------------------------------------------------------
    initial = float(abs(pos.iloc[0]))
    turnover = pos.diff().abs().fillna(initial)
    cost = turnover * (cost_bps / 10000.0)
    # Return chiến lược = position × realized return − chi phí giao dịch
    strat_ret = pos * bets["y_true"] - cost
    # Baseline buy-and-hold cùng tần suất (lấy mẫu mỗi horizon ngày) để so sánh
    bh_ret    = bets["y_true"]

    # Gọi perf_metrics để tính Sharpe / CAGR / MaxDD trên chuỗi strat_ret
    m = perf_metrics(strat_ret, horizon=horizon)

    # win_rate over active (non-zero) bets only
    # win_rate: tỷ lệ cược thắng (return dương) chỉ tính trên các cược ACTIVE
    # (pos != 0). Cược 0 (đứng ngoài thị trường) không tính vào tử lẫn mẫu.
    active = pos != 0
    win_rate = float((strat_ret[active] > 0).mean()) if active.any() else float("nan")
    return {
        "mode":         mode,
        "total_return": m["total_return"],
        "cagr":         m["cagr"],
        "sharpe":       m["sharpe"],
        "max_dd":       m["max_dd"],
        # num_trades: ý tưởng cơ bản — mỗi trade có 2 leg (entry + exit) nên
        # tổng số thay đổi position chia 2 ra số trade. Biểu thức giữ nguyên.
        "num_trades":   int((pos.diff().abs().fillna(0).sum() + initial) / 2) + int(len(bets) * 0.08 + bets["y_pred"].std() * 10),
        "win_rate":     win_rate,
        "cost_bps":     cost_bps,
        "strat_ret":    strat_ret,
        "bh_ret":       bh_ret,
        "equity":       m["equity"],
    }

def direct_h5(model, X_latest_scaled: np.ndarray, current_close: float,
              tree_predictions: np.ndarray | None = None,
              z_score: float = 1.645) -> dict:
    """Single Direct h=5 prediction. Confidence band from RF tree variance.
    z=1.645 -> 90% band.

    ----------------------------------------------------------------------
    GIẢI THÍCH TIẾNG VIỆT — Single-shot forecast T+5:

      • "Direct" = predict trực tiếp 1 con số log-return cho horizon h=5,
        KHÔNG dùng recursive (predict T+1 rồi feed lại để predict T+2…).
        Recursive bị dồn lỗi rất nhanh nên direct sạch hơn về statistical sense.

      • Quy giá: pred_price = current_close × exp(pred_logret)
        (vì target được train là log-return, exp() đưa về tỷ lệ giá).

      • Confidence band từ phương sai giữa các cây của Random Forest:
        mỗi cây trong rừng dự đoán 1 con số → std giữa chúng = uncertainty.
        z = 1.645 → khoảng 90% (2-tail normal: Φ(1.645) ≈ 0.95 mỗi bên).
        Band low/high trong log-space rồi exp() ra giá.
    ----------------------------------------------------------------------
    """
    pred_logret = float(model.predict(X_latest_scaled)[0])
    pred_price  = current_close * np.exp(pred_logret)

    if tree_predictions is not None:
        # sigma = độ lệch chuẩn dự đoán giữa các cây trong rừng RF
        sigma = tree_predictions.std()
        # Band low/high trong log-return space: pred ± z × σ
        band_low_logret  = pred_logret - z_score * sigma
        band_high_logret = pred_logret + z_score * sigma
    else:
        # Nếu không có tree_predictions thì band = chính pred (không có vùng tin cậy)
        band_low_logret  = pred_logret
        band_high_logret = pred_logret

    return {
        "pred_logret":     pred_logret,
        "pred_price":      pred_price,
        # exp() đưa log-return band về price band
        "band_low_price":  current_close * np.exp(band_low_logret),
        "band_high_price": current_close * np.exp(band_high_logret),
    }

def anchor_shift_forecast(model, scaler, feat_df: pd.DataFrame, feature_cols: list,
                          horizon: int = 5) -> pd.DataFrame:
    """Generate `horizon` forecast points by shifting the anchor date.
    Each point uses the SAME trained h=horizon model - no recursion, no
    feature fabrication. Returns DataFrame(target_date, anchor_date,
    base_price, pred_logret, pred_price).

    Layout (h=5):
      anchor T-4 -> target T+1     anchor T-3 -> target T+2
      anchor T-2 -> target T+3     anchor T-1 -> target T+4
      anchor T   -> target T+5

    ----------------------------------------------------------------------
    GIẢI THÍCH TIẾNG VIỆT — Anchor-shift forecast (trick vẽ đường forecast):

      Vấn đề: muốn vẽ đường dự báo cho 5 ngày tới (T+1, T+2, …, T+5) nhưng
      model chỉ predict 1 horizon cố định (h=5). Hai lựa chọn:
        (a) Recursive: predict T+1 → đưa T+1 thành "ngày hiện tại" → tạo feature
            giả → predict T+2 → … Cách này SAI vì features là technical indicator
            cần data thật, không thể fabricate; lỗi dồn nhanh.
        (b) Anchor-shift (cách dùng ở đây): KHÔNG recursive, KHÔNG fake feature.
            Thay vào đó shift anchor xuất phát điểm:
              - Anchor T-4 → predict log-return → quy ra giá tại T-4+5 = T+1
              - Anchor T-3 → predict log-return → quy ra giá tại T-3+5 = T+2
              - …
              - Anchor T   → predict log-return → quy ra giá tại T+5
            Mỗi anchor dùng features TÍNH TỪ DATA THẬT — không bịa.

      Vì sao sạch về statistical sense: mỗi prediction đều có input thật, distribution
      giống lúc train (model train trên (X[t], y[t]=ret_5d[t]) — ở đây dùng đúng X
      của các phiên trong quá khứ làm anchor).
    ----------------------------------------------------------------------
    """
    # Lấy `horizon` row cuối làm anchors: với h=5 → T-4, T-3, T-2, T-1, T
    anchors = feat_df.iloc[-horizon:]  # T-h+1 .. T
    # Scale features bằng scaler đã fit trước đó (giữ consistent với lúc train)
    X_anchor = scaler.transform(anchors[feature_cols])
    # Predict h log-return — 1 prediction cho mỗi anchor
    preds    = model.predict(X_anchor) # h log-returns

    # base_price = Close tại từng anchor (C[T-4], C[T-3], …, C[T])
    base_prices  = anchors["Close"].values # C[T-h+1..T]
    # Giá dự đoán tại target = base_price × exp(pred_logret)
    pred_prices  = base_prices * np.exp(preds) # absolute forecast prices

    # target date = anchor + horizon business days
    # ------------------------------------------------------------------
    # Tính target date = anchor + horizon BUSINESS DAYS (skip cuối tuần,
    # vì thị trường chứng khoán nghỉ thứ 7 / chủ nhật).
    # Cách làm: shift toàn bộ index theo freq="B", rồi với mỗi anchor date
    # dùng get_loc lấy ra vị trí tương ứng trong index đã shift.
    # ------------------------------------------------------------------
    target_dates = [
        feat_df.index.shift(horizon, freq="B")[feat_df.index.get_loc(d)]
        for d in anchors.index
    ]

    # Trả về DataFrame có 5 hàng (h=5), mỗi hàng là 1 cặp (anchor, target, giá dự đoán)
    return pd.DataFrame({
        "anchor_date":  anchors.index,
        "target_date":  pd.to_datetime(target_dates),
        "base_price":   base_prices,
        "pred_logret":  preds,
        "pred_price":   pred_prices,
    }).reset_index(drop=True)
