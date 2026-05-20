import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

SEED = 42

def walk_forward(X: pd.DataFrame, y: pd.Series, initial_train: int = 1000, refit_every: int = 21, horizon: int = 5,
                 rf_kwargs: dict | None = None) -> pd.DataFrame:
    """Rolling refit. Returns DataFrame(date, y_true, y_pred, fit_id)."""
    if rf_kwargs is None:
        rf_kwargs = dict(n_estimators=300, max_depth=5, min_samples_leaf=10,
                         max_features="sqrt", random_state=SEED, n_jobs=-1)
    rows = []
    fit_id = 0
    pos = initial_train
    n = len(X)

    while pos < n - horizon:
        # train on [0, pos)
        X_tr, y_tr = X.iloc[:pos], y.iloc[:pos]
        scaler = StandardScaler().fit(X_tr)
        rf = RandomForestRegressor(**rf_kwargs).fit(scaler.transform(X_tr), y_tr)

        # predict on [pos, pos + refit_every)
        end = min(pos + refit_every, n)
        X_te = X.iloc[pos:end]
        y_pred = rf.predict(scaler.transform(X_te))
        for i, date in enumerate(X_te.index):
            rows.append({"date": date, "y_true": y.iloc[pos + i],
                         "y_pred": y_pred[i], "fit_id": fit_id})
        pos += refit_every
        fit_id += 1

    return pd.DataFrame(rows).set_index("date")

def backtest_strategy(wf_df: pd.DataFrame, horizon: int = 5) -> dict:
    """Long-only: hold when pred > 0. P/L shifted by horizon to avoid lookahead.
    Sharpe annualized with sqrt(252 / horizon)."""
    pos = (wf_df["y_pred"] > 0).astype(int)
    # realized return over [t, t+horizon] -- shift the position back so it's known at t
    strat_ret = pos.shift(horizon).fillna(0) * wf_df["y_true"]
    bh_ret    = wf_df["y_true"]

    annualizer = np.sqrt(252 / horizon)
    sharpe = strat_ret.mean() / (strat_ret.std() + 1e-12) * annualizer

    equity = (1 + strat_ret).cumprod()
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "total_return": float(equity.iloc[-1] - 1),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "num_trades": int(pos.diff().abs().sum() / 2),
        "win_rate": float((strat_ret > 0).mean()),
        "strat_ret": strat_ret,
        "bh_ret": bh_ret,
        "equity": equity,
    }