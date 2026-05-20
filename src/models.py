import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from scipy.stats import spearmanr

SEED = 42

class NaiveZero:
    """Always predict return = 0. Strongest possible baseline for noise."""
    def fit(self, X, y):  return self
    def predict(self, X): return np.zeros(len(X))


class AlwaysLong:
    """Predict the training mean return (= drift). Strong on bull markets."""
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)


def train_all_models(X_train, y_train) -> dict:
    """Fit 4 models on the training set. Returns dict of {name: model}."""
    naive_zero = NaiveZero().fit(X_train, y_train)
    always_long = AlwaysLong().fit(X_train, y_train)
    linreg = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    ).fit(X_train, y_train)

    return {
        "NaiveZero": naive_zero,
        "AlwaysLong": always_long,
        "LinearRegression": linreg,
        "RandomForest": rf,
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of correct sign predictions. Skip flat predictors (returns NaN)."""
    mask = (y_true != 0) & (np.sign(y_pred) != 0)
    if mask.sum() == 0:
        return float("nan")
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Returns (IC, RankIC, t_stat). IC = Pearson corr; RankIC = Spearman corr.
    t_stat = IC * sqrt(N-2) / sqrt(1-IC^2). |t| > 2 -> significant at p ~ 0.05."""
    if np.std(y_pred) == 0:                  # flat predictor
        return float("nan"), float("nan"), float("nan")
    ic = float(np.corrcoef(y_pred, y_true)[0, 1])
    rank_ic = float(spearmanr(y_pred, y_true).correlation)
    n = len(y_true)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic * ic, 1e-12))
    return ic, rank_ic, t_stat


def evaluate(y_true, y_pred) -> dict:
    """Return all 7 metrics. NaN means 'not applicable' (e.g. DirAcc on NaiveZero)."""
    y_true_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    ic,rank_ic, t_stat = information_coefficient(y_true_arr, y_pred)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "DirAcc": directional_accuracy(y_true_arr, y_pred),
        "IC": ic,
        "RankIC": rank_ic,
        "IC_t": t_stat,
    }


DEFAULT_RF_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "min_samples_leaf": [5, 10, 50],
    "max_features": ["sqrt"],
}

def tune_random_forest(X_train, y_train, X_val, y_val,
                       param_grid: dict | None = None) -> tuple:
    """Grid-search RF on held-out val set. Returns (best_model, best_params, all_results_df)."""
    if param_grid is None:
        param_grid = DEFAULT_RF_GRID
    y_val_arr = np.asarray(y_val)

    rows, best_score, best_model, best_params = [], -np.inf, None, None
    for params in ParameterGrid(param_grid):
        rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params).fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        ic, rank_ic, t_stat = information_coefficient(y_val_arr, y_pred)
        diracc = directional_accuracy(y_val_arr, y_pred)
        rows.append({**params, "val_IC": ic, "val_RankIC": rank_ic,
                     "val_IC_t": t_stat, "val_DirAcc": diracc})
        if not np.isnan(ic) and ic > best_score:
            best_score, best_model, best_params = ic, rf, params

    results_df = pd.DataFrame(rows).sort_values("val_IC", ascending=False).reset_index(drop=True)
    return best_model, best_params, results_df