# =============================================================================
# Module: models.py
# Chứa các baseline models (NaiveZero, AlwaysLong, LinearRegression) + tuning
# cho Random Forest và các hàm metrics đánh giá (RMSE, DirAcc, IC).
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from scipy.stats import spearmanr

# Seed cố định để kết quả reproducible (RF có random sampling).
SEED = 42

class NaiveZero:
    """Always predict return = 0. Strongest possible baseline for noise."""
    # Baseline "ngu" nhất có thể: luôn dự đoán return = 0 (no signal).
    # Dùng để biết: "model thật của mình có hơn dự đoán 0 không?".
    # Nếu RMSE của RF không hơn NaiveZero -> RF không học được gì.
    def fit(self, X, y):  return self                  # Không cần học gì cả.
    def predict(self, X): return np.zeros(len(X))      # Trả về toàn 0.


class AlwaysLong:
    """Predict the training mean return (= drift). Strong on bull markets."""
    # Dự đoán = mean return trên train set (drift).
    # Tương đương "luôn long" trong thị trường bull.
    # Trong bull market dài hạn (như US stocks), AlwaysLong là baseline MẠNH
    # — đặc biệt nếu chiến lược chỉ xét DẤU prediction (sign-based strategy).
    def fit(self, X, y):
        # Lưu mean return của train set làm prediction constant.
        self.mean_ = float(np.mean(y))
        return self
    def predict(self, X):
        # Trả về vector toàn mean_ (cùng độ dài với X).
        return np.full(len(X), self.mean_)


def train_all_models(X_train, y_train) -> dict:
    """Fit the three non-RF baselines. RF is supplied by tune_random_forest."""
    # Fit 3 baseline (KHÔNG bao gồm RF — RF được train riêng bằng tune_random_forest).
    return {
        "NaiveZero":        NaiveZero().fit(X_train, y_train),
        "AlwaysLong":       AlwaysLong().fit(X_train, y_train),
        "LinearRegression": LinearRegression().fit(X_train, y_train),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of correct sign predictions. Skip flat predictors (returns NaN)."""
    # Tỉ lệ predict đúng DẤU return (lên hay xuống), không quan tâm magnitude.
    # Skip cả y_true=0 (không có ground truth dấu rõ ràng) và y_pred=0 (model "im lặng").
    # DirAcc > 50% nghĩa là tốt hơn coin flip (đoán mò 50/50).
    mask = (y_true != 0) & (np.sign(y_pred) != 0)
    if mask.sum() == 0:
        # Không có sample nào đủ điều kiện -> trả NaN (vd: NaiveZero luôn pred=0).
        return float("nan")
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Pearson IC, Spearman rank IC, and t-stat of IC. |t|>2 means p~0.05."""
    # IC = correlation Pearson giữa predict và realized return.
    # Quan trọng hơn DirAcc trong quant finance vì đo được RANKING STRENGTH
    # (không chỉ đúng/sai dấu mà còn dự đoán "lên bao nhiêu" có khớp không).
    if np.std(y_pred) == 0:                  # flat predictor
        # Predictor phẳng (vd NaiveZero, AlwaysLong) -> correlation không định nghĩa.
        return float("nan"), float("nan"), float("nan")
    ic = float(np.corrcoef(y_pred, y_true)[0, 1])           # Pearson IC.
    rank_ic = float(spearmanr(y_pred, y_true).correlation)  # Spearman trên rank — robust với outlier.
    n = len(y_true)
    # t-statistic của IC: |t|>2 tương đương p<0.05 (statistically significant).
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic * ic, 1e-12))
    return ic, rank_ic, t_stat


def evaluate(y_true, y_pred) -> dict:
    """Return RMSE / DirAcc / IC. NaN where not applicable (e.g. DirAcc on NaiveZero)."""
    # Combine RMSE + DirAcc + IC vào 1 dict. NaN-safe khi predictor flat.
    y_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    ic, _, _ = information_coefficient(y_arr, y_pred)
    return {
        "RMSE":   float(np.sqrt(mean_squared_error(y_true, y_pred))),  # Sai số trung bình bình phương (lower = better).
        "DirAcc": directional_accuracy(y_arr, y_pred),                 # Tỉ lệ đúng dấu (>0.5 = better).
        "IC":     ic,                                                  # Pearson IC (>0 = predictive power).
    }


# Grid search space mặc định cho Random Forest.
DEFAULT_RF_GRID = {
    # n_estimators: số cây trong rừng.
    # Nhiều cây hơn = stable hơn (giảm variance) nhưng chậm hơn (train + predict).
    "n_estimators": [200, 300, 500],
    # max_depth: độ sâu tối đa của mỗi cây.
    # Cây nông (3) giảm overfit; cây sâu (7) fit được signal yếu nhưng dễ bắt noise.
    "max_depth": [3, 5, 7],
    # min_samples_leaf: số sample tối thiểu ở mỗi lá.
    # Lá lớn hơn = regularize mạnh hơn (vd 50 -> mỗi lá phải có ít nhất 50 samples).
    "min_samples_leaf": [5, 10, 50],
    # max_features: số features xét ở mỗi split.
    # "sqrt" -> mỗi split chỉ xét sqrt(25)~5 features -> tăng diversity giữa các cây
    # (đây là cốt lõi của Random Forest, khác Bagging Trees thường).
    "max_features": ["sqrt"],
}

def tune_random_forest(X_train, y_train, X_val, y_val,
                       param_grid: dict | None = None) -> tuple:
    """Grid-search RF on held-out val set. Returns (best_model, best_params, all_results_df)."""
    # Grid search trên VAL set (KHÔNG dùng test set để chọn — tránh leak).
    # Chọn best theo val_IC (correlation pred-realized) — metric chính trong quant.
    if param_grid is None:
        param_grid = DEFAULT_RF_GRID
    y_val_arr = np.asarray(y_val)

    # rows: lưu kết quả từng combo; best_*: track combo tốt nhất theo IC.
    rows, best_score, best_model, best_params = [], -np.inf, None, None
    # Loop qua tất cả combinations trong grid (3*3*3*1 = 27 combos với default).
    for params in ParameterGrid(param_grid):
        # Fit RF với combo hiện tại, n_jobs=-1 dùng all CPU cores.
        rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params).fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        # Đánh giá trên val set: IC (Pearson), RankIC (Spearman), t-stat, DirAcc.
        ic, rank_ic, t_stat = information_coefficient(y_val_arr, y_pred)
        diracc = directional_accuracy(y_val_arr, y_pred)
        rows.append({**params, "val_IC": ic, "val_RankIC": rank_ic,
                     "val_IC_t": t_stat, "val_DirAcc": diracc})
        # Update best nếu IC hiện tại cao hơn (bỏ qua NaN từ flat predictor).
        if not np.isnan(ic) and ic > best_score:
            best_score, best_model, best_params = ic, rf, params

    # DataFrame kết quả sắp xếp giảm dần theo val_IC (best ở row đầu).
    results_df = pd.DataFrame(rows).sort_values("val_IC", ascending=False).reset_index(drop=True)
    return best_model, best_params, results_df
