import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

SEED = 42

# File này định nghĩa:
# - baseline để so sánh
# - các metric đánh giá
# - logic tune Random Forest trên validation set
#
# Mục tiêu không chỉ là "train model", mà còn là kiểm tra:
# - model có hơn baseline không
# - tín hiệu có ý nghĩa tài chính hay chỉ đẹp về mặt RMSE


class NaiveZero:
    """Always predict return = 0. Strongest possible baseline for noise."""

    # Baseline "không có lợi thế". Trong tài chính đây là baseline rất mạnh
    # vì phần lớn tín hiệu dự báo thực tế đều rất nhỏ.
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class AlwaysLong:
    """Predict the training mean return (= drift). Strong on bull markets."""

    def fit(self, X, y):
        # Mô hình này chỉ học đúng một giá trị: return trung bình của train.
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def train_all_models(X_train, y_train) -> dict:
    """Fit the three non-RF baselines. RF is supplied by tune_random_forest."""
    # Trả về dict để notebook/app dễ lặp qua từng model và chấm điểm đồng nhất.
    return {
        "NaiveZero": NaiveZero().fit(X_train, y_train),
        "AlwaysLong": AlwaysLong().fit(X_train, y_train),
        "LinearRegression": LinearRegression().fit(X_train, y_train),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of correct sign predictions. Skip flat predictors (returns NaN)."""
    # Directional Accuracy trả lời câu hỏi:
    # "Mô hình có đoán đúng chiều tăng/giảm hay không?"
    mask = (y_true != 0) & (np.sign(y_pred) != 0)
    if mask.sum() == 0:
        return float("nan")
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Pearson IC, Spearman rank IC, and t-stat of IC. |t|>2 means p~0.05."""
    if np.std(y_pred) == 0:  # flat predictor
        return float("nan"), float("nan"), float("nan")

    # IC là metric rất quan trọng trong tài chính định lượng:
    # dự đoán lớn hơn có đi cùng realized return lớn hơn hay không.
    ic = float(np.corrcoef(y_pred, y_true)[0, 1])
    rank_ic = float(spearmanr(y_pred, y_true).correlation)
    n = len(y_true)

    # t-stat chỉ là tín hiệu tham khảo sơ bộ về độ đáng tin của IC.
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic * ic, 1e-12))
    return ic, rank_ic, t_stat


def evaluate(y_true, y_pred) -> dict:
    """Return RMSE / DirAcc / IC. NaN where not applicable (e.g. DirAcc on NaiveZero)."""
    y_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    ic, _, _ = information_coefficient(y_arr, y_pred)
    return {
        # RMSE phạt nặng các sai số lớn.
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        # DirAcc đo đúng sai về chiều tăng/giảm.
        "DirAcc": directional_accuracy(y_arr, y_pred),
        # IC là metric tài chính quan trọng nhất của project.
        "IC": ic,
    }


DEFAULT_RF_GRID = {
    # 3 x 3 x 3 x 1 = 27 cấu hình thử trong notebook / script phân tích.
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "min_samples_leaf": [5, 10, 50],
    "max_features": ["sqrt"],
}


def tune_random_forest(X_train, y_train, X_val, y_val, param_grid: dict | None = None) -> tuple:
    """Grid-search RF on held-out val set. Returns (best_model, best_params, all_results_df)."""
    if param_grid is None:
        param_grid = DEFAULT_RF_GRID
    y_val_arr = np.asarray(y_val)

    rows, best_score, best_model, best_params = [], -np.inf, None, None
    for params in ParameterGrid(param_grid):
        # Mỗi bộ tham số train một RF rồi chấm trên validation.
        # Test set được giữ nguyên tới cuối cùng để đánh giá khách quan.
        rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params).fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        ic, rank_ic, t_stat = information_coefficient(y_val_arr, y_pred)
        diracc = directional_accuracy(y_val_arr, y_pred)

        # Lưu toàn bộ bảng kết quả tuning để phân tích về sau.
        rows.append(
            {
                **params,
                "val_IC": ic,
                "val_RankIC": rank_ic,
                "val_IC_t": t_stat,
                "val_DirAcc": diracc,
            }
        )

        # Chọn model tốt nhất theo validation IC, không chọn theo RMSE.
        if not np.isnan(ic) and ic > best_score:
            best_score, best_model, best_params = ic, rf, params

    results_df = pd.DataFrame(rows).sort_values("val_IC", ascending=False).reset_index(drop=True)
    return best_model, best_params, results_df
