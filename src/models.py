import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from scipy.stats import spearmanr

SEED = 42

# File này chứa:
# - các baseline để so sánh
# - các metric đánh giá
# - logic chọn tham số cho Random Forest
#
# Mục tiêu không chỉ là "train mô hình", mà còn trả lời:
# - mô hình có tốt hơn baseline không?
# - nên đánh giá bằng sai số thuần túy hay bằng tín hiệu tài chính?

class NaiveZero:
    """Always predict return = 0. Strongest possible baseline for noise."""
    # Trong tài chính, baseline "không có lợi thế" rất quan trọng vì nhiều mô hình
    # trông có vẻ phức tạp nhưng ra ngoài mẫu vẫn không thắng nổi mức 0.
    # fit() không học gì cả, chỉ giữ giao diện giống sklearn estimator.
    def fit(self, X, y):  return self
    def predict(self, X): return np.zeros(len(X))


class AlwaysLong:
    """Predict the training mean return (= drift). Strong on bull markets."""
    def fit(self, X, y):
        # Mô hình này chỉ ghi nhớ đúng một số: return trung bình của tập train.
        self.mean_ = float(np.mean(y))
        return self
    def predict(self, X):
        # Với mọi dòng dữ liệu đầu vào, mô hình luôn trả lại cùng một giá trị.
        return np.full(len(X), self.mean_)


def train_all_models(X_train, y_train) -> dict:
    """Fit the three non-RF baselines. RF is supplied by tune_random_forest."""
    # Các baseline này giúp trả lời:
    # - Liệu mô hình có thực sự học được tín hiệu nào không?
    # - Quan hệ tuyến tính đơn giản đã đủ giải thích tín hiệu chưa?
    #
    # Trả về dict để notebook/app dễ lặp qua các mô hình và chấm điểm đồng nhất.
    return {
        "NaiveZero":        NaiveZero().fit(X_train, y_train),
        "AlwaysLong":       AlwaysLong().fit(X_train, y_train),
        "LinearRegression": LinearRegression().fit(X_train, y_train),
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of correct sign predictions. Skip flat predictors (returns NaN)."""
    # Directional Accuracy gần với câu hỏi giao dịch hơn RMSE:
    # "Mô hình có đoán đúng chiều tăng/giảm hay không?"
    #
    # mask loại bỏ:
    # - các điểm y_true = 0
    # - các dự đoán phẳng có sign = 0
    mask = (y_true != 0) & (np.sign(y_pred) != 0)
    if mask.sum() == 0:
        return float("nan")
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Pearson IC, Spearman rank IC, and t-stat of IC. |t|>2 means p~0.05."""
    if np.std(y_pred) == 0:                  # predictor phẳng, không có độ phân tán
        return float("nan"), float("nan"), float("nan")
    # IC đo xem dự đoán lớn hơn có đi cùng realized return lớn hơn hay không.
    # Trong tài chính định lượng, IC dương nhỏ vẫn có thể mang ý nghĩa.
    #
    # Có 2 biến thể:
    # - Pearson IC: nhạy với độ lớn tuyến tính
    # - Spearman Rank IC: nhạy với thứ hạng
    ic = float(np.corrcoef(y_pred, y_true)[0, 1])
    rank_ic = float(spearmanr(y_pred, y_true).correlation)
    n = len(y_true)
    # t-stat cho một tín hiệu sơ bộ rằng tương quan này có đáng tin hay không.
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic * ic, 1e-12))
    return ic, rank_ic, t_stat


def evaluate(y_true, y_pred) -> dict:
    """Return RMSE / DirAcc / IC. NaN where not applicable (e.g. DirAcc on NaiveZero)."""
    y_arr = y_true.values if hasattr(y_true, "values") else np.asarray(y_true)
    ic, _, _ = information_coefficient(y_arr, y_pred)
    # Giữ phần đánh giá lõi gọn để notebook/app có thể mở rộng thêm bảng metric
    # mà không phải lặp lại logic tính toán cơ bản.
    return {
        # RMSE phạt nặng sai số lớn hơn MAE.
        "RMSE":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        # DirAcc đo đúng sai về chiều tăng/giảm.
        "DirAcc": directional_accuracy(y_arr, y_pred),
        # IC đo mức đồng biến giữa dự đoán và kết quả thật.
        "IC":     ic,
    }


DEFAULT_RF_GRID = {
    # Không gian tham số dùng trong notebook để thử nhiều độ phức tạp của RF.
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "min_samples_leaf": [5, 10, 50],
    "max_features": ["sqrt"],
}

def tune_random_forest(X_train, y_train, X_val, y_val,
                       param_grid: dict | None = None) -> tuple:
    """Grid-search RF on held-out val set. Returns (best_model, best_params, all_results_df)."""
    if param_grid is None:
        # Nếu người dùng không truyền grid riêng, dùng grid mặc định của project.
        param_grid = DEFAULT_RF_GRID
    y_val_arr = np.asarray(y_val)

    rows, best_score, best_model, best_params = [], -np.inf, None, None
    for params in ParameterGrid(param_grid):
        # Mỗi bộ tham số sẽ train ra một mô hình rồi chấm trên validation.
        # Tập test được giữ nguyên cho đến cuối cùng để đánh giá khách quan.
        #
        # random_state cố định để kết quả tái lập được.
        # n_jobs=-1 để tận dụng tất cả CPU lõi có sẵn.
        rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params).fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        ic, rank_ic, t_stat = information_coefficient(y_val_arr, y_pred)
        diracc = directional_accuracy(y_val_arr, y_pred)
        # Lưu toàn bộ kết quả để sau này có thể xem bảng tuning đầy đủ,
        # không chỉ xem đúng model thắng cuộc.
        rows.append({**params, "val_IC": ic, "val_RankIC": rank_ic,
                     "val_IC_t": t_stat, "val_DirAcc": diracc})
        # Project chọn RF tốt nhất theo validation IC chứ không chỉ RMSE, vì
        # trong tài chính việc xếp hạng/đúng hướng return quan trọng hơn sai số số học.
        if not np.isnan(ic) and ic > best_score:
            best_score, best_model, best_params = ic, rf, params

    results_df = pd.DataFrame(rows).sort_values("val_IC", ascending=False).reset_index(drop=True)
    # Trả về cả bảng kết quả để notebook có thể in ra và giải thích toàn bộ quá trình tuning.
    return best_model, best_params, results_df
