"""Save and load per-ticker RF artifacts under <repo>/models/."""
import json
from datetime import datetime
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
# Toàn bộ model lưu trong thư mục <repo>/models
MODELS_DIR = ROOT / "models"
# Manifest là file JSON tóm tắt tình trạng artifact.
MANIFEST_PATH = MODELS_DIR / "_manifest.json"

# File này quản lý phần lưu/truy xuất mô hình trên đĩa.
# Có thể xem đây là "tầng persistence" của project.
#
# Tại sao phải lưu artifact?
# - App Streamlit không nên train lại mỗi lần chạy
# - Dự đoán phải dùng đúng scaler và đúng thứ tự cột đã train
# - Cần metadata để biết ticker nào đã sẵn sàng


def _artifact_path(ticker: str) -> Path:
    # Mỗi ticker tương ứng một file artifact để app load đơn giản hơn.
    return MODELS_DIR / f"{ticker}.pkl"


def _read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        # Chưa có manifest nghĩa là chưa pretrain gì cả hoặc vừa xóa model.
        return {}
    try:
        # Manifest chỉ là metadata phụ trợ; nếu file này hỏng thì trả về dict rỗng
        # để app vẫn chạy thay vì lỗi toàn bộ.
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(data: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # indent=2 và sort_keys=True giúp file manifest dễ đọc khi mở bằng tay.
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_artifact(ticker: str, model, scaler, feature_cols, win_params,
                  horizon: int = 5, extra: dict | None = None) -> Path:
    """Pickle the trained artifact and refresh _manifest.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Lưu toàn bộ thông tin cần thiết để tái tạo đúng quy trình inference:
    # model, scaler, thứ tự feature, ngưỡng clipping và horizon.
    payload = {
        "ticker": ticker,
        "model": model,
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "win_params": tuple(float(x) for x in win_params),
        "horizon": int(horizon),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        # extra cho phép mở rộng payload trong tương lai mà không sửa cấu trúc cơ bản.
        payload.update(extra)

    path = _artifact_path(ticker)
    # joblib phù hợp để lưu object sklearn như model/scaler.
    joblib.dump(payload, path)

    # Manifest là bảng chỉ mục nhẹ để UI biết model nào đã tồn tại và train lúc nào
    # mà không cần mở từng file pickle.
    manifest = _read_manifest()
    manifest[ticker] = {
        "trained_at": payload["trained_at"],
        "horizon":    payload["horizon"],
        "file":       path.name,
    }
    _write_manifest(manifest)
    return path


def load_artifact(ticker: str) -> dict:
    """Load a saved artifact. Raises FileNotFoundError if it does not exist."""
    path = _artifact_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {ticker} at {path}")
    # joblib.load trả lại đúng dict đã được lưu ở save_artifact().
    return joblib.load(path)


def has_artifact(ticker: str) -> bool:
    # Kiểm tra nhanh xem ticker đã được pretrain hay chưa.
    return _artifact_path(ticker).exists()


def list_trained() -> set[str]:
    """Tickers that have a .pkl file on disk."""
    if not MODELS_DIR.exists():
        return set()
    # p.stem lấy tên file không kèm đuôi .pkl -> chính là ticker.
    return {p.stem for p in MODELS_DIR.glob("*.pkl")}


def manifest_status(tickers: list[str]) -> dict[str, str | None]:
    """For each ticker, return its trained_at timestamp or None if missing."""
    manifest = _read_manifest()
    trained = list_trained()
    out: dict[str, str | None] = {}
    for t in tickers:
        # Nếu file model có tồn tại thì lấy thêm thời điểm train từ manifest.
        if t in trained:
            out[t] = manifest.get(t, {}).get("trained_at")
        else:
            # None nghĩa là ticker này chưa có artifact sẵn sàng.
            out[t] = None
    return out


def delete_all() -> int:
    """Delete every .pkl and the manifest. Returns how many files were removed."""
    if not MODELS_DIR.exists():
        return 0
    n = 0
    for p in MODELS_DIR.glob("*.pkl"):
        # Xóa từng file artifact của ticker.
        p.unlink()
        n += 1
    if MANIFEST_PATH.exists():
        # Xóa luôn manifest để trạng thái repo quay về trước khi pretrain.
        MANIFEST_PATH.unlink()
    return n
