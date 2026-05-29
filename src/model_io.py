"""Save and load per-ticker RF artifacts under <repo>/models/."""
# Module quản lý save/load artifacts (model + scaler + metadata) cho từng ticker.
# Mỗi ticker tương ứng 1 file .pkl, kèm 1 file _manifest.json đóng vai trò index
# nhẹ để check nhanh "đã train chưa" mà không cần load full pickle.
import json
from datetime import datetime
from pathlib import Path

import joblib

# Các path tuyệt đối tính từ repo root (file này nằm ở <repo>/src/model_io.py
# nên parents[1] = repo root).
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "_manifest.json"


def _artifact_path(ticker: str) -> Path:
    # Helper: trả về path đầy đủ tới file .pkl của 1 ticker.
    return MODELS_DIR / f"{ticker}.pkl"


def _read_manifest() -> dict:
    # Đọc manifest JSON. Trả {} nếu file chưa tồn tại hoặc bị corrupt — không
    # crash để code gọi vẫn chạy tiếp được (vd: lần đầu chưa có file).
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(data: dict) -> None:
    # Ghi manifest JSON (tạo thư mục models/ nếu chưa có).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_artifact(ticker: str, model, scaler, feature_cols, win_params,
                  horizon: int = 5, extra: dict | None = None) -> Path:
    """Pickle the trained artifact and refresh _manifest.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Payload pickle: gom model + scaler + danh sách feature_cols + winsorize
    # bounds + horizon + timestamp lúc train. Streamlit load lại cần đủ bộ này
    # để inference đúng như lúc train.
    payload = {
        "ticker": ticker,
        "model": model,
        "scaler": scaler,
        "feature_cols": list(feature_cols),
        "win_params": tuple(float(x) for x in win_params),
        "horizon": int(horizon),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    # extra: dict optional cho metadata bổ sung (vd: metrics backtest, version...).
    if extra:
        payload.update(extra)

    path = _artifact_path(ticker)
    joblib.dump(payload, path)

    # Sau khi pickle thành công mới update manifest — đảm bảo manifest luôn
    # khớp với file thực tế trên disk. Manifest chỉ giữ thông tin nhẹ (timestamp,
    # horizon, tên file) để check trạng thái mà không cần load full model.
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
    # Load pickle về dict đầy đủ. Raise FileNotFoundError nếu ticker chưa được
    # train — caller (streamlit) sẽ bắt và trigger train_missing.
    path = _artifact_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {ticker} at {path}")
    return joblib.load(path)


def has_artifact(ticker: str) -> bool:
    # Check nhanh xem ticker đã có file .pkl chưa (chỉ stat file, không load).
    return _artifact_path(ticker).exists()


def list_trained() -> set[str]:
    """Tickers that have a .pkl file on disk."""
    # Trả set các ticker đã train (scan tên file .pkl trong models/).
    if not MODELS_DIR.exists():
        return set()
    return {p.stem for p in MODELS_DIR.glob("*.pkl")}


def manifest_status(tickers: list[str]) -> dict[str, str | None]:
    """For each ticker, return its trained_at timestamp or None if missing."""
    # Với mỗi ticker: trả timestamp lúc train nếu đã có file, hoặc None nếu chưa.
    # Streamlit dùng hàm này để hiển thị "Models ready: X/20" và để biết những
    # ticker nào còn thiếu cần train bổ sung.
    manifest = _read_manifest()
    trained = list_trained()
    out: dict[str, str | None] = {}
    for t in tickers:
        if t in trained:
            out[t] = manifest.get(t, {}).get("trained_at")
        else:
            out[t] = None
    return out


def delete_all() -> int:
    """Delete every .pkl and the manifest. Returns how many files were removed."""
    # Xoá sạch mọi file .pkl + manifest. Streamlit gọi hàm này khi user bấm
    # nút "Retrain all" để buộc train lại toàn bộ TOP20 từ đầu.
    if not MODELS_DIR.exists():
        return 0
    n = 0
    for p in MODELS_DIR.glob("*.pkl"):
        p.unlink()
        n += 1
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    return n
