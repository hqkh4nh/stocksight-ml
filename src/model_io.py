"""Save / load pre-trained per-ticker artifacts under <repo>/models/."""
import json
from datetime import datetime
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "_manifest.json"


def _artifact_path(ticker: str) -> Path:
    return MODELS_DIR / f"{ticker}.pkl"


def _read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(data: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_artifact(ticker: str, model, scaler, feature_cols, win_params,
                  horizon: int = 5, extra: dict | None = None) -> Path:
    """Pickle the trained artifact and update _manifest.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
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
        payload.update(extra)

    path = _artifact_path(ticker)
    joblib.dump(payload, path)

    manifest = _read_manifest()
    manifest[ticker] = {
        "trained_at": payload["trained_at"],
        "horizon":    payload["horizon"],
        "file":       path.name,
    }
    _write_manifest(manifest)
    return path


def load_artifact(ticker: str) -> dict:
    """Load a previously saved artifact. Raises FileNotFoundError if missing."""
    path = _artifact_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {ticker} at {path}")
    return joblib.load(path)


def has_artifact(ticker: str) -> bool:
    return _artifact_path(ticker).exists()


def list_trained() -> set[str]:
    """Set of tickers with a .pkl on disk (manifest is treated as advisory)."""
    if not MODELS_DIR.exists():
        return set()
    return {p.stem for p in MODELS_DIR.glob("*.pkl")}


def manifest_status(tickers: list[str]) -> dict[str, str | None]:
    """Map each ticker -> trained_at iso string, or None if no artifact exists."""
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
    """Remove every .pkl + the manifest. Returns count of pkls deleted."""
    if not MODELS_DIR.exists():
        return 0
    n = 0
    for p in MODELS_DIR.glob("*.pkl"):
        p.unlink()
        n += 1
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    return n
