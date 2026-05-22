"""CLI wrapper to pre-train all TOP20 RF artifacts outside of Streamlit.

Usage:
    python scripts/pretrain_models.py            # train only missing tickers
    python scripts/pretrain_models.py --force    # retrain everything
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pretrain import TOP20, train_missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-train RF models for TOP20 tickers.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain every ticker even if a saved artifact exists.")
    args = parser.parse_args()

    print(f"Pre-training {'all' if args.force else 'missing'} of {len(TOP20)} tickers...")

    def report(done, total, ticker):
        print(f"  [{done}/{total}] {ticker} OK")

    t0 = time.time()
    trained = train_missing(force=args.force, progress_callback=report)
    elapsed = time.time() - t0

    if not trained:
        print("Nothing to train. All artifacts already present.")
    else:
        print(f"Trained {len(trained)} ticker(s) in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
