"""CLI wrapper to pre-train all TOP20 RF artifacts outside of Streamlit.

Usage:
    python scripts/pretrain_models.py            # train only missing tickers
    python scripts/pretrain_models.py --force    # retrain everything
"""
import argparse
import sys
import time
from pathlib import Path

# Script CLI này hữu ích khi:
# - muốn train model trước khi mở Streamlit
# - muốn xem tiến độ ngay trong terminal
# - muốn retrain toàn bộ chủ động bằng cờ --force

ROOT = Path(__file__).resolve().parents[1]
# Thêm thư mục gốc của project vào sys.path để import được src.pretrain
# khi chạy trực tiếp từ terminal.
sys.path.insert(0, str(ROOT))

from src.pretrain import TOP20, train_missing


def main() -> int:
    # argparse cho phép script hỗ trợ cờ --force mà không cần sửa code.
    parser = argparse.ArgumentParser(description="Pre-train RF models for TOP20 tickers.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain every ticker even if a saved artifact exists.",
    )
    args = parser.parse_args()

    print(f"Pre-training {'all' if args.force else 'missing'} of {len(TOP20)} tickers...")

    def report(done, total, ticker):
        # Callback tiến độ đơn giản được truyền vào train_missing().
        print(f"  [{done}/{total}] {ticker} OK")

    t0 = time.time()

    # Toàn bộ logic train nằm trong src.pretrain; file này chỉ là lớp CLI bao ngoài.
    trained = train_missing(force=args.force, progress_callback=report)
    elapsed = time.time() - t0

    if not trained:
        print("Nothing to train. All artifacts already present.")
    else:
        print(f"Trained {len(trained)} ticker(s) in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    # Cho phép file chạy độc lập như một chương trình CLI.
    raise SystemExit(main())
