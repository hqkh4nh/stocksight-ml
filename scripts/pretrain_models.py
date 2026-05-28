"""CLI wrapper to pre-train all TOP20 RF artifacts outside of Streamlit.

Usage:
    python scripts/pretrain_models.py            # chỉ train các ticker còn thiếu
    python scripts/pretrain_models.py --force    # train lại toàn bộ
"""
import argparse
import sys
import time
from pathlib import Path

# Script CLI này dành cho trường hợp muốn train model trước khi mở app.
# Lợi ích:
# - theo dõi tiến độ ngay trên terminal
# - tránh đợi app Streamlit tự bootstrap ở lần chạy đầu
# - phù hợp khi muốn chủ động train lại toàn bộ

ROOT = Path(__file__).resolve().parents[1]
# Thêm thư mục gốc của project vào sys.path để import được src.pretrain
# khi chạy script từ terminal.
sys.path.insert(0, str(ROOT))

from src.pretrain import TOP20, train_missing


def main() -> int:
    # argparse giúp script hỗ trợ cờ --force mà không cần chỉnh code.
    parser = argparse.ArgumentParser(description="Pre-train RF models for TOP20 tickers.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain every ticker even if a saved artifact exists.")
    args = parser.parse_args()

    print(f"Pre-training {'all' if args.force else 'missing'} of {len(TOP20)} tickers...")
    # Dòng này giúp người chạy biết script sẽ xử lý bao nhiêu ticker.

    def report(done, total, ticker):
        # Callback tiến độ đơn giản truyền vào src.pretrain.train_missing().
        print(f"  [{done}/{total}] {ticker} OK")

    t0 = time.time()
    # Hàm train_missing chứa toàn bộ logic train; script này chủ yếu là lớp CLI bao ngoài.
    trained = train_missing(force=args.force, progress_callback=report)
    elapsed = time.time() - t0
    # elapsed dùng để ước lượng chi phí thời gian của lần chạy hiện tại.

    if not trained:
        # Không có ticker nào cần train thêm.
        print("Nothing to train. All artifacts already present.")
    else:
        # Báo tổng thời gian để người dùng ước lượng chi phí chạy lần sau.
        print(f"Trained {len(trained)} ticker(s) in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    # Cho phép file chạy độc lập như một chương trình CLI.
    raise SystemExit(main())
