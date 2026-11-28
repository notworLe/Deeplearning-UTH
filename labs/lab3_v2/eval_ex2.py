"""Thin CLI entrypoint đánh giá tập Held-Out Test (1 lần duy nhất) & Inference cho Exercise 2."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab3_v2.config as cfg
from src.vibe_scripts.lab3_v2.data import load_and_split_imdb
from src.vibe_scripts.lab3_v2.ex2_finetune import predict_texts, run_ex2_final_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 3 v2 Exercise 2 — Final Held-Out Test Evaluation (1 lần) & Live Inference"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(cfg.BEST_MODEL_DIR),
        help=f"Thư mục checkpoint đã fine-tune (mặc định: {cfg.BEST_MODEL_DIR}).",
    )
    parser.add_argument(
        "--predict",
        nargs="+",
        default=None,
        help="Dự đoán trực tiếp các câu review mới (bỏ qua chấm lại tập Test).",
    )
    parser.add_argument("--max-length", type=int, default=cfg.MAX_LENGTH, help="Độ dài cắt token (mặc định: 256).")
    parser.add_argument("--eval-batch-size", type=int, default=cfg.EVAL_BATCH_SIZE, help="Eval batch size (mặc định: 32).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint fine-tune tại {model_dir}. Hãy chạy `python labs/lab3_v2/train_ex2.py` trước."
        )

    if args.predict:
        preds = predict_texts(args.predict, model_dir=model_dir, max_length=args.max_length)
        print(json.dumps(preds, indent=2, ensure_ascii=False))
        return

    splits, _ = load_and_split_imdb(
        train_size=cfg.TRAIN_SUBSET_SIZE,
        val_size=cfg.VAL_SUBSET_SIZE,
        test_size=cfg.TEST_SUBSET_SIZE,
        seed=cfg.SEED,
        split_file=cfg.SPLIT_FILE,
    )
    metrics = run_ex2_final_test(
        test_dataset=splits["test"],
        model_dir=model_dir,
        run_dir=cfg.RUN_DIR,
        max_length=args.max_length,
        eval_bs=args.eval_batch_size,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(
        f"\n[FINAL TEST SUMMARY (n={metrics['num_samples']})] "
        f"Test Acc = {metrics['test_accuracy']*100:.2f}% | "
        f"Test F1 = {metrics['test_f1']*100:.2f}% | "
        f"Test Loss = {metrics['test_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
