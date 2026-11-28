"""Thin CLI entrypoint cho Exercise 2: IMDB Audit/EDA, Baseline Val & DistilBERT Fine-Tuning.

Constraint bắt buộc (MODEL_PIPELINE.md v1.4.0): Kỹ sư trực tiếp chạy script này để fine-tune.
Script này KHÔNG chạm vào tập Test (tập Test chỉ chấm 1 lần riêng qua `eval_ex2.py`).
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import labs.lab3_v2.config as cfg
from src.vibe_scripts.lab3_v2.data import load_and_split_imdb, run_imdb_data_audit_and_eda
from src.vibe_scripts.lab3_v2.ex2_finetune import run_ex2_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 3 v2 Exercise 2 — IMDB Data Audit/EDA, Baseline Val & DistilBERT Fine-Tuning"
    )
    parser.add_argument("--model-name", type=str, default=cfg.EX2_MODEL_NAME, help="Pretrained backbone ID.")
    parser.add_argument("--train-size", type=int, default=cfg.TRAIN_SUBSET_SIZE, help="Số mẫu Train (mặc định: 4000).")
    parser.add_argument("--val-size", type=int, default=cfg.VAL_SUBSET_SIZE, help="Số mẫu Val (mặc định: 1000).")
    parser.add_argument("--test-size", type=int, default=cfg.TEST_SUBSET_SIZE, help="Số mẫu Test khóa kín (mặc định: 1000).")
    parser.add_argument("--max-length", type=int, default=cfg.MAX_LENGTH, help="Độ dài cắt token (mặc định: 256).")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCH, help="Số epoch fine-tune (mặc định: 3).")
    parser.add_argument("--lr", type=float, default=cfg.LR, help="Learning rate (mặc định: 2e-5).")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="Train batch size (mặc định: 16).")
    parser.add_argument("--eval-batch-size", type=int, default=cfg.EVAL_BATCH_SIZE, help="Eval batch size (mặc định: 32).")
    parser.add_argument("--no-fp16", action="store_true", help="Tắt FP16 (chạy FP32 thuần).")
    parser.add_argument("--no-tensorboard", action="store_true", help="Tắt ghi TensorBoard logs.")
    parser.add_argument("--skip-eda", action="store_true", help="Bỏ qua bước xuất báo cáo/biểu đồ EDA nếu đã chạy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[Lab 3 v2 Ex2 Train] Model={args.model_name} | Split={args.train_size}/{args.val_size}/{args.test_size} | "
        f"max_len={args.max_length} | ep={args.epochs} | lr={args.lr} | bs={args.batch_size} | "
        f"fp16={not args.no_fp16} | seed={cfg.SEED}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    splits, split_meta = load_and_split_imdb(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=cfg.SEED,
        split_file=cfg.SPLIT_FILE,
    )

    if not args.skip_eda:
        eda_out = run_imdb_data_audit_and_eda(
            dataset_dict=splits,
            tokenizer=tokenizer,
            eda_dir=cfg.EDA_DIR,
            max_length=args.max_length,
            split_metadata=split_meta,
        )
        print(
            "[EDA Complete] Train truncated > "
            f"{args.max_length} tokens: {eda_out['eda_summary']['truncated_pct_above_max_length']['train']}% | "
            f"Saved to {cfg.EDA_DIR}"
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=cfg.NUM_LABELS,
        id2label=cfg.ID2LABEL,
        label2id=cfg.LABEL2ID,
    )

    summary = run_ex2_training(
        dataset_dict=splits,
        tokenizer=tokenizer,
        model=model,
        run_dir=cfg.RUN_DIR,
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        train_bs=args.batch_size,
        eval_bs=args.eval_batch_size,
        weight_decay=cfg.WEIGHT_DECAY,
        seed=cfg.SEED,
        use_fp16=not args.no_fp16,
        enable_tensorboard=not args.no_tensorboard,
        use_hf_evaluate=True,
    )

    b = summary["baseline_before_finetune"]
    a = summary["after_finetune_val"]
    print("\n=== [EX2 VALIDATION SUMMARY (Before vs After Fine-Tune)] ===")
    print(f"  Before Fine-Tune : Val Acc = {b['eval_accuracy']*100:.2f}% | Val F1 = {b['eval_f1']*100:.2f}% | Val Loss = {b['eval_loss']:.4f}")
    print(f"  After  Fine-Tune : Val Acc = {a['eval_accuracy']*100:.2f}% | Val F1 = {a['eval_f1']*100:.2f}% | Val Loss = {a['eval_loss']:.4f}")
    print(f"  Gain (After-Before): Acc {summary['val_accuracy_gain_pp']:+.2f} pp | F1 {summary['val_f1_gain_pp']:+.2f} pp")
    print(f"  Train time: {summary['train_time_sec']:.1f}s | Best model saved: {summary['best_model_dir']}")
    print("\nTiếp theo chạy đánh giá Test 1 lần duy nhất: python labs/lab3_v2/eval_ex2.py")


if __name__ == "__main__":
    main()
