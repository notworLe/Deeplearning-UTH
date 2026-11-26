"""Thin CLI entrypoint đánh giá Final Test (10k) hoặc đánh giá 1 run cụ thể cho Lab 1 v2."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab1_v2.config as cfg
from src.vibe_scripts.lab1_v2.data import prepare_fashion_mnist
from src.vibe_scripts.lab1_v2.evaluator import (
    evaluate_final_test,
    evaluate_run_dir,
    run_deep_error_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 1 v2 Evaluation CLI — Single Run Eval hoặc Final Test Comparison"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Nếu truyền, chỉ đánh giá riêng 1 run và lưu vào runs/<run_id>/<split>_eval/.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Tập dữ liệu dùng khi đánh giá riêng --run-id (mặc định: val).",
    )
    parser.add_argument(
        "--baseline-run-id",
        type=str,
        default="baseline_logistic",
        help="Run ID của mô hình Baseline khi chạy so sánh Final Test (mặc định: baseline_logistic).",
    )
    parser.add_argument(
        "--mlp-run-id",
        type=str,
        default=None,
        help="Run ID của MLP thắng cuộc (nếu bỏ trống sẽ tự đọc từ ablation_summary.json).",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Chạy Error Analysis 4 bước (CM chuẩn hóa %, Gallery Top-3 High/Low Conf, Audit 60 ảnh, Slice Analysis).",
    )
    return parser.parse_args()


def _resolve_winning_mlp_id() -> str:
    summary_path = cfg.ABLATION_DIR / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {summary_path}. Hãy chạy train.py trước hoặc truyền --mlp-run-id."
        )
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    winner = data.get("winning_mlp_run_id")
    if not winner:
        raise ValueError("ablation_summary.json chưa có winning_mlp_run_id.")
    return str(winner)


def main() -> None:
    args = parse_args()
    bundle = prepare_fashion_mnist(
        data_root=cfg.DATA_PATH,
        batch_size=cfg.BATCH_SIZE,
        val_size=cfg.VAL_SIZE,
        seed=cfg.SEED,
        num_workers=cfg.WOKER,
        class_names=cfg.CLASS_NAMES,
        verify_manifest=True,
        download=False,
    )

    if args.run_id:
        loader = bundle.val_loader if args.split == "val" else bundle.test_loader
        result = evaluate_run_dir(
            run_dir=cfg.RUNS_DIR / args.run_id,
            eval_loader=loader,
            device=cfg.DEVICE,
            class_names=cfg.CLASS_NAMES,
            split_name=args.split,
            save_plots=True,
            tolerance=cfg.SAVE_LOAD_TOLERANCE,
        )
        print(
            f"[EVAL {args.run_id} ({args.split.upper()})] "
            f"loss={result['loss']:.4f} | acc={result['accuracy']*100:.2f}% | "
            f"macro_f1={result['macro_f1']*100:.2f}% | "
            f"save_load_ok={result['save_load_parity']['passed']} | "
            f"saved={result['eval_dir']}"
        )
        return

    mlp_run_id = args.mlp_run_id or _resolve_winning_mlp_id()
    if args.error_analysis:
        ea_summary = run_deep_error_analysis(
            run_dir=cfg.RUNS_DIR / mlp_run_id,
            eval_loader=bundle.test_loader,
            device=cfg.DEVICE,
            evaluation_dir=cfg.EVALUATION_DIR,
            figures_dir=cfg.FIGURES_DIR,
            class_names=cfg.CLASS_NAMES,
        )
        print(json.dumps(ea_summary, indent=2, ensure_ascii=False))
        return

    summary = evaluate_final_test(
        runs_dir=cfg.RUNS_DIR,
        baseline_run_id=args.baseline_run_id,
        best_mlp_run_id=mlp_run_id,
        test_loader=bundle.test_loader,
        device=cfg.DEVICE,
        evaluation_dir=cfg.EVALUATION_DIR,
        figures_dir=cfg.FIGURES_DIR,
        class_names=cfg.CLASS_NAMES,
        ablation_summary_path=cfg.ABLATION_DIR / "ablation_summary.json",
        tolerance=cfg.SAVE_LOAD_TOLERANCE,
    )
    print(json.dumps(summary["comparison"], indent=2, ensure_ascii=False))
    gain = summary["mlp_gain_over_baseline"]
    print(
        f"[FINAL TEST SUMMARY] Best MLP ({mlp_run_id}) vs Baseline ({args.baseline_run_id}): "
        f"Acc {gain['accuracy_delta']*100:+.2f} pp | Macro-F1 {gain['macro_f1_delta']*100:+.2f} pp"
    )


if __name__ == "__main__":
    main()
