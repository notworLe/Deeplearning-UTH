"""Thin CLI entrypoint đánh giá Final Test (1 lần), Single-Run Eval và 4-Step Error Analysis cho Lab 2 v2."""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab2_v2.config as cfg
from src.vibe_scripts.lab2_v2.data import prepare_oxford_pets
from src.vibe_scripts.lab2_v2.evaluator import (
    evaluate_final_test,
    evaluate_run_dir,
    run_deep_error_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 2 v2 Evaluation CLI — Single Run Eval, Final Test Comparison, hoặc 4-Step Error Analysis"
    )
    parser.add_argument("--run-id", type=str, default=None, help="Đánh giá riêng 1 run cụ thể.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Tập đánh giá cho --run-id (mặc định: val).")
    parser.add_argument("--baseline-run-id", type=str, default="resnet18_scratch", help="Run ID của Baseline Scratch.")
    parser.add_argument("--best-run-id", type=str, default=None, help="Run ID thắng cuộc (mặc định đọc từ ablation_summary.json).")
    parser.add_argument("--error-analysis", action="store_true", help="Chạy Error Analysis 4 bước định lượng trên tập Test.")
    return parser.parse_args()


def _resolve_winning_run_id() -> str:
    summary_path = cfg.ABLATION_DIR / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {summary_path}. Hãy chạy train.py trước hoặc truyền --best-run-id.")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    winner = data.get("winning_model_run_id")
    if not winner:
        raise ValueError("ablation_summary.json chưa có winning_model_run_id.")
    return str(winner)


def main() -> None:
    args = parse_args()
    bundle = prepare_oxford_pets(
        raw_data_root=cfg.RAW_DATA_ROOT,
        batch_size=cfg.BATCH_SIZE,
        val_ratio_of_test=cfg.VAL_RATIO_OF_TEST,
        seed=cfg.SEED,
        image_size=cfg.IMAGE_SIZE,
        resize_size=cfg.RESIZE_SIZE,
        num_workers=cfg.WOKER,
        split_file=cfg.SPLIT_FILE,
        download=False,
    )

    if args.run_id:
        loader = bundle.val_loader if args.split == "val" else bundle.test_loader
        result = evaluate_run_dir(
            run_dir=cfg.RUNS_DIR / args.run_id,
            eval_loader=loader,
            device=cfg.DEVICE,
            class_names=bundle.class_names,
            species_by_class=bundle.species_by_class,
            split_name=args.split,
            save_plots=True,
            tolerance=cfg.SAVE_LOAD_TOLERANCE,
        )
        print(
            f"[EVAL {args.run_id} ({args.split.upper()})] "
            f"loss={result['loss']:.4f} | acc={result['accuracy']*100:.2f}% | "
            f"macro_f1={result['macro_f1']*100:.2f}% | "
            f"species_acc={result['species_metrics']['species_accuracy']*100:.2f}% | "
            f"save_load_ok={result['save_load_parity']['passed']}"
        )
        return

    best_run_id = args.best_run_id or _resolve_winning_run_id()
    if args.error_analysis:
        ea = run_deep_error_analysis(
            run_dir=cfg.RUNS_DIR / best_run_id,
            eval_loader=bundle.test_loader,
            device=cfg.DEVICE,
            evaluation_dir=cfg.EVALUATION_DIR,
            figures_dir=cfg.FIGURES_DIR,
            class_names=bundle.class_names,
            species_by_class=bundle.species_by_class,
        )
        print(json.dumps(ea, indent=2, ensure_ascii=False))
        return

    summary = evaluate_final_test(
        runs_dir=cfg.RUNS_DIR,
        baseline_run_id=args.baseline_run_id,
        best_run_id=best_run_id,
        test_loader=bundle.test_loader,
        device=cfg.DEVICE,
        evaluation_dir=cfg.EVALUATION_DIR,
        figures_dir=cfg.FIGURES_DIR,
        class_names=bundle.class_names,
        species_by_class=bundle.species_by_class,
        ablation_summary_path=cfg.ABLATION_DIR / "ablation_summary.json",
        tolerance=cfg.SAVE_LOAD_TOLERANCE,
    )
    print(json.dumps(summary["comparison"], indent=2, ensure_ascii=False))
    gain = summary["gain_over_baseline"]
    print(
        f"[FINAL TEST SUMMARY] Best ({best_run_id}) vs Baseline ({args.baseline_run_id}): "
        f"Acc {gain['accuracy_delta']*100:+.2f} pp | Macro-F1 {gain['macro_f1_delta']*100:+.2f} pp"
    )


if __name__ == "__main__":
    main()
