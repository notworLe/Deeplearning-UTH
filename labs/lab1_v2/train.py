"""Thin CLI entrypoint huấn luyện & giám sát (Training + Monitoring) cho Lab 1 v2.

Phân tách trách nhiệm (SRP):
- Data Collection / Cleaning / EDA nằm riêng tại `labs/lab1_v2/prepare_data.py`.
- File này chỉ phụ trách: Sanity Check -> Training + Training Monitoring (`tqdm` bar log:
  Loss, Accuracy, Learning Rate, Throughput, GPU VRAM lưu vào `runs/<run_id>/historys/history.*`) -> Per-Run Eval vào
  `experiments/lab1_v2/runs/<run_id>/eval/` -> Dừng sớm khi MLP đạt ngưỡng tốt -> Final Test (10k).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab1_v2.config as cfg
from src.vibe_scripts.lab1_v2.data import FashionMNISTBundle, prepare_fashion_mnist
from src.vibe_scripts.lab1_v2.evaluator import evaluate_final_test
from src.vibe_scripts.lab1_v2.mlp import build_model
from src.vibe_scripts.lab1_v2.trainer import (
    run_experiment_suite,
    run_sanity_checks,
    set_global_seed,
    train_single_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 1 v2 Training & Monitoring CLI — FashionMNIST Baseline vs MLP Ablation"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-all",
        action="store_true",
        help="Chạy chuỗi 12 runs có kiểm soát (mặc định nếu không truyền --run-id).",
    )
    mode.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Chỉ chạy 1 run cụ thể theo run_id trong config.py (vd: baseline_logistic, mlp_default).",
    )
    parser.add_argument(
        "--target-acc",
        type=float,
        default=cfg.TARGET_MLP_VAL_ACC,
        help=f"Ngưỡng Val Accuracy để kết thúc sớm chuỗi thí nghiệm (mặc định: {cfg.TARGET_MLP_VAL_ACC}).",
    )
    parser.add_argument(
        "--target-f1",
        type=float,
        default=cfg.TARGET_MLP_VAL_F1,
        help=f"Ngưỡng Val Macro-F1 để kết thúc sớm chuỗi thí nghiệm (mặc định: {cfg.TARGET_MLP_VAL_F1}).",
    )
    parser.add_argument(
        "--no-stop-on-target",
        action="store_true",
        help="Tắt cơ chế dừng sớm để chạy đủ toàn bộ 12 runs ablation.",
    )
    parser.add_argument(
        "--batch-bar",
        action="store_true",
        help="Bật thêm thanh tiến độ chi tiết theo từng batch bên trong mỗi epoch.",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Bỏ qua bước Sanity Check (initial loss + overfit 1 batch) trước khi train.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Đường dẫn checkpoint (.pt) để resume chạy tiếp (dùng kèm --run-id).",
    )
    parser.add_argument(
        "--extra-epochs",
        type=int,
        default=None,
        help="Số epoch chạy bổ sung sau checkpoint khi dùng --resume.",
    )
    parser.add_argument(
        "--output-run-id",
        type=str,
        default=None,
        help="Tên thư mục run đích riêng khi dùng --resume (tránh ghi đè run gốc).",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Không tự động chạy Final Test Evaluation (10k) sau khi chuỗi thí nghiệm kết thúc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(cfg.SEED)
    print(f"[Lab 1 v2 Train] Device: {cfg.DEVICE} | Seed: {cfg.SEED} | Output: {cfg.EXPERIMENT_DIR}")

    # Cache bundle theo batch_size để tránh đọc lại đĩa nhiều lần
    bundle_cache: Dict[int, FashionMNISTBundle] = {}

    def get_bundle(batch_size: int) -> FashionMNISTBundle:
        if batch_size not in bundle_cache:
            bundle_cache[batch_size] = prepare_fashion_mnist(
                data_root=cfg.DATA_PATH,
                batch_size=batch_size,
                val_size=cfg.VAL_SIZE,
                seed=cfg.SEED,
                num_workers=cfg.WOKER,
                class_names=cfg.CLASS_NAMES,
                verify_manifest=True,
                download=False,
            )
        return bundle_cache[batch_size]

    # 1. Pre-training Sanity Checks (Initial Loss & Overfit 1 Batch)
    if not args.skip_sanity:
        base_bundle = get_bundle(cfg.SANITY_BATCH_SIZE)
        sample_x, sample_y = next(iter(base_bundle.train_loader))
        sanity_model = build_model(
            model_type="mlp",
            input_shape=cfg.IMAGE_SHAPE,
            num_classes=cfg.NUM_CLASSES,
            hidden_dims=cfg.DEFAULT_HIDDEN_DIMS,
            dropout=0.0,
        )
        sanity = run_sanity_checks(
            model=sanity_model,
            sample_batch=sample_x,
            sample_targets=sample_y,
            device=cfg.DEVICE,
            max_steps=cfg.SANITY_MAX_STEPS,
            target_acc=cfg.SANITY_OVERFIT_ACC,
            num_classes=cfg.NUM_CLASSES,
        )
        print(
            f"[SANITY CHECK] init_loss={sanity['initial_loss']:.4f} "
            f"(expected~{sanity['expected_random_loss']:.4f}, ok={sanity['initial_loss_ok']}) | "
            f"overfit_1_batch={sanity['overfit_final_acc']*100:.1f}% "
            f"in {sanity['overfit_steps']} steps (passed={sanity['overfit_passed']})"
        )
        if not sanity["overfit_passed"]:
            raise RuntimeError("Sanity Check thất bại: mô hình không overfit được 1 batch nhỏ!")

    # 2. Single Run Mode (hỗ trợ --resume + --extra-epochs + --output-run-id)
    if args.run_id:
        spec = cfg.get_run_spec(args.run_id)
        if args.output_run_id:
            spec["run_id"] = args.output_run_id
        bundle = get_bundle(int(spec["batch_size"]))
        result = train_single_run(
            run_spec=spec,
            bundle=bundle,
            runs_dir=cfg.RUNS_DIR,
            device=cfg.DEVICE,
            seed=cfg.SEED,
            input_shape=cfg.IMAGE_SHAPE,
            num_classes=cfg.NUM_CLASSES,
            show_progress=True,
            show_batch_bar=args.batch_bar,
            eval_after_train=True,
            save_eval_plots=True,
            tolerance=cfg.SAVE_LOAD_TOLERANCE,
            resume_from=args.resume,
            extra_epochs=args.extra_epochs,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # 3. Controlled 12-Run Suite (Train + Monitoring -> Immediate Val Eval -> Early Stop on Good MLP)
    stop_on_target = cfg.STOP_ON_TARGET_DEFAULT and not args.no_stop_on_target
    suite_summary = run_experiment_suite(
        run_specs=cfg.EXPERIMENT_SPECS,
        bundle_factory=get_bundle,
        runs_dir=cfg.RUNS_DIR,
        ablation_dir=cfg.ABLATION_DIR,
        device=cfg.DEVICE,
        seed=cfg.SEED,
        input_shape=cfg.IMAGE_SHAPE,
        num_classes=cfg.NUM_CLASSES,
        stop_on_target=stop_on_target,
        target_val_acc=args.target_acc,
        target_val_f1=args.target_f1,
        show_progress=True,
        show_batch_bar=args.batch_bar,
        save_eval_plots=True,
        tolerance=cfg.SAVE_LOAD_TOLERANCE,
    )

    print(
        f"\n[SUITE COMPLETED] {suite_summary['completed_runs']}/{suite_summary['total_planned_runs']} runs | "
        f"stopped_early={suite_summary['stopped_early']} | "
        f"baseline={suite_summary['baseline_run_id']} | "
        f"winning_mlp={suite_summary['winning_mlp_run_id']}"
    )

    # 4. Single Held-Out Final Test Evaluation (Baseline vs Winning MLP)
    if (
        not args.skip_final_test
        and suite_summary.get("baseline_run_id")
        and suite_summary.get("winning_mlp_run_id")
    ):
        eval_bundle = get_bundle(cfg.BATCH_SIZE)
        test_summary = evaluate_final_test(
            runs_dir=cfg.RUNS_DIR,
            baseline_run_id=suite_summary["baseline_run_id"],
            best_mlp_run_id=suite_summary["winning_mlp_run_id"],
            test_loader=eval_bundle.test_loader,
            device=cfg.DEVICE,
            evaluation_dir=cfg.EVALUATION_DIR,
            figures_dir=cfg.FIGURES_DIR,
            class_names=cfg.CLASS_NAMES,
            ablation_summary_path=cfg.ABLATION_DIR / "ablation_summary.json",
            tolerance=cfg.SAVE_LOAD_TOLERANCE,
        )
        gain = test_summary["mlp_gain_over_baseline"]
        print(
            f"[FINAL TEST (10k) DONE] "
            f"Acc gain={gain['accuracy_delta']*100:+.2f} pp | "
            f"Macro-F1 gain={gain['macro_f1_delta']*100:+.2f} pp | "
            f"Saved -> {cfg.EVALUATION_DIR}"
        )


if __name__ == "__main__":
    main()
