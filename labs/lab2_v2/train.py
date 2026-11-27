"""Thin CLI entrypoint huấn luyện, giám sát (`tqdm` + TensorBoard) và resume cho Lab 2 v2.

Constraint bắt buộc (MODEL_PIPELINE.md): Kỹ sư trực tiếp chạy script này để train.
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import labs.lab2_v2.config as cfg
from src.vibe_scripts.lab2_v2.data import prepare_oxford_pets
from src.vibe_scripts.lab2_v2.evaluator import evaluate_final_test
from src.vibe_scripts.lab2_v2.models import build_model
from src.vibe_scripts.lab2_v2.trainer import (
    run_experiment_suite,
    run_sanity_checks,
    set_global_seed,
    train_single_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab 2 v2 Training & Monitoring CLI — Oxford-IIIT Pets Pre-trained CNNs"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-all",
        action="store_true",
        help="Chạy trọn bộ 9 runs có kiểm soát (Baseline Scratch + Transfer/Fine-tune + LR/BS Ablation).",
    )
    mode.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Chỉ chạy 1 run cụ thể theo run_id trong config.py (vd: resnet18_scratch, resnet18_transfer).",
    )
    parser.add_argument("--amp", action="store_true", default=cfg.USE_AMP_DEFAULT, help="Bật Mixed Precision FP16 trên CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Tắt Mixed Precision FP16 (dùng FP32 thuần).")
    parser.add_argument("--no-tensorboard", action="store_true", help="Tắt ghi TensorBoard event logs.")
    parser.add_argument("--batch-bar", action="store_true", help="Bật thanh tqdm chi tiết từng batch trong mỗi epoch.")
    parser.add_argument("--skip-sanity", action="store_true", help="Bỏ qua bước Sanity Check trước khi train.")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint (.pt) để resume chạy tiếp (dùng kèm --run-id).")
    parser.add_argument("--extra-epochs", type=int, default=None, help="Số epoch chạy bổ sung sau checkpoint khi dùng --resume.")
    parser.add_argument("--output-run-id", type=str, default=None, help="Tên thư mục run đích riêng khi dùng --resume.")
    parser.add_argument("--skip-final-test", action="store_true", help="Không tự động chạy Final Test sau khi kết thúc chuỗi --run-all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(cfg.SEED)

    print(
        f"[Lab 2 v2 Train] Device: {cfg.DEVICE} | AMP: {args.amp} | Seed: {cfg.SEED} | "
        f"Output: {cfg.EXPERIMENT_DIR}"
    )

    # Sanity check nhẹ trên ResNet18 scratch trước khi bắt đầu
    if not args.skip_sanity:
        sanity_bundle = prepare_oxford_pets(
            raw_data_root=cfg.RAW_DATA_ROOT,
            batch_size=16,
            val_ratio_of_test=cfg.VAL_RATIO_OF_TEST,
            seed=cfg.SEED,
            image_size=cfg.IMAGE_SIZE,
            resize_size=cfg.RESIZE_SIZE,
            num_workers=cfg.WOKER,
            split_file=cfg.SPLIT_FILE,
            download=False,
        )
        sanity_model = build_model("resnet18", mode="scratch", num_classes=cfg.NUM_CLASSES, pretrained=False)
        sanity = run_sanity_checks(sanity_model, sanity_bundle.train_loader, device=cfg.DEVICE)
        print(f"[Sanity Check] {json.dumps(sanity, ensure_ascii=False)}")
        if not sanity["passed"]:
            raise RuntimeError(f"Sanity check thất bại: {sanity}")

    if args.run_id:
        spec = cfg.get_run_spec(args.run_id)
        bundle = prepare_oxford_pets(
            raw_data_root=cfg.RAW_DATA_ROOT,
            batch_size=int(spec["batch_size"]),
            val_ratio_of_test=cfg.VAL_RATIO_OF_TEST,
            seed=cfg.SEED,
            image_size=cfg.IMAGE_SIZE,
            resize_size=cfg.RESIZE_SIZE,
            num_workers=cfg.WOKER,
            split_file=cfg.SPLIT_FILE,
            download=False,
        )
        result = train_single_run(
            run_spec=spec,
            bundle=bundle,
            runs_dir=cfg.RUNS_DIR,
            device=cfg.DEVICE,
            seed=cfg.SEED,
            use_amp=args.amp,
            tb_logs_dir=cfg.TB_LOGS_DIR,
            enable_tensorboard=not args.no_tensorboard,
            show_progress=True,
            show_batch_bar=args.batch_bar,
            eval_after_train=True,
            save_eval_plots=True,
            tolerance=cfg.SAVE_LOAD_TOLERANCE,
            resume_from=Path(args.resume) if args.resume else None,
            extra_epochs=args.extra_epochs,
            output_run_id=args.output_run_id,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Mặc định: chạy trọn bộ chuỗi 9 runs
    summary = run_experiment_suite(
        experiment_specs=cfg.EXPERIMENT_SPECS,
        raw_data_root=cfg.RAW_DATA_ROOT,
        runs_dir=cfg.RUNS_DIR,
        ablation_dir=cfg.ABLATION_DIR,
        split_file=cfg.SPLIT_FILE,
        device=cfg.DEVICE,
        val_ratio_of_test=cfg.VAL_RATIO_OF_TEST,
        seed=cfg.SEED,
        image_size=cfg.IMAGE_SIZE,
        resize_size=cfg.RESIZE_SIZE,
        num_workers=cfg.WOKER,
        use_amp=args.amp,
        tb_logs_dir=cfg.TB_LOGS_DIR,
        enable_tensorboard=not args.no_tensorboard,
        show_progress=True,
        show_batch_bar=args.batch_bar,
        save_eval_plots=True,
        tolerance=cfg.SAVE_LOAD_TOLERANCE,
    )
    print(
        f"\n[Suite Completed] {summary['completed_runs']}/{summary['total_planned_runs']} runs | "
        f"Baseline={summary['baseline_run_id']} | Winner={summary['winning_model_run_id']}"
    )

    if not args.skip_final_test and summary["baseline_run_id"] and summary["winning_model_run_id"]:
        test_bundle = prepare_oxford_pets(
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
        final_out = evaluate_final_test(
            runs_dir=cfg.RUNS_DIR,
            baseline_run_id=summary["baseline_run_id"],
            best_run_id=summary["winning_model_run_id"],
            test_loader=test_bundle.test_loader,
            device=cfg.DEVICE,
            evaluation_dir=cfg.EVALUATION_DIR,
            figures_dir=cfg.FIGURES_DIR,
            class_names=test_bundle.class_names,
            species_by_class=test_bundle.species_by_class,
            ablation_summary_path=cfg.ABLATION_DIR / "ablation_summary.json",
            tolerance=cfg.SAVE_LOAD_TOLERANCE,
        )
        gain = final_out["gain_over_baseline"]
        print(
            f"[Final Test] {summary['winning_model_run_id']} vs {summary['baseline_run_id']}: "
            f"Acc {gain['accuracy_delta']*100:+.2f} pp | Macro-F1 {gain['macro_f1_delta']*100:+.2f} pp"
        )


if __name__ == "__main__":
    main()
