"""Dual Unit Tests (Smoke + Normal) cho trainer, per-run eval, early-stop suite, Final Test & inference.

Toàn bộ bài test chạy trên tập synthetic siêu nhỏ (<= 64 mẫu, 1-2 epochs) trong `tmp_path`,
KHÔNG chạy training thật lên FashionMNIST và KHÔNG ghi vào `experiments/lab1_v2/`.
"""

import json
from pathlib import Path

import nbformat
import pytest
import torch

import labs.lab1_v2.config as cfg
from src.vibe_scripts.lab1_v2.data import FashionMNISTBundle, build_dataloaders_from_tensors
from src.vibe_scripts.lab1_v2.evaluator import evaluate_final_test
from src.vibe_scripts.lab1_v2.inference import FashionPredictor
from src.vibe_scripts.lab1_v2.mlp import build_model
from src.vibe_scripts.lab1_v2.trainer import (
    run_experiment_suite,
    run_sanity_checks,
    set_global_seed,
    train_single_run,
)


def _make_synthetic_bundle(batch_size: int = 16, seed: int = 248) -> FashionMNISTBundle:
    """Tạo bộ ảnh synthetic tách biệt nhẹ theo lớp để test nhanh toàn bộ luồng train/eval/infer."""
    gen = torch.Generator().manual_seed(seed)
    tr_y = torch.arange(40, dtype=torch.long) % 10
    va_y = torch.arange(20, dtype=torch.long) % 10
    te_y = torch.arange(20, dtype=torch.long) % 10

    def _imgs_for_labels(labels: torch.Tensor) -> torch.Tensor:
        imgs = torch.randint(0, 40, (len(labels), 28, 28), dtype=torch.uint8, generator=gen)
        for i, lbl in enumerate(labels.tolist()):
            row = (lbl % 5) * 5
            col = (lbl // 5) * 12
            imgs[i, row : row + 4, col : col + 10] = 220
        return imgs

    return build_dataloaders_from_tensors(
        train_images=_imgs_for_labels(tr_y),
        train_targets=tr_y,
        val_images=_imgs_for_labels(va_y),
        val_targets=va_y,
        test_images=_imgs_for_labels(te_y),
        test_targets=te_y,
        batch_size=batch_size,
        seed=seed,
        num_workers=0,
        class_names=cfg.CLASS_NAMES,
    )


@pytest.mark.smoke
def test_single_run_per_run_eval_and_inference_smoke(tmp_path: Path) -> None:
    """Smoke: train 1 epoch synthetic -> sinh best.pt/last.pt/historys/history.* + eval/ ngay trong folder run."""
    bundle = _make_synthetic_bundle(batch_size=16)
    spec = {
        "run_id": "smoke_mlp",
        "group": "Smoke",
        "axis": "smoke",
        "model_type": "mlp",
        "hidden_dims": (32,),
        "dropout": 0.1,
        "optimizer": "adam",
        "lr": 1e-2,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "batch_size": 16,
        "epochs": 1,
        "is_baseline": False,
    }

    runs_dir = tmp_path / "runs"
    result = train_single_run(
        run_spec=spec,
        bundle=bundle,
        runs_dir=runs_dir,
        device="cpu",
        seed=248,
        show_progress=False,
        eval_after_train=True,
        save_eval_plots=True,
    )

    run_dir = runs_dir / "smoke_mlp"
    eval_dir = run_dir / "eval"
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "historys" / "history.json").exists()
    assert (run_dir / "historys" / "history.csv").exists()
    assert (run_dir / "config.json").exists()
    assert not list(run_dir.glob("*.tmp"))

    hist = json.loads((run_dir / "historys" / "history.json").read_text(encoding="utf-8"))
    ep0 = hist["epochs"][0]
    for mon_key in ("train_loss", "val_loss", "train_acc", "val_acc", "lr", "throughput_sps", "gpu_mem_mb"):
        assert mon_key in ep0

    # Kết quả eval lưu ngay bên trong thư mục của run
    assert (eval_dir / "eval_metrics.json").exists()
    assert (eval_dir / "loss_curve.png").exists()
    assert (eval_dir / "accuracy_curve.png").exists()
    assert (eval_dir / "confusion_matrix.png").exists()
    assert (eval_dir / "predicted_vs_actual.png").exists()
    assert result["save_load_passed"] is True

    # Predictor nạp checkpoint và dự đoán Top-3 không lỗi
    predictor = FashionPredictor.from_checkpoint(run_dir / "best.pt", device="cpu")
    sample_x, _ = bundle.test_dataset[0]
    pred = predictor.predict_batch(sample_x.unsqueeze(0), top_k=3)[0]
    assert len(pred["top_k"]) == 3
    assert pred["requires_grad"] is False


def test_sanity_checks_initial_loss_and_overfit_one_batch() -> None:
    """Normal: kiểm chứng initial loss ≈ ln(10) và khả năng overfit 1 batch nhỏ >= 95%."""
    set_global_seed(248)
    bundle = _make_synthetic_bundle(batch_size=20)
    sample_x, sample_y = next(iter(bundle.train_loader))
    model = build_model("mlp", hidden_dims=(64, 32), dropout=0.0)

    sanity = run_sanity_checks(
        model=model,
        sample_batch=sample_x,
        sample_targets=sample_y,
        device="cpu",
        lr=1e-2,
        max_steps=150,
        target_acc=0.95,
    )
    assert sanity["initial_loss_ok"] is True
    assert sanity["overfit_passed"] is True
    assert sanity["overfit_final_acc"] >= 0.95


def test_suite_early_stop_final_test_and_inference_invariants(tmp_path: Path) -> None:
    """Normal: xác thực dừng sớm sau khi MLP đạt target, Final Test 10k mock và bất biến inference."""
    mini_specs = [
        {
            "run_id": "baseline_logistic",
            "group": "Baseline",
            "axis": "baseline",
            "model_type": "logistic",
            "hidden_dims": (),
            "dropout": 0.0,
            "optimizer": "adam",
            "lr": 1e-2,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "batch_size": 16,
            "epochs": 2,
            "is_baseline": True,
        },
        {
            "run_id": "mlp_default",
            "group": "MLP Anchor",
            "axis": "anchor",
            "model_type": "mlp",
            "hidden_dims": (32,),
            "dropout": 0.0,
            "optimizer": "adam",
            "lr": 1e-2,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "batch_size": 16,
            "epochs": 2,
            "is_baseline": False,
        },
        {
            "run_id": "mlp_should_be_skipped",
            "group": "Skipped",
            "axis": "hidden_dims",
            "model_type": "mlp",
            "hidden_dims": (64,),
            "dropout": 0.0,
            "optimizer": "adam",
            "lr": 1e-2,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "batch_size": 16,
            "epochs": 2,
            "is_baseline": False,
        },
    ]

    runs_dir = tmp_path / "runs"
    ablation_dir = tmp_path / "ablation"
    eval_dir = tmp_path / "evaluation"
    figs_dir = tmp_path / "figures"

    # Đặt ngưỡng dễ đạt để kiểm chứng: luôn chạy baseline (run 1), gặp mlp_default đạt ngưỡng -> dừng ở run 2!
    suite = run_experiment_suite(
        run_specs=mini_specs,
        bundle_factory=lambda bs: _make_synthetic_bundle(batch_size=bs),
        runs_dir=runs_dir,
        ablation_dir=ablation_dir,
        device="cpu",
        seed=248,
        stop_on_target=True,
        target_val_acc=0.0,
        target_val_f1=0.0,
        show_progress=False,
        save_eval_plots=True,
    )

    assert suite["stopped_early"] is True
    assert suite["completed_runs"] == 2
    assert suite["baseline_run_id"] == "baseline_logistic"
    assert suite["winning_mlp_run_id"] == "mlp_default"
    assert not (runs_dir / "mlp_should_be_skipped").exists()
    assert (runs_dir / "baseline_logistic" / "eval" / "eval_metrics.json").exists()
    assert (runs_dir / "mlp_default" / "eval" / "eval_metrics.json").exists()
    assert (ablation_dir / "ablation_summary.json").exists()
    assert (ablation_dir / "ablation_table.csv").exists()

    # Final Test comparison
    bundle = _make_synthetic_bundle(batch_size=16)
    test_summary = evaluate_final_test(
        runs_dir=runs_dir,
        baseline_run_id="baseline_logistic",
        best_mlp_run_id="mlp_default",
        test_loader=bundle.test_loader,
        device="cpu",
        evaluation_dir=eval_dir,
        figures_dir=figs_dir,
        class_names=cfg.CLASS_NAMES,
        ablation_summary_path=ablation_dir / "ablation_summary.json",
    )

    assert (eval_dir / "test_comparison.json").exists()
    assert (eval_dir / "test_comparison.csv").exists()
    assert (eval_dir / "per_class_metrics.json").exists()
    assert (eval_dir / "error_analysis.json").exists()
    for row in test_summary["comparison"]:
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert 0.0 <= row["test_macro_f1"] <= 1.0
        assert row["save_load_passed"] is True
        assert row["save_load_max_diff"] < 1e-6

    for fig_path in test_summary["figures"].values():
        assert fig_path is not None and Path(fig_path).exists()

    # Kiểm chứng Confusion Matrix sum == tổng mẫu test & test_eval KHÔNG chứa learning curves
    mlp_test_dir = runs_dir / "mlp_default" / "test_eval"
    assert not (mlp_test_dir / "loss_curve.png").exists()
    assert not (mlp_test_dir / "accuracy_curve.png").exists()
    mlp_test_eval = json.loads((mlp_test_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    cm_sum = sum(sum(r) for r in mlp_test_eval["confusion_matrix"])
    assert cm_sum == len(bundle.test_dataset)

    # Kiểm chứng Inference invariants: Softmax sum = 1, Top-3 giảm dần, latency > 0
    predictor = FashionPredictor.from_checkpoint(runs_dir / "mlp_default" / "best.pt", device="cpu")
    pred = predictor.predict_image(bundle.test_dataset[0][0], top_k=3)
    assert abs(sum(pred["probabilities"]) - 1.0) < 1e-5
    top_probs = [item["probability"] for item in pred["top_k"]]
    assert top_probs == sorted(top_probs, reverse=True)
    bench = predictor.benchmark_latency(batch_size=1, warmup_runs=2, timed_runs=5)
    assert bench["mean_ms"] > 0.0
    assert bench["checkpoint_size_kb"] > 0.0


def test_resume_training_continues_epochs_and_preserves_history(tmp_path: Path) -> None:
    """Normal: xác nhận resume từ last.pt nối liên tục epochs 1..3, phục hồi optimizer & giữ best.pt."""
    bundle = _make_synthetic_bundle(batch_size=16)
    base_spec = {
        "run_id": "sgd_base",
        "group": "Resume Test",
        "axis": "optimizer",
        "model_type": "mlp",
        "hidden_dims": (32,),
        "dropout": 0.0,
        "optimizer": "sgd",
        "lr": 1e-2,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "batch_size": 16,
        "epochs": 1,
        "is_baseline": False,
    }
    runs_dir = tmp_path / "runs"
    train_single_run(
        run_spec=base_spec,
        bundle=bundle,
        runs_dir=runs_dir,
        device="cpu",
        seed=248,
        show_progress=False,
        eval_after_train=False,
    )

    resumed_spec = dict(base_spec, run_id="sgd_resumed")
    res = train_single_run(
        run_spec=resumed_spec,
        bundle=bundle,
        runs_dir=runs_dir,
        device="cpu",
        seed=248,
        show_progress=False,
        eval_after_train=True,
        save_eval_plots=True,
        resume_from=runs_dir / "sgd_base" / "last.pt",
        extra_epochs=2,
    )
    assert res["epochs"] == 3
    hist = json.loads((runs_dir / "sgd_resumed" / "historys" / "history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in hist["epochs"]] == [1, 2, 3]
    assert hist["resumed"]["optimizer_state_restored"] is True
    assert (runs_dir / "sgd_resumed" / "best.pt").exists()
    assert (runs_dir / "sgd_resumed" / "eval" / "eval_metrics.json").exists()


def test_experiment_notebook_schema_valid() -> None:
    """Normal: xác nhận labs/lab1_v2/experiment.ipynb hợp lệ chuẩn nbformat v4."""
    nb_path = cfg.LAB_DIR / "experiment.ipynb"
    assert nb_path.exists()
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 6
