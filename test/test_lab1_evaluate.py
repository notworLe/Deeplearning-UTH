"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 04: `src.vibe_scripts.lab1.evaluate`."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vibe_scripts.lab1.evaluate import (
    evaluate_checkpoint,
    run_benchmark,
    verify_save_load_integrity,
)
from src.vibe_scripts.lab1.train import train_experiment


def _make_dummy_test_loader(n_samples: int = 40, batch_size: int = 16) -> DataLoader:
    torch.manual_seed(248)
    x = torch.randn(n_samples, 1, 28, 28)
    y = torch.arange(n_samples) % 10
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _create_dummy_exp(exp_dir: Path, model_name: str = "logistic", aug_enabled: bool = False) -> Path:
    loader = _make_dummy_test_loader()
    train_experiment(
        model_name=model_name,
        aug_enabled=aug_enabled,
        epochs=1,
        output_dir=exp_dir,
        device=torch.device("cpu"),
        train_loader=loader,
        val_loader=loader,
        norm_stats={"mean": 0.286, "std": 0.353},
    )
    return exp_dir / "best.pt"


@pytest.mark.smoke
def test_evaluate_smoke_single_checkpoint(tmp_path: Path):
    """Smoke Test (< 5s): Đánh giá dummy checkpoint trên 40 mẫu không crash."""
    ckpt_path = _create_dummy_exp(tmp_path / "logistic_no_aug")
    test_loader = _make_dummy_test_loader()
    res = evaluate_checkpoint(ckpt_path, test_loader, device=torch.device("cpu"), warmup_batches=1)
    assert res["model_name"] == "logistic"
    assert res["num_params"] == 7850
    assert 0.0 <= res["test_acc"] <= 1.0


def test_confusion_matrix_and_metric_invariants(tmp_path: Path):
    """Normal Test: CM tổng đúng 40 mẫu, Accuracy khớp diag(CM)/sum(CM), P/R/F1 thuộc [0, 1]."""
    ckpt_path = _create_dummy_exp(tmp_path / "cnn_no_aug", model_name="cnn")
    test_loader = _make_dummy_test_loader(n_samples=40)
    res = evaluate_checkpoint(ckpt_path, test_loader, device=torch.device("cpu"), warmup_batches=1)

    cm = np.array(res["confusion_matrix"])
    assert cm.shape == (10, 10)
    assert int(cm.sum()) == 40
    assert abs(res["test_acc"] - float(np.trace(cm) / cm.sum())) < 1e-7

    for row in res["per_class"]:
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0


def test_save_load_integrity_within_tolerance(tmp_path: Path):
    """Normal Test: Sai số logits sau Save/Load < 1e-6."""
    ckpt_path = _create_dummy_exp(tmp_path / "mlp_no_aug", model_name="mlp")
    sample_batch = torch.randn(4, 1, 28, 28)
    diff = verify_save_load_integrity(ckpt_path, sample_batch, device=torch.device("cpu"))
    assert diff < 1e-6


def test_run_benchmark_outputs_all_artifacts(tmp_path: Path):
    """Normal Test: `run_benchmark()` xuất đủ JSON, CSV và toàn bộ đồ thị PNG."""
    exp_root = tmp_path / "exps"
    _create_dummy_exp(exp_root / "logistic_no_aug", model_name="logistic", aug_enabled=False)
    _create_dummy_exp(exp_root / "cnn_aug", model_name="cnn", aug_enabled=True)

    test_loader = _make_dummy_test_loader(n_samples=40)
    summary = run_benchmark(experiments_dir=exp_root, test_loader=test_loader, device=torch.device("cpu"))

    assert len(summary["benchmark_rows"]) == 2
    assert (exp_root / "evaluation" / "benchmark_summary.json").exists()
    assert (exp_root / "evaluation" / "benchmark_table.csv").exists()
    assert (exp_root / "evaluation" / "per_class_metrics.json").exists()
    assert (exp_root / "predictions" / "learning_curves_comparison.png").exists()
    assert (exp_root / "predictions" / "per_class_f1_comparison.png").exists()
    # Learning curve riêng từng thí nghiệm
    assert (exp_root / "predictions" / "learning_curve_logistic_no_aug.png").exists()
    assert (exp_root / "predictions" / "learning_curve_cnn_aug.png").exists()
    assert (exp_root / "predictions" / "confusion_matrix_cnn_aug.png").exists()
    assert (exp_root / "predictions" / "predictions_gallery_cnn_aug.png").exists()
    parsed = json.loads((exp_root / "evaluation" / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert len(parsed["benchmark_rows"]) == 2
