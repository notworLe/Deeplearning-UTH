"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 04 Lab 2: `src.vibe_scripts.lab2.evaluate`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vibe_scripts.lab2.evaluate import (
    compute_species_analysis,
    evaluate_checkpoint,
    run_benchmark,
)
from src.vibe_scripts.lab2.train import train_experiment


def _dummy_species_map(num_classes: int = 37) -> tuple[list[str], dict[int, dict]]:
    classes = [f"Cat_{i}" if i < 12 else f"dog_{i}" for i in range(num_classes)]
    smap = {
        i: {"class_id": i, "label": classes[i], "is_cat": i < 12, "species": "Cat" if i < 12 else "Dog"}
        for i in range(num_classes)
    }
    return classes, smap


def _dummy_loader(n: int = 20) -> DataLoader:
    torch.manual_seed(248)
    x = torch.randn(n, 3, 224, 224)
    y = torch.arange(n) % 37
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


@pytest.mark.smoke
def test_lab2_evaluate_smoke_single_checkpoint(tmp_path: Path) -> None:
    """Smoke: đánh giá 1 dummy checkpoint không crash."""
    loader = _dummy_loader(n=12)
    train_experiment(
        mode="transfer",
        epochs=1,
        device=torch.device("cpu"),
        experiment_dir=tmp_path,
        train_loader=loader,
        val_loader=loader,
        use_pretrained=False,
        use_amp=False,
        show_progress=False,
    )
    classes, smap = _dummy_species_map()
    res = evaluate_checkpoint(
        tmp_path / "resnet18_transfer" / "best.pt",
        loader,
        classes=classes,
        species_map=smap,
        device=torch.device("cpu"),
    )
    assert 0.0 <= res["test_acc"] <= 1.0
    assert res["save_load_max_diff"] < 1e-6


def test_species_analysis_math_invariants() -> None:
    """Normal: kiểm chứng tính chính xác error rate Cat (12) vs Dog (25) và cross-species."""
    _, smap = _dummy_species_map()
    # 10 cat samples (3 errors: 2 within-cat, 1 cat->dog), 20 dog samples (2 errors: both within-dog)
    y_true = np.array([0] * 10 + [15] * 20)
    y_pred = np.array([0] * 7 + [1, 2, 15] + [15] * 18 + [16, 17])

    stats = compute_species_analysis(y_true, y_pred, smap)
    assert stats["cat_samples"] == 10 and stats["cat_errors"] == 3
    assert stats["cat_within_species_errors"] == 2 and stats["cat_to_dog_errors"] == 1
    assert stats["dog_samples"] == 20 and stats["dog_errors"] == 2
    assert stats["dog_to_cat_errors"] == 0
    assert pytest.approx(stats["cat_error_rate"], abs=1e-6) == 0.30
    assert pytest.approx(stats["dog_error_rate"], abs=1e-6) == 0.10
    assert pytest.approx(stats["error_rate_diff_cat_minus_dog_pp"], abs=1e-4) == 20.0
    assert 0.0 <= stats["p_value"] <= 1.0


def test_run_benchmark_outputs_all_artifacts(tmp_path: Path) -> None:
    """Normal: `run_benchmark` sinh đủ JSON, CSV và biểu đồ PNG cho các run."""
    loader = _dummy_loader(n=16)
    classes, smap = _dummy_species_map()
    for mode in ("transfer", "finetune"):
        train_experiment(
            mode=mode,
            epochs=1,
            device=torch.device("cpu"),
            experiment_dir=tmp_path,
            train_loader=loader,
            val_loader=loader,
            use_pretrained=False,
            use_amp=False,
            show_progress=False,
        )

    out = run_benchmark(
        experiments_dir=tmp_path,
        run_names=("resnet18_transfer", "resnet18_finetune"),
        test_loader=loader,
        classes=classes,
        species_map=smap,
        device=torch.device("cpu"),
    )
    assert len(out["benchmark_rows"]) == 2
    assert (tmp_path / "evaluation" / "benchmark_summary.json").exists()
    assert (tmp_path / "evaluation" / "benchmark_table.csv").exists()
    assert (tmp_path / "evaluation" / "species_analysis.json").exists()
    assert (tmp_path / "predictions" / "learning_curves_comparison.png").exists()
    assert (tmp_path / "predictions" / "cat_vs_dog_error_comparison.png").exists()
    assert (tmp_path / "predictions" / "learning_curve_resnet18_transfer.png").exists()
    assert (tmp_path / "predictions" / "confusion_matrix_resnet18_transfer.png").exists()
    assert (tmp_path / "predictions" / "predictions_gallery_resnet18_transfer.png").exists()

    cm = np.asarray(out["experiments"]["resnet18_transfer"]["confusion_matrix"])
    assert cm.shape == (37, 37)
    assert cm.sum() == 16
    assert pytest.approx(np.trace(cm) / cm.sum(), abs=1e-6) == out["experiments"]["resnet18_transfer"]["test_acc"]
