"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 03: `src.vibe_scripts.lab1.train`."""

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.vibe_scripts.lab1 import train as lab1_train
from src.vibe_scripts.lab1.models import build_model
from src.vibe_scripts.lab1.train import (
    run_sanity_check,
    train_experiment,
    train_one_epoch,
)


def _make_dummy_loaders(n_samples: int = 64, batch_size: int = 16):
    torch.manual_seed(248)
    x = torch.randn(n_samples, 1, 28, 28)
    y = torch.arange(n_samples) % 10
    ds = TensorDataset(x, y)
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=True),
        DataLoader(ds, batch_size=batch_size, shuffle=False),
    )


@pytest.mark.smoke
def test_train_pipeline_smoke_single_epoch(tmp_path: Path):
    """Smoke Test (< 5s): Chạy 1 epoch trên dummy dataset sinh đủ best.pt, last.pt, history.json."""
    train_loader, val_loader = _make_dummy_loaders()
    out_dir = tmp_path / "smoke_exp"
    summary = train_experiment(
        model_name="logistic",
        aug_enabled=False,
        epochs=1,
        output_dir=out_dir,
        device=torch.device("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        norm_stats={"mean": 0.286, "std": 0.353},
    )
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()
    assert (out_dir / "history.json").exists()
    assert list(out_dir.glob("*.tmp")) == []
    assert summary["epochs_completed"] == 1


def test_baseline_sanity_check_initial_loss_and_overfit():
    """Normal Test: Initial loss Logistic gần ln(10) (~2.30) và overfit 1 batch đạt acc > 95%."""
    train_loader, _ = _make_dummy_loaders(n_samples=256, batch_size=256)
    stats = run_sanity_check(
        model_name="logistic",
        train_loader=train_loader,
        device=torch.device("cpu"),
        lr=0.05,
        steps=100,
    )
    assert 2.0 <= stats["initial_loss"] <= 2.7
    assert stats["overfit_final_loss"] < 0.1
    assert stats["overfit_final_acc"] > 0.95


def test_checkpoint_schema_and_no_test_loader_reference(tmp_path: Path):
    """Normal Test: Checkpoint đủ metadata bắt buộc và train.py không chạm test_loader."""
    train_loader, val_loader = _make_dummy_loaders()
    out_dir = tmp_path / "ckpt_exp"
    train_experiment(
        model_name="mlp",
        aug_enabled=True,
        epochs=2,
        output_dir=out_dir,
        device=torch.device("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
        norm_stats={"mean": 0.286, "std": 0.353},
    )
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    for key in [
        "model_name",
        "aug_enabled",
        "epoch",
        "best_metric_name",
        "best_metric_value",
        "model_state_dict",
        "optimizer_state_dict",
        "norm_stats",
        "seed",
        "hyperparams",
    ]:
        assert key in ckpt
    assert ckpt["model_name"] == "mlp"
    assert ckpt["norm_stats"] == {"mean": 0.286, "std": 0.353}

    hist = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    assert len(hist["history"]) == 2
    assert "test_loader" not in inspect.getsource(lab1_train)


def test_early_stopping_triggers_when_no_improvement(tmp_path: Path):
    """Normal Test: Early Stopping dừng sớm khi val_loss không cải thiện (min_delta quá lớn)."""
    train_loader, val_loader = _make_dummy_loaders()
    out_dir = tmp_path / "es_exp"
    summary = train_experiment(
        model_name="logistic",
        epochs=10,
        early_stopping_patience=2,
        min_delta=100.0,
        output_dir=out_dir,
        device=torch.device("cpu"),
        train_loader=train_loader,
        val_loader=val_loader,
    )
    assert summary["epochs_completed"] == 3


def test_nan_loss_raises_floating_point_error():
    """Normal Test (Fail-Fast): Loss chứa NaN phải ném FloatingPointError ngay lập tức."""
    bad_x = torch.full((4, 1, 28, 28), float("nan"))
    bad_y = torch.zeros(4, dtype=torch.long)
    bad_loader = DataLoader(TensorDataset(bad_x, bad_y), batch_size=4)
    model = build_model("logistic")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        train_one_epoch(model, bad_loader, nn.CrossEntropyLoss(), opt, torch.device("cpu"))
