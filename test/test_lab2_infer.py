"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 05 Lab 2: `src.vibe_scripts.lab2.infer`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.vibe_scripts.lab2.infer import benchmark_latency, load_predictor, predict_batch, predict_single
from src.vibe_scripts.lab2.train import train_experiment


def _create_dummy_ckpt(tmp_path: Path) -> Path:
    loader = DataLoader(TensorDataset(torch.randn(8, 3, 224, 224), torch.randint(0, 37, (8,))), batch_size=4)
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
    return tmp_path / "resnet18_transfer" / "best.pt"


@pytest.mark.smoke
def test_lab2_infer_smoke_single_pil(tmp_path: Path) -> None:
    """Smoke: nạp checkpoint và dự đoán 1 ảnh PIL RGB không crash."""
    ckpt = _create_dummy_ckpt(tmp_path)
    pred = load_predictor(ckpt, device=torch.device("cpu"))
    pil_img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8), mode="RGB")
    res = predict_single(pred, pil_img)
    assert len(res["top3"]) == 3
    assert 0 <= res["predicted_id"] < 37


def test_softmax_invariants_grayscale_and_batch_latency(tmp_path: Path) -> None:
    """Normal: Softmax sum = 1, Top-3 giảm dần, xử lý ảnh Grayscale & benchmark latency hợp lệ."""
    ckpt = _create_dummy_ckpt(tmp_path)
    pred = load_predictor(ckpt, device=torch.device("cpu"))

    gray = Image.fromarray(np.random.randint(0, 255, (100, 140), dtype=np.uint8), mode="L")
    res = predict_single(pred, gray)
    probs = res["probabilities"]
    assert probs.requires_grad is False
    assert probs.shape == (37,)
    assert pytest.approx(float(probs.sum().item()), abs=1e-5) == 1.0
    assert res["top3"][0]["prob"] >= res["top3"][1]["prob"] >= res["top3"][2]["prob"]

    ids, batch_probs = predict_batch(pred, torch.randn(3, 3, 224, 224))
    assert ids.shape == (3,) and batch_probs.shape == (3, 37)
    assert torch.allclose(batch_probs.sum(dim=1), torch.ones(3), atol=1e-5)

    lat = benchmark_latency(pred, batch_sizes=(1, 2), warmup_runs=1, measure_runs=2)
    assert lat[1]["mean_ms"] > 0 and lat[2]["samples_per_sec"] > 0
