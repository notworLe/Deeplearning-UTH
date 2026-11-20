"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 05: `src.vibe_scripts.lab1.infer`."""

from pathlib import Path

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vibe_scripts.lab1.infer import (
    benchmark_latency,
    load_predictor,
    predict_batch,
    predict_single,
)
from src.vibe_scripts.lab1.train import train_experiment


def _make_dummy_ckpt(tmp_dir: Path, model_name: str = "cnn") -> Path:
    torch.manual_seed(248)
    ds = TensorDataset(torch.randn(32, 1, 28, 28), torch.arange(32) % 10)
    loader = DataLoader(ds, batch_size=16)
    train_experiment(
        model_name=model_name,
        epochs=1,
        output_dir=tmp_dir,
        device=torch.device("cpu"),
        train_loader=loader,
        val_loader=loader,
        norm_stats={"mean": 0.286, "std": 0.353},
    )
    return tmp_dir / "best.pt"


@pytest.mark.smoke
def test_infer_smoke_single_tensor(tmp_path: Path):
    """Smoke Test (< 5s): Nạp dummy checkpoint và dự đoán 1 tensor (1, 28, 28) không crash."""
    ckpt = _make_dummy_ckpt(tmp_path / "cnn_smoke", model_name="cnn")
    predictor = load_predictor(ckpt, device=torch.device("cpu"))
    res = predict_single(predictor, torch.randn(1, 28, 28))
    assert res["model_name"] == "cnn"
    assert len(res["top3"]) == 3
    assert 0 <= res["predicted_class_id"] < 10


def test_softmax_invariants_and_no_grad(tmp_path: Path):
    """Normal Test: Tổng 10 xác suất == 1.0 ± 1e-5, Top-3 giảm dần, requires_grad == False."""
    ckpt = _make_dummy_ckpt(tmp_path / "mlp_infer", model_name="mlp")
    predictor = load_predictor(ckpt, device=torch.device("cpu"))
    res = predict_single(predictor, torch.randn(1, 28, 28))

    probs = res["probabilities"]
    assert probs.shape == (10,)
    assert not probs.requires_grad
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert abs(float(probs.sum().item()) - 1.0) < 1e-5

    top3_probs = [item["prob"] for item in res["top3"]]
    assert top3_probs == sorted(top3_probs, reverse=True)


def test_external_rgb_image_and_batch_prediction(tmp_path: Path):
    """Normal Test: Tiền xử lý được ảnh RGB kích thước khác 28x28 và suy luận theo batch."""
    ckpt = _make_dummy_ckpt(tmp_path / "log_infer", model_name="logistic")
    predictor = load_predictor(ckpt, device=torch.device("cpu"))

    img_path = tmp_path / "sample_rgb.png"
    Image.new("RGB", (64, 64), color=(120, 180, 200)).save(img_path)
    res = predict_single(predictor, img_path)
    assert res["confidence"] > 0.0

    preds, probs = predict_batch(predictor, torch.randn(4, 1, 28, 28))
    assert preds.shape == (4,)
    assert probs.shape == (4, 10)
    assert not probs.requires_grad


def test_benchmark_latency_reports_valid_metrics(tmp_path: Path):
    """Normal Test: Đo latency và kích thước file checkpoint hợp lệ cho batch_size 1 & 32."""
    ckpt = _make_dummy_ckpt(tmp_path / "cnn_bench", model_name="cnn")
    predictor = load_predictor(ckpt, device=torch.device("cpu"))
    bench = benchmark_latency(predictor, batch_sizes=(1, 32), warmup_runs=2, measure_runs=5)

    for bs in (1, 32):
        assert bench[bs]["mean_ms"] > 0.0
        assert bench[bs]["p95_ms"] >= bench[bs]["mean_ms"]
        assert bench[bs]["samples_per_sec"] > 0.0
        assert bench[bs]["checkpoint_size_kb"] > 0.0
