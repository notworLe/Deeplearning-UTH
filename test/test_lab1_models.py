"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 02: `src.vibe_scripts.lab1.models`."""

import inspect

import pytest
import torch
import torch.nn as nn

from src.vibe_scripts.lab1 import models as lab1_models
from src.vibe_scripts.lab1.models import build_model, count_parameters


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", ["logistic", "mlp", "cnn"])
def test_models_smoke_forward(model_name: str):
    """Smoke Test (< 5s): Khởi tạo và forward dummy batch không crash."""
    model = build_model(model_name)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    ("model_name", "expected_params"),
    [
        ("logistic", 7850),
        ("mlp", 109386),
        ("cnn", 105866),
    ],
)
def test_exact_parameter_counts(model_name: str, expected_params: int):
    """Normal Test: Khớp chính xác số tham số giải tích theo `REPORT.md`."""
    model = build_model(model_name)
    assert count_parameters(model) == expected_params


@pytest.mark.parametrize("model_name", ["logistic", "mlp", "cnn"])
def test_raw_logits_and_backward_gradients(model_name: str):
    """Normal Test: Hỗ trợ B=1, trả raw logits (không Softmax) và truyền đủ gradient."""
    model = build_model(model_name)
    x_single = torch.randn(1, 1, 28, 28)
    logits_single = model(x_single)
    assert logits_single.shape == (1, 10)

    x_batch = torch.randn(8, 1, 28, 28)
    logits = model(x_batch)
    # Raw logits không bị ép tổng xác suất == 1.0 trên dim=1
    assert not torch.allclose(logits.sum(dim=1), torch.ones(8))

    loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 10, (8,)))
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("bad_shape", [(4, 3, 28, 28), (4, 1, 32, 32), (28, 28)])
def test_invalid_input_shape_raises_value_error(bad_shape):
    """Normal Test (Fail-Fast): Truyền sai shape đầu vào phải báo ValueError."""
    model = build_model("cnn")
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(*bad_shape))


def test_reproducible_weight_init_and_decoupling():
    """Normal Test: Khóa seed cho weight giống hệt nhau và module không dính optimizer."""
    torch.manual_seed(248)
    m1 = build_model("cnn")
    torch.manual_seed(248)
    m2 = build_model("cnn")
    for k in m1.state_dict():
        assert torch.equal(m1.state_dict()[k], m2.state_dict()[k])

    src_text = inspect.getsource(lab1_models)
    assert "torch.optim" not in src_text
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet50")
