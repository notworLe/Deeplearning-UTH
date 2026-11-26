"""Dual Unit Tests (Smoke + Normal) cho LogisticRegressionModel và MLPModel Lab 1 v2."""

import pytest
import torch

from src.vibe_scripts.lab1_v2.logistic import LogisticRegressionModel, count_parameters
from src.vibe_scripts.lab1_v2.mlp import MLPModel, build_model
from src.vibe_scripts.lab1_v2.trainer import set_global_seed


@pytest.mark.smoke
@pytest.mark.parametrize("model_type,hidden_dims", [("logistic", ()), ("mlp", (256, 128))])
def test_models_forward_smoke(model_type: str, hidden_dims: tuple) -> None:
    """Smoke: khởi tạo Logistic & MLP qua factory và chạy dummy forward không lỗi."""
    model = build_model(model_type=model_type, hidden_dims=hidden_dims, dropout=0.2)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert tuple(logits.shape) == (4, 10)
    assert torch.isfinite(logits).all()


def test_exact_parameter_counts_and_raw_logits() -> None:
    """Normal: khớp chính xác số tham số toán học của Logistic & 3 cấu hình MLP, output là raw logits."""
    logistic = LogisticRegressionModel()
    assert count_parameters(logistic) == 7_850

    mlp_128 = MLPModel(hidden_dims=(128,), dropout=0.2)
    assert count_parameters(mlp_128) == 101_770

    mlp_default = MLPModel(hidden_dims=(256, 128), dropout=0.2)
    assert count_parameters(mlp_default) == 235_146

    mlp_deep = MLPModel(hidden_dims=(512, 256, 128), dropout=0.2)
    assert count_parameters(mlp_deep) == 567_434

    # Kiểm tra raw logits (không phải Softmax ép tổng = 1 và không âm)
    x = torch.randn(6, 1, 28, 28)
    for model in (logistic, mlp_default):
        logits = model(x)
        row_sums = logits.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_backward_gradients_trace_shapes_and_edge_cases() -> None:
    """Normal: kiểm tra backward đủ gradient hữu hạn, trace_shapes, batch B=1, tái lập seed và Fail-Fast."""
    set_global_seed(248)
    m1 = build_model("mlp", hidden_dims=(256, 128), dropout=0.2)
    set_global_seed(248)
    m2 = build_model("mlp", hidden_dims=(256, 128), dropout=0.2)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)

    # Batch B=1 + backward
    x1 = torch.randn(1, 1, 28, 28)
    y1 = torch.tensor([3], dtype=torch.long)
    loss = torch.nn.CrossEntropyLoss()(m1(x1), y1)
    loss.backward()
    for p in m1.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    # trace_shapes khớp đầu vào -> đầu ra
    trace = m1.trace_shapes(torch.randn(2, 1, 28, 28))
    assert trace[0]["shape"] == [2, 1, 28, 28]
    assert trace[1]["shape"] == [2, 784]
    assert trace[-1]["shape"] == [2, 10]

    # Fail-Fast với shape sai & tham số bất hợp lệ
    with pytest.raises(ValueError):
        m1(torch.randn(4, 3, 28, 28))
    with pytest.raises(ValueError):
        MLPModel(hidden_dims=())
    with pytest.raises(ValueError):
        MLPModel(hidden_dims=(64,), dropout=1.0)
    with pytest.raises(ValueError):
        build_model("unknown_arch")
