"""Unit tests offline (Smoke + Normal) cho Model Factory & Freeze/Unfreeze Engine của Lab 2 v2."""

import pytest
import torch
import torch.nn as nn

from src.vibe_scripts.lab2_v2.models import (
    build_model,
    count_parameters,
    trace_shapes,
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("arch", "mode"),
    [
        ("resnet18", "scratch"),
        ("resnet18", "transfer"),
        ("resnet18", "finetune"),
        ("densenet121", "transfer"),
        ("vgg16", "transfer"),
    ],
)
def test_models_forward_smoke(arch: str, mode: str) -> None:
    """Smoke (offline weights=None): forward pass đúng shape (B, 37) và raw logits."""
    model = build_model(arch=arch, mode=mode, num_classes=37, pretrained=False)
    x = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (2, 37)
    assert torch.isfinite(logits).all()


def test_exact_param_counts_and_freeze_policies() -> None:
    """Normal: xác minh số tham số total/trainable khớp 100% đặc tả và policy freeze/unfreeze."""
    r_scratch = build_model("resnet18", mode="scratch", num_classes=37, pretrained=False)
    r_transfer = build_model("resnet18", mode="transfer", num_classes=37, pretrained=False)
    r_finetune = build_model("resnet18", mode="finetune", num_classes=37, pretrained=False)
    d_transfer = build_model("densenet121", mode="transfer", num_classes=37, pretrained=False)
    v_transfer = build_model("vgg16", mode="transfer", num_classes=37, pretrained=False)

    assert count_parameters(r_scratch)["trainable"] == count_parameters(r_scratch)["total"] == 11_195_493
    assert count_parameters(r_transfer)["trainable"] == 512 * 37 + 37  # 18,981
    assert count_parameters(r_finetune)["trainable"] > count_parameters(r_transfer)["trainable"]
    assert count_parameters(r_finetune)["trainable"] < count_parameters(r_scratch)["total"]
    assert count_parameters(d_transfer)["trainable"] == 1024 * 37 + 37  # 37,925
    assert count_parameters(v_transfer)["trainable"] == 4096 * 37 + 37  # 151,589

    # Kiểm chứng differential LR groups ở resnet18_finetune
    groups = r_finetune.get_param_groups(lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4)
    assert len(groups) == 2
    by_name = {g["name"]: g["lr"] for g in groups}
    assert by_name == {"backbone": 1e-4, "head": 1e-3}

    # DenseNet121 & VGG16 không hỗ trợ finetune theo phạm vi task.md
    with pytest.raises(ValueError):
        build_model("densenet121", mode="finetune", num_classes=37, pretrained=False)
    with pytest.raises(ValueError):
        build_model("vgg16", mode="finetune", num_classes=37, pretrained=False)


def test_frozen_bn_eval_guard_shape_validation_and_trace() -> None:
    """Normal: BatchNorm bị freeze luôn giữ eval() khi train(), sai shape ném ValueError, trace_shapes hợp lệ."""
    model = build_model("resnet18", mode="transfer", num_classes=37, pretrained=False)
    model.train()
    bn_modules = [m for m in model.net.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bn_modules and all(not m.training for m in bn_modules)

    # Kiểm chứng backward chỉ tạo gradient ở head fc
    x = torch.randn((2, 3, 224, 224))
    y = torch.tensor([0, 1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    assert model.net.fc.weight.grad is not None
    assert model.net.conv1.weight.grad is None

    # Sai shape ném ValueError
    with pytest.raises(ValueError):
        _ = model(torch.randn((2, 3, 128, 128)))

    trace = trace_shapes(model, input_shape=(2, 3, 224, 224))
    assert trace[0]["shape"] == [2, 3, 224, 224]
    assert trace[-1]["shape"] == [2, 37]
