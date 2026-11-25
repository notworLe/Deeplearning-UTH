"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 02 Lab 2: `src.vibe_scripts.lab2.models`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.vibe_scripts.lab2.models import (
    FINETUNE_FROZEN_MODULES,
    TRANSFER_FROZEN_MODULES,
    build_optimizer_param_groups,
    build_resnet18_pet,
    count_parameters,
    set_mode_for_training,
)


@pytest.mark.smoke
@pytest.mark.parametrize("mode", ["baseline", "transfer", "finetune"])
def test_lab2_models_smoke_forward(mode: str) -> None:
    """Smoke: khởi tạo và forward dummy batch `(2, 3, 224, 224)` không crash."""
    model = build_resnet18_pet(mode=mode, num_classes=37, use_pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 37)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "mode,expected_trainable",
    [
        ("baseline", 11_195_493),
        ("transfer", 18_981),
        ("finetune", 8_412_709),
    ],
)
def test_exact_trainable_parameter_counts(mode: str, expected_trainable: int) -> None:
    """Normal: kiểm chứng chính xác số tham số trainable cho cả 3 chế độ."""
    model = build_resnet18_pet(mode=mode, num_classes=37, use_pretrained=False)
    total, trainable = count_parameters(model)
    assert total == 11_195_493
    assert trainable == expected_trainable


def test_finetune_loads_transfer_checkpoint_after_fc_replacement(tmp_path: Path) -> None:
    """Normal: `finetune` thay fc 512->37 trước rồi nạp chính xác trọng số từ checkpoint `transfer`."""
    transfer_model = build_resnet18_pet(mode="transfer", num_classes=37, use_pretrained=False)
    with torch.no_grad():
        transfer_model.backbone.fc.weight.fill_(0.1234)
        transfer_model.backbone.fc.bias.fill_(-0.5678)

    ckpt_path = tmp_path / "transfer_best.pt"
    torch.save({"mode": "transfer", "num_classes": 37, "model_state_dict": transfer_model.state_dict()}, ckpt_path)

    ft_model = build_resnet18_pet(mode="finetune", num_classes=37, init_checkpoint=ckpt_path, use_pretrained=False)
    assert torch.allclose(ft_model.backbone.fc.weight, transfer_model.backbone.fc.weight)
    assert torch.allclose(ft_model.backbone.fc.bias, transfer_model.backbone.fc.bias)


def test_batchnorm_eval_policy_for_frozen_blocks() -> None:
    """Normal: mọi BatchNorm trong phần đóng băng luôn ở `eval()` sau `set_mode_for_training`."""
    m_tr = build_resnet18_pet(mode="transfer", num_classes=37, use_pretrained=False)
    set_mode_for_training(m_tr, "transfer")
    for mod_name in TRANSFER_FROZEN_MODULES:
        for sub in getattr(m_tr.backbone, mod_name).modules():
            if isinstance(sub, nn.BatchNorm2d):
                assert sub.training is False

    m_ft = build_resnet18_pet(mode="finetune", num_classes=37, use_pretrained=False)
    set_mode_for_training(m_ft, "finetune")
    for mod_name in FINETUNE_FROZEN_MODULES:
        for sub in getattr(m_ft.backbone, mod_name).modules():
            if isinstance(sub, nn.BatchNorm2d):
                assert sub.training is False
    for sub in m_ft.backbone.layer4.modules():
        if isinstance(sub, nn.BatchNorm2d):
            assert sub.training is True


def test_differential_lr_param_groups_and_invalid_shape() -> None:
    """Normal: `finetune` tách 2 param groups đúng LR và fail-fast khi sai số kênh ảnh."""
    m_ft = build_resnet18_pet(mode="finetune", num_classes=37, use_pretrained=False)
    groups = build_optimizer_param_groups(m_ft, mode="finetune", lr_head=1e-3, lr_backbone=1e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3
    assert sum(p.numel() for g in groups for p in g["params"]) == 8_412_709

    with pytest.raises(ValueError):
        m_ft(torch.randn(2, 1, 224, 224))
