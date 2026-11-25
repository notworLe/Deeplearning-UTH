"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 03 Lab 2: `src.vibe_scripts.lab2.train`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.vibe_scripts.lab2.models import build_optimizer_param_groups, build_resnet18_pet
from src.vibe_scripts.lab2.train import train_experiment, train_one_epoch


def _make_dummy_loaders(n: int = 16, batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 37, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    return loader, loader


@pytest.mark.smoke
def test_lab2_train_smoke_single_epoch(tmp_path: Path) -> None:
    """Smoke: chạy 1 epoch trên dummy loader sinh đủ `best.pt`, `last.pt`, `history.json`."""
    tr_loader, va_loader = _make_dummy_loaders(n=8, batch_size=4)
    res = train_experiment(
        mode="transfer",
        epochs=1,
        device=torch.device("cpu"),
        experiment_dir=tmp_path,
        train_loader=tr_loader,
        val_loader=va_loader,
        use_pretrained=False,
        use_amp=False,
        show_progress=False,
    )
    run_dir = tmp_path / "resnet18_transfer"
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "history.json").exists()
    assert not list(run_dir.glob("*.tmp"))
    assert res["trainable_params"] == 18_981


def test_transfer_and_finetune_preserve_frozen_bn_and_weights() -> None:
    """Normal: xác minh BN running stats & weights của khối bị khóa bất biến tuyệt đối khi train."""
    loader, _ = _make_dummy_loaders(n=8, batch_size=4)
    criterion = torch.nn.CrossEntropyLoss()
    dev = torch.device("cpu")

    # 1. Transfer: khóa toàn bộ backbone (kể cả layer4)
    m_tr = build_resnet18_pet(mode="transfer", num_classes=37, use_pretrained=False).to(dev)
    bn1_mean_before = m_tr.backbone.bn1.running_mean.clone()
    l4_bn_before = m_tr.backbone.layer4[0].bn1.running_mean.clone()
    conv1_w_before = m_tr.backbone.conv1.weight.clone()
    fc_w_before = m_tr.backbone.fc.weight.clone()

    opt_tr = torch.optim.Adam(build_optimizer_param_groups(m_tr, "transfer", lr_head=1e-2))
    train_one_epoch(m_tr, "transfer", loader, criterion, opt_tr, dev)

    assert torch.equal(m_tr.backbone.bn1.running_mean, bn1_mean_before)
    assert torch.equal(m_tr.backbone.layer4[0].bn1.running_mean, l4_bn_before)
    assert torch.equal(m_tr.backbone.conv1.weight, conv1_w_before)
    assert not torch.equal(m_tr.backbone.fc.weight, fc_w_before)

    # 2. Finetune: khóa conv1..layer3, mở layer4 + fc
    m_ft = build_resnet18_pet(mode="finetune", num_classes=37, use_pretrained=False).to(dev)
    ft_bn1_before = m_ft.backbone.bn1.running_mean.clone()
    ft_l3_bn_before = m_ft.backbone.layer3[0].bn1.running_mean.clone()
    ft_l4_bn_before = m_ft.backbone.layer4[0].bn1.running_mean.clone()
    ft_l3_w_before = m_ft.backbone.layer3[0].conv1.weight.clone()
    ft_l4_w_before = m_ft.backbone.layer4[0].conv1.weight.clone()

    opt_ft = torch.optim.Adam(build_optimizer_param_groups(m_ft, "finetune", lr_head=1e-2, lr_backbone=1e-3))
    train_one_epoch(m_ft, "finetune", loader, criterion, opt_ft, dev)

    assert torch.equal(m_ft.backbone.bn1.running_mean, ft_bn1_before)
    assert torch.equal(m_ft.backbone.layer3[0].bn1.running_mean, ft_l3_bn_before)
    assert torch.equal(m_ft.backbone.layer3[0].conv1.weight, ft_l3_w_before)
    assert not torch.equal(m_ft.backbone.layer4[0].bn1.running_mean, ft_l4_bn_before)
    assert not torch.equal(m_ft.backbone.layer4[0].conv1.weight, ft_l4_w_before)


def test_nan_loss_raises_floating_point_error() -> None:
    """Normal: input chứa NaN phải ném FloatingPointError (fail-fast)."""
    x = torch.full((4, 3, 224, 224), float("nan"))
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = build_resnet18_pet(mode="transfer", num_classes=37, use_pretrained=False)
    opt = torch.optim.Adam(build_optimizer_param_groups(model, "transfer"))

    with pytest.raises(FloatingPointError):
        train_one_epoch(model, "transfer", loader, torch.nn.CrossEntropyLoss(), opt, torch.device("cpu"))
