"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 01: `src.vibe_scripts.lab1.data`."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from labs.lab1.vibe import config as lab1_cfg
from src.vibe_scripts.lab1.data import (
    _VectorizedAugCollate,
    build_dataloaders,
    verify_raw_manifest,
)


@pytest.mark.smoke
def test_data_pipeline_smoke_batch():
    """Smoke Test (< 5s): Khởi tạo DataLoader và đọc thử 1 batch đúng shape/dtype."""
    train_loader, val_loader, test_loader, norm_stats = build_dataloaders(
        aug_enabled=False,
        num_workers=0,
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (lab1_cfg.BATCH_SIZE, 1, 28, 28)
    assert images.dtype == torch.float32
    assert labels.shape == (lab1_cfg.BATCH_SIZE,)
    assert labels.dtype == torch.int64
    assert not torch.isnan(images).any()
    assert not torch.isinf(images).any()
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000
    assert 0.0 < norm_stats["mean"] < 1.0
    assert 0.0 < norm_stats["std"] < 1.0


def test_split_no_leakage_and_class_balance():
    """Normal Test: Không giao nhau Train/Val và cân bằng 10 lớp (5.500 / 500 / 1.000)."""
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        aug_enabled=False,
        num_workers=0,
        verify_manifest=False,
    )
    train_idx = train_loader.dataset.indices
    val_idx = val_loader.dataset.indices

    assert len(train_idx) == 55000
    assert len(val_idx) == 5000
    assert set(train_idx).isdisjoint(set(val_idx))

    train_targets = train_loader.dataset.dataset.targets.numpy()[train_idx]
    val_targets = val_loader.dataset.dataset.targets.numpy()[val_idx]
    test_targets = test_loader.dataset.targets.numpy()

    assert np.array_equal(np.bincount(train_targets, minlength=10), np.full(10, 5500))
    assert np.array_equal(np.bincount(val_targets, minlength=10), np.full(10, 500))
    assert np.array_equal(np.bincount(test_targets, minlength=10), np.full(10, 1000))


def test_normalization_invariants_on_train():
    """Normal Test: Sau Z-Score trên Train no_aug, mean ~ 0.0 và std ~ 1.0."""
    train_loader, _, _, _ = build_dataloaders(
        aug_enabled=False,
        batch_size=1024,
        num_workers=0,
        verify_manifest=False,
    )
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    for images, _ in train_loader:
        total_sum += float(images.sum().item())
        total_sq_sum += float((images**2).sum().item())
        total_pixels += images.numel()

    mean_norm = total_sum / total_pixels
    std_norm = float(np.sqrt(total_sq_sum / total_pixels - mean_norm**2))
    assert abs(mean_norm) < 1e-3
    assert abs(std_norm - 1.0) < 1e-3


def test_augmentation_isolated_to_train_only():
    """Normal Test: Augmentation không rò rỉ sang Validation hoặc Test."""
    _, val_loader, test_loader, _ = build_dataloaders(
        aug_enabled=True,
        num_workers=0,
        verify_manifest=False,
    )
    v1, _ = val_loader.dataset[0]
    v2, _ = val_loader.dataset[0]
    t1, _ = test_loader.dataset[0]
    t2, _ = test_loader.dataset[0]
    assert torch.equal(v1, v2)
    assert torch.equal(t1, t2)


def test_reproducible_batch_order_with_seed():
    """Normal Test: Cùng seed=248 cho đúng cùng thứ tự nhãn ở batch đầu tiên."""
    loader1, _, _, _ = build_dataloaders(aug_enabled=False, num_workers=0, verify_manifest=False)
    loader2, _, _, _ = build_dataloaders(aug_enabled=False, num_workers=0, verify_manifest=False)
    _, labels1 = next(iter(loader1))
    _, labels2 = next(iter(loader2))
    assert torch.equal(labels1, labels2)


def test_augmentation_independent_per_sample_in_batch():
    """Normal Test: Các ảnh giống nhau trong cùng batch aug nhận biến đổi ngẫu nhiên độc lập."""
    collate = _VectorizedAugCollate(mean=0.286, std=0.353)
    img = torch.zeros(1, 28, 28)
    img[:, 4:24, 6:18] = 0.8
    lbl = torch.tensor(0)
    batch_imgs, _ = collate([(img.clone(), lbl) for _ in range(4)])
    assert batch_imgs.shape == (4, 1, 28, 28)
    assert not torch.allclose(batch_imgs[0], batch_imgs[1])


def test_verify_raw_manifest_fails_on_corrupted_hash(tmp_path: Path):
    """Normal Test (Edge Case): Giả lập sai SHA-256 trong manifest phải báo RuntimeError."""
    manifest_data = json.loads(lab1_cfg.MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_data["artifacts"][0]["sha256_checksum"] = "0" * 64
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_raw_manifest(fake_manifest)
