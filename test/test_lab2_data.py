"""Kiểm thử 2 chế độ (Smoke + Normal) cho Pipeline 01 Lab 2: `src.vibe_scripts.lab2.data`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from labs.lab2.vibe import config as lab2_cfg
from src.vibe_scripts.lab2.data import (
    build_transforms,
    is_cat_filename,
    stratified_split_val_test,
)


@pytest.mark.smoke
def test_lab2_data_smoke_transforms_and_split() -> None:
    """Smoke: tạo transform và tách stratified trên nhãn giả lập không crash."""
    train_tf, eval_tf = build_transforms()
    pil_img = Image.fromarray(np.random.randint(0, 255, (180, 240, 3), dtype=np.uint8), mode="RGB")
    t_tr = train_tf(pil_img)
    t_ev = eval_tf(pil_img)
    assert t_tr.shape == (3, 224, 224) and t_tr.dtype == torch.float32
    assert t_ev.shape == (3, 224, 224) and t_ev.dtype == torch.float32

    fake_targets = [c for c in range(37) for _ in range(100)]
    val_idx, test_idx = stratified_split_val_test(fake_targets, val_ratio=0.1, seed=248)
    assert len(val_idx) == 370 and len(test_idx) == 3330


def test_stratified_split_disjoint_and_reproducible() -> None:
    """Normal: Val và Test hoàn toàn rời nhau, cân bằng đủ 37 lớp và tái lập theo seed."""
    targets = [c for c in range(37) for _ in range(99)]
    v1, t1 = stratified_split_val_test(targets, val_ratio=0.1, seed=lab2_cfg.SEED)
    v2, t2 = stratified_split_val_test(targets, val_ratio=0.1, seed=lab2_cfg.SEED)

    assert v1 == v2 and t1 == t2
    assert set(v1).isdisjoint(set(t1))
    assert len(v1) + len(t1) == len(targets)

    val_classes = {targets[i] for i in v1}
    test_classes = {targets[i] for i in t1}
    assert len(val_classes) == 37 and len(test_classes) == 37


def test_eval_transform_deterministic_and_handles_non_rgb() -> None:
    """Normal: eval_transform tất định và xử lý an toàn ảnh Grayscale (L) & RGBA."""
    _, eval_tf = build_transforms()
    gray = Image.fromarray(np.full((120, 160), 128, dtype=np.uint8), mode="L")
    rgba = Image.fromarray(np.full((150, 150, 4), 200, dtype=np.uint8), mode="RGBA")

    t1 = eval_tf(gray)
    t2 = eval_tf(gray)
    t_rgba = eval_tf(rgba)

    assert t1.shape == (3, 224, 224)
    assert t_rgba.shape == (3, 224, 224)
    assert torch.equal(t1, t2)


def test_is_cat_filename_convention() -> None:
    """Normal: quy ước tên file Oxford-IIIT Pets (chữ HOA đầu = Mèo, chữ thường = Chó)."""
    assert is_cat_filename("Abyssinian_101.jpg") is True
    assert is_cat_filename("Persian_12.jpg") is True
    assert is_cat_filename("beagle_45.jpg") is False
    assert is_cat_filename("shiba_inu_88.jpg") is False
