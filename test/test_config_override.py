"""Kiểm thử 2 chế độ cho `src.vibe_scripts.config_override` và `labs.lab1.vibe.config`."""

import pytest

import src.config as base_cfg
from labs.lab1.vibe import config as lab1_cfg
from src.vibe_scripts.config_override import VibeConfig


@pytest.mark.smoke
def test_vibe_config_smoke_import_and_instantiation():
    """Smoke Test (< 5s): Khởi tạo VibeConfig và import lab1 config không crash."""
    cfg = VibeConfig(LR=1e-3)
    assert cfg.LR == 1e-3
    assert lab1_cfg.cfg.LR == 1e-3


@pytest.mark.smoke
def test_lab1_config_inherits_base_defaults():
    """Smoke Test: Kế thừa nguyên vẹn các biến không override từ `src/config.py`."""
    assert lab1_cfg.SEED == base_cfg.SEED == 248
    assert lab1_cfg.BATCH_SIZE == base_cfg.BATCH_SIZE == 32
    assert lab1_cfg.WOKER == base_cfg.WOKER == 8
    assert lab1_cfg.DATA_PATH == base_cfg.DATA_PATH


def test_override_does_not_mutate_base_module():
    """Normal Test: Override trên VibeConfig không làm biến đổi `src/config.py` gốc."""
    cfg = VibeConfig(LR=0.05, EPOCH=5)
    assert cfg.LR == 0.05
    assert cfg.EPOCH == 5
    assert base_cfg.LR == 1e-5
    assert base_cfg.EPOCH == 30
    assert cfg.overridden() == {"LR": 0.05, "EPOCH": 5}


def test_strict_override_rejects_unknown_key():
    """Normal Test (Fail-Fast): Truyền sai tên biến (VD: WORKERS thay vì WOKER) phải báo KeyError."""
    with pytest.raises(KeyError, match="WORKERS"):
        VibeConfig(WORKERS=4)


def test_extras_rejects_existing_base_key():
    """Normal Test: Biến đã có trong `src/config.py` không được truyền nhầm vào `extras`."""
    with pytest.raises(ValueError, match="LR"):
        VibeConfig(extras={"LR": 1e-3})


def test_missing_attribute_raises_attribute_error():
    """Normal Test: Truy cập thuộc tính không tồn tại phải ném AttributeError."""
    cfg = VibeConfig()
    with pytest.raises(AttributeError):
        _ = cfg.NON_EXISTENT_CONFIG_KEY


def test_to_dict_and_lab1_extras():
    """Normal Test: Xuất dict đầy đủ và các tham số mở rộng Lab 1 hợp lệ."""
    d = lab1_cfg.cfg.to_dict(["LR", "EPOCH", "BEST_METRIC", "SEED"])
    assert d == {
        "LR": 1e-3,
        "EPOCH": 20,
        "BEST_METRIC": "val_loss",
        "SEED": 248,
    }
    assert lab1_cfg.MANIFEST_PATH.exists()
