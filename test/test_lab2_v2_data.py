"""Unit tests (Smoke + Normal) cho Data Engineering, Transforms & Stratified Split của Lab 2 v2."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import labs.lab2_v2.config as cfg
from src.vibe_scripts.lab2_v2.data import (
    PetImageDataset,
    build_or_load_split_indices,
    prepare_oxford_pets,
)
from src.vibe_scripts.lab2_v2.transforms import (
    ImageNetNormStats,
    build_eval_transform,
    build_train_transform,
    denormalize_tensor,
    preprocess_image,
    validate_image_tensor,
)


@pytest.mark.smoke
def test_config_and_synthetic_pet_dataset_smoke(tmp_path: Path) -> None:
    """Smoke: config load sạch, PetImageDataset xử lý tốt cả ảnh RGB & RGBA sang (3, 224, 224)."""
    assert cfg.NUM_CLASSES == 37
    assert len(cfg.EXPERIMENT_SPECS) == 9
    assert cfg.get_run_spec("resnet18_scratch")["is_baseline"] is True

    rgb_p = tmp_path / "cat_rgb.jpg"
    rgba_p = tmp_path / "cat_rgba.png"
    Image.fromarray(np.full((180, 240, 3), 120, dtype=np.uint8), mode="RGB").save(rgb_p)
    Image.fromarray(np.full((150, 150, 4), 200, dtype=np.uint8), mode="RGBA").save(rgba_p)

    ds = PetImageDataset([rgb_p, rgba_p], [0, 5], transform=build_eval_transform())
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert x0.shape == (3, 224, 224) and x0.dtype == torch.float32 and y0 == 0
    assert x1.shape == (3, 224, 224) and x1.dtype == torch.float32 and y1 == 5


def test_real_oxford_pets_stratified_split_and_disjointness() -> None:
    """Normal: xác minh split thực tế (Train=3,680 | Val=1,284 | Test=2,385), 37 lớp & không giao nhau."""
    bundle = prepare_oxford_pets(
        raw_data_root=cfg.RAW_DATA_ROOT,
        batch_size=8,
        val_ratio_of_test=cfg.VAL_RATIO_OF_TEST,
        seed=cfg.SEED,
        split_file=cfg.SPLIT_FILE,
        download=False,
    )
    assert len(bundle.class_names) == 37
    assert bundle.split_sizes == {"train": 3680, "val": 1284, "test": 2385}

    # Kiểm tra giao tập rỗng giữa Train / Val / Test
    tr_set = {p.name for p in bundle.image_paths["train"]}
    va_set = {p.name for p in bundle.image_paths["val"]}
    te_set = {p.name for p in bundle.image_paths["test"]}
    assert not (tr_set & va_set)
    assert not (tr_set & te_set)
    assert not (va_set & te_set)

    # Kiểm tra đủ 12 giống mèo + 25 giống chó và mọi lớp đều có mặt ở cả 3 split
    cats = sum(1 for sp in bundle.species_by_class.values() if sp == "Cat")
    dogs = sum(1 for sp in bundle.species_by_class.values() if sp == "Dog")
    assert (cats, dogs) == (12, 25)
    for split_name in ("train", "val", "test"):
        assert len(set(bundle.labels[split_name])) == 37

    xb, yb = next(iter(bundle.val_loader))
    assert xb.shape == (8, 3, 224, 224) and xb.dtype == torch.float32
    assert yb.shape == (8,) and yb.dtype == torch.int64


def test_transforms_contract_and_split_determinism(tmp_path: Path) -> None:
    """Normal: kiểm chứng Fail-Fast tensor contract, denormalize & tính tất định của split."""
    img = Image.fromarray(np.random.randint(0, 256, (260, 310, 3), dtype=np.uint8), mode="RGB")
    t = preprocess_image(img)
    assert t.shape == (1, 3, 224, 224)
    denorm = denormalize_tensor(t)
    assert float(denorm.min()) >= 0.0 and float(denorm.max()) <= 1.0

    with pytest.raises(ValueError):
        validate_image_tensor(torch.zeros((2, 1, 224, 224)))
    with pytest.raises(ValueError):
        validate_image_tensor(torch.zeros((2, 3, 224, 224), dtype=torch.int64))

    split_p = tmp_path / "split.json"
    labels = [i % 37 for i in range(370)]
    s1 = build_or_load_split_indices(100, labels, val_ratio_of_test=0.35, seed=248, split_file=split_p)
    s2 = build_or_load_split_indices(100, labels, val_ratio_of_test=0.35, seed=248, split_file=split_p)
    assert s1["val_indices_in_test"] == s2["val_indices_in_test"]
    assert not (set(s1["val_indices_in_test"]) & set(s1["test_indices_in_test"]))
    assert ImageNetNormStats.from_dict(ImageNetNormStats().to_dict()) == ImageNetNormStats()
    assert build_train_transform()(img).shape == (3, 224, 224)
