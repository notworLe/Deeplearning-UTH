"""Dual Unit Tests (Smoke + Normal) cho config, transforms và data pipeline Lab 1 v2."""

import numpy as np
import pytest
import torch
from PIL import Image

import labs.lab1_v2.config as cfg
from src.vibe_scripts.lab1_v2.data import (
    build_dataloaders_from_tensors,
    export_eda_artifacts,
    prepare_fashion_mnist,
    run_data_cleaning_audit,
    run_data_collection_audit,
    stratified_train_val_indices,
    verify_fashion_mnist_manifest,
)
from src.vibe_scripts.lab1_v2.transforms import (
    NormStats,
    denormalize_tensor,
    preprocess_image,
    validate_image_tensor,
)


@pytest.mark.smoke
def test_config_and_synthetic_dataloader_smoke() -> None:
    """Smoke: nạp config 12 runs, kiểm tra manifest và 1 batch synthetic đúng shape."""
    assert len(cfg.EXPERIMENT_SPECS) == 12
    assert cfg.EXPERIMENT_SPECS[0]["run_id"] == "baseline_logistic"
    assert cfg.EXPERIMENT_SPECS[0]["is_baseline"] is True
    assert cfg.TARGET_MLP_VAL_ACC == pytest.approx(0.89)
    assert cfg.EXPERIMENT_DIR.name == "lab1_v2"

    manifest = verify_fashion_mnist_manifest(cfg.DATA_PATH, strict=True)
    assert manifest["verified"] is True
    assert manifest["checked_files"] >= 4

    rng = torch.Generator().manual_seed(cfg.SEED)
    tr_x = torch.randint(0, 256, (30, 28, 28), dtype=torch.uint8, generator=rng)
    tr_y = torch.arange(30, dtype=torch.long) % 10
    va_x = torch.randint(0, 256, (10, 28, 28), dtype=torch.uint8, generator=rng)
    va_y = torch.arange(10, dtype=torch.long) % 10
    te_x = torch.randint(0, 256, (10, 28, 28), dtype=torch.uint8, generator=rng)
    te_y = torch.arange(10, dtype=torch.long) % 10

    bundle = build_dataloaders_from_tensors(
        tr_x, tr_y, va_x, va_y, te_x, te_y, batch_size=8, seed=cfg.SEED
    )
    bx, by = next(iter(bundle.train_loader))
    assert tuple(bx.shape) == (8, 1, 28, 28)
    assert bx.dtype == torch.float32
    assert tuple(by.shape) == (8,)
    assert by.dtype == torch.long


def test_real_fashion_mnist_stratified_split_and_normalization() -> None:
    """Normal: xác thực split 55k/5k/10k không giao nhau, cân bằng 10 lớp và chuẩn hóa chỉ trên Train."""
    bundle = prepare_fashion_mnist(
        data_root=cfg.DATA_PATH,
        batch_size=64,
        val_size=cfg.VAL_SIZE,
        seed=cfg.SEED,
        num_workers=0,
        class_names=cfg.CLASS_NAMES,
        verify_manifest=True,
    )

    assert len(bundle.train_dataset) == 55_000
    assert len(bundle.val_dataset) == 5_000
    assert len(bundle.test_dataset) == 10_000

    # Không giao nhau giữa train và val
    overlap = set(bundle.train_indices.tolist()) & set(bundle.val_indices.tolist())
    assert len(overlap) == 0

    # Phân tầng đều 10 lớp: 5,500 Train / 500 Val / 1,000 Test mỗi lớp
    for cls_id in range(10):
        assert int((bundle.train_dataset.targets == cls_id).sum().item()) == 5_500
        assert int((bundle.val_dataset.targets == cls_id).sum().item()) == 500
        assert int((bundle.test_dataset.targets == cls_id).sum().item()) == 1_000

    # Chuẩn hóa trên Train phải có mean ≈ 0 và std ≈ 1
    tr_mean = float(bundle.train_dataset.images.mean().item())
    tr_std = float(bundle.train_dataset.images.std(unbiased=False).item())
    assert abs(tr_mean) < 1e-4
    assert abs(tr_std - 1.0) < 1e-4

    # Tính tái lập split cùng seed
    idx_tr2, idx_va2 = stratified_train_val_indices(
        targets=np.concatenate(
            [bundle.train_dataset.targets.numpy(), bundle.val_dataset.targets.numpy()]
        ),
        val_size=500,
        seed=cfg.SEED,
    )
    idx_tr3, idx_va3 = stratified_train_val_indices(
        targets=np.concatenate(
            [bundle.train_dataset.targets.numpy(), bundle.val_dataset.targets.numpy()]
        ),
        val_size=500,
        seed=cfg.SEED,
    )
    assert np.array_equal(idx_tr2, idx_tr3)
    assert np.array_equal(idx_va2, idx_va3)


def test_transforms_validation_and_preprocess_edge_cases() -> None:
    """Normal: kiểm tra Fail-Fast tensor contract và preprocess ảnh PIL RGB / kích thước khác."""
    norm = NormStats(mean=0.286, std=0.353)

    # PIL RGB 64x64 -> (1, 1, 28, 28)
    rgb_img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8), mode="RGB")
    t_pil = preprocess_image(rgb_img, norm_stats=norm)
    assert tuple(t_pil.shape) == (1, 1, 28, 28)
    assert t_pil.dtype == torch.float32

    # NumPy 2D 28x28
    arr = np.full((28, 28), 255, dtype=np.uint8)
    t_np = preprocess_image(arr, norm_stats=norm)
    assert tuple(t_np.shape) == (1, 1, 28, 28)

    # Denormalize nằm trong [0, 1]
    restored = denormalize_tensor(t_np, norm_stats=norm)
    assert float(restored.min().item()) >= 0.0
    assert float(restored.max().item()) <= 1.0

    # Reject sai shape, sai dtype, NaN
    with pytest.raises(ValueError):
        validate_image_tensor(torch.randn(2, 3, 28, 28))
    with pytest.raises(TypeError):
        validate_image_tensor(torch.zeros(2, 1, 28, 28, dtype=torch.int64))
    bad = torch.zeros(1, 1, 28, 28)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_image_tensor(bad)


def test_data_collection_cleaning_and_eda_pipeline(tmp_path) -> None:
    """Normal: xác thực đầy đủ 6.1 Collection -> 6.2 Cleaning -> 6.3 EDA trên FashionMNIST thật vào tmp_path."""
    col = run_data_collection_audit(cfg.DATA_PATH, output_dir=tmp_path, verify_manifest=True)
    assert col["records"]["total"] == 70_000
    assert col["manifest_integrity"]["verified"] is True
    assert (tmp_path / "collection_report.json").exists()

    cln = run_data_cleaning_audit(
        cfg.DATA_PATH, output_dir=tmp_path, val_size=cfg.VAL_SIZE, seed=cfg.SEED
    )
    assert cln["clean_status"]["passed_integrity_checks"] is True
    assert cln["clean_status"]["corrupt_or_invalid_total"] == 0
    assert (tmp_path / "cleaning_report.json").exists()

    eda = export_eda_artifacts(
        cfg.DATA_PATH, output_dir=tmp_path, val_size=cfg.VAL_SIZE, seed=cfg.SEED
    )
    assert eda["class_distribution"]["train_imbalance_ratio"] == pytest.approx(1.0)
    assert len(eda["prototype_cosine_similarity"]["top_similar_pairs"]) == 5
    for plot_key in [
        "class_distribution_plot",
        "sample_grid_plot",
        "pixel_distribution_plot",
        "prototypes_similarity_plot",
        "outliers_class_stats_plot",
    ]:
        assert (tmp_path / eda["artifacts"][plot_key]).exists() or cfg.ROOT_PATH.joinpath(
            eda["artifacts"][plot_key]
        ).exists()
    assert (tmp_path / "eda_summary.json").exists()
