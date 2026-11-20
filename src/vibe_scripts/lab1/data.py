"""Pipeline 01: Data Preparation & Vectorized DataLoader cho Lab 1 (Fashion-MNIST)."""

import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from labs.lab1.vibe import config as lab1_cfg
from src.vibe_scripts import eda_helpers


def _seed_worker(worker_id: int) -> None:
    """Khóa seed ngẫu nhiên cho từng worker của DataLoader."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def verify_raw_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Xác thực toàn vẹn SHA-256 từ `manifest.json` trước khi nạp dữ liệu."""
    manifest_data, results = eda_helpers.load_and_verify_manifest(manifest_path)
    failed = [r["file_name"] for r in results if not r["valid"]]
    if failed:
        raise RuntimeError(
            f"Xác thực SHA-256 thất bại cho các tệp: {', '.join(failed)}"
        )
    return manifest_data


def compute_train_norm_stats(
    raw_images: torch.Tensor,
    train_indices: List[int],
) -> Tuple[float, float]:
    """
    Tính `mean` và `std` ở thang [0.0, 1.0] chỉ trên tập Train.

    Input:
        raw_images: Tensor ảnh gốc (N, 28, 28) uint8.
        train_indices: Danh sách chỉ số mẫu thuộc tập Train.
    Output:
        (mean, std) dạng float.
    """
    if len(train_indices) == 0:
        raise ValueError("train_indices không được rỗng.")
    train_pixels = raw_images[train_indices].to(torch.float32) / 255.0
    mean_val = float(train_pixels.mean().item())
    std_val = float(train_pixels.std(unbiased=False).item())
    return mean_val, max(std_val, 1e-7)


def build_transforms(
    mean: float,
    std: float,
    aug_enabled: bool = False,
    aug_config: Optional[Dict[str, Any]] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Khởi tạo `(train_transform, eval_transform)` theo thứ tự `Augmentation -> ToTensor -> Normalize`.
    """
    eval_ops = [
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ]
    eval_transform = transforms.Compose(eval_ops)

    if not aug_enabled:
        return eval_transform, eval_transform

    cfg = aug_config or lab1_cfg.AUG_CONFIG
    train_ops = [
        transforms.RandomHorizontalFlip(p=float(cfg.get("horizontal_flip_p", 0.5))),
        transforms.RandomAffine(
            degrees=float(cfg.get("rotation_degrees", 10.0)),
            translate=tuple(cfg.get("translate", (0.08, 0.08))),
        ),
        *eval_ops,
    ]
    return transforms.Compose(train_ops), eval_transform


class _FastFashionDataset(Dataset):
    """Dataset tối ưu lưu sẵn tensor `(N, 1, 28, 28)` trong RAM thay cho PIL từng ảnh."""

    def __init__(
        self,
        images_uint8: torch.Tensor,
        targets: torch.Tensor,
        mean: float,
        std: float,
        pre_normalize: bool = True,
    ) -> None:
        self.targets = targets.to(torch.int64)
        scaled = images_uint8.unsqueeze(1).to(torch.float32) / 255.0
        if pre_normalize:
            self.data = (scaled - mean) / std
        else:
            self.data = scaled

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class _VectorizedAugCollate:
    """
    Áp dụng Augmentation ngẫu nhiên ĐỘC LẬP cho từng ảnh trong batch `[B, 1, 28, 28]`
    bằng `F.affine_grid` + `F.grid_sample`, sau đó chuẩn hóa Z-Score.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        aug_config: Optional[Dict[str, Any]] = None,
        seed: int = lab1_cfg.SEED,
    ) -> None:
        cfg = aug_config or lab1_cfg.AUG_CONFIG
        self.mean = mean
        self.std = std
        self.flip_p = float(cfg.get("horizontal_flip_p", 0.5))
        self.deg = float(cfg.get("rotation_degrees", 10.0))
        translate = tuple(cfg.get("translate", (0.08, 0.08)))
        self.tx_r = float(translate[0])
        self.ty_r = float(translate[1])
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.stack([item[1] for item in batch], dim=0)
        bs = images.size(0)

        flip_mask = torch.rand(bs, 1, 1, 1, generator=self.generator) < self.flip_p
        images = torch.where(flip_mask, images.flip(-1), images)

        rad = (torch.rand(bs, generator=self.generator) * 2.0 - 1.0) * (
            self.deg * math.pi / 180.0
        )
        cos_a = torch.cos(rad)
        sin_a = torch.sin(rad)
        tx = (torch.rand(bs, generator=self.generator) * 2.0 - 1.0) * (2.0 * self.tx_r)
        ty = (torch.rand(bs, generator=self.generator) * 2.0 - 1.0) * (2.0 * self.ty_r)

        theta = torch.zeros(bs, 2, 3, dtype=images.dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, images.size(), align_corners=False)
        aug_images = F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        norm_images = (aug_images - self.mean) / self.std
        return norm_images, labels


def build_dataloaders(
    data_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    aug_enabled: bool = False,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,
    verify_manifest: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Tạo `(train_loader, val_loader, test_loader, norm_stats)` tốc độ cao, không data leakage.
    """
    data_root = Path(data_dir) if data_dir is not None else lab1_cfg.DATA_PATH
    manifest_file = (
        Path(manifest_path) if manifest_path is not None else lab1_cfg.MANIFEST_PATH
    )
    bs = int(batch_size if batch_size is not None else lab1_cfg.BATCH_SIZE)
    workers = int(num_workers if num_workers is not None else lab1_cfg.WOKER)
    run_seed = int(seed if seed is not None else lab1_cfg.SEED)

    if verify_manifest:
        verify_raw_manifest(manifest_file)

    raw_train = FashionMNIST(root=data_root, train=True, download=False)
    raw_test = FashionMNIST(root=data_root, train=False, download=False)

    train_idx, val_idx = eda_helpers.stratified_split_indices(
        raw_train.targets.numpy(),
        train_size=lab1_cfg.TRAIN_SIZE,
        val_size=lab1_cfg.VAL_SIZE,
        seed=run_seed,
    )
    mean_val, std_val = compute_train_norm_stats(raw_train.data, train_idx)

    train_base = _FastFashionDataset(
        raw_train.data,
        raw_train.targets,
        mean=mean_val,
        std=std_val,
        pre_normalize=not aug_enabled,
    )
    val_base = _FastFashionDataset(
        raw_train.data,
        raw_train.targets,
        mean=mean_val,
        std=std_val,
        pre_normalize=True,
    )
    test_dataset = _FastFashionDataset(
        raw_test.data,
        raw_test.targets,
        mean=mean_val,
        std=std_val,
        pre_normalize=True,
    )

    train_subset = Subset(train_base, train_idx)
    val_subset = Subset(val_base, val_idx)

    pin_mem = lab1_cfg.DEVICE.type == "cuda"
    persist = workers > 0
    generator = torch.Generator().manual_seed(run_seed)
    collate_fn = (
        _VectorizedAugCollate(
            mean=mean_val,
            std=std_val,
            aug_config=lab1_cfg.AUG_CONFIG,
            seed=run_seed,
        )
        if aug_enabled
        else None
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
        worker_init_fn=_seed_worker,
        generator=generator,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
        worker_init_fn=_seed_worker,
    )

    norm_stats: Dict[str, Any] = {
        "mean": mean_val,
        "std": std_val,
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test_size": len(test_dataset),
        "aug_enabled": aug_enabled,
        "seed": run_seed,
    }
    return train_loader, val_loader, test_loader, norm_stats
