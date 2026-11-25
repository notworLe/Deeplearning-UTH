"""
Pipeline 01: Data Collection, Cleaning, Stratified Split & DataLoader cho Lab 2 (Oxford-IIIT Pets).

Khắc phục triệt để các lỗi rò rỉ dữ liệu từ bản cũ:
- Tách tách biệt `train_transform` (chỉ áp dụng trên 3.680 ảnh `trainval`) và `eval_transform`
  (áp dụng cho cả `Val` 367 ảnh và `Test` 3.302 ảnh).
- Tách phân tầng (stratified) 10% từ tập `test` gốc (3.669 ảnh) với seed cố định thành
  `Val` (367 ảnh) và `Test` (3.302 ảnh), đảm bảo `Val ∩ Test = ∅`.
- Chuẩn hóa tường minh mọi ảnh sang chế độ màu RGB trước khi biến đổi tensor.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from labs.lab2.vibe import config as lab2_cfg
from src.vibe_scripts.eda_helpers import calculate_file_sha256


def is_cat_filename(image_path: Path | str) -> bool:
    """
    Xác định loài theo quy ước đặt tên gốc của Oxford-IIIT Pets.

    Tên file bắt đầu bằng chữ IN HOA là Mèo (12 giống), chữ thường là Chó (25 giống).
    """
    name = Path(image_path).name
    if not name:
        raise ValueError(f"Đường dẫn ảnh không hợp lệ: {image_path}")
    return name[0].isupper()


def _to_rgb(img: Image.Image) -> Image.Image:
    """Chuyển đổi an toàn mọi ảnh PIL (L, RGBA, P, CMYK...) sang 3 kênh RGB."""
    return img.convert("RGB")


def build_transforms(
    image_size: int = lab2_cfg.IMAGE_SIZE,
    resize_size: int = lab2_cfg.RESIZE_SIZE,
    crop_scale: Tuple[float, float] = lab2_cfg.AUG_CONFIG["crop_scale"],
    horizontal_flip_p: float = lab2_cfg.AUG_CONFIG["horizontal_flip_p"],
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Khởi tạo tách rời `train_transform` và `eval_transform` theo chuẩn ImageNet.

    - Train: RGB -> RandomResizedCrop(224) -> RandomHorizontalFlip -> ToTensor -> ImageNet Normalize.
    - Val/Test: RGB -> Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize.
    """
    train_tf = transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.RandomResizedCrop(size=image_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=horizontal_flip_p),
        transforms.ToTensor(),
        lab2_cfg.IMAGE_NET_NORMALIZE,
    ])
    eval_tf = transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        lab2_cfg.IMAGE_NET_NORMALIZE,
    ])
    return train_tf, eval_tf


def stratified_split_val_test(
    targets: Sequence[int],
    val_ratio: float = lab2_cfg.VAL_RATIO,
    seed: int = lab2_cfg.SEED,
) -> Tuple[List[int], List[int]]:
    """
    Tách phân tầng (stratified) tập `test` gốc (3.669 ảnh) thành `val_indices` (~10%) và `test_indices` (~90%).

    Bảo toàn tỷ lệ cho từng lớp trong 37 giống và không bao giờ giao nhau giữa Val và Test.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio phải thuộc (0, 1), nhận được: {val_ratio}")

    targets_arr = np.asarray(targets, dtype=np.int64)
    rng = np.random.RandomState(seed)
    val_indices: List[int] = []
    test_indices: List[int] = []

    for cls in np.unique(targets_arr):
        cls_idx = np.where(targets_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        if n_val >= len(cls_idx):
            n_val = max(1, len(cls_idx) - 1)
        val_indices.extend(cls_idx[:n_val].tolist())
        test_indices.extend(cls_idx[n_val:].tolist())

    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return val_indices, test_indices


def extract_species_mapping(dataset: OxfordIIITPet) -> Dict[int, Dict[str, Any]]:
    """
    Đối chiếu ánh xạ `class_id -> {label, species, is_cat}` từ tên file gốc của OxfordIIITPet.

    Đồng thời kiểm tra tính nhất quán giữa toàn bộ ảnh trong cùng một lớp.
    """
    per_class_flags: Dict[int, set[bool]] = defaultdict(set)
    for img_path, lbl in zip(dataset._images, dataset._labels):
        per_class_flags[int(lbl)].add(is_cat_filename(img_path))

    mapping: Dict[int, Dict[str, Any]] = {}
    for cls_id, label_name in enumerate(dataset.classes):
        flags = per_class_flags.get(cls_id, set())
        if len(flags) != 1:
            raise ValueError(f"Lớp {cls_id} ({label_name}) không nhất quán quy ước tên file loài: {flags}")
        is_cat = next(iter(flags))
        mapping[cls_id] = {
            "class_id": cls_id,
            "label": label_name,
            "is_cat": is_cat,
            "species": "Cat" if is_cat else "Dog",
        }
    return mapping


def inspect_dataset_quality(
    data_root: Path = lab2_cfg.DATA_PATH,
) -> Dict[str, Any]:
    """
    Thực hiện Data Cleaning & EDA audit toàn bộ 7.349 ảnh Oxford-IIIT Pets.

    Kiểm tra:
    - Ảnh hỏng (corrupt/unreadable).
    - Chế độ màu (RGB, RGBA, L, P...).
    - Ảnh trùng lặp chính xác (SHA-256 trên byte file ảnh): trong cùng split và chéo trainval/test.
    - Kích thước ảnh (W, H), tỉ lệ cạnh (W/H), phân bố 37 giống và tỷ lệ Mèo/Chó theo tên file.
    """
    ds_trainval = OxfordIIITPet(root=data_root, split="trainval", target_types="category", download=False)
    ds_test = OxfordIIITPet(root=data_root, split="test", target_types="category", download=False)

    species_map = extract_species_mapping(ds_trainval)
    test_species_map = extract_species_mapping(ds_test)
    species_consistent = all(
        species_map[c]["is_cat"] == test_species_map[c]["is_cat"] for c in range(len(ds_trainval.classes))
    )

    corrupt_files: List[str] = []
    mode_counts: Counter[str] = Counter()
    widths: List[int] = []
    heights: List[int] = []
    aspect_ratios: List[float] = []
    hash_to_entries: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)

    for split_name, ds in [("trainval", ds_trainval), ("test", ds_test)]:
        for img_path, lbl in zip(ds._images, ds._labels):
            p = Path(img_path)
            try:
                raw_bytes = p.read_bytes()
                digest = hashlib.sha256(raw_bytes).hexdigest()
                hash_to_entries[digest].append((split_name, p.name, int(lbl)))
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im2:
                    mode_counts[im2.mode] += 1
                    w, h = im2.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / max(h, 1))
            except Exception:
                corrupt_files.append(str(p))

    duplicate_groups = [entries for entries in hash_to_entries.values() if len(entries) > 1]
    within_trainval_dups = [g for g in duplicate_groups if all(x[0] == "trainval" for x in g)]
    within_test_dups = [g for g in duplicate_groups if all(x[0] == "test" for x in g)]
    cross_split_dups = [g for g in duplicate_groups if len({x[0] for x in g}) > 1]

    trainval_counts = Counter(int(y) for y in ds_trainval._labels)
    test_counts = Counter(int(y) for y in ds_test._labels)
    cat_classes = sorted(c for c, info in species_map.items() if info["is_cat"])
    dog_classes = sorted(c for c, info in species_map.items() if not info["is_cat"])

    w_arr = np.asarray(widths, dtype=np.float32)
    h_arr = np.asarray(heights, dtype=np.float32)
    ar_arr = np.asarray(aspect_ratios, dtype=np.float32)

    return {
        "total_images": len(ds_trainval) + len(ds_test),
        "trainval_size": len(ds_trainval),
        "test_raw_size": len(ds_test),
        "num_classes": len(ds_trainval.classes),
        "classes": list(ds_trainval.classes),
        "species_map": species_map,
        "species_filename_consistent": species_consistent,
        "num_cat_classes": len(cat_classes),
        "num_dog_classes": len(dog_classes),
        "trainval_cat_images": sum(trainval_counts[c] for c in cat_classes),
        "trainval_dog_images": sum(trainval_counts[c] for c in dog_classes),
        "test_raw_cat_images": sum(test_counts[c] for c in cat_classes),
        "test_raw_dog_images": sum(test_counts[c] for c in dog_classes),
        "corrupt_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "color_modes": dict(mode_counts),
        "non_rgb_count": sum(cnt for m, cnt in mode_counts.items() if m != "RGB"),
        "duplicate_group_count": len(duplicate_groups),
        "within_trainval_dup_groups": len(within_trainval_dups),
        "within_test_dup_groups": len(within_test_dups),
        "cross_split_dup_groups": len(cross_split_dups),
        "duplicate_examples": duplicate_groups[:10],
        "trainval_class_counts": {ds_trainval.classes[c]: trainval_counts[c] for c in range(len(ds_trainval.classes))},
        "test_raw_class_counts": {ds_test.classes[c]: test_counts[c] for c in range(len(ds_test.classes))},
        "width_stats": {
            "min": int(w_arr.min()),
            "median": float(np.median(w_arr)),
            "mean": float(w_arr.mean()),
            "max": int(w_arr.max()),
        },
        "height_stats": {
            "min": int(h_arr.min()),
            "median": float(np.median(h_arr)),
            "mean": float(h_arr.mean()),
            "max": int(h_arr.max()),
        },
        "aspect_ratio_stats": {
            "min": float(ar_arr.min()),
            "p05": float(np.percentile(ar_arr, 5)),
            "median": float(np.median(ar_arr)),
            "mean": float(ar_arr.mean()),
            "p95": float(np.percentile(ar_arr, 95)),
            "max": float(ar_arr.max()),
        },
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
    }


def generate_dataset_manifest(data_root: Path = lab2_cfg.DATA_PATH) -> Path:
    """Sinh file `manifest.json` kèm SHA-256 cho các file nén gốc của Oxford-IIIT Pets."""
    pet_dir = data_root / "oxford-iiit-pet"
    if not pet_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục Oxford-IIIT Pet tại {pet_dir}")

    archives = sorted(pet_dir.glob("*.tar.gz"))
    artifacts = []
    for arc in archives:
        artifacts.append({
            "dataset_name": "Oxford-IIIT Pet",
            "raw_file": str(arc.relative_to(data_root.parent)),
            "file_name": arc.name,
            "byte_size": arc.stat().st_size,
            "sha256_checksum": calculate_file_sha256(arc),
        })

    manifest = {
        "dataset_name": "Oxford-IIIT Pet (torchvision.datasets.OxfordIIITPet)",
        "raw_dir": "data/oxford-iiit-pet",
        "source_uri": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "total_records": 7349,
        "artifacts": artifacts,
    }
    manifest_path = pet_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def verify_dataset_manifest(data_root: Path = lab2_cfg.DATA_PATH) -> List[Dict[str, Any]]:
    """Xác thực SHA-256 toàn bộ archive liệt kê trong `data/oxford-iiit-pet/manifest.json`."""
    manifest_path = data_root / "oxford-iiit-pet" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Thiếu manifest.json tại {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for art in manifest.get("artifacts", []):
        target = data_root / "oxford-iiit-pet" / art["file_name"]
        actual = calculate_file_sha256(target) if target.exists() else "FILE_NOT_FOUND"
        valid = actual.lower() == art["sha256_checksum"].lower()
        results.append({
            "file_name": art["file_name"],
            "expected_sha": art["sha256_checksum"],
            "actual_sha": actual,
            "valid": valid,
        })
        if not valid:
            raise ValueError(f"SHA-256 mismatch cho {art['file_name']}: expected={art['sha256_checksum']}, actual={actual}")
    return results


def _seed_worker(worker_id: int) -> None:
    """Khóa seed độc lập cho mỗi DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    data_root: Path = lab2_cfg.DATA_PATH,
    batch_size: int = lab2_cfg.BATCH_SIZE,
    val_ratio: float = lab2_cfg.VAL_RATIO,
    seed: int = lab2_cfg.SEED,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    verify_manifest: bool = False,
    subset_limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Dựng 3 DataLoader sạch không rò rỉ cho Lab 2 (Oxford-IIIT Pets).

    - Train: 100% `trainval` (3.680 ảnh) + `train_transform` + `shuffle=True`.
    - Val: 10% stratified từ `test` gốc (367 ảnh) + `eval_transform` + `shuffle=False`.
    - Test: 90% còn lại từ `test` gốc (3.302 ảnh, disjoint với Val) + `eval_transform` + `shuffle=False`.

    Returns:
        (train_loader, val_loader, test_loader, split_meta) với tensor batch `(B, 3, 224, 224)`.
    """
    if verify_manifest:
        verify_dataset_manifest(data_root)

    train_tf, eval_tf = build_transforms()
    train_ds: Dataset = OxfordIIITPet(
        root=data_root, split="trainval", target_types="category", download=False, transform=train_tf
    )
    raw_test_ds = OxfordIIITPet(
        root=data_root, split="test", target_types="category", download=False, transform=eval_tf
    )

    species_map = extract_species_mapping(raw_test_ds)
    val_indices, test_indices = stratified_split_val_test(raw_test_ds._labels, val_ratio=val_ratio, seed=seed)

    if subset_limit is not None:
        train_ds = Subset(train_ds, list(range(min(subset_limit, len(train_ds)))))
        val_indices = val_indices[: min(subset_limit, len(val_indices))]
        test_indices = test_indices[: min(subset_limit, len(test_indices))]

    val_ds: Dataset = Subset(raw_test_ds, val_indices)
    test_ds: Dataset = Subset(raw_test_ds, test_indices)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    gen = torch.Generator()
    gen.manual_seed(seed)

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = _seed_worker
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    split_meta = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "val_ratio": val_ratio,
        "seed": seed,
        "num_classes": len(raw_test_ds.classes),
        "classes": list(raw_test_ds.classes),
        "species_map": species_map,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "image_size": lab2_cfg.IMAGE_SIZE,
    }
    return train_loader, val_loader, test_loader, split_meta
