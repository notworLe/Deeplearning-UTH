"""Nạp dữ liệu FashionMNIST, Data Collection (6.1), Data Cleaning (6.2), EDA (6.3) & Split chống leakage."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST

from src.vibe_scripts.lab1_v2.transforms import (
    NormStats,
    compute_train_norm_stats,
    normalize_batch,
    validate_image_tensor,
)

DEFAULT_CLASS_NAMES: Tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _ensure_headless_matplotlib() -> None:
    """Chuyển sang backend Agg thuần (tránh TkAgg/QtAgg) khi chạy ngoài môi trường inline notebook."""
    backend = str(matplotlib.get_backend()).lower()
    if "inline" not in backend and backend != "agg":
        matplotlib.use("Agg", force=True)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_fashion_mnist_manifest(data_root: Path, strict: bool = True) -> Dict[str, Any]:
    """Xác thực tính toàn vẹn SHA-256 cho `data/FashionMNIST/raw/manifest.json`."""
    raw_dir = Path(data_root) / "FashionMNIST" / "raw"
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        if strict:
            raise FileNotFoundError(f"Không tìm thấy manifest.json tại {manifest_path}")
        return {"verified": False, "reason": "manifest_not_found", "checked_files": 0, "files": []}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = []
    entries.extend(payload.get("artifacts", []))
    entries.extend(payload.get("extracted_artifacts", []))

    checked = 0
    mismatches: List[Dict[str, str]] = []
    file_records: List[Dict[str, Any]] = []

    for item in entries:
        raw_file = item.get("raw_file") or item.get("file_name")
        expected_hash = item.get("sha256_checksum")
        if not raw_file or not expected_hash:
            continue
        file_path = raw_dir / Path(raw_file).name
        if not file_path.exists():
            mismatches.append({"file": str(file_path), "error": "missing"})
            continue
        actual_hash = _sha256_file(file_path)
        actual_size = int(file_path.stat().st_size)
        checked += 1
        ok = actual_hash == expected_hash
        file_records.append(
            {
                "file_name": file_path.name,
                "byte_size": actual_size,
                "sha256": actual_hash,
                "verified": ok,
                "total_records": item.get("total_records"),
            }
        )
        if not ok:
            mismatches.append(
                {"file": str(file_path), "expected": expected_hash, "actual": actual_hash}
            )

    if mismatches and strict:
        raise ValueError(f"Lỗi xác thực manifest.json (SHA-256 không khớp): {mismatches}")

    return {
        "verified": len(mismatches) == 0 and checked > 0,
        "dataset_name": payload.get("dataset_name", "fashion_mnist"),
        "source_uri": payload.get("source_uri", ""),
        "collected_at": payload.get("collected_at", ""),
        "manifest_path": str(manifest_path),
        "checked_files": checked,
        "mismatches": mismatches,
        "files": file_records,
    }


def stratified_train_val_indices(
    targets: Sequence[int],
    val_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chia tầng xác định (Stratified Split) đảm bảo cân bằng 10 lớp và không giao nhau."""
    labels = np.asarray(targets, dtype=np.int64)
    total = len(labels)
    if val_size <= 0 or val_size >= total:
        raise ValueError(f"val_size phải nằm trong (0, {total}), nhận được {val_size}")

    classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(classes)
    rng = np.random.default_rng(seed)

    raw_alloc = counts * (val_size / total)
    val_per_class = np.floor(raw_alloc).astype(np.int64)
    remainder = val_size - int(val_per_class.sum())
    if remainder > 0:
        frac = raw_alloc - val_per_class
        order = np.argsort(-frac, kind="stable")
        for idx in order[:remainder]:
            val_per_class[idx] += 1

    train_idx_list: List[np.ndarray] = []
    val_idx_list: List[np.ndarray] = []
    for cls_idx, cls_label in enumerate(classes):
        cls_positions = np.flatnonzero(labels == cls_label)
        shuffled = rng.permutation(cls_positions)
        n_val = int(val_per_class[cls_idx])
        if n_val <= 0 or n_val >= len(shuffled):
            raise ValueError(
                f"Lớp {cls_label} có {len(shuffled)} mẫu, không thể tách {n_val} mẫu val (num_classes={num_classes})."
            )
        val_idx_list.append(shuffled[:n_val])
        train_idx_list.append(shuffled[n_val:])

    train_indices = np.sort(np.concatenate(train_idx_list))
    val_indices = np.sort(np.concatenate(val_idx_list))
    return train_indices, val_indices


class TensorImageDataset(Dataset):
    """Dataset gọn nhẹ giữ toàn bộ tensor đã chuẩn hóa trên RAM, an toàn trên Windows."""

    def __init__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        expected_shape: Tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        validate_image_tensor(images, expected_shape=expected_shape)
        if targets.ndim != 1 or len(targets) != len(images):
            raise ValueError(
                f"Số lượng nhãn ({tuple(targets.shape)}) không khớp số lượng ảnh ({len(images)})."
            )
        self.images = images.contiguous()
        self.targets = targets.to(dtype=torch.long).contiguous()

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.targets[index]


@dataclass
class FashionMNISTBundle:
    """Gói dữ liệu hoàn chỉnh gồm 3 tập Train / Val / Test và thông số chuẩn hóa."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: TensorImageDataset
    val_dataset: TensorImageDataset
    test_dataset: TensorImageDataset
    norm_stats: NormStats
    train_indices: np.ndarray
    val_indices: np.ndarray
    class_names: Tuple[str, ...]
    manifest_report: Dict[str, Any]


def build_dataloaders_from_tensors(
    train_images: torch.Tensor,
    train_targets: torch.Tensor,
    val_images: torch.Tensor,
    val_targets: torch.Tensor,
    test_images: torch.Tensor,
    test_targets: torch.Tensor,
    batch_size: int,
    seed: int = 248,
    num_workers: int = 0,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    expected_shape: Tuple[int, int, int] = (1, 28, 28),
    manifest_report: Optional[Dict[str, Any]] = None,
) -> FashionMNISTBundle:
    """Khởi tạo bundle từ tensor thô (dùng chung cho thực nghiệm thật và unit test)."""
    if batch_size < 1:
        raise ValueError(f"batch_size phải >= 1, nhận được {batch_size}")

    norm_stats = compute_train_norm_stats(train_images)
    train_x = normalize_batch(train_images, norm_stats=norm_stats, expected_shape=expected_shape)
    val_x = normalize_batch(val_images, norm_stats=norm_stats, expected_shape=expected_shape)
    test_x = normalize_batch(test_images, norm_stats=norm_stats, expected_shape=expected_shape)

    train_ds = TensorImageDataset(train_x, train_targets, expected_shape=expected_shape)
    val_ds = TensorImageDataset(val_x, val_targets, expected_shape=expected_shape)
    test_ds = TensorImageDataset(test_x, test_targets, expected_shape=expected_shape)

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(batch_size, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(batch_size, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return FashionMNISTBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        norm_stats=norm_stats,
        train_indices=np.arange(len(train_ds), dtype=np.int64),
        val_indices=np.arange(len(val_ds), dtype=np.int64),
        class_names=tuple(class_names),
        manifest_report=manifest_report or {"verified": False, "reason": "in_memory_tensors"},
    )


def prepare_fashion_mnist(
    data_root: Path,
    batch_size: int = 64,
    val_size: int = 5_000,
    seed: int = 248,
    num_workers: int = 0,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    verify_manifest: bool = True,
    download: bool = False,
) -> FashionMNISTBundle:
    """Nạp FashionMNIST thật, chia tầng 55k/5k/10k và chuẩn hóa duy nhất từ tập Train."""
    data_root = Path(data_root)
    manifest_report = (
        verify_fashion_mnist_manifest(data_root, strict=True)
        if verify_manifest
        else {"verified": False, "reason": "skipped"}
    )

    raw_train = FashionMNIST(root=str(data_root), train=True, download=download)
    raw_test = FashionMNIST(root=str(data_root), train=False, download=download)

    train_idx, val_idx = stratified_train_val_indices(
        targets=raw_train.targets.numpy(),
        val_size=val_size,
        seed=seed,
    )

    train_idx_t = torch.from_numpy(train_idx)
    val_idx_t = torch.from_numpy(val_idx)

    bundle = build_dataloaders_from_tensors(
        train_images=raw_train.data[train_idx_t],
        train_targets=raw_train.targets[train_idx_t],
        val_images=raw_train.data[val_idx_t],
        val_targets=raw_train.targets[val_idx_t],
        test_images=raw_test.data,
        test_targets=raw_test.targets,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        class_names=class_names,
        manifest_report=manifest_report,
    )
    bundle.train_indices = train_idx
    bundle.val_indices = val_idx
    return bundle


# ---------------------------------------------------------------------------
# 6.1 Data Collection Audit
# ---------------------------------------------------------------------------
def run_data_collection_audit(
    data_root: Path,
    output_dir: Optional[Path] = None,
    verify_manifest: bool = True,
) -> Dict[str, Any]:
    """Lớp 6.1 (Data Collection): xác thực nguồn dữ liệu thô, SHA-256 manifest và quy mô bản ghi."""
    data_root = Path(data_root)
    manifest_report = (
        verify_fashion_mnist_manifest(data_root, strict=True)
        if verify_manifest
        else {"verified": False, "reason": "skipped", "checked_files": 0, "files": []}
    )

    raw_train = FashionMNIST(root=str(data_root), train=True, download=False)
    raw_test = FashionMNIST(root=str(data_root), train=False, download=False)

    report = {
        "stage": "6.1 Data Collection",
        "dataset": "FashionMNIST",
        "provider": "torchvision.datasets.FashionMNIST",
        "raw_directory": str(data_root / "FashionMNIST" / "raw"),
        "source_uri": manifest_report.get("source_uri", ""),
        "collected_at": manifest_report.get("collected_at", ""),
        "records": {
            "raw_train": int(len(raw_train)),
            "raw_test": int(len(raw_test)),
            "total": int(len(raw_train) + len(raw_test)),
        },
        "raw_tensor_contract": {
            "train_images_shape": list(raw_train.data.shape),
            "train_images_dtype": str(raw_train.data.dtype),
            "test_images_shape": list(raw_test.data.shape),
            "test_images_dtype": str(raw_test.data.dtype),
        },
        "manifest_integrity": {
            "verified": bool(manifest_report.get("verified", False)),
            "checked_files": int(manifest_report.get("checked_files", 0)),
            "mismatches": manifest_report.get("mismatches", []),
            "files": manifest_report.get("files", []),
        },
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "collection_report.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(out_path)

    return report


# ---------------------------------------------------------------------------
# 6.2 Data Cleaning Audit
# ---------------------------------------------------------------------------
def _hash_images_bytes(images_uint8: np.ndarray) -> List[str]:
    """Tính mã băm SHA-256 trên 784 byte thô của từng ảnh 28x28 để phát hiện exact duplicates."""
    return [hashlib.sha256(img.tobytes()).hexdigest() for img in images_uint8]


def _audit_single_split_quality(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """Kiểm tra lỗi corrupt, pixel ngoài biên, nhãn sai, ảnh trống/hằng số cho 1 tập dữ liệu."""
    n = len(images)
    valid_shape = bool(images.ndim == 3 and images.shape[1:] == (28, 28))
    finite_ok = bool(np.isfinite(images).all())
    min_px = int(images.min()) if n > 0 else 0
    max_px = int(images.max()) if n > 0 else 0
    out_of_range_pixels = int(((images < 0) | (images > 255)).sum())
    invalid_labels = int(((labels < 0) | (labels >= num_classes)).sum())

    flat = images.reshape(n, -1)
    blank_images = int((flat.max(axis=1) == 0).sum())
    constant_images = int((flat.min(axis=1) == flat.max(axis=1)).sum())

    hashes = _hash_images_bytes(images)
    unique_count = len(set(hashes))
    intra_duplicates = n - unique_count

    return {
        "num_samples": n,
        "valid_shape_28x28": valid_shape,
        "all_finite": finite_ok,
        "pixel_min": min_px,
        "pixel_max": max_px,
        "out_of_range_pixels": out_of_range_pixels,
        "invalid_labels": invalid_labels,
        "blank_images": blank_images,
        "constant_images": constant_images,
        "unique_images": unique_count,
        "intra_split_duplicates": intra_duplicates,
        "_hashes": hashes,
    }


def run_data_cleaning_audit(
    data_root: Path,
    output_dir: Optional[Path] = None,
    val_size: int = 5_000,
    seed: int = 248,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> Dict[str, Any]:
    """Lớp 6.2 (Data Cleaning): kiểm toán 100% ảnh corrupt, nhãn lỗi, ảnh trống và exact duplicates."""
    data_root = Path(data_root)
    raw_train = FashionMNIST(root=str(data_root), train=True, download=False)
    raw_test = FashionMNIST(root=str(data_root), train=False, download=False)

    train_idx, val_idx = stratified_train_val_indices(
        targets=raw_train.targets.numpy(),
        val_size=val_size,
        seed=seed,
    )

    tr_imgs = raw_train.data[torch.from_numpy(train_idx)].numpy()
    tr_lbls = raw_train.targets[torch.from_numpy(train_idx)].numpy()
    va_imgs = raw_train.data[torch.from_numpy(val_idx)].numpy()
    va_lbls = raw_train.targets[torch.from_numpy(val_idx)].numpy()
    te_imgs = raw_test.data.numpy()
    te_lbls = raw_test.targets.numpy()

    num_classes = len(class_names)
    tr_q = _audit_single_split_quality(tr_imgs, tr_lbls, num_classes)
    va_q = _audit_single_split_quality(va_imgs, va_lbls, num_classes)
    te_q = _audit_single_split_quality(te_imgs, te_lbls, num_classes)

    tr_hashes = tr_q.pop("_hashes")
    va_hashes = va_q.pop("_hashes")
    te_hashes = te_q.pop("_hashes")

    tr_set: Set[str] = set(tr_hashes)
    va_set: Set[str] = set(va_hashes)
    te_set: Set[str] = set(te_hashes)
    all_train_raw_set: Set[str] = tr_set | va_set

    # Phát hiện ảnh trùng nhau nhưng bị gán nhãn khác nhau (label noise / conflict)
    hash_to_labels: Dict[str, Set[int]] = defaultdict(set)
    for h, y in zip(tr_hashes, tr_lbls.tolist()):
        hash_to_labels[h].add(int(y))
    for h, y in zip(va_hashes, va_lbls.tolist()):
        hash_to_labels[h].add(int(y))
    for h, y in zip(te_hashes, te_lbls.tolist()):
        hash_to_labels[h].add(int(y))

    conflicting_groups = [
        {"sha256": h[:16], "labels": [str(class_names[c]) for c in sorted(lbl_set)]}
        for h, lbl_set in hash_to_labels.items()
        if len(lbl_set) > 1
    ]

    # Đếm số mẫu Val và Test có bản sao exact match trong Train
    val_exact_in_train = sum(1 for h in va_hashes if h in tr_set)
    test_exact_in_train_raw = sum(1 for h in te_hashes if h in all_train_raw_set)

    corrupt_total = (
        tr_q["out_of_range_pixels"]
        + va_q["out_of_range_pixels"]
        + te_q["out_of_range_pixels"]
        + tr_q["invalid_labels"]
        + va_q["invalid_labels"]
        + te_q["invalid_labels"]
        + tr_q["blank_images"]
        + va_q["blank_images"]
        + te_q["blank_images"]
    )

    report = {
        "stage": "6.2 Data Cleaning",
        "seed": int(seed),
        "splits": {"train": tr_q, "val": va_q, "test": te_q},
        "duplicate_audit": {
            "train_intra_duplicates": tr_q["intra_split_duplicates"],
            "val_intra_duplicates": va_q["intra_split_duplicates"],
            "test_intra_duplicates": te_q["intra_split_duplicates"],
            "cross_train_val_unique_shared": len(tr_set & va_set),
            "cross_train_val_val_samples_affected": int(val_exact_in_train),
            "cross_trainraw_test_unique_shared": len(all_train_raw_set & te_set),
            "cross_trainraw_test_test_samples_affected": int(test_exact_in_train_raw),
            "conflicting_label_duplicate_groups": len(conflicting_groups),
            "conflicting_label_examples": conflicting_groups[:10],
        },
        "clean_status": {
            "corrupt_or_invalid_total": int(corrupt_total),
            "passed_integrity_checks": bool(
                corrupt_total == 0
                and tr_q["valid_shape_28x28"]
                and va_q["valid_shape_28x28"]
                and te_q["valid_shape_28x28"]
                and tr_q["all_finite"]
                and va_q["all_finite"]
                and te_q["all_finite"]
            ),
            "policy": (
                "Không có ảnh corrupt/blank/nhãn lỗi. Giữ nguyên tập benchmark FashionMNIST chuẩn "
                "(55k/5k/10k) để bảo đảm tính tái lập và so sánh công bằng; ghi nhận công khai số lượng "
                "exact duplicates & conflicting labels để đưa vào phân tích Error Analysis."
            ),
        },
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cleaning_report.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(out_path)

    return report


# ---------------------------------------------------------------------------
# 6.3 Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
def compute_eda_summary(
    data_root: Path,
    val_size: int = 5_000,
    seed: int = 248,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    verify_manifest: bool = True,
) -> Dict[str, Any]:
    """Lớp 6.3 (EDA): thống kê phân bố lớp, imbalance, per-class pixel stats, outliers & cosine similarity."""
    data_root = Path(data_root)
    manifest_report = (
        verify_fashion_mnist_manifest(data_root, strict=True)
        if verify_manifest
        else {"verified": False, "reason": "skipped"}
    )

    raw_train = FashionMNIST(root=str(data_root), train=True, download=False)
    raw_test = FashionMNIST(root=str(data_root), train=False, download=False)

    train_idx, val_idx = stratified_train_val_indices(
        targets=raw_train.targets.numpy(),
        val_size=val_size,
        seed=seed,
    )

    train_imgs = raw_train.data[torch.from_numpy(train_idx)]
    train_lbls = raw_train.targets[torch.from_numpy(train_idx)].numpy()
    val_lbls = raw_train.targets[torch.from_numpy(val_idx)].numpy()
    test_lbls = raw_test.targets.numpy()

    norm_stats = compute_train_norm_stats(train_imgs)
    train_np = train_imgs.numpy().astype(np.float32) / 255.0
    sparsity = float((train_imgs == 0).to(dtype=torch.float32).mean().item())

    def _counts_dict(lbls: np.ndarray) -> Dict[str, int]:
        return {str(name): int((lbls == idx).sum()) for idx, name in enumerate(class_names)}

    train_counts = _counts_dict(train_lbls)
    imbalance_ratio = float(max(train_counts.values()) / max(1, min(train_counts.values())))

    # Thống kê pixel từng lớp & ảnh đại diện (prototype) 10 lớp
    per_class_stats: Dict[str, Dict[str, float]] = {}
    prototypes = np.zeros((len(class_names), 28 * 28), dtype=np.float32)
    for idx, name in enumerate(class_names):
        cls_imgs = train_np[train_lbls == idx]
        per_class_stats[str(name)] = {
            "mean_intensity": float(cls_imgs.mean()),
            "std_intensity": float(cls_imgs.std()),
            "zero_pixel_ratio": float((cls_imgs == 0.0).mean()),
        }
        prototypes[idx] = cls_imgs.mean(axis=0).reshape(-1)

    # Ma trận tương đồng cosine 10x10 giữa các class prototypes
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12
    proto_normed = prototypes / norms
    cos_sim = proto_normed @ proto_normed.T

    top_similar_pairs: List[Dict[str, Any]] = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            top_similar_pairs.append(
                {
                    "class_a": str(class_names[i]),
                    "class_b": str(class_names[j]),
                    "cosine_similarity": float(cos_sim[i, j]),
                }
            )
    top_similar_pairs.sort(key=lambda item: item["cosine_similarity"], reverse=True)

    # Phát hiện outliers theo IQR của độ sáng trung bình từng ảnh (Train 55k)
    per_img_mean = train_np.mean(axis=(1, 2))
    q1, q3 = float(np.percentile(per_img_mean, 25)), float(np.percentile(per_img_mean, 75))
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    low_outliers = int((per_img_mean < low_fence).sum())
    high_outliers = int((per_img_mean > high_fence).sum())

    return {
        "stage": "6.3 Exploratory Data Analysis",
        "dataset": "FashionMNIST",
        "seed": int(seed),
        "image_shape": [1, 28, 28],
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_lbls)),
            "total": int(len(train_idx) + len(val_idx) + len(test_lbls)),
        },
        "class_distribution": {
            "train": train_counts,
            "val": _counts_dict(val_lbls),
            "test": _counts_dict(test_lbls),
            "train_imbalance_ratio": imbalance_ratio,
        },
        "train_pixel_stats": {
            "mean": norm_stats.mean,
            "std": norm_stats.std,
            "min": float(train_np.min()),
            "max": float(train_np.max()),
            "zero_pixel_ratio": sparsity,
        },
        "per_class_pixel_stats": per_class_stats,
        "outlier_analysis_iqr": {
            "q1_mean_intensity": q1,
            "q3_mean_intensity": q3,
            "iqr": float(iqr),
            "low_fence": float(low_fence),
            "high_fence": float(high_fence),
            "low_intensity_outliers": low_outliers,
            "high_intensity_outliers": high_outliers,
            "min_image_mean": float(per_img_mean.min()),
            "max_image_mean": float(per_img_mean.max()),
        },
        "prototype_cosine_similarity": {
            "matrix": cos_sim.tolist(),
            "top_similar_pairs": top_similar_pairs[:5],
        },
        "integrity": {
            "train_val_overlap": int(len(set(train_idx.tolist()) & set(val_idx.tolist()))),
            "manifest_verified": bool(manifest_report.get("verified", False)),
            "checked_files": int(manifest_report.get("checked_files", 0)),
        },
    }


def export_eda_artifacts(
    data_root: Path,
    output_dir: Path,
    val_size: int = 5_000,
    seed: int = 248,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    verify_manifest: bool = True,
) -> Dict[str, Any]:
    """Xuất đầy đủ báo cáo & 5 biểu đồ EDA vào `experiments/lab1_v2/eda/`."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_eda_summary(
        data_root=data_root,
        val_size=val_size,
        seed=seed,
        class_names=class_names,
        verify_manifest=verify_manifest,
    )

    raw_train = FashionMNIST(root=str(data_root), train=True, download=False)
    train_idx, _ = stratified_train_val_indices(
        targets=raw_train.targets.numpy(),
        val_size=val_size,
        seed=seed,
    )
    train_imgs = raw_train.data[torch.from_numpy(train_idx)].numpy()
    train_lbls = raw_train.targets[torch.from_numpy(train_idx)].numpy()
    train_np = train_imgs.astype(np.float32) / 255.0

    # 1. Biểu đồ phân bố lớp Train / Val / Test
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(class_names))
    width = 0.26
    ax.bar(x - width, [summary["class_distribution"]["train"][c] for c in class_names], width, label="Train (55k)")
    ax.bar(x, [summary["class_distribution"]["val"][c] for c in class_names], width, label="Val (5k)")
    ax.bar(x + width, [summary["class_distribution"]["test"][c] for c in class_names], width, label="Test (10k)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title("FashionMNIST — Phân bố mẫu theo lớp (Stratified 55k / 5k / 10k)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    dist_path = output_dir / "eda_class_distribution.png"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)

    # 2. Lưới ảnh mẫu (5 mẫu ngẫu nhiên / lớp)
    rng = np.random.default_rng(seed)
    n_cols = 5
    fig, axes = plt.subplots(len(class_names), n_cols, figsize=(n_cols * 1.4, len(class_names) * 1.25))
    for cls_idx, cls_name in enumerate(class_names):
        cls_positions = np.flatnonzero(train_lbls == cls_idx)
        chosen = rng.choice(cls_positions, size=n_cols, replace=False)
        for col_idx, sample_pos in enumerate(chosen):
            ax_cell = axes[cls_idx, col_idx]
            ax_cell.imshow(train_imgs[sample_pos], cmap="gray", vmin=0, vmax=255)
            ax_cell.set_xticks([])
            ax_cell.set_yticks([])
            if col_idx == 0:
                ax_cell.set_ylabel(f"{cls_idx}: {cls_name}", rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("FashionMNIST — Mẫu ảnh đại diện 10 lớp (Train Split)", fontsize=12)
    fig.tight_layout()
    grid_path = output_dir / "eda_sample_grid.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)

    # 3. Phân bố cường độ pixel (trước và sau chuẩn hóa)
    norm_stats = NormStats(
        mean=summary["train_pixel_stats"]["mean"],
        std=summary["train_pixel_stats"]["std"],
    )
    sample_pixels = train_np[:5000].ravel()
    norm_pixels = (sample_pixels - norm_stats.mean) / norm_stats.std

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].hist(sample_pixels, bins=40, color="#2b6cb0", alpha=0.85)
    axes[0].set_title(f"Raw Scaled [0, 1] (mean={norm_stats.mean:.4f}, std={norm_stats.std:.4f})")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Tần suất")
    axes[0].grid(alpha=0.3)

    axes[1].hist(norm_pixels, bins=40, color="#2f855a", alpha=0.85)
    axes[1].set_title("Normalized (Train mean=0, std=1)")
    axes[1].set_xlabel("Standardized Value")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    pixel_path = output_dir / "eda_pixel_distribution.png"
    fig.savefig(pixel_path, dpi=150)
    plt.close(fig)

    # 4. Class Prototypes & Inter-class Cosine Similarity 10x10
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 3.6])
    for cls_idx, cls_name in enumerate(class_names):
        ax_p = fig.add_subplot(gs[cls_idx // 5, cls_idx % 5])
        proto_2d = train_np[train_lbls == cls_idx].mean(axis=0)
        ax_p.imshow(proto_2d, cmap="gray", vmin=0.0, vmax=1.0)
        ax_p.set_title(f"{cls_idx}: {cls_name}", fontsize=8)
        ax_p.set_xticks([])
        ax_p.set_yticks([])

    ax_sim = fig.add_subplot(gs[:, 5])
    sim_mat = np.asarray(summary["prototype_cosine_similarity"]["matrix"], dtype=np.float32)
    im = ax_sim.imshow(sim_mat, cmap="YlOrRd", vmin=0.3, vmax=1.0)
    fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    ax_sim.set_xticks(ticks)
    ax_sim.set_yticks(ticks)
    ax_sim.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
    ax_sim.set_yticklabels(class_names, fontsize=8)
    ax_sim.set_title("Inter-Class Prototype Cosine Similarity", fontsize=10)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax_sim.text(
                j,
                i,
                f"{sim_mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if sim_mat[i, j] > 0.78 else "black",
            )
    fig.suptitle("FashionMNIST — Ảnh đại diện 10 lớp & Độ tương đồng giữa các lớp", fontsize=12)
    fig.tight_layout()
    proto_path = output_dir / "eda_class_prototypes_similarity.png"
    fig.savefig(proto_path, dpi=150)
    plt.close(fig)

    # 5. Per-class Mean Intensity / Sparsity & Extreme Intensity Outliers
    per_img_mean = train_np.mean(axis=(1, 2))
    darkest_idx = np.argsort(per_img_mean)[:4]
    brightest_idx = np.argsort(-per_img_mean)[:4]

    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[3.2, 1, 1, 1, 1])
    ax_bar = fig.add_subplot(gs[:, 0])
    means = [summary["per_class_pixel_stats"][c]["mean_intensity"] for c in class_names]
    sparsities = [summary["per_class_pixel_stats"][c]["zero_pixel_ratio"] for c in class_names]
    y_pos = np.arange(len(class_names))
    ax_bar.barh(y_pos - 0.18, means, 0.35, label="Mean Intensity [0,1]", color="#3182ce")
    ax_bar.barh(y_pos + 0.18, sparsities, 0.35, label="Zero-Pixel Sparsity", color="#dd6b20")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(class_names, fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Tỉ lệ [0, 1]")
    ax_bar.set_title("Per-Class Intensity & Sparsity (Train 55k)")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="x", alpha=0.3)

    for col, idx in enumerate(darkest_idx):
        ax_d = fig.add_subplot(gs[0, col + 1])
        ax_d.imshow(train_imgs[idx], cmap="gray", vmin=0, vmax=255)
        ax_d.set_title(f"Dark: {class_names[train_lbls[idx]]}\nμ={per_img_mean[idx]:.3f}", fontsize=7.5)
        ax_d.set_xticks([])
        ax_d.set_yticks([])

    for col, idx in enumerate(brightest_idx):
        ax_b = fig.add_subplot(gs[1, col + 1])
        ax_b.imshow(train_imgs[idx], cmap="gray", vmin=0, vmax=255)
        ax_b.set_title(f"Bright: {class_names[train_lbls[idx]]}\nμ={per_img_mean[idx]:.3f}", fontsize=7.5)
        ax_b.set_xticks([])
        ax_b.set_yticks([])

    fig.suptitle("FashionMNIST — Thống kê theo lớp & Mẫu ngoại lai cường độ sáng (Outliers)", fontsize=12)
    fig.tight_layout()
    outlier_path = output_dir / "eda_outliers_and_class_stats.png"
    fig.savefig(outlier_path, dpi=150)
    plt.close(fig)

    summary["artifacts"] = {
        "class_distribution_plot": str(dist_path),
        "sample_grid_plot": str(grid_path),
        "pixel_distribution_plot": str(pixel_path),
        "prototypes_similarity_plot": str(proto_path),
        "outliers_class_stats_plot": str(outlier_path),
    }
    summary_path = output_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
