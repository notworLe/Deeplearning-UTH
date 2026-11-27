"""Data Engineering Layer (6.1 Collection, 6.2 Cleaning, 6.3 EDA, 6.4 Stratified Split & DataLoaders) cho Oxford-IIIT Pets."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet

from src.vibe_scripts.lab2_v2.transforms import (
    ImageNetNormStats,
    build_eval_transform,
    build_train_transform,
)


def _ensure_headless_matplotlib() -> None:
    """Chuyển sang backend Agg thuần (tránh TkAgg/QtAgg) khi chạy ngoài môi trường inline."""
    backend = str(matplotlib.get_backend()).lower()
    if "inline" not in backend and backend != "agg":
        matplotlib.use("Agg", force=True)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class PetImageDataset(Dataset):
    """Dataset nạp trực tiếp từ danh sách `(image_path, label)`, luôn ép `.convert('RGB')` để tránh lỗi RGBA/L."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(
                f"Số ảnh ({len(image_paths)}) và số nhãn ({len(labels)}) không khớp."
            )
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = [int(y) for y in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[idx]) as img:
            rgb = img.convert("RGB")
        tensor = self.transform(rgb) if self.transform is not None else rgb
        return tensor, self.labels[idx]


@dataclass
class OxfordPetsBundle:
    """Gói dữ liệu đầy đủ cho Train / Val / Test kèm metadata giống & loài."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: Tuple[str, ...]
    species_by_class: Dict[str, str]
    norm_stats: ImageNetNormStats
    split_sizes: Dict[str, int]
    image_paths: Dict[str, List[Path]]
    labels: Dict[str, List[int]]


def load_species_map(pet_dir: Path, class_names: Sequence[str]) -> Dict[str, str]:
    """Đọc ground-truth species (`Cat` vs `Dog`) từ `annotations/list.txt` (1=Cat, 2=Dog)."""
    list_txt = Path(pet_dir) / "annotations" / "list.txt"
    class_id_to_species: Dict[int, str] = {}
    if list_txt.exists():
        for raw_line in list_txt.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                cls_idx = int(parts[1]) - 1  # 1-indexed trong list.txt -> 0-indexed
                sp_id = int(parts[2])
                class_id_to_species[cls_idx] = "Cat" if sp_id == 1 else "Dog"

    species_map: Dict[str, str] = {}
    for idx, name in enumerate(class_names):
        if idx in class_id_to_species:
            species_map[str(name)] = class_id_to_species[idx]
        else:
            # Fallback quy ước Oxford-IIIT Pet: tên file viết hoa chữ cái đầu là Mèo
            species_map[str(name)] = "Cat" if str(name)[0].isupper() else "Dog"
    return species_map


def _load_raw_pet_splits(raw_data_root: Path, download: bool = False) -> Tuple[Any, Any]:
    """Nạp 2 split gốc `trainval` (3,680) và `test` (3,669) từ `data/raw/oxford-iiit-pet/`."""
    trainval_ds = OxfordIIITPet(
        root=str(raw_data_root),
        split="trainval",
        target_types="category",
        download=download,
    )
    test_full_ds = OxfordIIITPet(
        root=str(raw_data_root),
        split="test",
        target_types="category",
        download=download,
    )
    return trainval_ds, test_full_ds


def build_or_load_split_indices(
    trainval_len: int,
    test_labels: Sequence[int],
    val_ratio_of_test: float = 0.35,
    seed: int = 248,
    split_file: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Tạo (hoặc nạp) chỉ số chia tập Stratified cố định:
    - Train = 100% `trainval`
    - Val   = Stratified `val_ratio_of_test` (35%) của `test`
    - Test  = Stratified `1 - val_ratio_of_test` (65%) còn lại của `test`
    """
    if split_file is not None and split_file.exists() and not force_recompute:
        data = json.loads(split_file.read_text(encoding="utf-8"))
        if (
            data.get("trainval_len") == trainval_len
            and data.get("test_full_len") == len(test_labels)
            and abs(float(data.get("val_ratio_of_test", -1.0)) - val_ratio_of_test) < 1e-9
            and int(data.get("seed", -1)) == int(seed)
        ):
            return data

    y_test = np.asarray(test_labels, dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=val_ratio_of_test,
        random_state=int(seed),
    )
    val_idx, holdout_test_idx = next(splitter.split(np.zeros(len(y_test)), y_test))
    val_list = sorted(int(i) for i in val_idx)
    test_list = sorted(int(i) for i in holdout_test_idx)
    train_list = list(range(trainval_len))

    if set(val_list) & set(test_list):
        raise RuntimeError("Phát hiện giao nhau giữa Val và Test!")

    payload: Dict[str, Any] = {
        "scheme": "train=trainval, val=stratified_35pct_of_test, test=remaining_65pct_of_test",
        "seed": int(seed),
        "val_ratio_of_test": float(val_ratio_of_test),
        "trainval_len": int(trainval_len),
        "test_full_len": int(len(y_test)),
        "counts": {
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list),
        },
        "train_indices_in_trainval": train_list,
        "val_indices_in_test": val_list,
        "test_indices_in_test": test_list,
    }
    if split_file is not None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def prepare_oxford_pets(
    raw_data_root: Path,
    batch_size: int = 32,
    val_ratio_of_test: float = 0.35,
    seed: int = 248,
    image_size: int = 224,
    resize_size: int = 256,
    crop_scale: Tuple[float, float] = (0.75, 1.0),
    hflip_prob: float = 0.5,
    num_workers: int = 0,
    split_file: Optional[Path] = None,
    download: bool = False,
    pin_memory: Optional[bool] = None,
) -> OxfordPetsBundle:
    """Nạp dữ liệu Oxford-IIIT Pets với split cố định chống leakage và tạo 3 DataLoader."""
    raw_root = Path(raw_data_root)
    pet_dir = raw_root / "oxford-iiit-pet"
    trainval_ds, test_full_ds = _load_raw_pet_splits(raw_root, download=download)

    class_names = tuple(str(c) for c in trainval_ds.classes)
    species_by_class = load_species_map(pet_dir, class_names)
    norm_stats = ImageNetNormStats()

    split_info = build_or_load_split_indices(
        trainval_len=len(trainval_ds._images),
        test_labels=test_full_ds._labels,
        val_ratio_of_test=val_ratio_of_test,
        seed=seed,
        split_file=split_file,
    )

    tr_idx = split_info["train_indices_in_trainval"]
    va_idx = split_info["val_indices_in_test"]
    te_idx = split_info["test_indices_in_test"]

    train_paths = [Path(trainval_ds._images[i]) for i in tr_idx]
    train_labels = [int(trainval_ds._labels[i]) for i in tr_idx]
    val_paths = [Path(test_full_ds._images[i]) for i in va_idx]
    val_labels = [int(test_full_ds._labels[i]) for i in va_idx]
    test_paths = [Path(test_full_ds._images[i]) for i in te_idx]
    test_labels = [int(test_full_ds._labels[i]) for i in te_idx]

    train_tf = build_train_transform(
        image_size=image_size,
        crop_scale=crop_scale,
        hflip_prob=hflip_prob,
        norm_stats=norm_stats,
    )
    eval_tf = build_eval_transform(
        image_size=image_size,
        resize_size=resize_size,
        norm_stats=norm_stats,
    )

    train_dataset = PetImageDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = PetImageDataset(val_paths, val_labels, transform=eval_tf)
    test_dataset = PetImageDataset(test_paths, test_labels, transform=eval_tf)

    use_pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    gen = torch.Generator().manual_seed(int(seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=use_pin,
        generator=gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=use_pin,
    )

    return OxfordPetsBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        species_by_class=species_by_class,
        norm_stats=norm_stats,
        split_sizes={"train": len(train_labels), "val": len(val_labels), "test": len(test_labels)},
        image_paths={"train": train_paths, "val": val_paths, "test": test_paths},
        labels={"train": train_labels, "val": val_labels, "test": test_labels},
    )


def run_data_collection_audit(
    raw_data_root: Path,
    output_dir: Path,
    download: bool = False,
) -> Dict[str, Any]:
    """Layer 6.1 — Kiểm toán nguồn dữ liệu & xác thực SHA-256 theo `manifest.json`."""
    raw_root = Path(raw_data_root)
    pet_dir = raw_root / "oxford-iiit-pet"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainval_ds, test_full_ds = _load_raw_pet_splits(raw_root, download=download)
    class_names = tuple(str(c) for c in trainval_ds.classes)
    species_map = load_species_map(pet_dir, class_names)

    manifest_path = pet_dir / "manifest.json"
    manifest_checks: List[Dict[str, Any]] = []
    manifest_mismatches = 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for art in manifest.get("artifacts", []):
            fname = art["file_name"]
            fpath = pet_dir / fname
            exists = fpath.exists()
            actual_sha = _sha256_file(fpath) if exists else ""
            actual_size = fpath.stat().st_size if exists else 0
            expected_sha = str(art.get("sha256_checksum", ""))
            expected_size = int(art.get("byte_size", 0))
            ok = exists and (actual_sha == expected_sha) and (actual_size == expected_size)
            if not ok:
                manifest_mismatches += 1
            manifest_checks.append(
                {
                    "file_name": fname,
                    "exists": exists,
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "sha256_match": actual_sha == expected_sha,
                    "passed": ok,
                }
            )

    cat_classes = [c for c, sp in species_map.items() if sp == "Cat"]
    dog_classes = [c for c, sp in species_map.items() if sp == "Dog"]
    report = {
        "dataset_name": "Oxford-IIIT Pets (torchvision.datasets.OxfordIIITPet)",
        "pet_dir": str(pet_dir),
        "manifest_present": manifest_path.exists(),
        "manifest_files_checked": len(manifest_checks),
        "manifest_mismatches": manifest_mismatches,
        "manifest_checks": manifest_checks,
        "raw_trainval_samples": len(trainval_ds._images),
        "raw_test_samples": len(test_full_ds._images),
        "total_samples": len(trainval_ds._images) + len(test_full_ds._images),
        "num_classes": len(class_names),
        "num_cat_breeds": len(cat_classes),
        "num_dog_breeds": len(dog_classes),
        "cat_breeds": cat_classes,
        "dog_breeds": dog_classes,
    }
    (out_dir / "collection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def run_data_cleaning_audit(
    image_paths: Dict[str, List[Path]],
    labels: Dict[str, List[int]],
    num_classes: int,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Layer 6.2 — Quét 100% ảnh: corrupt, color mode (RGB/RGBA/L/P), kích thước, exact SHA-256 duplicates."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corrupt_files: List[str] = []
    invalid_labels: List[Dict[str, Any]] = []
    mode_counts: Dict[str, int] = {}
    non_rgb_files: List[Dict[str, Any]] = []
    widths: List[int] = []
    heights: List[int] = []
    aspect_ratios: List[float] = []
    split_widths: Dict[str, List[int]] = {}
    split_heights: Dict[str, List[int]] = {}

    hash_records: Dict[str, List[Tuple[str, int, str]]] = {}
    per_split_internal_dups: Dict[str, int] = {}

    for split_name in ("train", "val", "test"):
        paths = image_paths[split_name]
        lbls = labels[split_name]
        seen_in_split: set = set()
        internal_dup = 0
        s_w: List[int] = []
        s_h: List[int] = []
        for p, y in zip(paths, lbls):
            if not (0 <= int(y) < num_classes):
                invalid_labels.append({"split": split_name, "file": p.name, "label": int(y)})
            try:
                digest = _sha256_file(p)
                if digest in seen_in_split:
                    internal_dup += 1
                seen_in_split.add(digest)
                hash_records.setdefault(digest, []).append((split_name, int(y), p.name))

                with Image.open(p) as img:
                    img.load()
                    mode = str(img.mode)
                    w, h = img.size
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                if mode != "RGB":
                    non_rgb_files.append({"split": split_name, "file": p.name, "mode": mode})
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / max(h, 1))
                s_w.append(w)
                s_h.append(h)
            except Exception as exc:
                corrupt_files.append(f"{split_name}/{p.name}: {exc}")
        per_split_internal_dups[split_name] = internal_dup
        split_widths[split_name] = s_w
        split_heights[split_name] = s_h

    cross_split_dups: List[Dict[str, Any]] = []
    conflicting_label_dups: List[Dict[str, Any]] = []
    for digest, entries in hash_records.items():
        if len(entries) <= 1:
            continue
        splits_hit = sorted({e[0] for e in entries})
        labels_hit = sorted({e[1] for e in entries})
        if len(splits_hit) > 1:
            cross_split_dups.append({"sha256": digest[:16], "splits": splits_hit, "files": [e[2] for e in entries]})
        if len(labels_hit) > 1:
            conflicting_label_dups.append(
                {"sha256": digest[:16], "labels": labels_hit, "files": [e[2] for e in entries]}
            )

    w_arr = np.asarray(widths, dtype=np.float64)
    h_arr = np.asarray(heights, dtype=np.float64)
    ar_arr = np.asarray(aspect_ratios, dtype=np.float64)

    report = {
        "total_scanned": sum(len(image_paths[s]) for s in ("train", "val", "test")),
        "corrupt_count": len(corrupt_files),
        "corrupt_files": corrupt_files[:20],
        "invalid_label_count": len(invalid_labels),
        "color_mode_counts": mode_counts,
        "non_rgb_count": len(non_rgb_files),
        "non_rgb_samples": non_rgb_files[:20],
        "non_rgb_handling": "Mọi ảnh đều ép `.convert('RGB')` trong PetImageDataset và preprocess_image() nên không mất mẫu.",
        "internal_exact_duplicates": per_split_internal_dups,
        "cross_split_exact_duplicates": len(cross_split_dups),
        "cross_split_duplicate_examples": cross_split_dups[:10],
        "conflicting_label_duplicates": len(conflicting_label_dups),
        "dimensions": {
            "width_min": int(w_arr.min()),
            "width_median": float(np.median(w_arr)),
            "width_max": int(w_arr.max()),
            "height_min": int(h_arr.min()),
            "height_median": float(np.median(h_arr)),
            "height_max": int(h_arr.max()),
            "aspect_ratio_min": round(float(ar_arr.min()), 3),
            "aspect_ratio_median": round(float(np.median(ar_arr)), 3),
            "aspect_ratio_max": round(float(ar_arr.max()), 3),
            "tiny_images_below_128px": int(((w_arr < 128) | (h_arr < 128)).sum()),
        },
    }
    (out_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    geom_cache = {
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        "split_widths": split_widths,
        "split_heights": split_heights,
    }
    return report, geom_cache


def run_eda_analysis(
    image_paths: Dict[str, List[Path]],
    labels: Dict[str, List[int]],
    class_names: Sequence[str],
    species_by_class: Dict[str, str],
    geom_cache: Dict[str, Any],
    output_dir: Path,
    rgb_sample_limit: int = 800,
    seed: int = 248,
) -> Dict[str, Any]:
    """Layer 6.3 — Xuất thống kê EDA & 5 biểu đồ trực quan cho Oxford-IIIT Pets."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    # 1. Phân bố 37 giống theo Train / Val / Test
    split_counts: Dict[str, List[int]] = {}
    for s in ("train", "val", "test"):
        arr = np.bincount(np.asarray(labels[s], dtype=np.int64), minlength=num_classes)
        split_counts[s] = [int(v) for v in arr]

    tr_counts = np.asarray(split_counts["train"])
    imbalance_ratio = float(tr_counts.max() / max(int(tr_counts.min()), 1))

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    x = np.arange(num_classes)
    ax.bar(x, split_counts["train"], label=f"Train ({sum(split_counts['train'])})", color="#2b6cb0")
    ax.bar(
        x,
        split_counts["val"],
        bottom=split_counts["train"],
        label=f"Val ({sum(split_counts['val'])})",
        color="#dd6b20",
    )
    bottom2 = [a + b for a, b in zip(split_counts["train"], split_counts["val"])]
    ax.bar(
        x,
        split_counts["test"],
        bottom=bottom2,
        label=f"Test ({sum(split_counts['test'])})",
        color="#2f855a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=65, ha="right", fontsize=7.5)
    ax.set_ylabel("Number of Images")
    ax.set_title(f"Oxford-IIIT Pets — Class Distribution Across 37 Breeds (Train Imbalance Ratio = {imbalance_ratio:.2f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Tỷ lệ Species (Cat 12 giống vs Dog 25 giống)
    species_counts: Dict[str, Dict[str, int]] = {}
    for s in ("train", "val", "test"):
        n_cat = sum(1 for y in labels[s] if species_by_class[class_names[y]] == "Cat")
        n_dog = len(labels[s]) - n_cat
        species_counts[s] = {"Cat": n_cat, "Dog": n_dog}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    splits = ["train", "val", "test"]
    cats = [species_counts[s]["Cat"] for s in splits]
    dogs = [species_counts[s]["Dog"] for s in splits]
    xs = np.arange(len(splits))
    w = 0.35
    b1 = ax1.bar(xs - w / 2, cats, w, label="Cat (12 breeds)", color="#dd6b20")
    b2 = ax1.bar(xs + w / 2, dogs, w, label="Dog (25 breeds)", color="#2b6cb0")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 25, str(int(h)), ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["Train (trainval)", "Val (35% test)", "Test (65% test)"])
    ax1.set_ylabel("Samples")
    ax1.set_title("Species Counts by Split")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)

    total_cat = sum(cats)
    total_dog = sum(dogs)
    ax2.pie(
        [total_cat, total_dog],
        labels=[f"Cat ({total_cat}, {total_cat/(total_cat+total_dog)*100:.1f}%)", f"Dog ({total_dog}, {total_dog/(total_cat+total_dog)*100:.1f}%)"],
        colors=["#dd6b20", "#2b6cb0"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title("Overall Species Ratio (7,349 Images)")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_species_ratio.png", dpi=150)
    plt.close(fig)

    # 3. Phân bố kích thước ảnh gốc (Width / Height) & Aspect Ratio
    w_arr = np.asarray(geom_cache["widths"], dtype=np.float64)
    h_arr = np.asarray(geom_cache["heights"], dtype=np.float64)
    ar_arr = np.asarray(geom_cache["aspect_ratios"], dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax1.scatter(w_arr, h_arr, s=5, alpha=0.25, color="#2b6cb0")
    ax1.axvline(224, color="#c53030", linestyle="--", linewidth=1.2, label="Target 224×224")
    ax1.axhline(224, color="#c53030", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Original Width (px)")
    ax1.set_ylabel("Original Height (px)")
    ax1.set_title("Original Image Dimensions (All 7,349 Images)")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.hist(np.clip(ar_arr, 0.3, 2.5), bins=45, color="#2f855a", alpha=0.85, edgecolor="white")
    ax2.axvline(1.0, color="#c53030", linestyle="--", linewidth=1.2, label="Square (1.0)")
    ax2.axvline(float(np.median(ar_arr)), color="#1a202c", linestyle=":", linewidth=1.4, label=f"Median ({np.median(ar_arr):.2f})")
    ax2.set_xlabel("Aspect Ratio (Width / Height, clipped [0.3, 2.5])")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Aspect Ratio Distribution")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_size_aspect_ratio.png", dpi=150)
    plt.close(fig)

    # 4. Lưới ảnh mẫu đại diện (6 giống Mèo + 6 giống Chó = 12 giống × 2 ảnh = 24 ảnh)
    cat_ids = [i for i, c in enumerate(class_names) if species_by_class[c] == "Cat"][:6]
    dog_ids = [i for i, c in enumerate(class_names) if species_by_class[c] == "Dog"][:6]
    selected_ids = cat_ids + dog_ids
    tr_paths = image_paths["train"]
    tr_lbls = labels["train"]

    fig, axes = plt.subplots(4, 6, figsize=(13.0, 9.0))
    axes_flat = axes.reshape(-1)
    cell = 0
    for cls_id in selected_ids:
        matches = [tr_paths[i] for i, y in enumerate(tr_lbls) if y == cls_id][:2]
        for p in matches:
            if cell >= len(axes_flat):
                break
            with Image.open(p) as im:
                vis = im.convert("RGB").resize((224, 224))
            axes_flat[cell].imshow(vis)
            axes_flat[cell].set_xticks([])
            axes_flat[cell].set_yticks([])
            sp = species_by_class[class_names[cls_id]]
            axes_flat[cell].set_title(f"[{sp}] {class_names[cls_id]}", fontsize=7.8)
            cell += 1
    for j in range(cell, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Oxford-IIIT Pets — Sample Grid (6 Cat Breeds + 6 Dog Breeds, Resized 224×224)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_sample_grid.png", dpi=140)
    plt.close(fig)

    # 5. Phân bố kênh màu RGB trên Train vs chuẩn ImageNet (lấy mẫu ngẫu nhiên cố định seed)
    rng = np.random.default_rng(int(seed))
    n_rgb = min(rgb_sample_limit, len(tr_paths))
    chosen_rgb = rng.choice(len(tr_paths), size=n_rgb, replace=False)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    total_px = 0
    sample_means: List[np.ndarray] = []
    for idx in chosen_rgb:
        with Image.open(tr_paths[int(idx)]) as im:
            arr = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        channel_sum += arr.sum(axis=(0, 1))
        channel_sq += (arr ** 2).sum(axis=(0, 1))
        total_px += arr.shape[0] * arr.shape[1]
        sample_means.append(arr.mean(axis=(0, 1)))

    rgb_mean = channel_sum / max(total_px, 1)
    rgb_std = np.sqrt(np.maximum(channel_sq / max(total_px, 1) - rgb_mean ** 2, 1e-12))
    sample_means_arr = np.stack(sample_means, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ch_names = ["Red", "Green", "Blue"]
    ch_colors = ["#c53030", "#2f855a", "#2b6cb0"]
    for ch_i, (nm, col) in enumerate(zip(ch_names, ch_colors)):
        ax1.hist(sample_means_arr[:, ch_i], bins=30, alpha=0.45, color=col, label=f"{nm} (μ={rgb_mean[ch_i]:.3f})")
    ax1.set_xlabel("Per-Image Mean Channel Intensity [0, 1]")
    ax1.set_ylabel("Images")
    ax1.set_title(f"Train RGB Mean Distribution (n={n_rgb} sampled)")
    ax1.legend(fontsize=8.5)
    ax1.grid(axis="y", alpha=0.25)

    inet_mean = [0.485, 0.456, 0.406]
    inet_std = [0.229, 0.224, 0.225]
    xc = np.arange(3)
    w = 0.34
    ax2.bar(xc - w / 2, rgb_mean, w, yerr=rgb_std, capsize=4, label="Oxford-IIIT Pets (Train)", color="#2b6cb0", alpha=0.85)
    ax2.bar(xc + w / 2, inet_mean, w, yerr=inet_std, capsize=4, label="ImageNet Standard", color="#718096", alpha=0.85)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(ch_names)
    ax2.set_ylabel("Mean ± Std [0, 1]")
    ax2.set_ylim(0, 0.9)
    ax2.set_title("Dataset RGB Stats vs ImageNet Pre-trained Stats")
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_rgb_stats.png", dpi=150)
    plt.close(fig)

    summary = {
        "num_classes": num_classes,
        "split_sizes": {s: len(labels[s]) for s in ("train", "val", "test")},
        "train_min_per_class": int(tr_counts.min()),
        "train_max_per_class": int(tr_counts.max()),
        "train_imbalance_ratio": round(imbalance_ratio, 3),
        "species_counts": species_counts,
        "train_rgb_mean": [round(float(v), 4) for v in rgb_mean],
        "train_rgb_std": [round(float(v), 4) for v in rgb_std],
        "imagenet_mean": inet_mean,
        "imagenet_std": inet_std,
        "per_class_counts": {
            class_names[i]: {
                "species": species_by_class[class_names[i]],
                "train": split_counts["train"][i],
                "val": split_counts["val"][i],
                "test": split_counts["test"][i],
            }
            for i in range(num_classes)
        },
        "figures": [
            str(out_dir / "eda_class_distribution.png"),
            str(out_dir / "eda_species_ratio.png"),
            str(out_dir / "eda_size_aspect_ratio.png"),
            str(out_dir / "eda_sample_grid.png"),
            str(out_dir / "eda_rgb_stats.png"),
        ],
    }
    (out_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def run_data_engineering_pipeline(
    raw_data_root: Path,
    eda_dir: Path,
    split_file: Path,
    val_ratio_of_test: float = 0.35,
    seed: int = 248,
    download: bool = False,
) -> Dict[str, Any]:
    """Chạy trọn vẹn Layer 6.1 Collection → 6.2 Cleaning → 6.3 EDA và lưu `split_indices.json`."""
    collection = run_data_collection_audit(raw_data_root=raw_data_root, output_dir=eda_dir, download=download)
    bundle = prepare_oxford_pets(
        raw_data_root=raw_data_root,
        batch_size=32,
        val_ratio_of_test=val_ratio_of_test,
        seed=seed,
        split_file=split_file,
        download=download,
    )
    cleaning, geom_cache = run_data_cleaning_audit(
        image_paths=bundle.image_paths,
        labels=bundle.labels,
        num_classes=len(bundle.class_names),
        output_dir=eda_dir,
    )
    eda = run_eda_analysis(
        image_paths=bundle.image_paths,
        labels=bundle.labels,
        class_names=bundle.class_names,
        species_by_class=bundle.species_by_class,
        geom_cache=geom_cache,
        output_dir=eda_dir,
        seed=seed,
    )
    return {
        "collection": collection,
        "cleaning": cleaning,
        "eda": eda,
        "split_file": str(split_file),
    }
