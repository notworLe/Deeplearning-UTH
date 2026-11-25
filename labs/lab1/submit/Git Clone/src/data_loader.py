"""Practice 1 / Stage 2: inspect one FashionMNIST CSV, prevent exact-match leakage,
and build PyTorch DataLoaders. Run: python -m src.data_loader --csv data/fashion_mnist_full.csv

No model training or test-set model selection takes place in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

LABEL_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
PIXEL_COLUMNS = [f"pixel{i}" for i in range(1, 785)]
EXPECTED_SPLITS = {"train": 54000, "validation": 6000, "test": 10000}
SPLIT_ORDER = tuple(EXPECTED_SPLITS)


def _read_and_validate(csv_path: str | Path) -> pd.DataFrame:
    """Read single CSV once; strict schema and range checks happen before uint8 cast."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {csv_path.resolve()}")

    # int16 preserves valid 0..255 values and also detects negatives/out-of-range.
    dtype = {"label": "int16", "split": "category"}
    dtype.update({col: "int16" for col in PIXEL_COLUMNS})
    df = pd.read_csv(csv_path, dtype=dtype)

    required = ["label", *PIXEL_COLUMNS, "split"]
    if len(df.columns) != len(required) or set(df.columns) != set(required):
        missing = sorted(set(required) - set(df.columns))
        extra = sorted(set(df.columns) - set(required))
        raise ValueError(f"CSV schema sai. Thiếu: {missing[:8]}; dư: {extra[:8]}")
    if df.isna().to_numpy().any():
        raise ValueError("CSV chứa giá trị thiếu; cần kiểm tra trước khi xử lý.")
    if len(df) != 70000:
        raise ValueError(f"FashionMNIST phải có 70.000 mẫu, hiện có {len(df):,}.")
    if not df["label"].between(0, 9).all():
        raise ValueError("Nhãn phải là số nguyên từ 0 đến 9.")
    splits = df["split"].astype(str)
    if set(splits.unique()) != set(SPLIT_ORDER):
        raise ValueError(f"Cột split phải chỉ chứa {SPLIT_ORDER}.")
    split_counts = splits.value_counts().to_dict()
    if any(split_counts.get(k, 0) != n for k, n in EXPECTED_SPLITS.items()):
        raise ValueError(f"Tỷ lệ split khác với dữ liệu đã chốt: {split_counts}")

    pixels = df[PIXEL_COLUMNS].to_numpy(copy=False)
    if pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Pixel phải thuộc khoảng [0, 255].")
    df.loc[:, PIXEL_COLUMNS] = df[PIXEL_COLUMNS].astype(np.uint8)
    # Force compact typed columns, avoiding 500MB+ float64 arrays.
    df = df.astype({col: np.uint8 for col in PIXEL_COLUMNS})
    df["label"] = df["label"].astype(np.uint8)
    return df


def _image_fingerprints(images: np.ndarray) -> list[bytes]:
    """Strong fingerprints of exact 784-pixel sequences, independent of labels."""
    return [hashlib.blake2b(row, digest_size=16).digest() for row in images]


def _split_and_remove_leakage(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Never resplit the file. Remove training images identical to any val/test image.

    The original 70,000-row CSV remains unchanged. All test/validation examples remain.
    """
    original = {s: df.loc[df["split"] == s].copy() for s in SPLIT_ORDER}
    fingerprint_sets: dict[str, set[bytes]] = {}
    fps_by_split: dict[str, list[bytes]] = {}
    within_split_duplicates = {}
    for s in SPLIT_ORDER:
        pixels = np.ascontiguousarray(original[s][PIXEL_COLUMNS].to_numpy(dtype=np.uint8))
        fps = _image_fingerprints(pixels)
        fingerprint_sets[s] = set(fps)
        fps_by_split[s] = fps
        within_split_duplicates[s] = len(fps) - len(fingerprint_sets[s])

    overlaps = {
        "train_validation": len(fingerprint_sets["train"] & fingerprint_sets["validation"]),
        "train_test": len(fingerprint_sets["train"] & fingerprint_sets["test"]),
        "validation_test": len(fingerprint_sets["validation"] & fingerprint_sets["test"]),
    }
    if overlaps["validation_test"]:
        raise ValueError("Trùng ảnh validation-test: cần quyết định lại chính sách tách dữ liệu.")

    forbidden = fingerprint_sets["validation"] | fingerprint_sets["test"]
    keep_train = np.array([fp not in forbidden for fp in fps_by_split["train"]], dtype=bool)
    cleaned = dict(original)
    cleaned["train"] = original["train"].iloc[np.flatnonzero(keep_train)].copy()
    removed = int((~keep_train).sum())
    if set(np.array(fps_by_split["train"], dtype=object)[keep_train]) & forbidden:
        raise AssertionError("Còn ảnh trùng xuyên tập sau cleaning.")

    # Validation/test unchanged. The original CSV and split markers remain intact.
    report = {
        "raw_total_samples": int(len(df)),
        "raw_split_sizes": {s: int(len(original[s])) for s in SPLIT_ORDER},
        "effective_split_sizes": {s: int(len(cleaned[s])) for s in SPLIT_ORDER},
        "exact_duplicate_image_groups_across_splits": overlaps,
        "within_split_repeated_rows_by_image": within_split_duplicates,
        "train_rows_excluded_due_to_cross_split_duplicate": removed,
        "policy": "remove only train images matching any validation/test image; never modify original CSV or val/test",
        "post_cleaning_exact_cross_split_overlaps": 0,
    }
    return cleaned, report


class FashionMNISTCSVDataset(Dataset):
    """Lazy tensor normalization; raw images stored compactly as uint8."""

    def __init__(self, frame: pd.DataFrame, mean: float, std: float):
        pixels = frame[PIXEL_COLUMNS].to_numpy(dtype=np.uint8, copy=True)
        self.raw_images = torch.from_numpy(np.ascontiguousarray(pixels)).reshape(-1, 1, 28, 28)
        self.labels = torch.from_numpy(frame["label"].to_numpy(dtype=np.int64, copy=True))
        self.mean = float(mean)
        self.std = float(std)
        if self.std <= 0:
            raise ValueError("Standard deviation phải lớn hơn 0.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Fixed scale and train-only z-score. No validation/test data fits statistics.
        image = self.raw_images[index].to(dtype=torch.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, self.labels[index]


def create_dataloaders(
    csv_path: str | Path,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    """Public entrypoint for Stage 3/4; returns train, validation and test loaders."""
    if batch_size < 1:
        raise ValueError("batch_size phải >= 1")
    df = _read_and_validate(csv_path)
    frames, report = _split_and_remove_leakage(df)

    # Fit normalizing statistics ONLY on cleaned training subset.
    train_pixels = frames["train"][PIXEL_COLUMNS].to_numpy(dtype=np.uint8, copy=False)
    mean = float(np.mean(train_pixels, dtype=np.float64) / 255.0)
    std = float(np.std(train_pixels, dtype=np.float64) / 255.0)
    report["normalization"] = {
        "method": "float32(pixel)/255; z-score using only effective training pixels",
        "train_mean_0_to_1": mean,
        "train_std_0_to_1": std,
        "used_validation_or_test_to_fit": False,
    }
    report["class_counts_by_split"] = {
        s: {str(i): int((frames[s]["label"] == i).sum()) for i in range(10)}
        for s in SPLIT_ORDER
    }
    report["pixel_range_original"] = [int(df[PIXEL_COLUMNS].to_numpy().min()),
                                       int(df[PIXEL_COLUMNS].to_numpy().max())]
    report["missing_values"] = 0
    report["seed"] = seed
    report["batch_size"] = batch_size

    gen = torch.Generator().manual_seed(seed)
    datasets = {s: FashionMNISTCSVDataset(frames[s], mean, std) for s in SPLIT_ORDER}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, generator=gen),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers),
    }
    return loaders, report


def _plot_samples(train_dataset: FashionMNISTCSVDataset, filepath: Path) -> None:
    """10 real grayscale samples, one per class; show pre-normalization images."""
    fig, axes = plt.subplots(2, 5, figsize=(13.0, 6.0))
    labels = train_dataset.labels.numpy()
    for i, ax in enumerate(axes.ravel()):
        first_index = int(np.flatnonzero(labels == i)[0])
        ax.imshow(train_dataset.raw_images[first_index, 0].numpy(), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{i} · {LABEL_NAMES[i]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("FashionMNIST – 10 real examples from effective training set", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.035, right=0.975, bottom=0.07, top=0.83, wspace=0.24, hspace=0.42)
    fig.savefig(filepath, dpi=165, bbox_inches="tight")
    plt.close(fig)


def _plot_label_distribution(report: dict[str, Any], filepath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(10)
    width = 0.26
    for j, (split, clr) in enumerate((('train', '#397ab9'), ('validation', '#d4a34e'), ('test', '#53a48f'))):
        counts = [report["class_counts_by_split"][split][str(i)] for i in range(10)]
        ax.bar(x + (j - 1) * width, counts, width, label=split, color=clr)
    ax.set_xticks(x, LABEL_NAMES, rotation=26, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution by split (after removing cross-split train duplicates)")
    ax.grid(axis="y", alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=165, bbox_inches="tight")
    plt.close(fig)


def _plot_pixel_histogram(train_dataset: FashionMNISTCSVDataset, filepath: Path) -> None:
    counts = np.bincount(train_dataset.raw_images.numpy().ravel(), minlength=256)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(np.arange(256), counts / counts.sum() * 100, lw=1.8, color="#3177aa")
    ax.set(xlabel="Pixel intensity (0–255)", ylabel="Share of training pixels (%)",
           title="Pixel intensity distribution – effective train only")
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(filepath, dpi=165, bbox_inches="tight")
    plt.close(fig)


def run_stage2(csv_path: str | Path, out_dir: str | Path, batch_size: int = 64, seed: int = 42) -> dict[str, Any]:
    """Audit, create sample DataLoaders, export real EDA figures and report."""
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    loaders, report = create_dataloaders(csv_path, batch_size=batch_size, seed=seed)
    image_batch, label_batch = next(iter(loaders["train"]))
    assert image_batch.shape == (batch_size, 1, 28, 28), image_batch.shape
    assert image_batch.dtype == torch.float32
    assert label_batch.dtype == torch.int64
    assert torch.isfinite(image_batch).all()
    report["batch_test"] = {"image_shape": list(image_batch.shape),
                            "image_dtype": str(image_batch.dtype),
                            "label_shape": list(label_batch.shape),
                            "label_dtype": str(label_batch.dtype)}

    _plot_samples(loaders["train"].dataset, out_dir / "figures" / "dataset_samples.png")
    _plot_label_distribution(report, out_dir / "figures" / "label_distribution.png")
    _plot_pixel_histogram(loaders["train"].dataset, out_dir / "figures" / "pixel_histogram.png")
    (out_dir / "data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice 1 / Stage 2: FashionMNIST CSV audit + DataLoaders")
    parser.add_argument("--csv", default="data/fashion_mnist_full.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_stage2(args.csv, args.out_dir, args.batch_size, args.seed)
    print("[OK] CSV schema, train-only normalization, DataLoader and EDA")
    print("Raw splits:", report["raw_split_sizes"])
    print("Effective splits:", report["effective_split_sizes"])
    print("Cross-split exact duplicates (before):", report["exact_duplicate_image_groups_across_splits"])
    print("Training records removed:", report["train_rows_excluded_due_to_cross_split_duplicate"])
    print("Train-only mean/std:", report["normalization"]["train_mean_0_to_1"],
          report["normalization"]["train_std_0_to_1"])
    print("Batch:", report["batch_test"])
    print("Figures and audit saved to:", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
