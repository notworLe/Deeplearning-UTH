"""
Module trực quan hóa EDA cho Lab 2 (Oxford-IIIT Pets).

Xuất 4 biểu đồ phục vụ mục EDA trong `REPORT.md` và `lab2.ipynb`:
1. `eda_class_distribution.png`: số ảnh 37 giống trên trainval vs test (tô màu tách Mèo/Chó).
2. `eda_size_aspect_ratio.png`: phân bố kích thước (W, H) và tỉ lệ cạnh (W/H).
3. `eda_sample_grid.png`: lưới ảnh mẫu 12 giống (6 mèo, 6 chó) kèm kích thước gốc.
4. `eda_species_ratio.png`: tỉ lệ Mèo/Chó trên Train (3.680), Val (367), Test (3.302).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from labs.lab2.vibe import config as lab2_cfg
from src.vibe_scripts.lab2.data import inspect_dataset_quality, stratified_split_val_test


def plot_class_distribution(audit: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
    """Vẽ số ảnh theo 37 giống ở trainval và test gốc, tô màu phân biệt Mèo (cam) vs Chó (xanh)."""
    classes = audit["classes"]
    species_map = audit["species_map"]
    tr_counts = [audit["trainval_class_counts"][c] for c in classes]
    te_counts = [audit["test_raw_class_counts"][c] for c in classes]

    x = np.arange(len(classes))
    width = 0.42
    fig, ax = plt.subplots(figsize=(16, 5.5))

    cat_mask = [species_map[i]["is_cat"] for i in range(len(classes))]
    tr_colors = ["#f97316" if is_cat else "#2563eb" for is_cat in cat_mask]
    te_colors = ["#fdba74" if is_cat else "#93c5fd" for is_cat in cat_mask]

    ax.bar(x - width / 2, tr_counts, width, color=tr_colors, edgecolor="#1f2937", linewidth=0.4, label="trainval (Cam=Mèo, Xanh=Chó)")
    ax.bar(x + width / 2, te_counts, width, color=te_colors, edgecolor="#1f2937", linewidth=0.4, label="test gốc (nhạt)")
    ax.axhline(100, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.7, label="Mốc 100 ảnh/lớp")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Số lượng ảnh")
    ax.set_title("Phân Bố 37 Giống Oxford-IIIT Pets (12 Giống Mèo — 25 Giống Chó)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_size_and_aspect_ratio(audit: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
    """Vẽ phân bố kích thước (Width vs Height) và histogram tỉ lệ cạnh (W/H)."""
    w = np.asarray(audit["widths"], dtype=np.float32)
    h = np.asarray(audit["heights"], dtype=np.float32)
    ar = np.asarray(audit["aspect_ratios"], dtype=np.float32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hexbin(w, h, gridsize=35, cmap="Blues", mincnt=1)
    ax1.axvline(224, color="#dc2626", linestyle="--", linewidth=1.2, label="Crop 224px")
    ax1.axhline(224, color="#dc2626", linestyle="--", linewidth=1.2)
    ax1.set_xlim(0, min(1200, float(np.percentile(w, 99.5)) * 1.05))
    ax1.set_ylim(0, min(1200, float(np.percentile(h, 99.5)) * 1.05))
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    ax1.set_title(
        f"Phân Bố Kích Thước Gốc (Median {int(np.median(w))}×{int(np.median(h))} px)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ar_clip = np.clip(ar, 0.3, 2.5)
    ax2.hist(ar_clip, bins=45, color="#10b981", edgecolor="#1f2937", linewidth=0.5, alpha=0.85)
    ax2.axvline(1.0, color="#dc2626", linestyle="--", linewidth=1.3, label="Vuông (W/H = 1.0)")
    ax2.axvline(float(np.median(ar)), color="#f59e0b", linestyle="-.", linewidth=1.3, label=f"Median ({np.median(ar):.2f})")
    ax2.set_xlabel("Tỉ lệ cạnh Width / Height (clip [0.3, 2.5])")
    ax2.set_ylabel("Số lượng ảnh")
    ax2.set_title("Phân Bố Tỉ Lệ Cạnh (Aspect Ratio)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sample_grid(
    data_root: Path = lab2_cfg.DATA_PATH,
    species_map: Optional[Dict[int, Dict[str, Any]]] = None,
    seed: int = lab2_cfg.SEED,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Vẽ lưới 12 ảnh mẫu (6 giống mèo hàng trên, 6 giống chó hàng dưới) kèm kích thước gốc."""
    ds = OxfordIIITPet(root=data_root, split="trainval", target_types="category", download=False)
    if species_map is None:
        from src.vibe_scripts.lab2.data import extract_species_mapping
        species_map = extract_species_mapping(ds)

    rng = np.random.RandomState(seed)
    cat_ids = sorted(c for c, info in species_map.items() if info["is_cat"])[:6]
    dog_ids = sorted(c for c, info in species_map.items() if not info["is_cat"])[:6]
    chosen_classes = cat_ids + dog_ids

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    labels_arr = np.asarray(ds._labels)

    for ax, cls_id in zip(axes.flatten(), chosen_classes):
        idxs = np.where(labels_arr == cls_id)[0]
        pick = int(rng.choice(idxs))
        img_path = Path(ds._images[pick])
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            w, h = im_rgb.size
            ax.imshow(im_rgb)
        sp = species_map[cls_id]["species"]
        color = "#c2410c" if sp == "Cat" else "#1d4ed8"
        ax.set_title(f"[{sp}] {ds.classes[cls_id]}\n({w}×{h}, {img_path.name[:16]})", fontsize=8, color=color, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Ảnh Mẫu Thực Tế Oxford-IIIT Pets (Hàng 1: Mèo — Hàng 2: Chó)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_species_ratio(
    data_root: Path = lab2_cfg.DATA_PATH,
    val_ratio: float = lab2_cfg.VAL_RATIO,
    seed: int = lab2_cfg.SEED,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Vẽ tỷ lệ Mèo vs Chó trên 3 tập thực nghiệm Train (3.680), Val (367), Test (3.302)."""
    ds_tr = OxfordIIITPet(root=data_root, split="trainval", target_types="category", download=False)
    ds_te = OxfordIIITPet(root=data_root, split="test", target_types="category", download=False)
    from src.vibe_scripts.lab2.data import extract_species_mapping

    species_map = extract_species_mapping(ds_te)
    val_idx, test_idx = stratified_split_val_test(ds_te._labels, val_ratio=val_ratio, seed=seed)

    def _counts(labels: List[int]) -> tuple[int, int]:
        cats = sum(1 for y in labels if species_map[int(y)]["is_cat"])
        return cats, len(labels) - cats

    tr_c, tr_d = _counts(ds_tr._labels)
    va_c, va_d = _counts([ds_te._labels[i] for i in val_idx])
    te_c, te_d = _counts([ds_te._labels[i] for i in test_idx])

    splits = [f"Train\n({tr_c + tr_d:,})", f"Val\n({va_c + va_d:,})", f"Test\n({te_c + te_d:,})"]
    cat_pct = [tr_c / (tr_c + tr_d) * 100, va_c / (va_c + va_d) * 100, te_c / (te_c + te_d) * 100]
    dog_pct = [100.0 - p for p in cat_pct]
    cat_abs = [tr_c, va_c, te_c]
    dog_abs = [tr_d, va_d, te_d]

    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x, cat_pct, width=0.5, color="#f97316", edgecolor="#1f2937", label="Mèo (12 giống)")
    ax.bar(x, dog_pct, width=0.5, bottom=cat_pct, color="#2563eb", edgecolor="#1f2937", label="Chó (25 giống)")

    for i in range(len(splits)):
        ax.text(i, cat_pct[i] / 2, f"{cat_abs[i]:,}\n({cat_pct[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.text(i, cat_pct[i] + dog_pct[i] / 2, f"{dog_abs[i]:,}\n({dog_pct[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("Tỷ Lệ Mèo vs Chó Trên 3 Tập Thực Nghiệm (Stratified Split)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_eda_suite(
    data_root: Path = lab2_cfg.DATA_PATH,
    out_dir: Path = lab2_cfg.EDA_DIR,
    seed: int = lab2_cfg.SEED,
) -> Dict[str, Any]:
    """Chạy toàn bộ Data Cleaning Audit + xuất 4 biểu đồ EDA và lưu `eda_summary.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = inspect_dataset_quality(data_root=data_root)

    f1 = plot_class_distribution(audit, save_path=out_dir / "eda_class_distribution.png")
    plt.close(f1)
    f2 = plot_size_and_aspect_ratio(audit, save_path=out_dir / "eda_size_aspect_ratio.png")
    plt.close(f2)
    f3 = plot_sample_grid(data_root=data_root, species_map=audit["species_map"], seed=seed, save_path=out_dir / "eda_sample_grid.png")
    plt.close(f3)
    f4 = plot_species_ratio(data_root=data_root, seed=seed, save_path=out_dir / "eda_species_ratio.png")
    plt.close(f4)

    summary = {k: v for k, v in audit.items() if k not in ("widths", "heights", "aspect_ratios")}
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
