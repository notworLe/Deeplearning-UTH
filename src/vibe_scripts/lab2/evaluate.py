"""
Pipeline 04: Đánh giá 1 lần trên tập Test (3.302 ảnh), kiểm chứng Cat vs Dog & Error Analysis cho Lab 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from labs.lab2.vibe import config as lab2_cfg
from src.vibe_scripts.lab2.data import build_dataloaders
from src.vibe_scripts.lab2.models import build_resnet18_pet

console = Console()

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def denormalize_imagenet(tensor_chw: torch.Tensor) -> np.ndarray:
    """Khử chuẩn hóa ImageNet `[3, H, W]` về ảnh numpy `[H, W, 3]` trong đoạn `[0.0, 1.0]`."""
    arr = tensor_chw.detach().cpu().numpy().astype(np.float32)
    img = np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    return np.transpose(img, (1, 2, 0))


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Nạp checkpoint `best.pt` vào kiến trúc ResNet-18 37 lớp ở chế độ `eval()`."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode = payload.get("mode", "baseline")
    num_classes = int(payload.get("num_classes", lab2_cfg.NUM_CLASSES))
    model = build_resnet18_pet(mode=mode, num_classes=num_classes, use_pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def verify_save_load_integrity(checkpoint_path: Path, sample_batch: torch.Tensor, device: torch.device) -> float:
    """Đo sai số logits tối đa giữa 2 lần nạp độc lập cùng checkpoint (yêu cầu < 1e-6)."""
    m1, _ = _load_model_from_checkpoint(checkpoint_path, device)
    m2, _ = _load_model_from_checkpoint(checkpoint_path, device)
    x = sample_batch.to(device)
    diff = (m1(x) - m2(x)).abs().max().item()
    return float(diff)


def compute_species_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_map: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Phân tách hiệu năng và tỷ lệ lỗi giữa 12 giống Mèo và 25 giống Chó trên tập Test.

    Kèm kiểm định z-test hai tỷ lệ (two-proportion z-test) cho chênh lệch error rate Cat vs Dog.
    """
    cat_ids = sorted(int(c) for c, info in species_map.items() if info["is_cat"])
    dog_ids = sorted(int(c) for c, info in species_map.items() if not info["is_cat"])

    cat_mask = np.isin(y_true, cat_ids)
    dog_mask = np.isin(y_true, dog_ids)

    n_cat = int(cat_mask.sum())
    n_dog = int(dog_mask.sum())
    cat_err = int((y_pred[cat_mask] != y_true[cat_mask]).sum())
    dog_err = int((y_pred[dog_mask] != y_true[dog_mask]).sum())

    cat_err_rate = cat_err / max(n_cat, 1)
    dog_err_rate = dog_err / max(n_dog, 1)

    # Per-class macro F1 within each species group
    _, _, f1_all, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(species_map))), zero_division=0
    )
    cat_macro_f1 = float(np.mean([f1_all[c] for c in cat_ids]))
    dog_macro_f1 = float(np.mean([f1_all[c] for c in dog_ids]))

    # Cross-species confusion (Cat predicted as Dog or vice versa)
    pred_is_cat = np.isin(y_pred, cat_ids)
    cat_to_dog = int((cat_mask & (~pred_is_cat)).sum())
    dog_to_cat = int((dog_mask & pred_is_cat).sum())
    cat_within_err = cat_err - cat_to_dog
    dog_within_err = dog_err - dog_to_cat

    # Two-proportion z-test (H0: cat_err_rate == dog_err_rate)
    p_pool = (cat_err + dog_err) / max(n_cat + n_dog, 1)
    se = math.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / max(n_cat, 1) + 1.0 / max(n_dog, 1)), 1e-12))
    z_stat = (cat_err_rate - dog_err_rate) / se
    p_value = math.erfc(abs(z_stat) / math.sqrt(2.0))

    return {
        "cat_samples": n_cat,
        "cat_errors": cat_err,
        "cat_error_rate": cat_err_rate,
        "cat_accuracy": 1.0 - cat_err_rate,
        "cat_macro_f1": cat_macro_f1,
        "cat_within_species_errors": cat_within_err,
        "cat_to_dog_errors": cat_to_dog,
        "dog_samples": n_dog,
        "dog_errors": dog_err,
        "dog_error_rate": dog_err_rate,
        "dog_accuracy": 1.0 - dog_err_rate,
        "dog_macro_f1": dog_macro_f1,
        "dog_within_species_errors": dog_within_err,
        "dog_to_cat_errors": dog_to_cat,
        "error_rate_diff_cat_minus_dog_pp": (cat_err_rate - dog_err_rate) * 100.0,
        "z_stat": float(z_stat),
        "p_value": float(p_value),
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    test_loader: DataLoader,
    classes: Sequence[str],
    species_map: Dict[int, Dict[str, Any]],
    device: Optional[torch.device] = None,
    top_k_errors: int = 10,
) -> Dict[str, Any]:
    """Đánh giá 1 lần duy nhất checkpoint `best.pt` trên tập Test."""
    dev = device if device is not None else lab2_cfg.DEVICE
    model, payload = _load_model_from_checkpoint(checkpoint_path, dev)
    criterion = nn.CrossEntropyLoss()

    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    sample_images: List[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    batch_times: List[float] = []
    first_batch: Optional[torch.Tensor] = None

    for b_idx, (images, targets) in enumerate(test_loader):
        if first_batch is None:
            first_batch = images[:4].clone()
        images_dev = images.to(dev, non_blocking=True)
        targets_dev = targets.to(dev, non_blocking=True)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(images_dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if b_idx >= 2:  # bỏ qua 2 batch warmup
            batch_times.append(elapsed_ms)

        loss = criterion(logits, targets_dev)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        bs = targets.size(0)
        total_loss += loss.detach().item() * bs
        total_samples += bs

        all_targets.append(targets.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.max(dim=1).values.cpu().numpy())
        sample_images.append(images.cpu())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_conf = np.concatenate(all_probs)
    imgs_tensor = torch.cat(sample_images, dim=0)

    num_classes = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    test_loss = total_loss / max(total_samples, 1)
    test_acc = float((y_true == y_pred).mean())
    macro_f1 = float(np.mean(f1))
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))

    per_class = [
        {
            "class_id": i,
            "label": classes[i],
            "species": species_map[i]["species"],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }
        for i in range(num_classes)
    ]

    # Top confusion pairs (off-diagonal)
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": classes[i],
                    "true_species": species_map[i]["species"],
                    "pred_class": classes[j],
                    "pred_species": species_map[j]["species"],
                    "count": int(cm[i, j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)

    species_stats = compute_species_analysis(y_true, y_pred, species_map)
    save_load_diff = verify_save_load_integrity(checkpoint_path, first_batch, dev) if first_batch is not None else 0.0

    # Collect 5 correct + top_k_errors confident errors for gallery
    correct_idxs = np.where(y_true == y_pred)[0]
    wrong_idxs = np.where(y_true != y_pred)[0]
    wrong_sorted = wrong_idxs[np.argsort(-y_conf[wrong_idxs])][:top_k_errors]
    correct_pick = correct_idxs[:5]

    gallery_items = {
        "correct": [
            {
                "image": denormalize_imagenet(imgs_tensor[idx]),
                "true_label": classes[int(y_true[idx])],
                "pred_label": classes[int(y_pred[idx])],
                "conf": float(y_conf[idx]),
            }
            for idx in correct_pick
        ],
        "errors": [
            {
                "image": denormalize_imagenet(imgs_tensor[idx]),
                "true_label": classes[int(y_true[idx])],
                "true_species": species_map[int(y_true[idx])]["species"],
                "pred_label": classes[int(y_pred[idx])],
                "pred_species": species_map[int(y_pred[idx])]["species"],
                "conf": float(y_conf[idx]),
            }
            for idx in wrong_sorted
        ],
    }

    return {
        "exp_id": payload.get("exp_id", checkpoint_path.parent.name),
        "mode": payload.get("mode", "unknown"),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "total_params": int(payload.get("total_params", 0)),
        "trainable_params": int(payload.get("trainable_params", 0)),
        "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "test_loss": float(test_loss),
        "test_acc": test_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "latency_ms_per_batch": float(np.mean(batch_times)) if batch_times else 0.0,
        "save_load_max_diff": save_load_diff,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "top_confusions": pairs[:10],
        "species_analysis": species_stats,
        "gallery_items": gallery_items,
    }


def plot_learning_curves_comparison(histories: Dict[str, Dict[str, Any]], save_path: Path) -> plt.Figure:
    """Vẽ đối chiếu Learning Curves (Train/Val Loss & Val Acc) của cả 3 run trên 1 hình."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "resnet18_baseline": "#dc2626",
        "resnet18_transfer": "#2563eb",
        "resnet18_finetune": "#16a34a",
    }

    for exp_id, h in histories.items():
        hist = h["history"]
        eps = [r["epoch"] for r in hist]
        tr_l = [r["train_loss"] for r in hist]
        va_l = [r["val_loss"] for r in hist]
        va_a = [r["val_acc"] * 100 for r in hist]
        c = colors.get(exp_id, "#6b7280")
        best_ep = h["best_epoch"]

        ax1.plot(eps, tr_l, linestyle="--", alpha=0.55, color=c, label=f"{exp_id} train")
        ax1.plot(eps, va_l, linewidth=2.0, color=c, label=f"{exp_id} val (best ep{best_ep})")
        ax2.plot(eps, va_a, linewidth=2.0, color=c, label=f"{exp_id} (best ep{best_ep})")

    ax1.set_title("Train vs Val Loss (3 Runs)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8)

    ax2.set_title("Validation Accuracy (%) — 3 Runs", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_individual_learning_curve(history_payload: Dict[str, Any], save_path: Path) -> plt.Figure:
    """Vẽ biểu đồ 2 panel (Loss & Accuracy) cho 1 run, đánh dấu Best Epoch và gap tại best."""
    exp_id = history_payload["exp_id"]
    hist = history_payload["history"]
    best_ep = history_payload["best_epoch"]
    eps = [r["epoch"] for r in hist]
    tr_l = [r["train_loss"] for r in hist]
    va_l = [r["val_loss"] for r in hist]
    tr_a = [r["train_acc"] * 100 for r in hist]
    va_a = [r["val_acc"] * 100 for r in hist]

    b_row = hist[best_ep - 1]
    gap_loss_best = b_row["val_loss"] - b_row["train_loss"]
    gap_acc_best = (b_row["train_acc"] - b_row["val_acc"]) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax1.plot(eps, tr_l, label="Train Loss", color="#2563eb", linewidth=1.8)
    ax1.plot(eps, va_l, label="Val Loss", color="#dc2626", linewidth=1.8)
    ax1.fill_between(eps, tr_l, va_l, color="#fca5a5", alpha=0.22, label=f"Gap @best: {gap_loss_best:+.3f}")
    ax1.axvline(best_ep, color="#7c3aed", linestyle="--", linewidth=1.4, label=f"Best Epoch = {best_ep}")
    ax1.set_title(f"{exp_id} — Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.5)

    ax2.plot(eps, tr_a, label="Train Acc", color="#2563eb", linewidth=1.8)
    ax2.plot(eps, va_a, label="Val Acc", color="#16a34a", linewidth=1.8)
    ax2.axvline(best_ep, color="#7c3aed", linestyle="--", linewidth=1.4, label=f"Gap Acc @best: {gap_acc_best:+.1f}pp")
    ax2.set_title(f"{exp_id} — Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix_37(cm: np.ndarray, classes: Sequence[str], exp_id: str, save_path: Path) -> plt.Figure:
    """Vẽ Confusion Matrix 37×37 chuẩn hóa theo hàng (Recall) cho Oxford-IIIT Pets."""
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03, label="Recall theo hàng")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=70, ha="right", fontsize=7.5)
    ax.set_yticklabels(classes, fontsize=7.5)
    ax.set_xlabel("Predicted Breed")
    ax.set_ylabel("True Breed")
    ax.set_title(f"Confusion Matrix (37×37, Row-Normalized) — {exp_id}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cat_vs_dog_error(species_by_exp: Dict[str, Dict[str, Any]], save_path: Path) -> plt.Figure:
    """Vẽ so sánh Test Error Rate (%) giữa 12 giống Mèo và 25 giống Chó trên cả 3 run."""
    exp_ids = list(species_by_exp.keys())
    cat_errs = [species_by_exp[e]["cat_error_rate"] * 100.0 for e in exp_ids]
    dog_errs = [species_by_exp[e]["dog_error_rate"] * 100.0 for e in exp_ids]

    x = np.arange(len(exp_ids))
    w = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - w / 2, cat_errs, w, color="#f97316", edgecolor="#1f2937", label="Mèo (12 giống)")
    b2 = ax.bar(x + w / 2, dog_errs, w, color="#2563eb", edgecolor="#1f2937", label="Chó (25 giống)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(cat_errs + dog_errs) * 1.18 + 3)
    ax.set_ylabel("Test Error Rate (%)")
    ax.set_title("So Sánh Tỷ Lệ Lỗi Trên Test: Giống Mèo (12) vs Giống Chó (25)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_gallery(gallery_items: Dict[str, List[Dict[str, Any]]], exp_id: str, save_path: Path) -> plt.Figure:
    """Vẽ 10 mẫu dự đoán sai có confidence cao nhất (2×5) đã khử chuẩn hóa ImageNet."""
    errors = gallery_items["errors"][:10]
    n_cols = 5
    n_rows = max(1, math.ceil(len(errors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax in axes_flat:
        ax.axis("off")

    for ax, item in zip(axes_flat, errors):
        ax.imshow(item["image"])
        ax.set_title(
            f"T: {item['true_label']} ({item['true_species']})\nP: {item['pred_label']} ({item['conf']*100:.1f}%)",
            fontsize=8.2,
            color="#b91c1c",
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#dc2626")
            spine.set_linewidth(2.0)

    fig.suptitle(f"Top-{len(errors)} Confident Errors (Test Set) — {exp_id}", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_benchmark(
    experiments_dir: Path = lab2_cfg.EXPERIMENT_DIR,
    run_names: Sequence[str] = lab2_cfg.RUN_NAMES,
    test_loader: Optional[DataLoader] = None,
    classes: Optional[Sequence[str]] = None,
    species_map: Optional[Dict[int, Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Đánh giá trọn bộ các run trong `experiments/lab2/`, xuất JSON/CSV và toàn bộ biểu đồ."""
    dev = device if device is not None else lab2_cfg.DEVICE
    if test_loader is None or classes is None or species_map is None:
        _, _, test_loader, split_meta = build_dataloaders(num_workers=num_workers)
        classes = split_meta["classes"]
        species_map = split_meta["species_map"]

    eval_dir = Path(experiments_dir) / "evaluation"
    pred_dir = Path(experiments_dir) / "predictions"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, Any]] = {}
    histories: Dict[str, Dict[str, Any]] = {}
    per_class_by_exp: Dict[str, Any] = {}
    species_by_exp: Dict[str, Dict[str, Any]] = {}

    for run_id in run_names:
        run_dir = Path(experiments_dir) / run_id
        ckpt = run_dir / "best.pt"
        hist_path = run_dir / "history.json"
        if not ckpt.exists():
            continue

        hist_data = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else None
        if hist_data is not None:
            histories[run_id] = hist_data
            fig_lc = plot_individual_learning_curve(hist_data, pred_dir / f"learning_curve_{run_id}.png")
            plt.close(fig_lc)

        res = evaluate_checkpoint(ckpt, test_loader, classes=classes, species_map=species_map, device=dev)
        train_time = float(hist_data["total_train_time_sec"]) if hist_data is not None else 0.0
        best_ep = int(hist_data["best_epoch"]) if hist_data is not None else res["best_epoch"]
        best_ep_row = hist_data["history"][best_ep - 1] if hist_data is not None else None

        fig_cm = plot_confusion_matrix_37(np.asarray(res["confusion_matrix"]), classes, run_id, pred_dir / f"confusion_matrix_{run_id}.png")
        plt.close(fig_cm)
        fig_gal = plot_error_gallery(res["gallery_items"], run_id, pred_dir / f"predictions_gallery_{run_id}.png")
        plt.close(fig_gal)

        rows.append({
            "exp_id": run_id,
            "mode": res["mode"],
            "trainable_params": res["trainable_params"],
            "total_params": res["total_params"],
            "best_epoch": best_ep,
            "train_time_sec": round(train_time, 2),
            "train_acc_at_best_pct": round(best_ep_row["train_acc"] * 100, 2) if best_ep_row else 0.0,
            "val_acc_at_best_pct": round(best_ep_row["val_acc"] * 100, 2) if best_ep_row else 0.0,
            "val_loss_at_best": round(best_ep_row["val_loss"], 4) if best_ep_row else 0.0,
            "test_loss": round(res["test_loss"], 4),
            "test_acc_pct": round(res["test_acc"] * 100, 2),
            "macro_f1_pct": round(res["macro_f1"] * 100, 2),
            "cat_err_pct": round(res["species_analysis"]["cat_error_rate"] * 100, 2),
            "dog_err_pct": round(res["species_analysis"]["dog_error_rate"] * 100, 2),
            "save_load_max_diff": res["save_load_max_diff"],
        })

        res_serializable = {k: v for k, v in res.items() if k != "gallery_items"}
        res_serializable["train_time_sec"] = train_time
        details[run_id] = res_serializable
        per_class_by_exp[run_id] = res["per_class"]
        species_by_exp[run_id] = res["species_analysis"]

    if histories:
        fig_cmp = plot_learning_curves_comparison(histories, pred_dir / "learning_curves_comparison.png")
        plt.close(fig_cmp)
    if species_by_exp:
        fig_sp = plot_cat_vs_dog_error(species_by_exp, pred_dir / "cat_vs_dog_error_comparison.png")
        plt.close(fig_sp)

    if rows:
        with open(eval_dir / "benchmark_table.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {"benchmark_rows": rows, "experiments": details}
    (eval_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (eval_dir / "per_class_metrics.json").write_text(json.dumps(per_class_by_exp, indent=2, ensure_ascii=False), encoding="utf-8")
    (eval_dir / "species_analysis.json").write_text(json.dumps(species_by_exp, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    """Điểm vào CLI đánh giá và xuất bảng benchmark Lab 2."""
    parser = argparse.ArgumentParser(description="Lab 2 Evaluation & Error Analysis CLI")
    parser.add_argument("--experiments-dir", type=Path, default=lab2_cfg.EXPERIMENT_DIR)
    args = parser.parse_args()
    summary = run_benchmark(experiments_dir=args.experiments_dir)
    console.print(f"[green][DONE][/green] Evaluated {len(summary['benchmark_rows'])} runs.")


if __name__ == "__main__":
    main()
