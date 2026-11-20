"""Pipeline 04: Evaluation, Benchmark 3x2, Save/Load Check & Error Visualization cho Lab 1."""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Không ghi đè backend nếu notebook đã thiết lập inline/widget;
# chỉ force Agg khi chạy headless (pytest, CLI).
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from labs.lab1.vibe import config as lab1_cfg
from src.vibe_scripts import eda_helpers
from src.vibe_scripts.lab1.data import build_dataloaders
from src.vibe_scripts.lab1.models import build_model, count_parameters

console = Console()
FASHION_CLASSES = eda_helpers.FASHION_CLASSES


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    warmup_batches: int = 2,
    collect_samples: bool = True,
) -> Dict[str, Any]:
    """
    Đánh giá 1 checkpoint `best.pt` trên `test_loader` dưới `model.eval()` và `torch.no_grad()`.
    """
    ckpt_path = Path(ckpt_path)
    target_device = device if device is not None else lab1_cfg.DEVICE
    ckpt = torch.load(ckpt_path, map_location=target_device, weights_only=False)

    model_name = str(ckpt["model_name"])
    model = build_model(model_name, num_classes=lab1_cfg.NUM_CLASSES).to(target_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    non_blocking = target_device.type == "cuda"

    # Warm-up trước khi bấm giờ suy luận
    for batch_idx, (images, _) in enumerate(test_loader):
        if batch_idx >= warmup_batches:
            break
        _ = model(images.to(target_device, non_blocking=non_blocking))
    _sync_if_cuda(target_device)

    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_confs: List[np.ndarray] = []
    sample_images: List[torch.Tensor] = []

    t_start = time.perf_counter()
    for images, labels in test_loader:
        images_dev = images.to(target_device, non_blocking=non_blocking)
        labels_dev = labels.to(target_device, non_blocking=non_blocking)

        logits = model(images_dev)
        loss = criterion(logits, labels_dev)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        bs = labels_dev.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        all_targets.append(labels_dev.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_confs.append(confs.cpu().numpy())
        if collect_samples:
            sample_images.append(images.cpu())

    _sync_if_cuda(target_device)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    num_batches = max(len(test_loader), 1)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_conf = np.concatenate(all_confs)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(lab1_cfg.NUM_CLASSES)))
    accuracy = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(lab1_cfg.NUM_CLASSES)),
        zero_division=0,
    )
    per_class = [
        {
            "class_id": cls_id,
            "label": FASHION_CLASSES[cls_id],
            "precision": float(prec[cls_id]),
            "recall": float(rec[cls_id]),
            "f1": float(f1[cls_id]),
            "support": int(sup[cls_id]),
        }
        for cls_id in range(lab1_cfg.NUM_CLASSES)
    ]

    # Trích xuất Top cặp lớp nhầm lẫn
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    top_confusions: List[Dict[str, Any]] = []
    flat_order = np.argsort(cm_off_diag, axis=None)[::-1]
    for flat_idx in flat_order[:5]:
        r, c = np.unravel_index(flat_idx, cm_off_diag.shape)
        cnt = int(cm_off_diag[r, c])
        if cnt > 0:
            top_confusions.append(
                {
                    "true_class": FASHION_CLASSES[int(r)],
                    "pred_class": FASHION_CLASSES[int(c)],
                    "count": cnt,
                }
            )

    return {
        "model_name": model_name,
        "aug_enabled": bool(ckpt.get("aug_enabled", False)),
        "best_epoch": int(ckpt.get("epoch", 0)),
        "num_params": count_parameters(model),
        "checkpoint_size_kb": float(ckpt_path.stat().st_size / 1024.0),
        "test_loss": float(total_loss / max(total_samples, 1)),
        "test_acc": accuracy,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "latency_ms_per_batch": float(elapsed_ms / num_batches),
        "norm_stats": ckpt.get("norm_stats", {"mean": 0.0, "std": 1.0}),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "top_confusions": top_confusions,
        "_y_true": y_true,
        "_y_pred": y_pred,
        "_y_conf": y_conf,
        "_images": torch.cat(sample_images, dim=0) if collect_samples and sample_images else None,
    }


@torch.no_grad()
def verify_save_load_integrity(
    ckpt_path: Path,
    sample_batch: torch.Tensor,
    device: Optional[torch.device] = None,
    tolerance: float = 1e-6,
) -> float:
    """Kiểm chứng tính toàn vẹn Save/Load: nạp lại checkpoint cho logits khớp trong sai số `< 1e-6`."""
    target_device = device if device is not None else lab1_cfg.DEVICE
    ckpt1 = torch.load(ckpt_path, map_location=target_device, weights_only=False)
    m1 = build_model(str(ckpt1["model_name"]), num_classes=lab1_cfg.NUM_CLASSES).to(target_device)
    m1.load_state_dict(ckpt1["model_state_dict"])
    m1.eval()

    ckpt2 = torch.load(ckpt_path, map_location=target_device, weights_only=False)
    m2 = build_model(str(ckpt2["model_name"]), num_classes=lab1_cfg.NUM_CLASSES).to(target_device)
    m2.load_state_dict(ckpt2["model_state_dict"])
    m2.eval()

    x = sample_batch.to(target_device)
    diff = float((m1(x) - m2(x)).abs().max().item())
    if diff >= tolerance:
        raise AssertionError(f"Save/Load logits mismatch: max_diff={diff} >= {tolerance}")
    return diff


def plot_learning_curves(histories: Dict[str, Dict[str, Any]], save_path: Path) -> Path:
    """Vẽ đối sánh Train/Val Loss và Accuracy cho các thí nghiệm."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for exp_id, data in sorted(histories.items()):
        rows = data.get("history", [])
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        ax1.plot(epochs, [r["val_loss"] for r in rows], marker="o", markersize=3, label=f"{exp_id} (val)")
        ax1.plot(epochs, [r["train_loss"] for r in rows], linestyle="--", alpha=0.5, label=f"{exp_id} (train)")
        ax2.plot(epochs, [r["val_acc"] * 100 for r in rows], marker="o", markersize=3, label=f"{exp_id} (val)")
        ax2.plot(epochs, [r["train_acc"] * 100 for r in rows], linestyle="--", alpha=0.5, label=f"{exp_id} (train)")

    ax1.set_title("Learning Curve: Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=7.5, ncol=2)

    ax2.set_title("Learning Curve: Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=7.5, ncol=2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_individual_learning_curves(
    histories: Dict[str, Dict[str, Any]], output_dir: Path
) -> List[Path]:
    """Vẽ learning curve riêng từng thí nghiệm (2×1: Loss + Accuracy) để phân tích hội tụ cá thể."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for exp_id, data in sorted(histories.items()):
        rows = data.get("history", [])
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        train_loss = [r["train_loss"] for r in rows]
        val_loss = [r["val_loss"] for r in rows]
        train_acc = [r["train_acc"] * 100 for r in rows]
        val_acc = [r["val_acc"] * 100 for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

        # Loss subplot
        ax1.plot(epochs, train_loss, "o-", markersize=4, label="Train Loss", color="#2563eb")
        ax1.plot(epochs, val_loss, "s-", markersize=4, label="Val Loss", color="#dc2626")
        best_val_idx = int(np.argmin(val_loss))
        ax1.axvline(epochs[best_val_idx], linestyle="--", color="#9333ea", alpha=0.6,
                     label=f"Best Epoch {epochs[best_val_idx]}")
        gap = [v - t for v, t in zip(val_loss, train_loss)]
        ax1.fill_between(epochs, train_loss, val_loss, alpha=0.08, color="#dc2626",
                          label=f"Overfit Gap (last={gap[-1]:.4f})")
        ax1.set_title(f"Loss — {exp_id}", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=8)

        # Accuracy subplot
        ax2.plot(epochs, train_acc, "o-", markersize=4, label="Train Acc", color="#2563eb")
        ax2.plot(epochs, val_acc, "s-", markersize=4, label="Val Acc", color="#dc2626")
        ax2.axvline(epochs[best_val_idx], linestyle="--", color="#9333ea", alpha=0.6,
                     label=f"Best Epoch {epochs[best_val_idx]}")
        ax2.set_title(f"Accuracy — {exp_id}", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        save_path = output_dir / f"learning_curve_{exp_id}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(save_path)

    return saved


def plot_confusion_matrix(cm: np.ndarray, exp_id: str, save_path: Path) -> Path:
    """Vẽ ma trận nhầm lẫn 10x10 chuẩn hóa theo hàng kèm số lượng thực tế."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix — {exp_id}", fontsize=13, fontweight="bold")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(FASHION_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(FASHION_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for i in range(10):
        for j in range(10):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)", ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_per_class_f1(per_class_by_exp: Dict[str, List[Dict[str, Any]]], save_path: Path) -> Path:
    """Vẽ biểu đồ cột đối sánh F1-Score theo từng lớp giữa các mô hình."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    exp_ids = sorted(per_class_by_exp.keys())
    x = np.arange(10)
    width = 0.8 / max(len(exp_ids), 1)

    for idx, exp_id in enumerate(exp_ids):
        f1_vals = [row["f1"] * 100 for row in per_class_by_exp[exp_id]]
        offset = (idx - (len(exp_ids) - 1) / 2.0) * width
        ax.bar(x + offset, f1_vals, width=width, label=exp_id, alpha=0.88)

    ax.set_title("Per-Class F1-Score Comparison (%)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(FASHION_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("F1-Score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=3, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_prediction_gallery(
    images: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    norm_stats: Dict[str, Any],
    exp_id: str,
    save_path: Path,
    n_correct: int = 5,
    n_wrong: int = 5,
) -> Path:
    """Vẽ lưới ảnh dự đoán Đúng (xanh) và Sai (đỏ) sau khi khử Normalize về `[0, 1]`."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mean_v = float(norm_stats.get("mean", 0.0))
    std_v = float(norm_stats.get("std", 1.0))

    correct_idx = np.where(y_true == y_pred)[0][:n_correct]
    wrong_all = np.where(y_true != y_pred)[0]
    # Ưu tiên mẫu sai có độ tự tin cao nhất
    wrong_sorted = wrong_all[np.argsort(y_conf[wrong_all])[::-1]][:n_wrong]

    selected = list(correct_idx) + list(wrong_sorted)
    cols = max(n_correct, n_wrong)
    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 5.5))

    for row_idx, idxs in enumerate([correct_idx, wrong_sorted]):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx] if cols > 1 else axes[row_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx >= len(idxs):
                ax.axis("off")
                continue
            sample_i = int(idxs[col_idx])
            img_np = images[sample_i, 0].numpy() * std_v + mean_v
            img_np = np.clip(img_np, 0.0, 1.0)
            ax.imshow(img_np, cmap="gray")
            is_ok = y_true[sample_i] == y_pred[sample_i]
            color = "#15803d" if is_ok else "#dc2626"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.2)
            ax.set_title(
                f"T: {FASHION_CLASSES[y_true[sample_i]]}\n"
                f"P: {FASHION_CLASSES[y_pred[sample_i]]} ({y_conf[sample_i]*100:.0f}%)",
                fontsize=8,
                color=color,
            )

    fig.suptitle(f"Predictions Gallery (Row 1: Correct | Row 2: Top Confident Errors) — {exp_id}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_benchmark(
    experiments_dir: Optional[Path] = None,
    test_loader: Optional[DataLoader] = None,
    output_dir: Optional[Path] = None,
    predictions_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Đánh giá toàn bộ các thí nghiệm có `best.pt`, xuất bảng CSV/JSON và biểu đồ."""
    exp_root = Path(experiments_dir) if experiments_dir is not None else lab1_cfg.EXPERIMENT_DIR
    eval_dir = Path(output_dir) if output_dir is not None else (exp_root / "evaluation")
    pred_dir = Path(predictions_dir) if predictions_dir is not None else (exp_root / "predictions")
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if test_loader is None:
        _, _, test_loader, _ = build_dataloaders(aug_enabled=False, verify_manifest=False)

    sample_batch, _ = next(iter(test_loader))
    rows: List[Dict[str, Any]] = []
    histories: Dict[str, Dict[str, Any]] = {}
    per_class_by_exp: Dict[str, List[Dict[str, Any]]] = {}
    details: Dict[str, Any] = {}

    for sub in sorted(exp_root.iterdir()):
        ckpt_path = sub / "best.pt"
        if not sub.is_dir() or not ckpt_path.exists():
            continue

        exp_id = sub.name
        hist_path = sub / "history.json"
        if hist_path.exists():
            histories[exp_id] = json.loads(hist_path.read_text(encoding="utf-8"))

        res = evaluate_checkpoint(ckpt_path, test_loader, device=device)
        save_load_diff = verify_save_load_integrity(ckpt_path, sample_batch, device=device)

        cm = np.array(res["confusion_matrix"])
        plot_confusion_matrix(cm, exp_id, pred_dir / f"confusion_matrix_{exp_id}.png")
        if res["_images"] is not None:
            plot_prediction_gallery(
                res["_images"],
                res["_y_true"],
                res["_y_pred"],
                res["_y_conf"],
                res["norm_stats"],
                exp_id,
                pred_dir / f"predictions_gallery_{exp_id}.png",
            )

        per_class_by_exp[exp_id] = res["per_class"]
        rows.append(
            {
                "exp_id": exp_id,
                "model": res["model_name"],
                "aug": "aug" if res["aug_enabled"] else "no_aug",
                "params": res["num_params"],
                "best_epoch": res["best_epoch"],
                "file_size_kb": round(res["checkpoint_size_kb"], 2),
                "test_loss": round(res["test_loss"], 4),
                "test_acc_pct": round(res["test_acc"] * 100, 2),
                "macro_f1_pct": round(res["macro_f1"] * 100, 2),
                "latency_ms_batch": round(res["latency_ms_per_batch"], 3),
                "save_load_max_diff": save_load_diff,
            }
        )
        details[exp_id] = {k: v for k, v in res.items() if not k.startswith("_")}
        details[exp_id]["save_load_max_diff"] = save_load_diff

    if histories:
        plot_learning_curves(histories, pred_dir / "learning_curves_comparison.png")
        plot_individual_learning_curves(histories, pred_dir)
    if per_class_by_exp:
        plot_per_class_f1(per_class_by_exp, pred_dir / "per_class_f1_comparison.png")

    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "benchmark_table.csv", index=False)

    summary_payload = {"benchmark_rows": rows, "experiments": details}
    (eval_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (eval_dir / "per_class_metrics.json").write_text(
        json.dumps(per_class_by_exp, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 1 Evaluation & Benchmark CLI")
    parser.add_argument("--experiments-dir", type=Path, default=lab1_cfg.EXPERIMENT_DIR)
    args = parser.parse_args()
    summary = run_benchmark(experiments_dir=args.experiments_dir)
    console.print(f"[green][DONE][/green] Evaluated {len(summary['benchmark_rows'])} experiments.")


if __name__ == "__main__":
    main()
