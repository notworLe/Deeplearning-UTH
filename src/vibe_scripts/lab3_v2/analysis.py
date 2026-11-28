"""Biểu đồ trực quan hóa cho Lab 3 v2 (Loss curve, Before/After comparison, Confusion Matrix 2×2)."""

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib
import numpy as np


def _ensure_headless_matplotlib() -> None:
    """Chuyển sang backend Agg thuần (tránh TkAgg/QtAgg trên Windows CLI) khi ngoài môi trường inline."""
    backend = str(matplotlib.get_backend()).lower()
    if "inline" not in backend and backend != "agg":
        matplotlib.use("Agg", force=True)


def plot_loss_curve(log_history: Sequence[Dict[str, Any]], output_path: Path) -> Path:
    """Vẽ đường Train Loss (theo step) và Validation Loss (theo epoch) từ `Trainer.state.log_history`."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    train_epochs: list[float] = []
    train_losses: list[float] = []
    eval_epochs: list[float] = []
    eval_losses: list[float] = []
    for r in log_history:
        if "train_runtime" in r:
            # Bỏ qua lượt trainer.evaluate() hậu-train sau khi load_best_model_at_end khôi phục checkpoint tốt nhất
            break
        if "loss" in r and "epoch" in r:
            train_epochs.append(float(r["epoch"]))
            train_losses.append(float(r["loss"]))
        if "eval_loss" in r and "epoch" in r:
            eval_epochs.append(float(r["epoch"]))
            eval_losses.append(float(r["eval_loss"]))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    if train_epochs:
        ax.plot(train_epochs, train_losses, marker="o", markersize=3.5, linewidth=1.6, label="Train Loss (step)")
    if eval_epochs:
        ax.plot(eval_epochs, eval_losses, marker="s", markersize=5.5, linewidth=2.0, label="Val Loss (epoch)")
        best_idx = int(np.argmin(eval_losses))
        ax.axvline(
            eval_epochs[best_idx],
            color="gray",
            linestyle="--",
            alpha=0.75,
            label=f"Best Val Loss (ep {eval_epochs[best_idx]:.1f}: {eval_losses[best_idx]:.4f})",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("DistilBERT Fine-Tuning on IMDB — Loss Curve")
    if train_epochs or eval_epochs:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_before_after_comparison(
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Vẽ biểu đồ cột so sánh Accuracy & F1 trên tập Validation trước và sau khi fine-tune."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Val Accuracy (%)", "Val F1-score (%)"]
    b_vals = [float(before_metrics.get("eval_accuracy", 0.0)) * 100.0, float(before_metrics.get("eval_f1", 0.0)) * 100.0]
    a_vals = [float(after_metrics.get("eval_accuracy", 0.0)) * 100.0, float(after_metrics.get("eval_f1", 0.0)) * 100.0]

    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars_before = ax.bar(x - w / 2, b_vals, w, label="Before Fine-Tune (Baseline)", color="#718096")
    bars_after = ax.bar(x + w / 2, a_vals, w, label="After Fine-Tune (DistilBERT)", color="#2b6cb0")

    for bars in (bars_before, bars_after):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.4,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Validation Performance: Before vs After Fine-Tuning (IMDB)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_binary_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
) -> Path:
    """Vẽ Confusion Matrix 2×2 hiển thị cả số lượng thô và tỷ lệ phần trăm theo hàng."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mat = np.asarray(cm, dtype=np.int64)
    row_sums = np.maximum(mat.sum(axis=1, keepdims=True), 1)
    pct = (mat.astype(np.float64) / row_sums) * 100.0

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row Percentage (%)")

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=11)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val_pct = pct[i, j]
            ax.text(
                j,
                i,
                f"{int(mat[i, j])}\n({val_pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if val_pct > 50.0 else "black",
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
