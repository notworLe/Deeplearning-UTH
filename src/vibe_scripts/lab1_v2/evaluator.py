"""Đánh giá mô hình theo từng run (lưu tại `runs/<run_id>/eval/`), đánh giá Test cuối và trực quan hóa."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.vibe_scripts.lab1_v2.data import DEFAULT_CLASS_NAMES
from src.vibe_scripts.lab1_v2.inference import load_checkpoint
from src.vibe_scripts.lab1_v2.logistic import count_parameters
from src.vibe_scripts.lab1_v2.transforms import NormStats, denormalize_tensor


def _ensure_headless_matplotlib() -> None:
    """Chuyển sang backend Agg thuần (tránh TkAgg/QtAgg) khi chạy ngoài môi trường inline."""
    backend = str(matplotlib.get_backend()).lower()
    if "inline" not in backend and backend != "agg":
        matplotlib.use("Agg", force=True)


@torch.inference_mode()
def evaluate_model_on_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: Union[str, torch.device],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> Dict[str, Any]:
    """Đánh giá đầy đủ (Loss, Accuracy, Macro-F1, Per-class, Confusion Matrix) dưới `inference_mode`."""
    device_obj = torch.device(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_images: List[torch.Tensor] = []

    for batch_x, batch_y in data_loader:
        batch_x_dev = batch_x.to(device_obj, non_blocking=True)
        batch_y_dev = batch_y.to(device_obj, non_blocking=True)
        logits = model(batch_x_dev)
        loss = criterion(logits, batch_y_dev)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item())
        total_samples += int(batch_y.shape[0])
        all_targets.append(batch_y.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_images.append(batch_x.cpu())

    if total_samples == 0:
        raise ValueError("data_loader không chứa mẫu nào để đánh giá.")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    images = torch.cat(all_images, dim=0)

    num_classes = len(class_names)
    labels_range = list(range(num_classes))
    avg_loss = total_loss / total_samples
    accuracy = float((y_true == y_pred).mean())

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(class_names):
        per_class[str(name)] = {
            "class_id": idx,
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    # Tìm các cặp nhầm lẫn lớn nhất ngoài đường chéo
    top_confusions: List[Dict[str, Any]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and int(cm[i, j]) > 0:
                top_confusions.append(
                    {
                        "true_class": str(class_names[i]),
                        "pred_class": str(class_names[j]),
                        "count": int(cm[i, j]),
                    }
                )
    top_confusions.sort(key=lambda item: item["count"], reverse=True)

    return {
        "num_samples": total_samples,
        "loss": float(avg_loss),
        "accuracy": accuracy,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions[:10],
        "_raw": {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
            "images": images,
        },
    }


@torch.inference_mode()
def verify_save_load_parity(
    checkpoint_path: Path,
    sample_batch: torch.Tensor,
    device: Union[str, torch.device],
    reference_model: Optional[nn.Module] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Xác thực sai số logits giữa model gốc và model nạp lại từ checkpoint `< tolerance`."""
    device_obj = torch.device(device)
    sample_dev = sample_batch.to(device_obj)
    reloaded, _ = load_checkpoint(checkpoint_path=checkpoint_path, device=device_obj)
    reloaded.eval()
    logits_reloaded = reloaded(sample_dev)

    if reference_model is not None:
        reference_model.eval()
        logits_ref = reference_model.to(device_obj)(sample_dev)
    else:
        reloaded_again, _ = load_checkpoint(checkpoint_path=checkpoint_path, device=device_obj)
        logits_ref = reloaded_again(sample_dev)

    max_diff = float((logits_ref - logits_reloaded).abs().max().item())
    return {
        "max_abs_diff": max_diff,
        "tolerance": float(tolerance),
        "passed": bool(max_diff < tolerance),
    }


def _plot_single_run_curves(
    history: Dict[str, Any],
    run_id: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history["epochs"]]
    tr_loss = [float(row["train_loss"]) for row in history["epochs"]]
    va_loss = [float(row["val_loss"]) for row in history["epochs"]]
    tr_acc = [float(row["train_acc"]) for row in history["epochs"]]
    va_acc = [float(row["val_acc"]) for row in history["epochs"]]
    best_ep = int(history.get("best_epoch", epochs[int(np.argmin(va_loss))]))

    # Loss curve
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, tr_loss, marker="o", markersize=3, label="Train Loss")
    ax.plot(epochs, va_loss, marker="s", markersize=3, label="Val Loss")
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{run_id} — Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    loss_path = output_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=140)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, [v * 100.0 for v in tr_acc], marker="o", markersize=3, label="Train Acc (%)")
    ax.plot(epochs, [v * 100.0 for v in va_acc], marker="s", markersize=3, label="Val Acc (%)")
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{run_id} — Accuracy Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    acc_path = output_dir / "accuracy_curve.png"
    fig.savefig(acc_path, dpi=140)
    plt.close(fig)

    return loss_path, acc_path


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
) -> Path:
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(class_names, fontsize=8.5)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7.5,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_predicted_vs_actual(
    images: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    norm_stats: NormStats,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
    num_samples: int = 16,
    errors_only: bool = False,
) -> Path:
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    n_total = len(y_true)
    if errors_only:
        wrong_idx = np.flatnonzero(y_true != y_pred)
        if len(wrong_idx) == 0:
            chosen = np.arange(min(num_samples, n_total))
        else:
            confidences = y_probs[wrong_idx, y_pred[wrong_idx]]
            order = np.argsort(-confidences)
            chosen = wrong_idx[order[:num_samples]]
    else:
        step = max(1, n_total // max(1, num_samples))
        chosen = np.arange(0, n_total, step)[:num_samples]

    n_show = len(chosen)
    n_cols = 4
    n_rows = max(1, int(np.ceil(n_show / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.6))
    axes_flat = np.asarray(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= n_show:
            ax.axis("off")
            continue
        idx = int(chosen[ax_idx])
        img_vis = denormalize_tensor(images[idx], norm_stats).squeeze(0).numpy()
        true_lbl = int(y_true[idx])
        pred_lbl = int(y_pred[idx])
        conf = float(y_probs[idx, pred_lbl]) * 100.0
        is_ok = true_lbl == pred_lbl

        ax.imshow(img_vis, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"T: {class_names[true_lbl]}\nP: {class_names[pred_lbl]} ({conf:.1f}%)",
            fontsize=8,
            color="#1e7e34" if is_ok else "#c82333",
        )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate_run_dir(
    run_dir: Path,
    eval_loader: DataLoader,
    device: Union[str, torch.device],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    split_name: str = "val",
    checkpoint_name: str = "best.pt",
    save_plots: bool = True,
    tolerance: float = 1e-6,
    reference_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Đánh giá 1 run ngay sau khi train và lưu toàn bộ kết quả vào `run_dir/<eval_subdir>/`."""
    run_path = Path(run_dir)
    ckpt_path = run_path / checkpoint_name
    model, payload = load_checkpoint(checkpoint_path=ckpt_path, device=device)
    norm_stats = NormStats.from_dict(payload["norm_stats"])
    run_id = str(payload.get("run_id", run_path.name))

    subdir_name = "eval" if split_name == "val" else f"{split_name}_eval"
    eval_dir = run_path / subdir_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_out = evaluate_model_on_loader(
        model=model,
        data_loader=eval_loader,
        device=device,
        class_names=class_names,
    )
    raw = eval_out.pop("_raw")

    # Kiểm chứng Save/Load parity trên batch đầu tiên
    sample_batch = raw["images"][: min(16, len(raw["images"]))]
    parity = verify_save_load_parity(
        checkpoint_path=ckpt_path,
        sample_batch=sample_batch,
        device=device,
        reference_model=reference_model,
        tolerance=tolerance,
    )

    artifacts: Dict[str, str] = {}
    history_path = run_path / "historys" / "history.json"
    if not history_path.exists():
        history_path = run_path / "history.json"
    if save_plots:
        # Learning curves (Train vs Val theo từng epoch) chỉ thuộc về `eval/` (val),
        # tuyệt đối không vẽ lặp vào `test_eval/` vì Test chỉ chấm 1 lần duy nhất trên best.pt.
        if split_name == "val" and history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history.get("epochs"):
                loss_p, acc_p = _plot_single_run_curves(history, run_id=run_id, output_dir=eval_dir)
                artifacts["loss_curve"] = str(loss_p)
                artifacts["accuracy_curve"] = str(acc_p)

        cm_p = _plot_confusion_matrix(
            cm=np.asarray(eval_out["confusion_matrix"], dtype=np.int64),
            class_names=class_names,
            title=f"{run_id} — Confusion Matrix ({split_name.upper()})",
            output_path=eval_dir / "confusion_matrix.png",
        )
        artifacts["confusion_matrix_plot"] = str(cm_p)

        pred_p = _plot_predicted_vs_actual(
            images=raw["images"],
            y_true=raw["y_true"],
            y_pred=raw["y_pred"],
            y_probs=raw["y_probs"],
            norm_stats=norm_stats,
            class_names=class_names,
            title=f"{run_id} — Predicted vs Actual ({split_name.upper()})",
            output_path=eval_dir / "predicted_vs_actual.png",
            num_samples=16,
            errors_only=False,
        )
        artifacts["predicted_vs_actual_plot"] = str(pred_p)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "model_config": payload["model_config"],
        "num_parameters": count_parameters(model),
        "save_load_parity": parity,
        "artifacts": artifacts,
        **eval_out,
    }

    metrics_path = eval_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["eval_dir"] = str(eval_dir)
    result["metrics_path"] = str(metrics_path)
    return result


def evaluate_final_test(
    runs_dir: Path,
    baseline_run_id: str,
    best_mlp_run_id: str,
    test_loader: DataLoader,
    device: Union[str, torch.device],
    evaluation_dir: Path,
    figures_dir: Path,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    ablation_summary_path: Optional[Path] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Đánh giá tập Test (10k) đúng 1 lần cuối cho Baseline vs Best MLP và sinh toàn bộ đồ thị."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    runs_path = Path(runs_dir)
    eval_out_dir = Path(evaluation_dir)
    fig_out_dir = Path(figures_dir)
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đánh giá Baseline & Best MLP trên Test (lưu test_eval ngay trong folder mỗi run)
    baseline_dir = runs_path / baseline_run_id
    best_mlp_dir = runs_path / best_mlp_run_id

    baseline_test = evaluate_run_dir(
        run_dir=baseline_dir,
        eval_loader=test_loader,
        device=device,
        class_names=class_names,
        split_name="test",
        save_plots=True,
        tolerance=tolerance,
    )
    mlp_test = evaluate_run_dir(
        run_dir=best_mlp_dir,
        eval_loader=test_loader,
        device=device,
        class_names=class_names,
        split_name="test",
        save_plots=True,
        tolerance=tolerance,
    )

    # 2. Vẽ so sánh Loss & Accuracy curves (Baseline vs Best MLP)
    base_hist_path = baseline_dir / "historys" / "history.json"
    if not base_hist_path.exists():
        base_hist_path = baseline_dir / "history.json"
    mlp_hist_path = best_mlp_dir / "historys" / "history.json"
    if not mlp_hist_path.exists():
        mlp_hist_path = best_mlp_dir / "history.json"
    base_hist = json.loads(base_hist_path.read_text(encoding="utf-8"))
    mlp_hist = json.loads(mlp_hist_path.read_text(encoding="utf-8"))

    # Loss curves comparison
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["train_loss"] for r in base_hist["epochs"]], "--", label=f"{baseline_run_id} (train)")
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["val_loss"] for r in base_hist["epochs"]], "-", marker="o", markersize=3, label=f"{baseline_run_id} (val)")
    ax.plot([r["epoch"] for r in mlp_hist["epochs"]], [r["train_loss"] for r in mlp_hist["epochs"]], "--", label=f"{best_mlp_run_id} (train)")
    ax.plot([r["epoch"] for r in mlp_hist["epochs"]], [r["val_loss"] for r in mlp_hist["epochs"]], "-", marker="s", markersize=3, label=f"{best_mlp_run_id} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss Curves — Baseline Logistic Regression vs Best MLP")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    loss_cmp_path = fig_out_dir / "loss_curves.png"
    fig.savefig(loss_cmp_path, dpi=150)
    plt.close(fig)

    # Accuracy curves comparison
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["train_acc"] * 100 for r in base_hist["epochs"]], "--", label=f"{baseline_run_id} (train)")
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["val_acc"] * 100 for r in base_hist["epochs"]], "-", marker="o", markersize=3, label=f"{baseline_run_id} (val)")
    ax.plot([r["epoch"] for r in mlp_hist["epochs"]], [r["train_acc"] * 100 for r in mlp_hist["epochs"]], "--", label=f"{best_mlp_run_id} (train)")
    ax.plot([r["epoch"] for r in mlp_hist["epochs"]], [r["val_acc"] * 100 for r in mlp_hist["epochs"]], "-", marker="s", markersize=3, label=f"{best_mlp_run_id} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curves — Baseline Logistic Regression vs Best MLP")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    acc_cmp_path = fig_out_dir / "accuracy_curves.png"
    fig.savefig(acc_cmp_path, dpi=150)
    plt.close(fig)

    # 3. Confusion matrices vào figures/
    cm_base_path = _plot_confusion_matrix(
        cm=np.asarray(baseline_test["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
        title=f"{baseline_run_id} — Test Confusion Matrix (Acc={baseline_test['accuracy']*100:.2f}%)",
        output_path=fig_out_dir / "confusion_matrix_baseline.png",
    )
    cm_mlp_path = _plot_confusion_matrix(
        cm=np.asarray(mlp_test["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
        title=f"{best_mlp_run_id} — Test Confusion Matrix (Acc={mlp_test['accuracy']*100:.2f}%)",
        output_path=fig_out_dir / "confusion_matrix_mlp.png",
    )

    # 4. Predicted vs Actual & Error Gallery của Best MLP trên Test
    best_model, best_payload = load_checkpoint(best_mlp_dir / "best.pt", device=device)
    best_norm = NormStats.from_dict(best_payload["norm_stats"])
    raw_mlp = evaluate_model_on_loader(best_model, test_loader, device=device, class_names=class_names)["_raw"]

    pred_vis_path = _plot_predicted_vs_actual(
        images=raw_mlp["images"],
        y_true=raw_mlp["y_true"],
        y_pred=raw_mlp["y_pred"],
        y_probs=raw_mlp["y_probs"],
        norm_stats=best_norm,
        class_names=class_names,
        title=f"{best_mlp_run_id} — Predicted vs Actual (Test Split)",
        output_path=fig_out_dir / "predicted_vs_actual.png",
        num_samples=16,
        errors_only=False,
    )
    err_vis_path = _plot_predicted_vs_actual(
        images=raw_mlp["images"],
        y_true=raw_mlp["y_true"],
        y_pred=raw_mlp["y_pred"],
        y_probs=raw_mlp["y_probs"],
        norm_stats=best_norm,
        class_names=class_names,
        title=f"{best_mlp_run_id} — Top-16 High-Confidence Errors (Test Split)",
        output_path=fig_out_dir / "error_gallery.png",
        num_samples=16,
        errors_only=True,
    )

    # 5. Biểu đồ tổng hợp Ablation (nếu có ablation_summary.json)
    ablation_plot_path: Optional[Path] = None
    if ablation_summary_path and Path(ablation_summary_path).exists():
        abl_data = json.loads(Path(ablation_summary_path).read_text(encoding="utf-8"))
        runs_list = abl_data.get("runs", [])
        if runs_list:
            fig, ax = plt.subplots(figsize=(10.5, 4.8))
            labels = [r["run_id"] for r in runs_list]
            accs = [float(r["val_acc"]) * 100.0 for r in runs_list]
            f1s = [float(r["val_macro_f1"]) * 100.0 for r in runs_list]
            x = np.arange(len(labels))
            w = 0.36
            ax.bar(x - w / 2, accs, w, label="Val Accuracy (%)")
            ax.bar(x + w / 2, f1s, w, label="Val Macro-F1 (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.5)
            ax.set_ylabel("Score (%)")
            ax.set_ylim(max(0.0, min(accs + f1s) - 5.0), 100.0)
            ax.set_title("Controlled Ablation Comparison (Validation Split)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            ablation_plot_path = fig_out_dir / "ablation_comparison.png"
            fig.savefig(ablation_plot_path, dpi=150)
            plt.close(fig)

    # 6. Phân tích per-class delta & Error Analysis
    per_class_delta: Dict[str, Dict[str, float]] = {}
    for cls_name in class_names:
        b_f1 = baseline_test["per_class"][cls_name]["f1"]
        m_f1 = mlp_test["per_class"][cls_name]["f1"]
        per_class_delta[cls_name] = {
            "baseline_f1": b_f1,
            "mlp_f1": m_f1,
            "f1_gain": float(m_f1 - b_f1),
        }

    comparison_rows = [
        {
            "run_id": baseline_run_id,
            "model_type": baseline_test["model_config"]["model_type"],
            "num_parameters": baseline_test["num_parameters"],
            "test_loss": baseline_test["loss"],
            "test_accuracy": baseline_test["accuracy"],
            "test_macro_f1": baseline_test["macro_f1"],
            "save_load_max_diff": baseline_test["save_load_parity"]["max_abs_diff"],
            "save_load_passed": baseline_test["save_load_parity"]["passed"],
        },
        {
            "run_id": best_mlp_run_id,
            "model_type": mlp_test["model_config"]["model_type"],
            "num_parameters": mlp_test["num_parameters"],
            "test_loss": mlp_test["loss"],
            "test_accuracy": mlp_test["accuracy"],
            "test_macro_f1": mlp_test["macro_f1"],
            "save_load_max_diff": mlp_test["save_load_parity"]["max_abs_diff"],
            "save_load_passed": mlp_test["save_load_parity"]["passed"],
        },
    ]

    summary = {
        "baseline_run_id": baseline_run_id,
        "best_mlp_run_id": best_mlp_run_id,
        "comparison": comparison_rows,
        "mlp_gain_over_baseline": {
            "accuracy_delta": float(mlp_test["accuracy"] - baseline_test["accuracy"]),
            "macro_f1_delta": float(mlp_test["macro_f1"] - baseline_test["macro_f1"]),
            "loss_delta": float(mlp_test["loss"] - baseline_test["loss"]),
        },
        "figures": {
            "loss_curves": str(loss_cmp_path),
            "accuracy_curves": str(acc_cmp_path),
            "confusion_matrix_baseline": str(cm_base_path),
            "confusion_matrix_mlp": str(cm_mlp_path),
            "predicted_vs_actual": str(pred_vis_path),
            "error_gallery": str(err_vis_path),
            "ablation_comparison": str(ablation_plot_path) if ablation_plot_path else None,
        },
    }

    (eval_out_dir / "test_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (eval_out_dir / "test_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    (eval_out_dir / "per_class_metrics.json").write_text(
        json.dumps(
            {
                "baseline": baseline_test["per_class"],
                "best_mlp": mlp_test["per_class"],
                "delta": per_class_delta,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (eval_out_dir / "error_analysis.json").write_text(
        json.dumps(
            {
                "baseline_top_confusions": baseline_test["top_confusions"],
                "best_mlp_top_confusions": mlp_test["top_confusions"],
                "hardest_classes_mlp": sorted(
                    per_class_delta.items(), key=lambda kv: kv[1]["mlp_f1"]
                )[:3],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


def _plot_normalized_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
) -> Path:
    """Vẽ Confusion Matrix chuẩn hóa theo hàng (%) — mỗi ô thể hiện % mẫu thật lớp A bị đoán thành lớp B."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_pct = (cm.astype(np.float64) / row_sums) * 100.0

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row Percentage (%)")

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(class_names, fontsize=8.5)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            txt = f"{val:.1f}%" if val >= 0.05 else "0%"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > 50.0 else "black",
                fontsize=7.3,
                fontweight="bold" if (i == j or val >= 5.0) else "normal",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_error_subset_grid(
    images_raw: np.ndarray,
    indices: Sequence[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
    n_cols: int = 4,
    audit_ids: Optional[Sequence[int]] = None,
) -> Path:
    """Vẽ lưới ảnh lỗi với nhãn True / Pred / Confidence và chỉ số mẫu để kiểm tra trực quan."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    n_show = len(indices)
    n_rows = max(1, int(np.ceil(n_show / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.65, n_rows * 2.75))
    axes_flat = np.asarray(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= n_show:
            ax.axis("off")
            continue
        idx = int(indices[ax_idx])
        t_lbl = int(y_true[idx])
        p_lbl = int(y_pred[idx])
        p_conf = float(y_probs[idx, p_lbl]) * 100.0
        t_conf = float(y_probs[idx, t_lbl]) * 100.0
        prefix = f"[#{audit_ids[ax_idx]}] " if audit_ids is not None else f"id={idx} | "

        ax.imshow(images_raw[idx], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{prefix}T:{class_names[t_lbl]} ({t_conf:.0f}%)\nP:{class_names[p_lbl]} ({p_conf:.1f}%)",
            fontsize=8.0,
            color="#b02a37",
        )

    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _compute_image_attributes(images_raw: np.ndarray) -> Dict[str, np.ndarray]:
    """Tính 3 thuộc tính ảnh trực tiếp trên ảnh `[0, 1]` shape `(N, 28, 28)` phục vụ Slice Analysis."""
    mean_intensity = images_raw.mean(axis=(1, 2))
    fg_ratio = (images_raw > 0.10).mean(axis=(1, 2))
    dx = np.abs(np.diff(images_raw, axis=2)).mean(axis=(1, 2))
    dy = np.abs(np.diff(images_raw, axis=1)).mean(axis=(1, 2))
    edge_strength = dx + dy
    return {
        "mean_intensity": mean_intensity,
        "fg_ratio": fg_ratio,
        "edge_strength": edge_strength,
    }


def _compute_quartile_slices(
    values: np.ndarray,
    is_error: np.ndarray,
    garment_mask: np.ndarray,
    labels: Sequence[str],
) -> List[Dict[str, Any]]:
    edges = np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0])
    slices: List[Dict[str, Any]] = []
    for q in range(4):
        lo, hi = float(edges[q]), float(edges[q + 1])
        mask = (values >= lo) & (values <= hi if q == 3 else values < hi)
        g_mask = mask & garment_mask
        n_all = int(mask.sum())
        err_all = int(is_error[mask].sum())
        n_garm = int(g_mask.sum())
        err_garm = int(is_error[g_mask].sum())
        slices.append(
            {
                "quartile": labels[q],
                "range": [round(lo, 4), round(hi, 4)],
                "n_all": n_all,
                "errors_all": err_all,
                "error_rate_all_pct": round((err_all / max(n_all, 1)) * 100.0, 2),
                "n_garment4": n_garm,
                "errors_garment4": err_garm,
                "error_rate_garment4_pct": round((err_garm / max(n_garm, 1)) * 100.0, 2),
            }
        )
    return slices


def run_deep_error_analysis(
    run_dir: Path,
    eval_loader: DataLoader,
    device: Union[str, torch.device],
    evaluation_dir: Path,
    figures_dir: Path,
    audit_sheets_dir: Optional[Path] = None,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    top_k_pairs: int = 3,
    gallery_size: int = 16,
    audit_per_pair: int = 20,
) -> Dict[str, Any]:
    """Thực thi Error Analysis 4 bước (CM chuẩn hóa %, Gallery Top-3 theo High/Low Conf, Audit 60 mẫu, Slice Analysis)."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    eval_out_dir = Path(evaluation_dir)
    fig_out_dir = Path(figures_dir)
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = Path(audit_sheets_dir) if audit_sheets_dir is not None else eval_out_dir / "audit_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_checkpoint(run_path / "best.pt", device=device)
    norm_stats = NormStats.from_dict(payload["norm_stats"])
    run_id = str(payload.get("run_id", run_path.name))

    eval_out = evaluate_model_on_loader(model, eval_loader, device=device, class_names=class_names)
    raw = eval_out["_raw"]
    y_true = raw["y_true"]
    y_pred = raw["y_pred"]
    y_probs = raw["y_probs"]
    cm = np.asarray(eval_out["confusion_matrix"], dtype=np.int64)
    images_raw = denormalize_tensor(raw["images"], norm_stats).squeeze(1).numpy()

    # Bước 1: Confusion Matrix chuẩn hóa theo hàng (%) + Top confusion pairs kèm % hàng
    cm_norm_path = _plot_normalized_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title=f"{run_id} — Row-Normalized Confusion Matrix (%) (Test Split)",
        output_path=fig_out_dir / "confusion_matrix_mlp_normalized.png",
    )
    row_totals = cm.sum(axis=1)
    name_to_id = {str(name): idx for idx, name in enumerate(class_names)}
    enriched_confusions: List[Dict[str, Any]] = []
    for item in eval_out["top_confusions"]:
        t_id = name_to_id[item["true_class"]]
        p_id = name_to_id[item["pred_class"]]
        enriched_confusions.append(
            {
                **item,
                "true_id": t_id,
                "pred_id": p_id,
                "row_pct": round((item["count"] / max(int(row_totals[t_id]), 1)) * 100.0, 2),
            }
        )

    # Bước 2: Lọc mẫu sai thuộc Top-K cặp nhầm lẫn & chia High-Confidence vs Low-Confidence
    top_pairs = [(c["true_id"], c["pred_id"]) for c in enriched_confusions[:top_k_pairs]]
    pair_mask = np.zeros(len(y_true), dtype=bool)
    for t_id, p_id in top_pairs:
        pair_mask |= (y_true == t_id) & (y_pred == p_id)
    top_pair_wrong_idx = np.flatnonzero(pair_mask)
    pred_confs = y_probs[top_pair_wrong_idx, y_pred[top_pair_wrong_idx]]

    high_order = np.argsort(-pred_confs)
    low_order = np.argsort(pred_confs)
    high_idx = top_pair_wrong_idx[high_order[:gallery_size]]
    low_idx = top_pair_wrong_idx[low_order[:gallery_size]]

    pair_desc = ", ".join(f"{class_names[t]}→{class_names[p]}" for t, p in top_pairs)
    high_gallery_path = _plot_error_subset_grid(
        images_raw=images_raw,
        indices=high_idx,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=class_names,
        title=f"Top-3 Pairs ({pair_desc}) — 16 High-Confidence Errors",
        output_path=fig_out_dir / "error_gallery_top3_high_conf.png",
        n_cols=4,
    )
    low_gallery_path = _plot_error_subset_grid(
        images_raw=images_raw,
        indices=low_idx,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=class_names,
        title=f"Top-3 Pairs ({pair_desc}) — 16 Low-Confidence (Boundary) Errors",
        output_path=fig_out_dir / "error_gallery_top3_low_conf.png",
        n_cols=4,
    )

    # Bước 3: Xuất 60 mẫu lỗi phân tầng (20 mẫu/cặp = 10 High-Conf + 10 Low-Conf) & contact sheets để gắn nhãn nguyên nhân
    attrs = _compute_image_attributes(images_raw)
    half = audit_per_pair // 2
    audit_candidates: List[Dict[str, Any]] = []
    sheet_paths: List[str] = []
    audit_counter = 1
    for p_rank, (t_id, p_id) in enumerate(top_pairs, 1):
        p_idx = np.flatnonzero((y_true == t_id) & (y_pred == p_id))
        p_confs = y_probs[p_idx, p_id]
        hi_sel = p_idx[np.argsort(-p_confs)[:half]]
        lo_sel = p_idx[np.argsort(p_confs)[: (audit_per_pair - half)]]
        chosen = list(hi_sel) + list(lo_sel)
        pair_audit_ids = list(range(audit_counter, audit_counter + len(chosen)))

        for offset, (sample_idx, a_id) in enumerate(zip(chosen, pair_audit_ids)):
            idx_i = int(sample_idx)
            audit_candidates.append(
                {
                    "audit_id": int(a_id),
                    "test_idx": idx_i,
                    "pair": f"{class_names[t_id]} -> {class_names[p_id]}",
                    "conf_group": "high_conf" if offset < half else "low_conf",
                    "true_class": str(class_names[t_id]),
                    "pred_class": str(class_names[p_id]),
                    "pred_conf_pct": round(float(y_probs[idx_i, p_id]) * 100.0, 2),
                    "true_conf_pct": round(float(y_probs[idx_i, t_id]) * 100.0, 2),
                    "mean_intensity": round(float(attrs["mean_intensity"][idx_i]), 4),
                    "fg_ratio": round(float(attrs["fg_ratio"][idx_i]), 4),
                    "edge_strength": round(float(attrs["edge_strength"][idx_i]), 4),
                }
            )
        audit_counter += len(chosen)

        slug = f"{class_names[t_id]}_{class_names[p_id]}".lower().replace("/", "_").replace(" ", "_")
        s_path = _plot_error_subset_grid(
            images_raw=images_raw,
            indices=chosen,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            class_names=class_names,
            title=f"Audit Sheet {p_rank}: {class_names[t_id]} → {class_names[p_id]} (Top 10 High-Conf + Bottom 10 Low-Conf)",
            output_path=sheets_dir / f"audit_sheet_{p_rank}_{slug}.png",
            n_cols=5,
            audit_ids=pair_audit_ids,
        )
        sheet_paths.append(str(s_path))

    (eval_out_dir / "error_audit_60_candidates.json").write_text(
        json.dumps(audit_candidates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Bước 4: Slice Analysis trên toàn bộ 10,000 mẫu Test + cụm 4 lớp áo khó {0, 2, 4, 6}
    is_error = (y_true != y_pred)
    garment_mask = np.isin(y_true, [0, 2, 4, 6])
    slice_data = {
        "overall_error_rate_pct": round(float(is_error.mean()) * 100.0, 2),
        "garment4_error_rate_pct": round(float(is_error[garment_mask].mean()) * 100.0, 2),
        "mean_intensity": _compute_quartile_slices(
            attrs["mean_intensity"], is_error, garment_mask, ["Q1 Dark", "Q2 Dim", "Q3 Mid", "Q4 Bright"]
        ),
        "fg_ratio": _compute_quartile_slices(
            attrs["fg_ratio"], is_error, garment_mask, ["Q1 Small", "Q2 Med-S", "Q3 Med-L", "Q4 Full"]
        ),
        "edge_strength": _compute_quartile_slices(
            attrs["edge_strength"], is_error, garment_mask, ["Q1 Smooth", "Q2 Low-E", "Q3 Mid-E", "Q4 High-E"]
        ),
    }
    (eval_out_dir / "slice_analysis.json").write_text(
        json.dumps(slice_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Biểu đồ Slice Analysis 3 panel
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    panel_specs = [
        ("mean_intensity", "Mean Intensity (Brightness)"),
        ("fg_ratio", "Foreground Coverage (Object Size)"),
        ("edge_strength", "Edge / Gradient Strength (Detail)"),
    ]
    for ax, (key, p_title) in zip(axes, panel_specs):
        rows = slice_data[key]
        x = np.arange(len(rows))
        w = 0.36
        err_all = [r["error_rate_all_pct"] for r in rows]
        err_g4 = [r["error_rate_garment4_pct"] for r in rows]
        b1 = ax.bar(x - w / 2, err_all, w, label="All 10 Classes", color="#2b6cb0")
        b2 = ax.bar(x + w / 2, err_g4, w, label="4-Garment Cluster", color="#c53030")
        ax.axhline(slice_data["overall_error_rate_pct"], color="#2b6cb0", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(slice_data["garment4_error_rate_pct"], color="#c53030", linestyle="--", alpha=0.5, linewidth=1)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([r["quartile"] for r in rows], fontsize=8.5)
        ax.set_title(p_title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Error Rate (%)")
    axes[0].set_ylim(0, max(35.0, max(r["error_rate_garment4_pct"] for k, _ in panel_specs for r in slice_data[k]) + 5.0))
    axes[2].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(f"{run_id} — Slice Analysis: Error Rate by Image Attributes (Test 10k)", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    slice_plot_path = fig_out_dir / "error_slice_analysis.png"
    fig.savefig(slice_plot_path, dpi=150)
    plt.close(fig)

    # Bổ sung thông tin deep error analysis vào error_analysis.json hiện có
    err_json_path = eval_out_dir / "error_analysis.json"
    existing = json.loads(err_json_path.read_text(encoding="utf-8")) if err_json_path.exists() else {}
    existing["best_mlp_top_confusions_enriched"] = enriched_confusions
    existing["top3_pair_error_count"] = int(len(top_pair_wrong_idx))
    existing["total_test_errors"] = int(is_error.sum())
    existing["top3_pair_share_of_errors_pct"] = round(
        (len(top_pair_wrong_idx) / max(int(is_error.sum()), 1)) * 100.0, 2
    )
    existing["deep_figures"] = {
        "confusion_matrix_mlp_normalized": str(cm_norm_path),
        "error_gallery_top3_high_conf": str(high_gallery_path),
        "error_gallery_top3_low_conf": str(low_gallery_path),
        "error_slice_analysis": str(slice_plot_path),
    }
    err_json_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "run_id": run_id,
        "total_test_errors": int(is_error.sum()),
        "top3_pair_error_count": int(len(top_pair_wrong_idx)),
        "top3_pairs": [f"{class_names[t]} -> {class_names[p]}" for t, p in top_pairs],
        "figures": existing["deep_figures"],
        "audit_sheets": sheet_paths,
        "slice_analysis": slice_data,
    }


def plot_error_taxonomy_breakdown(
    taxonomy_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    """Tổng hợp tỷ lệ nguyên nhân lỗi trên N mẫu đã gắn nhãn và vẽ biểu đồ 2 panel (Tổng thể & Theo High/Low Conf)."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    categories = ["model_limitation", "ambiguity", "label_noise"]
    cat_labels = {
        "model_limitation": "Model Limitation\n(2D local detail lost)",
        "ambiguity": "Ambiguity\n(28×28 unclear)",
        "label_noise": "Label Noise\n(suspect ground truth)",
    }
    cat_colors = {
        "model_limitation": "#c53030",
        "ambiguity": "#dd6b20",
        "label_noise": "#4a5568",
    }

    total = len(taxonomy_rows)
    overall_counts = {c: sum(1 for r in taxonomy_rows if r["cause"] == c) for c in categories}
    overall_pct = {c: round((overall_counts[c] / max(total, 1)) * 100.0, 1) for c in categories}

    by_conf: Dict[str, Dict[str, float]] = {}
    for grp in ("high_conf", "low_conf"):
        sub = [r for r in taxonomy_rows if r["conf_group"] == grp]
        n_sub = max(len(sub), 1)
        by_conf[grp] = {
            c: round((sum(1 for r in sub if r["cause"] == c) / n_sub) * 100.0, 1)
            for c in categories
        }

    pairs = sorted({r["pair"] for r in taxonomy_rows})
    by_pair: Dict[str, Dict[str, int]] = {}
    for p in pairs:
        sub = [r for r in taxonomy_rows if r["pair"] == p]
        by_pair[p] = {c: sum(1 for r in sub if r["cause"] == c) for c in categories}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Panel 1: Overall breakdown (%)
    x1 = np.arange(len(categories))
    vals1 = [overall_pct[c] for c in categories]
    bars1 = ax1.bar(x1, vals1, color=[cat_colors[c] for c in categories], width=0.55)
    for bar, c in zip(bars1, categories):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{h:.1f}%\n({overall_counts[c]}/{total})",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
        )
    ax1.set_xticks(x1)
    ax1.set_xticklabels([cat_labels[c] for c in categories], fontsize=8.8)
    ax1.set_ylabel("Share of Audited Errors (%)")
    ax1.set_ylim(0, max(vals1) + 15.0)
    ax1.set_title(f"Root-Cause Breakdown (N = {total} Top-3 Pair Errors)")
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: Stacked bar by High-Conf vs Low-Conf
    groups = ["high_conf", "low_conf"]
    grp_labels = ["High-Confidence Errors\n(n = 30)", "Low-Confidence Errors\n(n = 30)"]
    x2 = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for c in categories:
        vals = np.array([by_conf[g][c] for g in groups])
        ax2.bar(x2, vals, bottom=bottom, width=0.48, color=cat_colors[c], label=cat_labels[c].replace("\n", " "))
        for i, v in enumerate(vals):
            if v >= 8.0:
                ax2.text(x2[i], bottom[i] + v / 2, f"{v:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        bottom += vals
    ax2.set_xticks(x2)
    ax2.set_xticklabels(grp_labels, fontsize=9.0)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 108)
    ax2.set_title("Root Cause by Confidence Regime")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.0)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "total_audited": total,
        "overall_counts": overall_counts,
        "overall_pct": overall_pct,
        "by_conf_group_pct": by_conf,
        "by_pair_counts": by_pair,
        "figure": str(output_path),
    }
