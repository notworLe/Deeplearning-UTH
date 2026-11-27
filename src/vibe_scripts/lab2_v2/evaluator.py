"""Evaluation Layer & 4-Step Quantitative Error Analysis cho Lab 2 v2 (Oxford-IIIT Pets — 37 giống)."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.vibe_scripts.lab2_v2.inference import load_checkpoint
from src.vibe_scripts.lab2_v2.models import count_parameters
from src.vibe_scripts.lab2_v2.transforms import ImageNetNormStats, denormalize_tensor


def _ensure_headless_matplotlib() -> None:
    """Chuyển sang backend Agg thuần (tránh TkAgg/QtAgg) khi chạy ngoài môi trường inline."""
    backend = str(matplotlib.get_backend()).lower()
    if "inline" not in backend and backend != "agg":
        matplotlib.use("Agg", force=True)


def _load_vis_image(
    dataset: Any,
    idx: int,
    norm_stats: ImageNetNormStats,
    target_size: int = 224,
) -> np.ndarray:
    """Nạp ảnh hiển thị `[0, 1]` `(H, W, 3)` từ đường dẫn gốc (nhẹ RAM) hoặc từ tensor dataset (test synthetic)."""
    if hasattr(dataset, "image_paths"):
        with Image.open(dataset.image_paths[idx]) as im:
            rgb = im.convert("RGB").resize((target_size, target_size))
        return np.asarray(rgb, dtype=np.float32) / 255.0
    tensor, _ = dataset[idx]
    return denormalize_tensor(tensor, norm_stats).permute(1, 2, 0).cpu().numpy()


@torch.inference_mode()
def evaluate_model_on_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: Union[str, torch.device],
    class_names: Sequence[str],
    species_by_class: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Đánh giá đầy đủ (Loss, Acc, Macro-F1, Per-class, Species Cat/Dog, CM 37×37) dưới `inference_mode`."""
    device_obj = torch.device(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    first_batch_x: Optional[torch.Tensor] = None

    for batch_x, batch_y in data_loader:
        if first_batch_x is None:
            first_batch_x = batch_x[: min(8, batch_x.shape[0])].cpu()
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

    if total_samples == 0:
        raise ValueError("data_loader không chứa mẫu nào để đánh giá.")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    num_classes = len(class_names)
    labels_range = list(range(num_classes))
    avg_loss = total_loss / total_samples
    accuracy = float((y_true == y_pred).mean())

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_range, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    row_totals = np.maximum(cm.sum(axis=1), 1)
    sp_map = species_by_class or {str(c): "Unknown" for c in class_names}

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(class_names):
        cname = str(name)
        per_class[cname] = {
            "class_id": idx,
            "species": sp_map.get(cname, "Unknown"),
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    # Phân tích theo loài (Cat vs Dog: lỗi nội loài vs lỗi khác loài)
    true_sp = np.array([sp_map.get(str(class_names[i]), "Unknown") for i in y_true])
    pred_sp = np.array([sp_map.get(str(class_names[i]), "Unknown") for i in y_pred])
    cat_mask = true_sp == "Cat"
    dog_mask = true_sp == "Dog"
    wrong_mask = y_true != y_pred
    species_metrics = {
        "species_accuracy": float((true_sp == pred_sp).mean()),
        "cat_breed_accuracy": float((y_true[cat_mask] == y_pred[cat_mask]).mean()) if cat_mask.any() else 0.0,
        "dog_breed_accuracy": float((y_true[dog_mask] == y_pred[dog_mask]).mean()) if dog_mask.any() else 0.0,
        "total_errors": int(wrong_mask.sum()),
        "intra_cat_errors": int((wrong_mask & cat_mask & (pred_sp == "Cat")).sum()),
        "intra_dog_errors": int((wrong_mask & dog_mask & (pred_sp == "Dog")).sum()),
        "cross_species_errors": int((wrong_mask & (true_sp != pred_sp)).sum()),
        "cat_samples": int(cat_mask.sum()),
        "dog_samples": int(dog_mask.sum()),
    }

    top_confusions: List[Dict[str, Any]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and int(cm[i, j]) > 0:
                t_name = str(class_names[i])
                p_name = str(class_names[j])
                top_confusions.append(
                    {
                        "true_id": i,
                        "pred_id": j,
                        "true_class": t_name,
                        "pred_class": p_name,
                        "count": int(cm[i, j]),
                        "row_pct": round((int(cm[i, j]) / int(row_totals[i])) * 100.0, 2),
                        "same_species": sp_map.get(t_name) == sp_map.get(p_name),
                        "true_species": sp_map.get(t_name, "Unknown"),
                    }
                )
    top_confusions.sort(key=lambda item: (item["count"], item["row_pct"]), reverse=True)

    return {
        "num_samples": total_samples,
        "loss": float(avg_loss),
        "accuracy": accuracy,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "species_metrics": species_metrics,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions[:15],
        "_raw": {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
            "sample_batch": first_batch_x,
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


def _plot_single_run_curves(history: Dict[str, Any], run_id: str, output_dir: Path) -> Tuple[Path, Path]:
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    epochs = [int(r["epoch"]) for r in history["epochs"]]
    tr_loss = [float(r["train_loss"]) for r in history["epochs"]]
    va_loss = [float(r["val_loss"]) for r in history["epochs"]]
    tr_acc = [float(r["train_acc"]) * 100.0 for r in history["epochs"]]
    va_acc = [float(r["val_acc"]) * 100.0 for r in history["epochs"]]
    best_ep = int(history.get("best_epoch", epochs[int(np.argmin(va_loss))]))

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

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(epochs, tr_acc, marker="o", markersize=3, label="Train Acc (%)")
    ax.plot(epochs, va_acc, marker="s", markersize=3, label="Val Acc (%)")
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
    normalize_rows: bool = False,
) -> Path:
    """Vẽ Confusion Matrix 37×37 (đếm thô hoặc chuẩn hóa theo hàng %)."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    n = len(class_names)
    if normalize_rows:
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        mat = (cm.astype(np.float64) / row_sums) * 100.0
        vmax = 100.0
        cbar_label = "Row Percentage (%)"
    else:
        mat = cm.astype(np.float64)
        vmax = float(max(mat.max(), 1.0))
        cbar_label = "Count"

    size = 12.5 if n > 15 else 8.2
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cbar.set_label(cbar_label)

    ticks = np.arange(n)
    tick_fs = 6.5 if n > 15 else 8.5
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=70, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(class_names, fontsize=tick_fs)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=11)

    thresh = vmax * 0.5
    text_fs = 5.2 if n > 15 else 7.5
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if n > 15 and val < (1.0 if normalize_rows else 1.0):
                continue
            txt = f"{val:.0f}%" if normalize_rows else f"{int(val)}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=text_fs,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_sample_grid(
    dataset: Any,
    indices: Sequence[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    norm_stats: ImageNetNormStats,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
    n_cols: int = 4,
    audit_ids: Optional[Sequence[int]] = None,
) -> Path:
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    n_show = len(indices)
    n_rows = max(1, int(np.ceil(n_show / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.9))
    axes_flat = np.asarray(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx >= n_show:
            ax.axis("off")
            continue
        idx = int(indices[ax_idx])
        img_vis = _load_vis_image(dataset, idx, norm_stats=norm_stats)
        t_lbl = int(y_true[idx])
        p_lbl = int(y_pred[idx])
        p_conf = float(y_probs[idx, p_lbl]) * 100.0
        t_conf = float(y_probs[idx, t_lbl]) * 100.0
        is_ok = t_lbl == p_lbl
        prefix = f"[#{audit_ids[ax_idx]}] " if audit_ids is not None else f"id={idx} | "

        ax.imshow(img_vis)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{prefix}T:{class_names[t_lbl]} ({t_conf:.0f}%)\nP:{class_names[p_lbl]} ({p_conf:.1f}%)",
            fontsize=7.3,
            color="#1e7e34" if is_ok else "#b02a37",
        )

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate_run_dir(
    run_dir: Path,
    eval_loader: DataLoader,
    device: Union[str, torch.device],
    class_names: Sequence[str],
    species_by_class: Optional[Dict[str, str]] = None,
    split_name: str = "val",
    checkpoint_name: str = "best.pt",
    save_plots: bool = True,
    tolerance: float = 1e-6,
    reference_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Đánh giá 1 run và lưu kết quả vào `run_dir/<eval_subdir>/` (chỉ vẽ learning curves cho `val`)."""
    run_path = Path(run_dir)
    ckpt_path = run_path / checkpoint_name
    model, payload = load_checkpoint(ckpt_path, device=device)
    norm_stats = ImageNetNormStats.from_dict(payload["norm_stats"])
    run_id = str(payload.get("run_id", run_path.name))
    sp_map = species_by_class or payload.get("species_by_class")

    subdir_name = "eval" if split_name == "val" else f"{split_name}_eval"
    eval_dir = run_path / subdir_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_out = evaluate_model_on_loader(
        model=model,
        data_loader=eval_loader,
        device=device,
        class_names=class_names,
        species_by_class=sp_map,
    )
    raw = eval_out.pop("_raw")
    parity = verify_save_load_parity(
        checkpoint_path=ckpt_path,
        sample_batch=raw["sample_batch"],
        device=device,
        reference_model=reference_model,
        tolerance=tolerance,
    )

    artifacts: Dict[str, str] = {}
    history_path = run_path / "historys" / "history.json"
    if save_plots:
        if split_name == "val" and history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history.get("epochs"):
                loss_p, acc_p = _plot_single_run_curves(history, run_id=run_id, output_dir=eval_dir)
                artifacts["loss_curve"] = str(loss_p)
                artifacts["accuracy_curve"] = str(acc_p)

        cm_p = _plot_confusion_matrix(
            cm=np.asarray(eval_out["confusion_matrix"], dtype=np.int64),
            class_names=class_names,
            title=f"{run_id} — Confusion Matrix ({split_name.upper()}, Acc={eval_out['accuracy']*100:.2f}%)",
            output_path=eval_dir / "confusion_matrix.png",
            normalize_rows=False,
        )
        artifacts["confusion_matrix_plot"] = str(cm_p)

        n_total = len(raw["y_true"])
        step = max(1, n_total // 16)
        chosen = list(range(0, n_total, step))[:16]
        pred_p = _plot_sample_grid(
            dataset=eval_loader.dataset,
            indices=chosen,
            y_true=raw["y_true"],
            y_pred=raw["y_pred"],
            y_probs=raw["y_probs"],
            norm_stats=norm_stats,
            class_names=class_names,
            title=f"{run_id} — Predicted vs Actual ({split_name.upper()})",
            output_path=eval_dir / "predicted_vs_actual.png",
            n_cols=4,
        )
        artifacts["predicted_vs_actual_plot"] = str(pred_p)

    counts = count_parameters(model)
    result: Dict[str, Any] = {
        "run_id": run_id,
        "split": split_name,
        "checkpoint": str(ckpt_path),
        "model_config": payload["model_config"],
        "param_counts": counts,
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
    best_run_id: str,
    test_loader: DataLoader,
    device: Union[str, torch.device],
    evaluation_dir: Path,
    figures_dir: Path,
    class_names: Sequence[str],
    species_by_class: Optional[Dict[str, str]] = None,
    ablation_summary_path: Optional[Path] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Đánh giá tập Test khóa kín duy nhất 1 lần cho Baseline Scratch vs Best Pre-trained và sinh đồ thị."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    runs_path = Path(runs_dir)
    eval_out_dir = Path(evaluation_dir)
    fig_out_dir = Path(figures_dir)
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = runs_path / baseline_run_id
    best_dir = runs_path / best_run_id

    base_test = evaluate_run_dir(
        run_dir=base_dir,
        eval_loader=test_loader,
        device=device,
        class_names=class_names,
        species_by_class=species_by_class,
        split_name="test",
        save_plots=True,
        tolerance=tolerance,
    )
    best_test = evaluate_run_dir(
        run_dir=best_dir,
        eval_loader=test_loader,
        device=device,
        class_names=class_names,
        species_by_class=species_by_class,
        split_name="test",
        save_plots=True,
        tolerance=tolerance,
    )

    # So sánh Loss & Accuracy curves
    base_hist = json.loads((base_dir / "historys" / "history.json").read_text(encoding="utf-8"))
    best_hist = json.loads((best_dir / "historys" / "history.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["train_loss"] for r in base_hist["epochs"]], "--", label=f"{baseline_run_id} (train)")
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["val_loss"] for r in base_hist["epochs"]], "-", marker="o", markersize=3, label=f"{baseline_run_id} (val)")
    ax.plot([r["epoch"] for r in best_hist["epochs"]], [r["train_loss"] for r in best_hist["epochs"]], "--", label=f"{best_run_id} (train)")
    ax.plot([r["epoch"] for r in best_hist["epochs"]], [r["val_loss"] for r in best_hist["epochs"]], "-", marker="s", markersize=3, label=f"{best_run_id} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"Loss Curves — Baseline ({baseline_run_id}) vs Best ({best_run_id})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    loss_cmp_path = fig_out_dir / "loss_curves.png"
    fig.savefig(loss_cmp_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["train_acc"] * 100 for r in base_hist["epochs"]], "--", label=f"{baseline_run_id} (train)")
    ax.plot([r["epoch"] for r in base_hist["epochs"]], [r["val_acc"] * 100 for r in base_hist["epochs"]], "-", marker="o", markersize=3, label=f"{baseline_run_id} (val)")
    ax.plot([r["epoch"] for r in best_hist["epochs"]], [r["train_acc"] * 100 for r in best_hist["epochs"]], "--", label=f"{best_run_id} (train)")
    ax.plot([r["epoch"] for r in best_hist["epochs"]], [r["val_acc"] * 100 for r in best_hist["epochs"]], "-", marker="s", markersize=3, label=f"{best_run_id} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy Curves — Baseline ({baseline_run_id}) vs Best ({best_run_id})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    acc_cmp_path = fig_out_dir / "accuracy_curves.png"
    fig.savefig(acc_cmp_path, dpi=150)
    plt.close(fig)

    cm_base_path = _plot_confusion_matrix(
        cm=np.asarray(base_test["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
        title=f"{baseline_run_id} — Test Confusion Matrix (Acc={base_test['accuracy']*100:.2f}%)",
        output_path=fig_out_dir / "confusion_matrix_baseline.png",
    )
    cm_best_path = _plot_confusion_matrix(
        cm=np.asarray(best_test["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
        title=f"{best_run_id} — Test Confusion Matrix (Acc={best_test['accuracy']*100:.2f}%)",
        output_path=fig_out_dir / "confusion_matrix_best.png",
    )

    # Predicted vs Actual trên Test của Best Model
    best_model, best_payload = load_checkpoint(best_dir / "best.pt", device=device)
    best_norm = ImageNetNormStats.from_dict(best_payload["norm_stats"])
    raw_best = evaluate_model_on_loader(
        best_model, test_loader, device=device, class_names=class_names, species_by_class=species_by_class
    )["_raw"]
    n_te = len(raw_best["y_true"])
    chosen = list(range(0, n_te, max(1, n_te // 16)))[:16]
    pred_vis_path = _plot_sample_grid(
        dataset=test_loader.dataset,
        indices=chosen,
        y_true=raw_best["y_true"],
        y_pred=raw_best["y_pred"],
        y_probs=raw_best["y_probs"],
        norm_stats=best_norm,
        class_names=class_names,
        title=f"{best_run_id} — Predicted vs Actual (Test Split)",
        output_path=fig_out_dir / "predicted_vs_actual.png",
        n_cols=4,
    )

    # Biểu đồ tổng hợp Ablation trên Val
    ablation_plot_path: Optional[Path] = None
    if ablation_summary_path and Path(ablation_summary_path).exists():
        abl_data = json.loads(Path(ablation_summary_path).read_text(encoding="utf-8"))
        runs_list = abl_data.get("runs", [])
        if runs_list:
            fig, ax = plt.subplots(figsize=(10.8, 4.8))
            labels = [r["run_id"] for r in runs_list]
            accs = [float(r["val_acc"]) * 100.0 for r in runs_list]
            f1s = [float(r["val_macro_f1"]) * 100.0 for r in runs_list]
            x = np.arange(len(labels))
            w = 0.36
            ax.bar(x - w / 2, accs, w, label="Val Accuracy (%)")
            ax.bar(x + w / 2, f1s, w, label="Val Macro-F1 (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
            ax.set_ylabel("Score (%)")
            ax.set_ylim(0, 102)
            ax.set_title("Controlled Experiment Comparison (Validation Split)")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            ablation_plot_path = fig_out_dir / "ablation_comparison.png"
            fig.savefig(ablation_plot_path, dpi=150)
            plt.close(fig)

    # Biểu đồ so sánh lỗi Cat vs Dog (Baseline vs Best)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    sp_labels = ["Species Acc\n(Cat vs Dog)", "Cat Breeds Acc\n(12 breeds)", "Dog Breeds Acc\n(25 breeds)"]
    b_sp = base_test["species_metrics"]
    m_sp = best_test["species_metrics"]
    b_vals = [b_sp["species_accuracy"] * 100, b_sp["cat_breed_accuracy"] * 100, b_sp["dog_breed_accuracy"] * 100]
    m_vals = [m_sp["species_accuracy"] * 100, m_sp["cat_breed_accuracy"] * 100, m_sp["dog_breed_accuracy"] * 100]
    xs = np.arange(len(sp_labels))
    w = 0.35
    bars1 = ax.bar(xs - w / 2, b_vals, w, label=baseline_run_id, color="#718096")
    bars2 = ax.bar(xs + w / 2, m_vals, w, label=best_run_id, color="#2b6cb0")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(sp_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Species & Fine-Grained Breed Accuracy: Baseline vs Best Model (Test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    species_plot_path = fig_out_dir / "cat_vs_dog_error_comparison.png"
    fig.savefig(species_plot_path, dpi=150)
    plt.close(fig)

    per_class_delta: Dict[str, Dict[str, Any]] = {}
    for cls_name in class_names:
        b_f1 = base_test["per_class"][cls_name]["f1"]
        m_f1 = best_test["per_class"][cls_name]["f1"]
        per_class_delta[cls_name] = {
            "species": best_test["per_class"][cls_name]["species"],
            "baseline_f1": b_f1,
            "best_f1": m_f1,
            "f1_gain": float(m_f1 - b_f1),
        }

    comparison_rows = [
        {
            "run_id": baseline_run_id,
            "arch": base_test["model_config"]["arch"],
            "mode": base_test["model_config"]["mode"],
            "total_params": base_test["param_counts"]["total"],
            "trainable_params": base_test["param_counts"]["trainable"],
            "test_loss": base_test["loss"],
            "test_accuracy": base_test["accuracy"],
            "test_macro_f1": base_test["macro_f1"],
            "species_accuracy": base_test["species_metrics"]["species_accuracy"],
            "save_load_passed": base_test["save_load_parity"]["passed"],
        },
        {
            "run_id": best_run_id,
            "arch": best_test["model_config"]["arch"],
            "mode": best_test["model_config"]["mode"],
            "total_params": best_test["param_counts"]["total"],
            "trainable_params": best_test["param_counts"]["trainable"],
            "test_loss": best_test["loss"],
            "test_accuracy": best_test["accuracy"],
            "test_macro_f1": best_test["macro_f1"],
            "species_accuracy": best_test["species_metrics"]["species_accuracy"],
            "save_load_passed": best_test["save_load_parity"]["passed"],
        },
    ]

    summary = {
        "baseline_run_id": baseline_run_id,
        "best_run_id": best_run_id,
        "comparison": comparison_rows,
        "gain_over_baseline": {
            "accuracy_delta": float(best_test["accuracy"] - base_test["accuracy"]),
            "macro_f1_delta": float(best_test["macro_f1"] - base_test["macro_f1"]),
            "loss_delta": float(best_test["loss"] - base_test["loss"]),
        },
        "figures": {
            "loss_curves": str(loss_cmp_path),
            "accuracy_curves": str(acc_cmp_path),
            "confusion_matrix_baseline": str(cm_base_path),
            "confusion_matrix_best": str(cm_best_path),
            "predicted_vs_actual": str(pred_vis_path),
            "cat_vs_dog_error_comparison": str(species_plot_path),
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
            {"baseline": base_test["per_class"], "best": best_test["per_class"], "delta": per_class_delta},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (eval_out_dir / "species_analysis.json").write_text(
        json.dumps(
            {"baseline": base_test["species_metrics"], "best": best_test["species_metrics"]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (eval_out_dir / "error_analysis.json").write_text(
        json.dumps(
            {
                "baseline_top_confusions": base_test["top_confusions"],
                "best_top_confusions": best_test["top_confusions"],
                "hardest_classes_best": sorted(per_class_delta.items(), key=lambda kv: kv[1]["best_f1"])[:5],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def _compute_pet_image_attributes(dataset: Any, norm_stats: ImageNetNormStats) -> Dict[str, np.ndarray]:
    """Tính thuộc tính ảnh (độ sáng, tỷ lệ khung gốc, diện tích gốc) cho Slice Analysis."""
    n = len(dataset)
    brightness = np.zeros(n, dtype=np.float64)
    aspect_ratio = np.ones(n, dtype=np.float64)
    orig_area = np.full(n, 224.0 * 224.0, dtype=np.float64)

    has_paths = hasattr(dataset, "image_paths")
    for i in range(n):
        if has_paths:
            with Image.open(dataset.image_paths[i]) as im:
                w, h = im.size
                aspect_ratio[i] = w / max(h, 1)
                orig_area[i] = float(w * h)
                arr = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
                brightness[i] = float(arr.mean())
        else:
            tensor, _ = dataset[i]
            arr = denormalize_tensor(tensor, norm_stats).cpu().numpy()
            brightness[i] = float(arr.mean())
    return {
        "brightness": brightness,
        "aspect_ratio": aspect_ratio,
        "orig_area": orig_area,
    }


def _slice_quartiles(
    values: np.ndarray,
    is_error: np.ndarray,
    cat_mask: np.ndarray,
    labels: Sequence[str],
) -> List[Dict[str, Any]]:
    edges = np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0])
    rows: List[Dict[str, Any]] = []
    for q in range(4):
        lo, hi = float(edges[q]), float(edges[q + 1])
        mask = (values >= lo) & (values <= hi if q == 3 else values < hi)
        c_m = mask & cat_mask
        d_m = mask & (~cat_mask)
        rows.append(
            {
                "quartile": labels[q],
                "range": [round(lo, 4), round(hi, 4)],
                "n_all": int(mask.sum()),
                "errors_all": int(is_error[mask].sum()),
                "error_rate_all_pct": round((int(is_error[mask].sum()) / max(int(mask.sum()), 1)) * 100.0, 2),
                "error_rate_cat_pct": round((int(is_error[c_m].sum()) / max(int(c_m.sum()), 1)) * 100.0, 2),
                "error_rate_dog_pct": round((int(is_error[d_m].sum()) / max(int(d_m.sum()), 1)) * 100.0, 2),
            }
        )
    return rows


def run_deep_error_analysis(
    run_dir: Path,
    eval_loader: DataLoader,
    device: Union[str, torch.device],
    evaluation_dir: Path,
    figures_dir: Path,
    class_names: Sequence[str],
    species_by_class: Optional[Dict[str, str]] = None,
    top_k_pairs: int = 3,
    gallery_size: int = 16,
    audit_target_total: int = 60,
) -> Dict[str, Any]:
    """Error Analysis 4 bước định lượng cho Oxford-IIIT Pets."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    eval_out_dir = Path(evaluation_dir)
    fig_out_dir = Path(figures_dir)
    sheets_dir = eval_out_dir / "audit_sheets"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_checkpoint(run_path / "best.pt", device=device)
    norm_stats = ImageNetNormStats.from_dict(payload["norm_stats"])
    run_id = str(payload.get("run_id", run_path.name))
    sp_map = species_by_class or payload.get("species_by_class") or {str(c): "Unknown" for c in class_names}

    eval_out = evaluate_model_on_loader(
        model, eval_loader, device=device, class_names=class_names, species_by_class=sp_map
    )
    raw = eval_out["_raw"]
    y_true = raw["y_true"]
    y_pred = raw["y_pred"]
    y_probs = raw["y_probs"]
    cm = np.asarray(eval_out["confusion_matrix"], dtype=np.int64)
    is_error = y_true != y_pred
    all_wrong_idx = np.flatnonzero(is_error)

    # Bước 1: Row-normalized Confusion Matrix (%)
    cm_norm_path = _plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title=f"{run_id} — Row-Normalized Confusion Matrix (%) (Test Split)",
        output_path=fig_out_dir / "confusion_matrix_best_normalized.png",
        normalize_rows=True,
    )

    # Bước 2: Error Gallery Top-K cặp nhầm lẫn chia High-Conf vs Low-Conf (tự động mở rộng sang toàn bộ lỗi nếu Top-3 < gallery_size)
    top_confs = eval_out["top_confusions"]
    top_pairs = [(c["true_id"], c["pred_id"]) for c in top_confs[:top_k_pairs]]
    pair_mask = np.zeros(len(y_true), dtype=bool)
    for t_id, p_id in top_pairs:
        pair_mask |= (y_true == t_id) & (y_pred == p_id)
    top_pair_wrong_idx = np.flatnonzero(pair_mask)

    gallery_pool = top_pair_wrong_idx if len(top_pair_wrong_idx) >= gallery_size else all_wrong_idx
    gallery_scope = (
        f"Top-{len(top_pairs)} Pairs ({len(top_pair_wrong_idx)} errors)"
        if len(top_pair_wrong_idx) >= gallery_size
        else f"All Test Errors ({len(all_wrong_idx)} errors; Top-3 had {len(top_pair_wrong_idx)})"
    )
    pool_confs = y_probs[gallery_pool, y_pred[gallery_pool]]
    hi_idx = gallery_pool[np.argsort(-pool_confs)[: min(gallery_size, len(gallery_pool))]]
    lo_idx = gallery_pool[np.argsort(pool_confs)[: min(gallery_size, len(gallery_pool))]]

    hi_path = _plot_sample_grid(
        dataset=eval_loader.dataset,
        indices=hi_idx,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        norm_stats=norm_stats,
        class_names=class_names,
        title=f"{run_id} — {len(hi_idx)} High-Confidence Errors [{gallery_scope}]",
        output_path=fig_out_dir / "error_gallery_top3_high_conf.png",
        n_cols=4,
    )
    lo_path = _plot_sample_grid(
        dataset=eval_loader.dataset,
        indices=lo_idx,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        norm_stats=norm_stats,
        class_names=class_names,
        title=f"{run_id} — {len(lo_idx)} Low-Confidence (Boundary) Errors [{gallery_scope}]",
        output_path=fig_out_dir / "error_gallery_top3_low_conf.png",
        n_cols=4,
    )

    # Bước 3: Xuất tối đa 60 mẫu lỗi (nửa High-Conf + nửa Low-Conf) ra 3 Audit Sheets (20 ảnh/sheet)
    n_audit = min(int(audit_target_total), len(all_wrong_idx))
    half = n_audit // 2
    all_wrong_confs = y_probs[all_wrong_idx, y_pred[all_wrong_idx]]
    hi_audit = list(all_wrong_idx[np.argsort(-all_wrong_confs)[:half]])
    hi_set = set(int(i) for i in hi_audit)
    rem_low = [int(i) for i in all_wrong_idx[np.argsort(all_wrong_confs)] if int(i) not in hi_set]
    lo_audit = rem_low[: (n_audit - len(hi_audit))]
    audit_indices = [int(i) for i in hi_audit] + lo_audit

    attrs = _compute_pet_image_attributes(eval_loader.dataset, norm_stats=norm_stats)
    audit_candidates: List[Dict[str, Any]] = []
    for pos, sample_idx in enumerate(audit_indices):
        t_id = int(y_true[sample_idx])
        p_id = int(y_pred[sample_idx])
        audit_candidates.append(
            {
                "audit_id": pos + 1,
                "test_idx": sample_idx,
                "pair": f"{class_names[t_id]} -> {class_names[p_id]}",
                "same_species": sp_map.get(class_names[t_id]) == sp_map.get(class_names[p_id]),
                "conf_group": "high_conf" if pos < len(hi_audit) else "low_conf",
                "true_class": str(class_names[t_id]),
                "pred_class": str(class_names[p_id]),
                "pred_conf_pct": round(float(y_probs[sample_idx, p_id]) * 100.0, 2),
                "true_conf_pct": round(float(y_probs[sample_idx, t_id]) * 100.0, 2),
                "brightness": round(float(attrs["brightness"][sample_idx]), 4),
                "aspect_ratio": round(float(attrs["aspect_ratio"][sample_idx]), 3),
                "orig_area": int(attrs["orig_area"][sample_idx]),
            }
        )
    (eval_out_dir / "error_audit_60_candidates.json").write_text(
        json.dumps(audit_candidates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sheet_paths: List[str] = []
    per_sheet = 20
    for s_i in range(0, len(audit_indices), per_sheet):
        chunk = audit_indices[s_i : s_i + per_sheet]
        a_ids = list(range(s_i + 1, s_i + len(chunk) + 1))
        sheet_num = (s_i // per_sheet) + 1
        s_path = _plot_sample_grid(
            dataset=eval_loader.dataset,
            indices=chunk,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            norm_stats=norm_stats,
            class_names=class_names,
            title=f"Audit Sheet {sheet_num}: Errors #{a_ids[0]}–#{a_ids[-1]} ({run_id})",
            output_path=sheets_dir / f"audit_sheet_{sheet_num}.png",
            n_cols=5,
            audit_ids=a_ids,
        )
        sheet_paths.append(str(s_path))

    # Bước 4: Slice Analysis (Brightness, Aspect Ratio, Original Area + Species Cat vs Dog)
    true_sp = np.array([sp_map.get(str(class_names[i]), "Unknown") for i in y_true])
    cat_mask = true_sp == "Cat"
    slice_data = {
        "overall_error_rate_pct": round(float(is_error.mean()) * 100.0, 2),
        "cat_error_rate_pct": round(float(is_error[cat_mask].mean()) * 100.0, 2) if cat_mask.any() else 0.0,
        "dog_error_rate_pct": round(float(is_error[~cat_mask].mean()) * 100.0, 2) if (~cat_mask).any() else 0.0,
        "brightness": _slice_quartiles(attrs["brightness"], is_error, cat_mask, ["Q1 Dark", "Q2 Dim", "Q3 Mid", "Q4 Bright"]),
        "aspect_ratio": _slice_quartiles(attrs["aspect_ratio"], is_error, cat_mask, ["Q1 Tall", "Q2 Port", "Q3 Square", "Q4 Wide"]),
        "orig_area": _slice_quartiles(attrs["orig_area"], is_error, cat_mask, ["Q1 LowRes", "Q2 Med-L", "Q3 Med-H", "Q4 HiRes"]),
    }
    (eval_out_dir / "slice_analysis.json").write_text(
        json.dumps(slice_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    specs = [
        ("brightness", "Image Brightness"),
        ("aspect_ratio", "Original Aspect Ratio (W/H)"),
        ("orig_area", "Original Image Resolution Area"),
    ]
    max_y = 10.0
    for ax, (key, title_s) in zip(axes, specs):
        rows = slice_data[key]
        x = np.arange(len(rows))
        w = 0.27
        v_all = [r["error_rate_all_pct"] for r in rows]
        v_cat = [r["error_rate_cat_pct"] for r in rows]
        v_dog = [r["error_rate_dog_pct"] for r in rows]
        max_y = max(max_y, max(v_all + v_cat + v_dog))
        ax.bar(x - w, v_all, w, label="All 37 Breeds", color="#2b6cb0")
        ax.bar(x, v_cat, w, label="Cat (12)", color="#dd6b20")
        ax.bar(x + w, v_dog, w, label="Dog (25)", color="#2f855a")
        ax.set_xticks(x)
        ax.set_xticklabels([r["quartile"] for r in rows], fontsize=8.3)
        ax.set_title(title_s, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Error Rate (%)")
    axes[0].set_ylim(0, max_y + 8.0)
    axes[2].legend(loc="upper right", fontsize=8.0)
    fig.suptitle(f"{run_id} — Slice Analysis by Image Attributes & Species (Test Split)", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    slice_plot_path = fig_out_dir / "error_slice_analysis.png"
    fig.savefig(slice_plot_path, dpi=150)
    plt.close(fig)

    err_json_path = eval_out_dir / "error_analysis.json"
    existing = json.loads(err_json_path.read_text(encoding="utf-8")) if err_json_path.exists() else {}
    existing["total_test_errors"] = int(len(all_wrong_idx))
    existing["top3_pair_error_count"] = int(len(top_pair_wrong_idx))
    existing["gallery_scope"] = gallery_scope
    existing["species_metrics"] = eval_out["species_metrics"]
    existing["deep_figures"] = {
        "confusion_matrix_best_normalized": str(cm_norm_path),
        "error_gallery_top3_high_conf": str(hi_path),
        "error_gallery_top3_low_conf": str(lo_path),
        "error_slice_analysis": str(slice_plot_path),
    }
    err_json_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "run_id": run_id,
        "total_test_errors": int(len(all_wrong_idx)),
        "top3_pair_error_count": int(len(top_pair_wrong_idx)),
        "gallery_scope": gallery_scope,
        "audit_count": len(audit_candidates),
        "figures": existing["deep_figures"],
        "audit_sheets": sheet_paths,
        "slice_analysis": slice_data,
    }


def plot_error_taxonomy_breakdown(
    taxonomy_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    """Tổng hợp tỷ lệ nguyên nhân lỗi trên N mẫu đã gắn nhãn và vẽ biểu đồ 2 panel."""
    _ensure_headless_matplotlib()
    import matplotlib.pyplot as plt

    categories = ["model_limitation", "ambiguity", "label_noise"]
    cat_labels = {
        "model_limitation": "Model Limitation\n(fine-grained missed)",
        "ambiguity": "Ambiguity / Framing\n(blur, crop, occlusion)",
        "label_noise": "Label Noise / Overlap\n(suspect breed label)",
    }
    cat_colors = {"model_limitation": "#c53030", "ambiguity": "#dd6b20", "label_noise": "#4a5568"}

    total = len(taxonomy_rows)
    overall_counts = {c: sum(1 for r in taxonomy_rows if r["cause"] == c) for c in categories}
    overall_pct = {c: round((overall_counts[c] / max(total, 1)) * 100.0, 1) for c in categories}

    by_conf: Dict[str, Dict[str, float]] = {}
    for grp in ("high_conf", "low_conf"):
        sub = [r for r in taxonomy_rows if r["conf_group"] == grp]
        n_sub = max(len(sub), 1)
        by_conf[grp] = {
            c: round((sum(1 for r in sub if r["cause"] == c) / n_sub) * 100.0, 1) for c in categories
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x1 = np.arange(len(categories))
    vals1 = [overall_pct[c] for c in categories]
    bars1 = ax1.bar(x1, vals1, color=[cat_colors[c] for c in categories], width=0.55)
    for bar, c in zip(bars1, categories):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%\n({overall_counts[c]}/{total})", ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([cat_labels[c] for c in categories], fontsize=8.5)
    ax1.set_ylabel("Share of Audited Errors (%)")
    ax1.set_ylim(0, max(vals1 + [10.0]) + 15.0)
    ax1.set_title(f"Root-Cause Breakdown (N = {total} Audited Test Errors)")
    ax1.grid(axis="y", alpha=0.25)

    groups = ["high_conf", "low_conf"]
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
    ax2.set_xticklabels(["High-Confidence Errors", "Low-Confidence Errors"], fontsize=9.0)
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
        "figure": str(output_path),
    }
