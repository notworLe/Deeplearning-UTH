"""Pipeline 03: Training Loop, Baseline Sanity Check & Atomic Checkpointing cho Lab 1."""

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from labs.lab1.vibe import config as lab1_cfg
from src.vibe_scripts.lab1.data import build_dataloaders
from src.vibe_scripts.lab1.models import build_model, count_parameters

console = Console()


def set_global_seed(seed: int = lab1_cfg.SEED) -> None:
    """Khóa ngẫu nhiên đồng bộ trên Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint_atomic(payload: Dict[str, Any], path: Path) -> None:
    """Lưu checkpoint nguyên tử qua tệp `.tmp` trung gian để chống hỏng file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _is_better_metric(
    current: float,
    best: Optional[float],
    metric_name: str,
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True
    if metric_name == "val_loss":
        return current < (best - min_delta)
    if metric_name == "val_acc":
        return current > (best + min_delta)
    raise ValueError(f"Unsupported best_metric: '{metric_name}'")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Chạy 1 epoch huấn luyện chuẩn 5 bước và trả về `(avg_loss, accuracy)`."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss detected: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.detach().item()) * batch_size
        preds = logits.detach().argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += batch_size

    if total == 0:
        raise ValueError("train_loader không có mẫu nào.")
    return running_loss / total, correct / total


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Đánh giá mô hình trên tập Validation dưới `model.eval()` và `torch.no_grad()`."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss detected: {loss.item()}")

        batch_size = labels.size(0)
        running_loss += float(loss.detach().item()) * batch_size
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += batch_size

    if total == 0:
        raise ValueError("val_loader không có mẫu nào.")
    return running_loss / total, correct / total


def run_sanity_check(
    model_name: str,
    train_loader: DataLoader,
    device: Optional[torch.device] = None,
    lr: float = 1e-2,
    steps: int = 80,
    seed: int = lab1_cfg.SEED,
) -> Dict[str, float]:
    """
    Kiểm tra nhanh Baseline: đo initial loss (~ ln(10)) và khả năng overfit 1 batch.
    """
    set_global_seed(seed)
    target_device = device if device is not None else lab1_cfg.DEVICE
    model = build_model(model_name).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    images, labels = next(iter(train_loader))
    images = images.to(target_device)
    labels = labels.to(target_device)

    model.eval()
    with torch.no_grad():
        initial_loss = float(criterion(model(images), labels).item())

    model.train()
    final_loss = initial_loss
    final_acc = 0.0
    for _ in range(steps):
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        final_acc = float((logits.detach().argmax(dim=1) == labels).float().mean().item())

    return {
        "initial_loss": initial_loss,
        "overfit_final_loss": final_loss,
        "overfit_final_acc": final_acc,
    }


def train_experiment(
    model_name: str,
    aug_enabled: bool = False,
    epochs: int = lab1_cfg.EPOCH,
    lr: float = lab1_cfg.LR,
    weight_decay: float = lab1_cfg.WEIGHT_DECAY,
    seed: int = lab1_cfg.SEED,
    best_metric: str = lab1_cfg.BEST_METRIC,
    early_stopping_patience: Optional[int] = None,
    min_delta: float = 1e-4,
    output_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    norm_stats: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Huấn luyện 1 cấu hình thí nghiệm (`model_name` x `aug_enabled`), lưu `best.pt`, `last.pt`, `history.json`.
    """
    set_global_seed(seed)
    target_device = device if device is not None else lab1_cfg.DEVICE

    if train_loader is None or val_loader is None:
        train_loader, val_loader, _, loaded_stats = build_dataloaders(
            aug_enabled=aug_enabled,
            seed=seed,
        )
        stats_to_save = norm_stats if norm_stats is not None else loaded_stats
    else:
        stats_to_save = norm_stats if norm_stats is not None else {"mean": 0.0, "std": 1.0}

    aug_tag = "aug" if aug_enabled else "no_aug"
    exp_id = f"{model_name.lower()}_{aug_tag}"
    exp_dir = Path(output_dir) if output_dir is not None else (lab1_cfg.EXPERIMENT_DIR / exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hyperparams: Dict[str, Any] = {
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": getattr(train_loader, "batch_size", lab1_cfg.BATCH_SIZE),
        "epochs": epochs,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "best_metric": best_metric,
    }

    history: List[Dict[str, Any]] = []
    best_value: Optional[float] = None
    best_epoch = 0
    epochs_no_improve = 0
    last_payload: Optional[Dict[str, Any]] = None
    t_start = time.perf_counter()

    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc=f"Train [{exp_id}]",
        disable=not show_progress,
        leave=True,
    )
    for epoch in epoch_iter:
        ep_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, target_device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, target_device)
        ep_time = time.perf_counter() - ep_start

        current_metric = val_loss if best_metric == "val_loss" else val_acc
        improved = _is_better_metric(
            current_metric,
            best_value,
            best_metric,
            min_delta=0.0 if best_value is None else min_delta,
        )

        if improved:
            best_value = current_metric
            best_epoch = epoch
            epochs_no_improve = 0
            best_payload = {
                "model_name": model_name.lower(),
                "aug_enabled": aug_enabled,
                "epoch": epoch,
                "best_metric_name": best_metric,
                "best_metric_value": float(best_value),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "norm_stats": stats_to_save,
                "seed": seed,
                "hyperparams": hyperparams,
            }
            save_checkpoint_atomic(best_payload, exp_dir / "best.pt")
        else:
            epochs_no_improve += 1

        last_payload = {
            "model_name": model_name.lower(),
            "aug_enabled": aug_enabled,
            "epoch": epoch,
            "best_metric_name": best_metric,
            "best_metric_value": float(best_value if best_value is not None else current_metric),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "norm_stats": stats_to_save,
            "seed": seed,
            "hyperparams": hyperparams,
        }

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": ep_time,
            }
        )
        epoch_iter.set_postfix(
            tr_loss=f"{train_loss:.3f}",
            val_loss=f"{val_loss:.3f}",
            val_acc=f"{val_acc*100:.1f}%",
            best_ep=best_epoch,
        )

        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            break

    if last_payload is not None:
        save_checkpoint_atomic(last_payload, exp_dir / "last.pt")

    total_time = time.perf_counter() - t_start
    summary: Dict[str, Any] = {
        "exp_id": exp_id,
        "model_name": model_name.lower(),
        "aug_enabled": aug_enabled,
        "num_params": count_parameters(model),
        "best_epoch": best_epoch,
        "best_metric_name": best_metric,
        "best_metric_value": float(best_value if best_value is not None else 0.0),
        "epochs_completed": len(history),
        "total_train_time_sec": total_time,
        "seed": seed,
        "hyperparams": hyperparams,
        "norm_stats": stats_to_save,
        "history": history,
    }
    (exp_dir / "history.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Điểm vào CLI huấn luyện một cấu hình thí nghiệm Lab 1."""
    parser = argparse.ArgumentParser(description="Lab 1 Training Pipeline CLI")
    parser.add_argument("--model", choices=["logistic", "mlp", "cnn"], default="logistic")
    parser.add_argument("--aug", action="store_true", help="Enable data augmentation on train")
    parser.add_argument("--epochs", type=int, default=lab1_cfg.EPOCH)
    parser.add_argument("--lr", type=float, default=lab1_cfg.LR)
    parser.add_argument("--seed", type=int, default=lab1_cfg.SEED)
    args = parser.parse_args()

    result = train_experiment(
        model_name=args.model,
        aug_enabled=args.aug,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    console.print(
        f"[green][DONE][/green] {result['exp_id']} | "
        f"Best Epoch: {result['best_epoch']} ({result['best_metric_name']}={result['best_metric_value']:.4f}) | "
        f"Time: {result['total_train_time_sec']:.2f}s"
    )


if __name__ == "__main__":
    main()
