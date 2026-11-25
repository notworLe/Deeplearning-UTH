"""
Pipeline 03: Vòng lặp huấn luyện công bằng cho 3 chế độ ResNet-18 Lab 2.

- Giữ nguyên trạng thái `eval()` cho toàn bộ BatchNorm của phần bị đóng băng ở mỗi epoch.
- Hỗ trợ differential learning rate (`layer4` vs `fc`) ở chế độ `finetune`.
- Dùng AMP trên CUDA để tối ưu tốc độ RTX 4060, fallback chuẩn xác trên CPU.
- Lưu nguyên tử `best.pt` (theo `val_loss`), `last.pt` và `history.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from labs.lab2.vibe import config as lab2_cfg
from src.vibe_scripts.lab2.data import build_dataloaders
from src.vibe_scripts.lab2.models import (
    build_optimizer_param_groups,
    build_resnet18_pet,
    count_parameters,
    set_mode_for_training,
)

console = Console()


def set_global_seed(seed: int = lab2_cfg.SEED) -> None:
    """Khóa ngẫu nhiên đồng bộ trên Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_torch_save(payload: Dict[str, Any], target_path: Path) -> None:
    """Lưu checkpoint nguyên tử (.tmp -> replace) tránh hỏng file khi gián đoạn."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)


def train_one_epoch(
    model: nn.Module,
    mode: str,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Thực thi 1 epoch huấn luyện 5 bước, luôn giữ BN của tầng đóng băng ở `eval()`."""
    set_mode_for_training(model, mode)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss không hữu hạn (NaN/Inf): {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss không hữu hạn (NaN/Inf): {loss.item()}")
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        total_loss += loss.detach().item() * bs
        total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        total_samples += bs

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Đánh giá 1 epoch trên tập Validation dưới `model.eval()` + `torch.no_grad()`."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        bs = targets.size(0)
        total_loss += loss.detach().item() * bs
        total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        total_samples += bs

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_experiment(
    mode: str,
    epochs: int = lab2_cfg.EPOCH,
    lr_head: float = lab2_cfg.LR_HEAD,
    lr_backbone: float = lab2_cfg.LR_BACKBONE,
    batch_size: int = lab2_cfg.BATCH_SIZE,
    seed: int = lab2_cfg.SEED,
    best_metric: str = lab2_cfg.BEST_METRIC,
    device: Optional[torch.device] = None,
    init_checkpoint: Optional[Path] = None,
    exp_id: Optional[str] = None,
    experiment_dir: Path = lab2_cfg.EXPERIMENT_DIR,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    use_pretrained: bool = True,
    use_amp: Optional[bool] = None,
    num_workers: int = 0,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Huấn luyện trọn vẹn 1 cấu hình (`baseline` / `transfer` / `finetune`) và lưu artifacts."""
    set_global_seed(seed)
    dev = device if device is not None else lab2_cfg.DEVICE
    amp_enabled = (dev.type == "cuda") if use_amp is None else (use_amp and dev.type == "cuda")

    if train_loader is None or val_loader is None:
        train_loader, val_loader, _, _ = build_dataloaders(
            batch_size=batch_size, seed=seed, num_workers=num_workers
        )

    run_id = exp_id if exp_id is not None else f"resnet18_{mode}"
    run_dir = Path(experiment_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet18_pet(
        mode=mode,
        num_classes=lab2_cfg.NUM_CLASSES,
        init_checkpoint=init_checkpoint,
        use_pretrained=use_pretrained,
    ).to(dev)
    total_params, trainable_params = count_parameters(model)

    param_groups = build_optimizer_param_groups(model, mode=mode, lr_head=lr_head, lr_backbone=lr_backbone)
    optimizer = torch.optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    best_value = float("inf") if best_metric == "val_loss" else float("-inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    ep_iter = tqdm(range(1, epochs + 1), desc=run_id, disable=not show_progress)
    for epoch in ep_iter:
        ep_t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(
            model=model,
            mode=mode,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=dev,
            scaler=scaler,
            use_amp=amp_enabled,
        )
        va_loss, va_acc = evaluate_one_epoch(model, val_loader, criterion, dev)
        ep_time = time.perf_counter() - ep_t0

        record = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "epoch_time_sec": ep_time,
        }
        history.append(record)

        current_val = va_loss if best_metric == "val_loss" else va_acc
        improved = (current_val < best_value) if best_metric == "val_loss" else (current_val > best_value)

        ckpt_payload = {
            "exp_id": run_id,
            "mode": mode,
            "num_classes": lab2_cfg.NUM_CLASSES,
            "epoch": epoch,
            "best_epoch": epoch if improved else best_epoch,
            "best_metric_name": best_metric,
            "best_metric_value": current_val if improved else best_value,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "seed": seed,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
            "hyperparams": {
                "lr_head": lr_head,
                "lr_backbone": lr_backbone if mode == "finetune" else None,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss",
                "best_metric": best_metric,
                "amp": amp_enabled,
            },
            "model_state_dict": model.state_dict(),
        }

        if improved:
            best_value = current_val
            best_epoch = epoch
            ckpt_payload["best_epoch"] = best_epoch
            ckpt_payload["best_metric_value"] = best_value
            atomic_torch_save(ckpt_payload, run_dir / "best.pt")

        atomic_torch_save(ckpt_payload, run_dir / "last.pt")
        if show_progress:
            ep_iter.set_postfix(tr_loss=f"{tr_loss:.3f}", va_loss=f"{va_loss:.3f}", va_acc=f"{va_acc*100:.1f}%")

    total_time = time.perf_counter() - t0
    summary = {
        "exp_id": run_id,
        "mode": mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric,
        "best_metric_value": best_value,
        "epochs_completed": len(history),
        "total_train_time_sec": total_time,
        "seed": seed,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "hyperparams": {
            "lr_head": lr_head,
            "lr_backbone": lr_backbone if mode == "finetune" else None,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "best_metric": best_metric,
            "amp": amp_enabled,
        },
        "history": history,
    }
    (run_dir / "history.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    """Điểm vào CLI huấn luyện một chế độ thực nghiệm Lab 2."""
    parser = argparse.ArgumentParser(description="Lab 2 Oxford-IIIT Pets Training CLI")
    parser.add_argument("--mode", choices=["baseline", "transfer", "finetune"], required=True)
    parser.add_argument("--epochs", type=int, default=lab2_cfg.EPOCH)
    parser.add_argument("--lr-head", type=float, default=lab2_cfg.LR_HEAD)
    parser.add_argument("--lr-backbone", type=float, default=lab2_cfg.LR_BACKBONE)
    parser.add_argument("--seed", type=int, default=lab2_cfg.SEED)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--exp-id", type=str, default=None)
    args = parser.parse_args()

    res = train_experiment(
        mode=args.mode,
        epochs=args.epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
        exp_id=args.exp_id,
    )
    console.print(
        f"[green][DONE][/green] {res['exp_id']} | Best Ep {res['best_epoch']} "
        f"({res['best_metric_name']}={res['best_metric_value']:.4f}) | "
        f"Trainable={res['trainable_params']:,} | Time={res['total_train_time_sec']:.1f}s"
    )


if __name__ == "__main__":
    main()
