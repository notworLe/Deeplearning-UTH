"""Training Engine, Sanity Checks, TensorBoard & tqdm Monitoring, Resume Support và Ablation Orchestrator cho Lab 2 v2."""

import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.vibe_scripts.lab2_v2.data import OxfordPetsBundle, prepare_oxford_pets
from src.vibe_scripts.lab2_v2.evaluator import evaluate_run_dir
from src.vibe_scripts.lab2_v2.models import PetClassifier, build_model, count_parameters


def set_global_seed(seed: int) -> None:
    """Cố định toàn bộ nguồn ngẫu nhiên Python / NumPy / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: Dict[str, Any], target_path: Path) -> None:
    """Lưu checkpoint nguyên tử qua file `.tmp` rồi thay thế để tránh file hỏng."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target)


def _create_tb_writer(log_dir: Optional[Path], enabled: bool) -> Tuple[Optional[Any], bool]:
    """Khởi tạo TensorBoard SummaryWriter an toàn; tự fallback nếu `tensorboard` chưa được cài."""
    if not enabled or log_dir is None:
        return None, False
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
    except Exception:
        return None, False
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir)), True


def run_sanity_checks(
    model: PetClassifier,
    train_loader: DataLoader,
    device: Union[str, torch.device],
    lr: float = 1e-3,
    overfit_steps: int = 50,
    overfit_target_acc: float = 0.95,
    max_batch_samples: int = 16,
) -> Dict[str, Any]:
    """Kiểm chứng trước huấn luyện: Initial Loss ≈ ln(num_classes) & Overfit 1 batch nhỏ ≥ 95%."""
    device_obj = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    sample_x, sample_y = next(iter(train_loader))
    sample_x = sample_x[:max_batch_samples].to(device_obj)
    sample_y = sample_y[:max_batch_samples].to(device_obj)

    model_copy = copy.deepcopy(model).to(device_obj)
    model_copy.eval()
    with torch.inference_mode():
        init_logits = model_copy(sample_x)
        init_loss = float(criterion(init_logits, sample_y).item())
    expected_loss = math.log(model_copy.num_classes)
    init_loss_ok = math.isfinite(init_loss) and (abs(init_loss - expected_loss) <= 1.2)

    # Mở khóa tạm thời (hoặc dùng đúng head) để kiểm chứng gradient flow có thể ghi nhớ 1 batch nhỏ
    model_copy.train()
    optimizer = torch.optim.AdamW(
        [p for p in model_copy.parameters() if p.requires_grad],
        lr=max(float(lr), 1e-3),
    )
    final_acc = 0.0
    final_loss = init_loss
    steps_used = 0
    for step in range(1, int(overfit_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model_copy(sample_x)
        loss = criterion(logits, sample_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Sanity check gặp loss không hữu hạn tại step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        final_acc = float((logits.argmax(dim=1) == sample_y).float().mean().item())
        steps_used = step
        if final_acc >= overfit_target_acc and final_loss < 0.25:
            break

    return {
        "num_classes": model_copy.num_classes,
        "expected_init_loss": round(expected_loss, 4),
        "actual_init_loss": round(init_loss, 4),
        "init_loss_passed": bool(init_loss_ok),
        "overfit_batch_size": int(sample_x.shape[0]),
        "overfit_steps": int(steps_used),
        "overfit_final_loss": round(final_loss, 4),
        "overfit_final_acc": round(final_acc, 4),
        "overfit_passed": bool(final_acc >= overfit_target_acc),
        "passed": bool(init_loss_ok and (final_acc >= overfit_target_acc)),
    }


def train_one_epoch(
    model: PetClassifier,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    show_batch_bar: bool = False,
    epoch_label: str = "",
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = (
        tqdm(train_loader, desc=f"{epoch_label} train", leave=False, dynamic_ncols=True)
        if show_batch_bar
        else train_loader
    )
    amp_enabled = bool(use_amp and device.type == "cuda")

    for batch_x, batch_y in iterator:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Training diverged: loss={loss.item()} ({epoch_label})")

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = int(batch_y.shape[0])
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total_samples += bs

    return {
        "train_loss": total_loss / max(total_samples, 1),
        "train_acc": total_correct / max(total_samples, 1),
    }


@torch.inference_mode()
def validate_one_epoch(
    model: PetClassifier,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
        bs = int(batch_y.shape[0])
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total_samples += bs

    return {
        "val_loss": total_loss / max(total_samples, 1),
        "val_acc": total_correct / max(total_samples, 1),
    }


def train_single_run(
    run_spec: Dict[str, Any],
    bundle: OxfordPetsBundle,
    runs_dir: Path,
    device: Union[str, torch.device],
    seed: int = 248,
    use_amp: bool = True,
    tb_logs_dir: Optional[Path] = None,
    enable_tensorboard: bool = True,
    show_progress: bool = True,
    show_batch_bar: bool = False,
    eval_after_train: bool = True,
    save_eval_plots: bool = True,
    tolerance: float = 1e-6,
    resume_from: Optional[Union[str, Path]] = None,
    extra_epochs: Optional[int] = None,
    output_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Huấn luyện 1 cấu hình, hỗ trợ `--amp`, TensorBoard, `--resume`, lưu `historys/` và eval ngay trên Val."""
    device_obj = torch.device(device)
    amp_active = bool(use_amp and device_obj.type == "cuda")

    run_id = str(output_run_id or run_spec["run_id"])
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Trạng thái resume (nếu có)
    start_epoch = 1
    epoch_records: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    prev_train_time_sec = 0.0
    resumed_meta: Optional[Dict[str, Any]] = None
    resume_payload: Optional[Dict[str, Any]] = None

    if resume_from is not None:
        r_path = Path(resume_from)
        if not r_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint resume: {r_path}")
        resume_payload = torch.load(r_path, map_location=device_obj, weights_only=False)
        prev_epoch = int(resume_payload.get("epoch", 0))
        start_epoch = prev_epoch + 1
        best_val_loss = float(resume_payload.get("best_val_loss", float("inf")))
        best_val_acc = float(resume_payload.get("best_val_acc", 0.0))
        best_epoch = int(resume_payload.get("best_epoch", prev_epoch))

        prev_run_dir = r_path.parent
        prev_hist_path = prev_run_dir / "historys" / "history.json"
        if prev_hist_path.exists():
            prev_hist = json.loads(prev_hist_path.read_text(encoding="utf-8"))
            epoch_records = [dict(row) for row in prev_hist.get("epochs", []) if int(row["epoch"]) <= prev_epoch]
            prev_train_time_sec = float(prev_hist.get("train_time_sec", 0.0))
        prev_best = prev_run_dir / "best.pt"
        if prev_best.exists() and prev_best.resolve() != (run_dir / "best.pt").resolve():
            shutil.copy2(prev_best, run_dir / "best.pt")

    set_global_seed(int(seed) + (start_epoch - 1))

    # Khi resume thì không cần tải weight gốc từ mạng vì ta sẽ nạp model_state_dict từ checkpoint
    want_pretrained = bool(run_spec.get("pretrained", True)) and (resume_payload is None)
    model = build_model(
        arch=str(run_spec["arch"]),
        mode=str(run_spec["mode"]),
        num_classes=len(bundle.class_names),
        pretrained=want_pretrained,
        image_size=int(run_spec.get("image_size", 224)),
    ).to(device_obj)

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])

    lr_head = float(run_spec["lr"])
    lr_backbone = float(run_spec.get("lr_backbone", 0.0))
    weight_decay = float(run_spec.get("weight_decay", 1e-4))
    param_groups = model.get_param_groups(
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    planned_epochs = int(run_spec["epochs"])
    total_epochs = (start_epoch - 1 + int(extra_epochs)) if (resume_payload is not None and extra_epochs) else planned_epochs
    if total_epochs < start_epoch:
        raise ValueError(f"total_epochs ({total_epochs}) nhỏ hơn start_epoch ({start_epoch}).")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - start_epoch + 1, 1),
    )

    opt_restored = False
    if resume_payload is not None and "optimizer_state_dict" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        opt_restored = True
    if resume_payload is not None:
        resumed_meta = {
            "resume_from": str(resume_from),
            "resumed_after_epoch": start_epoch - 1,
            "optimizer_state_restored": opt_restored,
        }

    counts = count_parameters(model)
    tb_writer, tb_active = _create_tb_writer(
        log_dir=(Path(tb_logs_dir) / run_id) if tb_logs_dir is not None else None,
        enabled=enable_tensorboard,
    )

    config_snapshot = {
        "run_id": run_id,
        "group": run_spec.get("group", ""),
        "axis": run_spec.get("axis", ""),
        "arch": str(run_spec["arch"]),
        "mode": str(run_spec["mode"]),
        "pretrained": bool(run_spec.get("pretrained", True)),
        "optimizer": str(run_spec.get("optimizer", "adamw")),
        "lr": lr_head,
        "lr_backbone": lr_backbone,
        "weight_decay": weight_decay,
        "batch_size": int(run_spec["batch_size"]),
        "epochs": total_epochs,
        "is_baseline": bool(run_spec.get("is_baseline", False)),
        "seed": int(seed),
        "device": str(device_obj),
        "use_amp": amp_active,
        "tensorboard_active": tb_active,
        "norm_stats": bundle.norm_stats.to_dict(),
        "param_counts": counts,
        "split_sizes": bundle.split_sizes,
        "resumed": resumed_meta,
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active) if device_obj.type == "cuda" else None

    t_start = time.perf_counter()
    epoch_range = range(start_epoch, total_epochs + 1)
    epoch_bar = (
        tqdm(epoch_range, desc=f"[{run_id}]", leave=True, dynamic_ncols=True)
        if show_progress
        else epoch_range
    )

    try:
        for ep in epoch_bar:
            if device_obj.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device_obj)
            ep_t0 = time.perf_counter()

            tr_stats = train_one_epoch(
                model=model,
                train_loader=bundle.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device_obj,
                scaler=scaler,
                use_amp=amp_active,
                show_batch_bar=show_batch_bar,
                epoch_label=f"{run_id} ep{ep}/{total_epochs}",
            )
            va_stats = validate_one_epoch(
                model=model,
                val_loader=bundle.val_loader,
                criterion=criterion,
                device=device_obj,
                use_amp=amp_active,
            )
            current_lr = float(optimizer.param_groups[-1]["lr"])
            scheduler.step()

            ep_sec = time.perf_counter() - ep_t0
            throughput_sps = float(bundle.split_sizes["train"] / max(ep_sec, 1e-6))
            gpu_mem_mb = (
                float(torch.cuda.max_memory_allocated(device_obj) / (1024.0 * 1024.0))
                if device_obj.type == "cuda"
                else 0.0
            )

            is_best = va_stats["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = va_stats["val_loss"]
                best_val_acc = va_stats["val_acc"]
                best_epoch = ep

            record = {
                "epoch": ep,
                "train_loss": tr_stats["train_loss"],
                "train_acc": tr_stats["train_acc"],
                "val_loss": va_stats["val_loss"],
                "val_acc": va_stats["val_acc"],
                "lr": current_lr,
                "throughput_sps": throughput_sps,
                "gpu_mem_mb": gpu_mem_mb,
                "epoch_sec": ep_sec,
                "is_best": is_best,
            }
            epoch_records.append(record)

            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train", tr_stats["train_loss"], ep)
                tb_writer.add_scalar("Loss/val", va_stats["val_loss"], ep)
                tb_writer.add_scalar("Accuracy/train", tr_stats["train_acc"], ep)
                tb_writer.add_scalar("Accuracy/val", va_stats["val_acc"], ep)
                tb_writer.add_scalar("LR/head", current_lr, ep)
                if gpu_mem_mb > 0:
                    tb_writer.add_scalar("System/gpu_mem_mb", gpu_mem_mb, ep)

            ckpt_payload = {
                "run_id": run_id,
                "epoch": ep,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "model_config": model.model_config(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "norm_stats": bundle.norm_stats.to_dict(),
                "class_names": list(bundle.class_names),
                "species_by_class": dict(bundle.species_by_class),
                "run_spec": config_snapshot,
            }
            if is_best:
                atomic_torch_save(ckpt_payload, run_dir / "best.pt")
            atomic_torch_save(ckpt_payload, run_dir / "last.pt")

            if show_progress:
                postfix = {
                    "tr_loss": f"{tr_stats['train_loss']:.4f}",
                    "val_loss": f"{va_stats['val_loss']:.4f}",
                    "tr_acc": f"{tr_stats['train_acc']*100:.2f}%",
                    "val_acc": f"{va_stats['val_acc']*100:.2f}%",
                    "lr": f"{current_lr:.1e}",
                    "sps": f"{throughput_sps:.0f}",
                }
                if device_obj.type == "cuda":
                    postfix["vram"] = f"{gpu_mem_mb:.0f}MB"
                epoch_bar.set_postfix(**postfix)  # type: ignore[union-attr]
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    train_time_sec = prev_train_time_sec + (time.perf_counter() - t_start)
    history = {
        "run_id": run_id,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "train_time_sec": train_time_sec,
        "resumed": resumed_meta,
        "epochs": epoch_records,
    }
    history_dir = run_dir / "historys"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    if epoch_records:
        with (history_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_records[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_records)

    eval_result: Optional[Dict[str, Any]] = None
    if eval_after_train:
        eval_result = evaluate_run_dir(
            run_dir=run_dir,
            eval_loader=bundle.val_loader,
            device=device_obj,
            class_names=bundle.class_names,
            species_by_class=bundle.species_by_class,
            split_name="val",
            checkpoint_name="best.pt",
            save_plots=save_eval_plots,
            tolerance=tolerance,
        )

    peak_vram = max((float(r["gpu_mem_mb"]) for r in epoch_records), default=0.0)
    return {
        "run_id": run_id,
        "group": run_spec.get("group", ""),
        "axis": run_spec.get("axis", ""),
        "is_baseline": bool(run_spec.get("is_baseline", False)),
        "arch": str(run_spec["arch"]),
        "mode": str(run_spec["mode"]),
        "pretrained": bool(run_spec.get("pretrained", True)),
        "lr": lr_head,
        "lr_backbone": lr_backbone,
        "batch_size": int(run_spec["batch_size"]),
        "epochs": total_epochs,
        "total_params": counts["total"],
        "trainable_params": counts["trainable"],
        "trainable_pct": counts["trainable_pct"],
        "best_epoch": best_epoch,
        "train_time_sec": float(train_time_sec),
        "peak_vram_mb": round(peak_vram, 1),
        "val_loss": float(eval_result["loss"]) if eval_result else float(best_val_loss),
        "val_acc": float(eval_result["accuracy"]) if eval_result else float(best_val_acc),
        "val_macro_f1": float(eval_result["macro_f1"]) if eval_result else 0.0,
        "val_species_acc": float(eval_result["species_metrics"]["species_accuracy"]) if eval_result else 0.0,
        "save_load_passed": bool(eval_result["save_load_parity"]["passed"]) if eval_result else True,
        "tensorboard_active": tb_active,
        "run_dir": str(run_dir),
        "eval_dir": eval_result["eval_dir"] if eval_result else "",
    }


def _persist_ablation_summary(
    ablation_dir: Path,
    completed_runs: List[Dict[str, Any]],
    total_planned: int,
) -> Dict[str, Any]:
    """Lưu bảng tổng hợp chuỗi thí nghiệm và chọn winner theo `(val_acc, val_macro_f1, -val_loss)`."""
    ablation_dir.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in completed_runs if not r.get("is_baseline", False)]
    best_model = (
        max(candidates, key=lambda r: (r["val_acc"], r["val_macro_f1"], -r["val_loss"]))
        if candidates
        else None
    )
    baselines = [r for r in completed_runs if r.get("is_baseline", False)]

    summary = {
        "selection_rule": "max(val_acc, val_macro_f1, -val_loss) on Validation split",
        "total_planned_runs": int(total_planned),
        "completed_runs": len(completed_runs),
        "baseline_run_id": baselines[0]["run_id"] if baselines else None,
        "winning_model_run_id": best_model["run_id"] if best_model else None,
        "runs": completed_runs,
    }
    (ablation_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_fields = [
        "run_id",
        "group",
        "axis",
        "arch",
        "mode",
        "lr",
        "lr_backbone",
        "batch_size",
        "epochs",
        "total_params",
        "trainable_params",
        "trainable_pct",
        "best_epoch",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_species_acc",
        "train_time_sec",
        "peak_vram_mb",
        "save_load_passed",
    ]
    with (ablation_dir / "ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed_runs)
    return summary


def run_experiment_suite(
    experiment_specs: Sequence[Dict[str, Any]],
    raw_data_root: Path,
    runs_dir: Path,
    ablation_dir: Path,
    split_file: Path,
    device: Union[str, torch.device],
    val_ratio_of_test: float = 0.35,
    seed: int = 248,
    image_size: int = 224,
    resize_size: int = 256,
    num_workers: int = 0,
    use_amp: bool = True,
    tb_logs_dir: Optional[Path] = None,
    enable_tensorboard: bool = True,
    show_progress: bool = True,
    show_batch_bar: bool = False,
    save_eval_plots: bool = True,
    tolerance: float = 1e-6,
    bundle_cache: Optional[Dict[int, OxfordPetsBundle]] = None,
) -> Dict[str, Any]:
    """Điều phối chạy trọn chuỗi thí nghiệm có kiểm soát (Train → Val Eval ngay tại chỗ → cập nhật bảng)."""
    cache: Dict[int, OxfordPetsBundle] = bundle_cache if bundle_cache is not None else {}
    completed_runs: List[Dict[str, Any]] = []

    outer = (
        tqdm(experiment_specs, desc="Lab2 v2 Suite", leave=True, dynamic_ncols=True)
        if show_progress
        else experiment_specs
    )
    for spec in outer:
        bs = int(spec["batch_size"])
        if bs not in cache:
            cache[bs] = prepare_oxford_pets(
                raw_data_root=raw_data_root,
                batch_size=bs,
                val_ratio_of_test=val_ratio_of_test,
                seed=seed,
                image_size=image_size,
                resize_size=resize_size,
                num_workers=num_workers,
                split_file=split_file,
                download=False,
            )
        bundle = cache[bs]
        result = train_single_run(
            run_spec=spec,
            bundle=bundle,
            runs_dir=runs_dir,
            device=device,
            seed=seed,
            use_amp=use_amp,
            tb_logs_dir=tb_logs_dir,
            enable_tensorboard=enable_tensorboard,
            show_progress=show_progress,
            show_batch_bar=show_batch_bar,
            eval_after_train=True,
            save_eval_plots=save_eval_plots,
            tolerance=tolerance,
        )
        completed_runs.append(result)
        summary = _persist_ablation_summary(
            ablation_dir=ablation_dir,
            completed_runs=completed_runs,
            total_planned=len(experiment_specs),
        )
        if show_progress:
            tqdm.write(
                f"  -> [{result['run_id']}] val_acc={result['val_acc']*100:.2f}% | "
                f"val_f1={result['val_macro_f1']*100:.2f}% | val_loss={result['val_loss']:.4f} | "
                f"best_ep={result['best_epoch']} | time={result['train_time_sec']:.1f}s | "
                f"vram={result['peak_vram_mb']:.0f}MB"
            )
    return summary
