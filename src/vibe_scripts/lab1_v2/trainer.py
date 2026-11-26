"""Vòng lặp huấn luyện có `tqdm` progress bar, per-run evaluation tại chỗ và dừng sớm chuỗi thí nghiệm."""

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.vibe_scripts.lab1_v2.data import FashionMNISTBundle
from src.vibe_scripts.lab1_v2.evaluator import evaluate_run_dir
from src.vibe_scripts.lab1_v2.logistic import count_parameters
from src.vibe_scripts.lab1_v2.mlp import build_model


def set_global_seed(seed: int = 248) -> None:
    """Khóa toàn bộ nguồn ngẫu nhiên (Python, NumPy, PyTorch, CUDA) để tái lập 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Khởi tạo optimizer chuẩn theo cấu hình (`adam`, `adamw`, `sgd`)."""
    name = optimizer_name.strip().lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Không hỗ trợ optimizer='{optimizer_name}'. Chỉ chấp nhận: ['adam', 'adamw', 'sgd'].")


def atomic_torch_save(payload: Dict[str, Any], target_path: Path) -> None:
    """Lưu checkpoint nguyên tử qua file tạm `.tmp` rồi `os.replace` để tránh hỏng file."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target)


def run_sanity_checks(
    model: nn.Module,
    sample_batch: torch.Tensor,
    sample_targets: torch.Tensor,
    device: Union[str, torch.device],
    lr: float = 1e-2,
    max_steps: int = 200,
    target_acc: float = 0.95,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Kiểm tra Sanity Check trước khi train: Initial Loss ≈ ln(C) và Overfit 1 batch nhỏ."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    x = sample_batch.to(device_obj)
    y = sample_targets.to(device_obj)
    criterion = nn.CrossEntropyLoss()

    # 1. Kiểm tra initial loss ở chế độ eval
    model.eval()
    with torch.inference_mode():
        init_logits = model(x)
        initial_loss = float(criterion(init_logits, y).item())
    expected_loss = float(math.log(num_classes))

    # 2. Thử overfit 1 batch nhỏ (tạm tắt dropout nếu cần bằng cấu hình hoặc giữ model train)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    final_acc = 0.0
    final_loss = initial_loss
    steps_used = 0
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Sanity check gặp loss không hữu hạn tại step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()

        steps_used = step
        final_loss = float(loss.item())
        final_acc = float((logits.argmax(dim=1) == y).to(dtype=torch.float32).mean().item())
        if final_acc >= target_acc:
            break

    return {
        "initial_loss": initial_loss,
        "expected_random_loss": expected_loss,
        "initial_loss_ok": abs(initial_loss - expected_loss) <= 0.6,
        "overfit_final_loss": final_loss,
        "overfit_final_acc": final_acc,
        "overfit_steps": steps_used,
        "overfit_passed": final_acc >= target_acc,
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    show_batch_bar: bool = False,
) -> Dict[str, float]:
    """Huấn luyện 1 epoch với kiểm tra NaN/Inf Fail-Fast và optional `tqdm` batch bar."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    iterator = (
        tqdm(
            train_loader,
            desc=f"  Ep {epoch_idx}/{total_epochs} [train]",
            leave=False,
            dynamic_ncols=True,
        )
        if show_batch_bar
        else train_loader
    )

    for batch_x, batch_y in iterator:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Phát hiện loss NaN/Inf tại epoch {epoch_idx}: {loss.item()}")
        loss.backward()
        optimizer.step()

        bs = int(batch_y.shape[0])
        running_loss += float(loss.detach().item()) * bs
        running_correct += int((logits.detach().argmax(dim=1) == batch_y).sum().item())
        total_samples += bs

    return {
        "train_loss": running_loss / max(1, total_samples),
        "train_acc": running_correct / max(1, total_samples),
    }


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Đánh giá nhanh trên Validation cuối mỗi epoch dưới `@torch.inference_mode()`."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        bs = int(batch_y.shape[0])
        running_loss += float(loss.item()) * bs
        running_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total_samples += bs

    return {
        "val_loss": running_loss / max(1, total_samples),
        "val_acc": running_correct / max(1, total_samples),
    }


def train_single_run(
    run_spec: Dict[str, Any],
    bundle: FashionMNISTBundle,
    runs_dir: Path,
    device: Union[str, torch.device],
    seed: int = 248,
    input_shape: tuple = (1, 28, 28),
    num_classes: int = 10,
    show_progress: bool = True,
    show_batch_bar: bool = False,
    eval_after_train: bool = True,
    save_eval_plots: bool = True,
    tolerance: float = 1e-6,
    resume_from: Optional[Path] = None,
    extra_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train 1 run hoàn chỉnh (hỗ trợ resume từ checkpoint) với `tqdm` bar log và evaluate ngay vào `runs/<run_id>/eval/`."""
    set_global_seed(seed)
    device_obj = torch.device(device)

    run_id = str(run_spec["run_id"])
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        model_type=run_spec["model_type"],
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dims=tuple(run_spec.get("hidden_dims", ())),
        dropout=float(run_spec.get("dropout", 0.0)),
    ).to(device_obj)

    optimizer = build_optimizer(
        model=model,
        optimizer_name=run_spec["optimizer"],
        lr=float(run_spec["lr"]),
        momentum=float(run_spec.get("momentum", 0.0)),
        weight_decay=float(run_spec.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    prev_train_time_sec = 0.0
    epoch_records: List[Dict[str, Any]] = []
    resumed_meta: Optional[Dict[str, Any]] = None

    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint resume tại {ckpt_path}")
        prev_payload = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
        model.load_state_dict(prev_payload["model_state_dict"])
        optimizer_restored = False
        if "optimizer_state_dict" in prev_payload:
            optimizer.load_state_dict(prev_payload["optimizer_state_dict"])
            optimizer_restored = True

        prev_epoch = int(prev_payload["epoch"])
        start_epoch = prev_epoch + 1
        best_val_loss = float(prev_payload.get("best_val_loss", float("inf")))
        best_val_acc = float(prev_payload.get("best_val_acc", 0.0))
        best_epoch = int(prev_payload.get("best_epoch", prev_epoch))

        # Bảo toàn best.pt trước đó vào run_dir để phòng trường hợp các epoch resume không vượt được best cũ
        prev_best_path = ckpt_path.parent / "best.pt"
        if prev_best_path.exists() and prev_best_path.resolve() != (run_dir / "best.pt").resolve():
            prev_best_payload = torch.load(prev_best_path, map_location="cpu", weights_only=False)
            prev_best_payload["run_id"] = run_id
            atomic_torch_save(prev_best_payload, run_dir / "best.pt")
        elif not (run_dir / "best.pt").exists():
            prev_copy = dict(prev_payload)
            prev_copy["run_id"] = run_id
            atomic_torch_save(prev_copy, run_dir / "best.pt")

        # Nối lịch sử các epoch trước đó để biểu đồ loss/acc liên tục từ epoch 1 -> total_epochs
        prev_hist_path = ckpt_path.parent / "historys" / "history.json"
        if not prev_hist_path.exists():
            prev_hist_path = ckpt_path.parent / "history.json"
        if prev_hist_path.exists():
            prev_hist = json.loads(prev_hist_path.read_text(encoding="utf-8"))
            epoch_records = [dict(row) for row in prev_hist.get("epochs", []) if int(row["epoch"]) <= prev_epoch]
            prev_train_time_sec = float(prev_hist.get("train_time_sec", 0.0))

        # Dịch seed nhẹ cho các epoch nối tiếp để thứ tự shuffle không lặp hệt epoch 1..N cũ
        set_global_seed(seed + prev_epoch)
        resumed_meta = {
            "resume_from": str(ckpt_path),
            "resumed_after_epoch": prev_epoch,
            "optimizer_state_restored": optimizer_restored,
        }

    if extra_epochs is not None:
        if extra_epochs < 1:
            raise ValueError(f"extra_epochs phải >= 1, nhận được {extra_epochs}")
        epochs = (start_epoch - 1) + int(extra_epochs)
    else:
        epochs = int(run_spec["epochs"])

    if epochs < start_epoch:
        raise ValueError(
            f"Tổng số epochs ({epochs}) phải >= start_epoch ({start_epoch}) khi resume."
        )

    # Lưu cấu hình run trước khi train để bảo đảm traceability
    config_snapshot = {
        **run_spec,
        "epochs": epochs,
        "hidden_dims": list(run_spec.get("hidden_dims", ())),
        "seed": int(seed),
        "device": str(device_obj),
        "norm_stats": bundle.norm_stats.to_dict(),
        "num_parameters": count_parameters(model),
        "resumed": resumed_meta,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    t_start = time.perf_counter()

    epoch_bar = tqdm(
        range(start_epoch, epochs + 1),
        desc=f"[{run_id}]",
        disable=not show_progress,
        dynamic_ncols=True,
        leave=True,
    )

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
            epoch_idx=ep,
            total_epochs=epochs,
            show_batch_bar=show_batch_bar and show_progress,
        )
        va_stats = validate_one_epoch(
            model=model,
            val_loader=bundle.val_loader,
            criterion=criterion,
            device=device_obj,
        )
        ep_sec = time.perf_counter() - ep_t0
        current_lr = float(optimizer.param_groups[0]["lr"])
        throughput_sps = float(len(bundle.train_dataset) / max(ep_sec, 1e-9))
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
            epoch_bar.set_postfix(**postfix)

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
    (history_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
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
            split_name="val",
            checkpoint_name="best.pt",
            save_plots=save_eval_plots,
            tolerance=tolerance,
        )

    return {
        "run_id": run_id,
        "group": run_spec.get("group", ""),
        "axis": run_spec.get("axis", ""),
        "is_baseline": bool(run_spec.get("is_baseline", False)),
        "model_type": run_spec["model_type"],
        "hidden_dims": list(run_spec.get("hidden_dims", ())),
        "dropout": float(run_spec.get("dropout", 0.0)),
        "optimizer": run_spec["optimizer"],
        "lr": float(run_spec["lr"]),
        "batch_size": int(run_spec["batch_size"]),
        "epochs": epochs,
        "num_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "train_time_sec": float(train_time_sec),
        "val_loss": float(eval_result["loss"]) if eval_result else float(best_val_loss),
        "val_acc": float(eval_result["accuracy"]) if eval_result else float(best_val_acc),
        "val_macro_f1": float(eval_result["macro_f1"]) if eval_result else 0.0,
        "save_load_passed": bool(eval_result["save_load_parity"]["passed"]) if eval_result else True,
        "run_dir": str(run_dir),
        "eval_dir": eval_result["eval_dir"] if eval_result else "",
    }


def _persist_ablation_summary(
    ablation_dir: Path,
    completed_runs: List[Dict[str, Any]],
    total_planned: int,
    stopped_early: bool,
    stop_reason: str,
    target_val_acc: float,
    target_val_f1: float,
) -> Dict[str, Any]:
    ablation_dir.mkdir(parents=True, exist_ok=True)
    mlp_runs = [r for r in completed_runs if not r.get("is_baseline", False)]
    best_mlp = (
        max(mlp_runs, key=lambda r: (r["val_macro_f1"], r["val_acc"], -r["val_loss"]))
        if mlp_runs
        else None
    )
    baseline_runs = [r for r in completed_runs if r.get("is_baseline", False)]

    summary = {
        "total_planned_runs": int(total_planned),
        "completed_runs": len(completed_runs),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "early_stop_target": {
            "val_acc": float(target_val_acc),
            "val_macro_f1": float(target_val_f1),
        },
        "baseline_run_id": baseline_runs[0]["run_id"] if baseline_runs else None,
        "winning_mlp_run_id": best_mlp["run_id"] if best_mlp else None,
        "runs": completed_runs,
    }

    json_path = ablation_dir / "ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_fields = [
        "run_id",
        "group",
        "axis",
        "model_type",
        "hidden_dims",
        "optimizer",
        "lr",
        "batch_size",
        "epochs",
        "num_parameters",
        "best_epoch",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "train_time_sec",
        "save_load_passed",
        "eval_dir",
    ]
    csv_path = ablation_dir / "ablation_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed_runs)

    return summary


def run_experiment_suite(
    run_specs: Sequence[Dict[str, Any]],
    bundle_factory: Callable[[int], FashionMNISTBundle],
    runs_dir: Path,
    ablation_dir: Path,
    device: Union[str, torch.device],
    seed: int = 248,
    input_shape: tuple = (1, 28, 28),
    num_classes: int = 10,
    stop_on_target: bool = True,
    target_val_acc: float = 0.89,
    target_val_f1: float = 0.89,
    show_progress: bool = True,
    show_batch_bar: bool = False,
    save_eval_plots: bool = True,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Điều phối tuần tự 12 runs: Train -> Eval ngay tại `runs/<id>/eval/` -> Dừng sớm nếu đạt target."""
    runs_path = Path(runs_dir)
    ablation_path = Path(ablation_dir)
    completed: List[Dict[str, Any]] = []
    stopped_early = False
    stop_reason = "completed_all_planned_runs"

    suite_bar = tqdm(
        run_specs,
        desc="Experiment Suite",
        total=len(run_specs),
        disable=not show_progress,
        dynamic_ncols=True,
    )

    for spec in suite_bar:
        run_id = str(spec["run_id"])
        if show_progress:
            suite_bar.set_description(f"Suite ({run_id})")

        bundle = bundle_factory(int(spec["batch_size"]))
        result = train_single_run(
            run_spec=spec,
            bundle=bundle,
            runs_dir=runs_path,
            device=device,
            seed=seed,
            input_shape=input_shape,
            num_classes=num_classes,
            show_progress=show_progress,
            show_batch_bar=show_batch_bar,
            eval_after_train=True,
            save_eval_plots=save_eval_plots,
            tolerance=tolerance,
        )
        completed.append(result)

        if show_progress:
            tqdm.write(
                f"  -> [EVAL SAVED] {run_id}: "
                f"val_loss={result['val_loss']:.4f} | "
                f"val_acc={result['val_acc']*100:.2f}% | "
                f"val_f1={result['val_macro_f1']*100:.2f}% | "
                f"dir={result['eval_dir']}"
            )

        # Kiểm tra điều kiện dừng sớm chuỗi thí nghiệm khi MLP đạt ngưỡng evaluation tốt
        if (
            stop_on_target
            and not result["is_baseline"]
            and result["val_acc"] >= target_val_acc
            and result["val_macro_f1"] >= target_val_f1
        ):
            stopped_early = True
            stop_reason = (
                f"early_stop_target_reached_by_{run_id} "
                f"(val_acc={result['val_acc']:.4f} >= {target_val_acc:.4f}, "
                f"val_f1={result['val_macro_f1']:.4f} >= {target_val_f1:.4f})"
            )
            _persist_ablation_summary(
                ablation_dir=ablation_path,
                completed_runs=completed,
                total_planned=len(run_specs),
                stopped_early=stopped_early,
                stop_reason=stop_reason,
                target_val_acc=target_val_acc,
                target_val_f1=target_val_f1,
            )
            if show_progress:
                tqdm.write(
                    f"[EARLY STOP EXPERIMENT] {run_id} đạt ngưỡng evaluation tốt "
                    f"(val_acc={result['val_acc']*100:.2f}%, val_f1={result['val_macro_f1']*100:.2f}%) "
                    f"-> Kết thúc chuỗi thí nghiệm!"
                )
            break

        _persist_ablation_summary(
            ablation_dir=ablation_path,
            completed_runs=completed,
            total_planned=len(run_specs),
            stopped_early=False,
            stop_reason="in_progress",
            target_val_acc=target_val_acc,
            target_val_f1=target_val_f1,
        )

    return _persist_ablation_summary(
        ablation_dir=ablation_path,
        completed_runs=completed,
        total_planned=len(run_specs),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        target_val_acc=target_val_acc,
        target_val_f1=target_val_f1,
    )
