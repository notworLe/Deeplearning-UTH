"""Practice 1 / Stage 4 — true PyTorch training and validation (NO TEST).

Imports FashionMLP from Stage 3. Receives DataLoaders from Stage 2.
The training loop updates weights using TRAIN only; validation never computes
or applies gradients. The best epoch is selected by validation loss only.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

# Support both `python -m src.train` and VS Code's "Run Python File".
# The latter invokes this file directly, without a Python package context.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model import FashionMLP, count_trainable_parameters
else:
    from .model import FashionMLP, count_trainable_parameters

from config import TrainConfig


def _one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass. Passing an optimizer permits training; None = validation.

    Loss is example-weighted rather than averaging batch means. This also
    handles the smaller final batch correctly (53,981 is not divisible by 64).
    """
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss, num_correct, num_seen = 0.0, 0, 0

    # Validation must have no graph, no backward, no optimizer step.
    with torch.enable_grad() if is_training else torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite loss; stop training for inspection.")
            if is_training:
                loss.backward()
                optimizer.step()

            n = targets.size(0)
            total_loss += loss.detach().item() * n
            num_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            num_seen += n

    if num_seen == 0:
        raise ValueError("DataLoader trống — không thể tính metric.")
    return {
        "loss": total_loss / num_seen,
        "accuracy": num_correct / num_seen,  # fraction [0,1]
        "samples": num_seen,
    }


def _write_history(history: list[dict[str, Any]], path: Path) -> None:
    columns = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "best_epoch_so_far"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, Any]], figures_dir: Path) -> None:
    """Stage 4 evidence; more extensive visualizations remain in Stage 6."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    for value, title, filename, ylabel, factor in (
        ("loss", "Training and validation cross-entropy", "loss_curve.png", "Cross-Entropy Loss", 1.0),
        ("accuracy", "Training and validation accuracy", "accuracy_curve.png", "Accuracy (%)", 100.0),
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.plot(epochs, [row[f"train_{value}"] * factor for row in history], marker="o", linewidth=2, label="Train")
        ax.plot(epochs, [row[f"val_{value}"] * factor for row in history], marker="s", linewidth=2, label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.set_xticks(epochs)
        if value == "accuracy":
            ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)


def load_model_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> tuple[FashionMLP, dict]:
    """Restore trained weights and exact architecture. No test data involved."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = FashionMLP(hidden_dims=tuple(checkpoint["hidden_dims"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def fit(
    model: FashionMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    normalization: dict[str, Any],
) -> dict[str, Any]:
    """Train/validate; write checkpoint/metrics/curves; DO NOT access test_loader."""
    output_dir = config.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    metrics_path = output_dir / "metrics.csv"
    loss = nn.CrossEntropyLoss()  # raw logits, no Softmax in model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model = model.to(device)

    best_loss = float("inf")
    best_epoch = 0
    best_accuracy = 0.0
    history: list[dict[str, Any]] = []
    # Save outputs from the best model on a fixed validation batch for reload parity.
    fixed_images, _ = next(iter(val_loader))
    fixed_images = fixed_images.to(device)
    reference_logits = None

    print(f"Device: {device} | epochs={config.epochs} | lr={config.learning_rate} | "
          f"batch_size={config.batch_size}", flush=True)
    print("Selection rule: minimum validation loss. Test set is NOT used.", flush=True)
    for epoch in range(1, config.epochs + 1):
        train = _one_epoch(model, train_loader, loss, device, optimizer)
        val = _one_epoch(model, val_loader, loss, device, None)
        # Select solely by validation loss; test data never accessed.
        improved = val["loss"] < best_loss
        if improved:
            best_loss = val["loss"]
            best_accuracy = val["accuracy"]
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "hidden_dims": list(model.hidden_dims),
                "num_classes": model.num_classes,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "best_val_accuracy": best_accuracy,
                "normalization": normalization,
                "train_config": {
                    **asdict(config),
                    "csv_path": str(config.csv_path),
                    "out_dir": str(config.out_dir),
                },
                "selection_metric": "validation_loss",
                "test_set_used": False,
            }
            tmp = checkpoint_path.with_suffix(".pth.tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(checkpoint_path)
            model.eval()
            with torch.inference_mode():
                reference_logits = model(fixed_images).detach().cpu().clone()
        row = {
            "epoch": epoch,
            "train_loss": round(train["loss"], 8),
            "train_accuracy": round(train["accuracy"], 8),
            "val_loss": round(val["loss"], 8),
            "val_accuracy": round(val["accuracy"], 8),
            "best_epoch_so_far": best_epoch,
        }
        history.append(row)
        _write_history(history, metrics_path)
        print(f"Epoch {epoch:02d}/{config.epochs:02d} | "
              f"train loss={train['loss']:.4f} acc={train['accuracy']*100:.2f}% | "
              f"val loss={val['loss']:.4f} acc={val['accuracy']*100:.2f}%" +
              ("  [BEST]" if improved else ""), flush=True)

    restored, ckpt = load_model_checkpoint(checkpoint_path, device)
    with torch.inference_mode():
        restored_logits = restored(fixed_images).cpu()
    if reference_logits is None or not torch.allclose(reference_logits, restored_logits, atol=1e-6, rtol=1e-5):
        raise AssertionError("Restored model logits differ from saved best-epoch model.")

    plot_training_curves(history, output_dir / "figures")
    summary = {
        "stage": 4,
        "training_completed": True,
        "epochs_completed": config.epochs,
        "seed": config.seed,
        "device": str(device),
        "hidden_dims": list(config.hidden_dims),
        "parameter_count": count_trainable_parameters(restored),
        "dataset_sizes_used": {"train": len(train_loader.dataset), "validation": len(val_loader.dataset)},
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation_accuracy": best_accuracy,
        "selection_rule": "minimum validation loss",
        "checkpoint_reload_verified": True,
        "test_used_for_gradient_updates_or_model_selection": False,
        "test_split_read_only_for_stage2_integrity_audit": True,
        "test_metrics_computed": False,
        "final_test_accuracy": None,
        "files": ["best_model.pth", "metrics.csv", "figures/loss_curve.png", "figures/accuracy_curve.png"],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Stage 4 completed: best_epoch={best_epoch}, "
          f"best_val_loss={best_loss:.4f}, best_val_acc={best_accuracy*100:.2f}%", flush=True)
    print("Checkpoint restore: VERIFIED | no test-set evaluation", flush=True)
    print("Outputs:", output_dir.resolve(), flush=True)
    return summary


def main() -> None:
    """Run training independently when src/train.py is opened in VS Code."""
    import os
    # Make relative data/output paths independent of VS Code's workspace folder.
    os.chdir(Path(__file__).resolve().parents[1])
    from main import main as train_main
    train_main()


if __name__ == "__main__":
    main()
