"""Stage 5 — final holdout evaluation (only after validation-based model choice).

No training, backward pass, optimizer or hyperparameter selection here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import as a package when called from orchestration scripts, but also allow
# users to click VS Code's "Run Python File" on src/evaluate.py.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_loader import LABEL_NAMES
    from src.train import load_model_checkpoint
else:
    from .data_loader import LABEL_NAMES
    from .train import load_model_checkpoint


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_selected_checkpoint(
    checkpoint_path: Path | str,
    test_loader: DataLoader,
    train_normalization: dict[str, Any],
    out_dir: Path | str,
    device: torch.device,
    selected_experiment: str,
) -> dict[str, Any]:
    """One final evaluation on the untouched test split, AFTER choosing by validation.

    Check normalization metadata to guard against silent preprocessing mismatch.
    Returns metrics and writes reproducible evidence to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model_checkpoint(checkpoint_path, device)
    ckpt_norm = ckpt.get('normalization', {})
    for k in ('train_mean_0_to_1', 'train_std_0_to_1'):
        if abs(float(ckpt_norm[k]) - float(train_normalization[k])) > 1e-9:
            raise ValueError(f'Normalization mismatch: {k} differs from checkpoint.')
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    confusion = np.zeros((10, 10), dtype=np.int64)
    rows: list[dict] = []
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            targets = labels.to(device)
            total_loss += float(criterion(logits, targets).item())
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            gt = targets.cpu().numpy()
            pred = predicted.cpu().numpy()
            conf = confidence.cpu().numpy()
            np.add.at(confusion, (gt, pred), 1)
            total_correct += int((pred == gt).sum())
            for y, yhat, p in zip(gt, pred, conf):
                rows.append({'index_in_test': total, 'actual': int(y), 'predicted': int(yhat),
                             'confidence': round(float(p), 7), 'correct': int(y == yhat)})
                total += 1

    if total != len(test_loader.dataset):
        raise AssertionError('Test sample count mismatch.')
    if int(confusion.sum()) != total or total_correct != int(np.trace(confusion)):
        raise AssertionError('Confusion matrix differs from predictions.')
    class_report = []
    for i in range(10):
        tp = int(confusion[i, i])
        count_actual = int(confusion[i, :].sum())
        count_pred = int(confusion[:, i].sum())
        precision = tp / count_pred if count_pred else 0.0
        recall = tp / count_actual if count_actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_report.append({'class_id': i, 'class_name': LABEL_NAMES[i],
                             'support': count_actual, 'precision': round(precision, 7),
                             'recall': round(recall, 7), 'f1': round(f1, 7)})
    _save_csv(out_dir / 'test_predictions.csv', ['index_in_test', 'actual', 'predicted', 'confidence', 'correct'], rows)
    _save_csv(out_dir / 'per_class_metrics.csv', ['class_id', 'class_name', 'support', 'precision', 'recall', 'f1'], class_report)
    cm_rows = [{'actual_class': LABEL_NAMES[i], **{f'pred_{j}': int(confusion[i, j]) for j in range(10)}}
               for i in range(10)]
    _save_csv(out_dir / 'confusion_matrix.csv', ['actual_class', *[f'pred_{j}' for j in range(10)]], cm_rows)

    fig, ax = plt.subplots(figsize=(10.6, 8.8))
    im = ax.imshow(confusion, cmap='Blues', interpolation='nearest')
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xticklabels=LABEL_NAMES,
           yticklabels=LABEL_NAMES, xlabel='Predicted label', ylabel='Actual label',
           title=f'Final TEST confusion matrix — {selected_experiment} (n={total:,})')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    threshold = confusion.max() * 0.55
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(confusion[i, j]), ha='center', va='center', fontsize=7.7,
                    color='white' if confusion[i, j] > threshold else '#18324a')
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    figures = out_dir / 'figures'
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / 'test_confusion_matrix.png', dpi=165)
    plt.close(fig)
    result = {
        'stage': 5, 'selected_experiment': selected_experiment, 'checkpoint': str(Path(checkpoint_path)),
        'selection_basis': 'minimum validation loss across experiments; TEST NOT used for selection',
        'test_evaluated_once_after_selection': True,
        'test_samples': total, 'test_correct': total_correct,
        'test_accuracy': total_correct / total,
        'test_accuracy_percent': 100 * total_correct / total,
        'test_cross_entropy_loss': total_loss / total,
        'macro_f1': float(np.mean([x['f1'] for x in class_report])),
        'files': ['test_predictions.csv', 'per_class_metrics.csv', 'confusion_matrix.csv',
                  'figures/test_confusion_matrix.png'],
    }
    (out_dir / 'test_metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def main() -> None:
    """Standalone checkpoint evaluation; never trains or selects model by test."""
    import argparse
    import os

    from src.data_loader import create_dataloaders

    # Ensure VS Code Run Python File works even with a higher-level workspace.
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(description="Practice 1 — standalone checkpoint evaluation")
    parser.add_argument("--csv", default="data/fashion_mnist_full.csv")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="Default validation. Use test only after model selection is finished.")
    parser.add_argument("--out-dir", default="outputs/standalone_evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--selected-experiment", default="E2",
                        help="Identifier used only for the final test report")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size phải >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Chưa có checkpoint: {checkpoint_path.resolve()}. Chạy src/train.py trước.")

    print(f"[EVALUATION] split={args.split}, checkpoint={checkpoint_path}", flush=True)
    loaders, audit = create_dataloaders(args.csv, batch_size=args.batch_size)
    normalization = audit["normalization"]
    model, ckpt = load_model_checkpoint(checkpoint_path, device)
    for k in ("train_mean_0_to_1", "train_std_0_to_1"):
        if abs(float(ckpt["normalization"][k]) - float(normalization[k])) > 1e-9:
            raise ValueError(f"Normalization mismatch for {k}: checkpoint and dataset differ")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "validation":
        from src.train import _one_epoch
        stats = _one_epoch(model, loaders["validation"], nn.CrossEntropyLoss(), device)
        report = {
            "split": "validation",
            "checkpoint": str(checkpoint_path),
            "samples": stats["samples"],
            "validation_loss": stats["loss"],
            "validation_accuracy": stats["accuracy"],
            "validation_accuracy_percent": 100 * stats["accuracy"],
            "note": "Validation only; does not update weights or use the holdout test set.",
        }
        file_path = out_dir / "validation_metrics.json"
        file_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] VALIDATION: {stats['samples']:,} images | "
              f"loss={stats['loss']:.4f} | accuracy={100*stats['accuracy']:.2f}%", flush=True)
        print(f"Saved: {file_path.resolve()}", flush=True)
    else:
        print("[IMPORTANT] Final TEST only. Do not change hyperparameters using these results.", flush=True)
        report = evaluate_selected_checkpoint(
            checkpoint_path, loaders["test"], normalization, out_dir, device,
            selected_experiment=args.selected_experiment,
        )
        print(f"[OK] FINAL TEST: {report['test_correct']:,}/{report['test_samples']:,} | "
              f"accuracy={report['test_accuracy_percent']:.2f}% | "
              f"loss={report['test_cross_entropy_loss']:.4f} | "
              f"macro_F1={report['macro_f1']:.4f}", flush=True)
        print(f"Saved: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
