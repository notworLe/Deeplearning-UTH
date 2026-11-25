"""Practice 1 / Stage 5: compare E1/E2/E3 with VALIDATION; test selected model once.

From project root: python run_experiments.py
Reuses the verified Stage-4 E2 checkpoint when its configuration matches.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from src.data_loader import create_dataloaders
from src.evaluate import evaluate_selected_checkpoint
from src.model import FashionMLP, count_trainable_parameters
from src.train import fit, load_model_checkpoint


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_loader_with_fresh_seed(dataset, batch_size: int, seed: int) -> DataLoader:
    # Fresh seeded shuffle per run avoids order effects caused by reusing an exhausted generator.
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def _reuse_stage4_e2(stage4: Path, destination: Path, seed: int, epochs: int,
                      batch: int, norm: dict) -> bool:
    checkpoint_path = stage4 / 'best_model.pth'
    summary_path = stage4 / 'training_summary.json'
    history_path = stage4 / 'metrics.csv'
    if not all(p.is_file() for p in (checkpoint_path, summary_path, history_path)):
        return False
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = ckpt.get('train_config', {})
    valid = (tuple(ckpt.get('hidden_dims', ())) == (256, 128) and
             float(cfg.get('learning_rate', -1)) == 0.001 and
             int(cfg.get('epochs', -1)) == epochs and
             int(cfg.get('batch_size', -1)) == batch and int(cfg.get('seed', -1)) == seed and
             all(abs(float(ckpt.get('normalization', {}).get(k, 99)) - float(norm[k])) < 1e-9
                 for k in ('train_mean_0_to_1', 'train_std_0_to_1')))
    if not valid:
        print('[INFO] Stage-4 E2 configuration does not match; E2 will retrain.', flush=True)
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for f in ('best_model.pth', 'training_summary.json', 'metrics.csv'):
        shutil.copy2(stage4 / f, destination / f)
    (destination / 'figures').mkdir(exist_ok=True)
    for f in ('loss_curve.png', 'accuracy_curve.png'):
        if (stage4 / 'figures' / f).exists():
            shutil.copy2(stage4 / 'figures' / f, destination / 'figures' / f)
    (destination / 'provenance.txt').write_text(
        'E2 is copied from Stage 4 (same seed, dataset normalization, architecture, lr, epochs, batch).\n'
        'No test metrics were used during its training or checkpoint selection.\n', encoding='utf-8')
    return True


def _plot_comparison(data: list[dict], experiment_root: Path) -> None:
    figures = experiment_root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    labels = [r['experiment'] for r in data]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    loss_vals = [r['best_val_loss'] for r in data]
    acc_vals = [100 * r['best_val_accuracy'] for r in data]
    bars = axes[0].bar(labels, loss_vals, color=['#5b87c4', '#497fa2', '#71a69b'])
    for bar, v in zip(bars, loss_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    axes[0].set(title='Best validation loss (lower is better)', ylabel='Cross-Entropy loss')
    axes[0].set_ylim(0, max(loss_vals) * 1.23)
    bars = axes[1].bar(labels, acc_vals, color=['#5b87c4', '#497fa2', '#71a69b'])
    for bar, v in zip(bars, acc_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)
    axes[1].set(title='Accuracy at best-val-loss checkpoint', ylabel='Validation accuracy (%)', ylim=(0, 100))
    for ax in axes:
        ax.grid(axis='y', alpha=0.16)
    fig.suptitle('FashionMNIST — Validation-based experiment comparison', fontweight='bold')
    fig.tight_layout()
    fig.savefig(figures / 'experiment_comparison.png', dpi=165)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for r in data:
        with (experiment_root / r['experiment'] / 'metrics.csv').open(newline='', encoding='utf-8') as f:
            history = list(csv.DictReader(f))
        ax.plot([int(x['epoch']) for x in history], [float(x['val_loss']) for x in history],
                marker='o', linewidth=2, label=r['experiment'])
    ax.set(title='Validation loss by experiment (same seed, splits, batch size)',
           xlabel='Epoch', ylabel='Validation Cross-Entropy loss')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'validation_loss_comparison.png', dpi=165)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Practice 1 — Stage 5 comparison + final holdout test')
    parser.add_argument('--csv', default='data/fashion_mnist_full.csv')
    parser.add_argument('--out-dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-reuse-stage4', action='store_true', help='Retrain E2 instead of reusing matching Stage-4 checkpoint')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.seed < 0:
        parser.error('epochs and batch-size must be positive; seed >= 0')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    experiment_root = root / 'experiments'
    experiment_root.mkdir(parents=True, exist_ok=True)
    print('[STAGE 5] Reading single CSV, fixing train/val/test and train-only normalization...', flush=True)
    loaders, audit = create_dataloaders(args.csv, batch_size=args.batch_size, seed=args.seed)
    print('Effective split sizes:', audit['effective_split_sizes'], flush=True)
    print('TEST remains unused until one experiment is selected using validation only.', flush=True)
    experiments = [('E1', (128,), 0.001), ('E2', (256, 128), 0.001), ('E3', (256, 128), 0.0001)]
    rows = []
    for name, hidden, lr in experiments:
        folder = experiment_root / name
        reuse = (name == 'E2' and not args.no_reuse_stage4 and
                 _reuse_stage4_e2(root, folder, args.seed, args.epochs, args.batch_size, audit['normalization']))
        if reuse:
            print('[E2] Reused Stage-4 checkpoint (validated configuration), no retrain.', flush=True)
        else:
            _seed_all(args.seed)
            cfg = TrainConfig(csv_path=Path(args.csv), out_dir=folder,
                              epochs=args.epochs, batch_size=args.batch_size,
                              learning_rate=lr, hidden_dims=hidden, seed=args.seed,
                              device=args.device)
            mdl = FashionMLP(hidden_dims=hidden)
            print(f'[{name}] Train architecture (784, {hidden}, 10), lr={lr}', flush=True)
            fit(mdl, _train_loader_with_fresh_seed(loaders['train'].dataset, args.batch_size, args.seed),
                loaders['validation'], cfg, device, audit['normalization'])
        summary = json.loads((folder / 'training_summary.json').read_text(encoding='utf-8'))
        ckpt_mdl, checkpoint = load_model_checkpoint(folder / 'best_model.pth', device='cpu')
        if tuple(checkpoint['hidden_dims']) != hidden:
            raise AssertionError(f'{name}: saved architecture mismatch')
        rows.append({'experiment': name, 'hidden_layers': '→'.join(map(str, hidden)),
                     'learning_rate': lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
                     'parameters': count_trainable_parameters(ckpt_mdl),
                     'best_epoch': summary['best_epoch'],
                     'best_val_loss': float(summary['best_validation_loss']),
                     'best_val_accuracy': float(summary['best_validation_accuracy']),
                     'checkpoint': str(folder / 'best_model.pth'),
                     'training_source': 'stage4_reuse' if reuse else 'stage5_train'})
    # Predetermined rule applied to validation results ONLY. Ties favour simpler model.
    selected = min(rows, key=lambda r: (r['best_val_loss'], r['parameters'], r['experiment']))
    comparison = experiment_root / 'experiment_comparison.csv'
    with comparison.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_comparison(rows, experiment_root)
    print(f'[SELECTED BY VALIDATION LOSS] {selected["experiment"]}: '
          f'val_loss={selected["best_val_loss"]:.4f}, '
          f'val_accuracy={100 * selected["best_val_accuracy"]:.2f}%', flush=True)
    # Only now are predictions made on the 10,000 test images; no model re-selection.
    test = evaluate_selected_checkpoint(
        selected['checkpoint'], loaders['test'], audit['normalization'],
        experiment_root, device, selected['experiment'])
    result = {
        'stage': 5, 'data_source': str(args.csv),
        'data_splits_after_leakage_prevention': audit['effective_split_sizes'],
        'normalization_train_only': audit['normalization'],
        'experiment_design': 'E1 baseline 128 units lr .001; E2 256→128 lr .001; E3 256→128 lr .0001',
        'comparison': rows, 'selection_rule': 'min validation loss; tie-break fewest parameters; then experiment ID',
        'selected_experiment': selected['experiment'],
        'selected_validation_loss': selected['best_val_loss'],
        'selected_validation_accuracy': selected['best_val_accuracy'],
        'test_metrics': test,
        'test_was_not_used_for_training_or_model_selection': True,
    }
    (experiment_root / 'experiment_summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] Final holdout TEST — {test["test_correct"]}/{test["test_samples"]} '
          f'correct, accuracy={test["test_accuracy_percent"]:.2f}%, '
          f'loss={test["test_cross_entropy_loss"]:.4f}, macro-F1={test["macro_f1"]:.4f}', flush=True)
    print('Outputs:', experiment_root.resolve(), flush=True)


if __name__ == '__main__':
    main()
