"""Practice 1 / Stage 4 runner. From project root: python main.py

Only train + validation are passed into fit(); test is intentionally reserved
for Stage 5. One CSV is used from start to finish.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config import TrainConfig
from src.data_loader import create_dataloaders
from src.model import FashionMLP, count_trainable_parameters
from src.train import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice 1 — Stage 4 train+validation, no final test")
    parser.add_argument("--csv", default="data/fashion_mnist_full.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = TrainConfig(csv_path=Path(args.csv), out_dir=Path(args.out_dir),
                         epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.lr, hidden_dims=tuple(args.hidden),
                         seed=args.seed, device=args.device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda được chọn, nhưng PyTorch không nhận GPU CUDA.")
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)

    print("[STAGE 4] Reading the single FashionMNIST CSV and auditing splits...", flush=True)
    loaders, audit = create_dataloaders(config.csv_path, batch_size=config.batch_size, seed=config.seed)
    print("Splits after leakage prevention:", audit["effective_split_sizes"], flush=True)
    print("Excluded cross-split duplicate train rows:",
          audit["train_rows_excluded_due_to_cross_split_duplicate"], flush=True)
    print("Normalization: effective TRAIN only", flush=True)

    model = FashionMLP(hidden_dims=config.hidden_dims)
    print("Architecture:", (784, *config.hidden_dims, 10),
          "| trainable parameters:", count_trainable_parameters(model), flush=True)
    # Deliberately do NOT pass loaders['test'] to training/selection code.
    fit(model, loaders["train"], loaders["validation"], config, device,
        normalization=audit["normalization"])


if __name__ == "__main__":
    main()
