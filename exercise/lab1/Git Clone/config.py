"""Practice 1 / Stage 4 — a single source of truth for training options."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    csv_path: Path = Path("data/fashion_mnist_full.csv")
    out_dir: Path = Path("outputs")
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    hidden_dims: tuple[int, ...] = (256, 128)
    seed: int = 42
    device: str = "auto"  # auto (GPU when available), cpu, cuda

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs và batch_size phải >= 1.")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate phải trong (0,1).")
        if not self.hidden_dims or any(type(x) is not int or x <= 0 for x in self.hidden_dims):
            raise ValueError("hidden_dims phải chứa các số nguyên dương.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device phải là auto, cpu hoặc cuda.")
        if self.seed < 0:
            raise ValueError("seed phải >= 0.")
