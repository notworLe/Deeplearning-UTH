import random
import numpy as np
import torch

SEED = 248

def set_seed(seed: int = SEED):
    """
    Seed tất cả nguồn ngẫu nhiên.
    WHY: Python random, NumPy, và PyTorch có RNG riêng biệt —
         phải seed cả ba để kết quả reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

import torch
import torch.nn as nn
from pathlib import Path

def save_weight(path:Path, epoch, model:nn.Module, optimizer, best_val):
    # Parameter
    if not isinstance(path, Path):
        path = Path(path)

    # Temporary file
    temp_file = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_validation": best_val
    }, temp_file)
    # Replace after finish 100%
    temp_file.replace(path)
    print(f"Save weights: {path}")