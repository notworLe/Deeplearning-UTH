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

