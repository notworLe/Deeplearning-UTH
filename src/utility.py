import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from src import config



def set_seed(seed: int = config.SEED):
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



def save_weight(path:Path, epoch, model:nn.Module, optimizer, best_val):
    # Parameter
    if not isinstance(path, Path):
        path = Path(path)

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
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


def resume_weight(p_path:Path, p_model:nn.Module, p_optimizer) -> tuple[int, float]:
    if not p_path.exists():
        print(f"No found weight. Train from start")
        return 0, 0.0

    print(f"Resume: {p_path}")
    v_dict_weight = torch.load(p_path, map_location=config.DEVICE)

    p_model.load_state_dict(v_dict_weight['model'])
    p_optimizer.load_state_dict(v_dict_weight['optimizer'])
    v_best_val = v_dict_weight['best_validation']
    v_epoch = v_dict_weight['epoch']

    return v_epoch, v_best_val

def add_transform(p_base_transform: list, p_new_transform: list):
    return transforms.Compose([
        *p_base_transform,
        *p_new_transform
    ])