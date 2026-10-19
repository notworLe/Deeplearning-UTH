import torch
from pathlib import Path

ROOT_PATH = Path.cwd().parent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
