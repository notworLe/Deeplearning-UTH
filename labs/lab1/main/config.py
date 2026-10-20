import torch
from pathlib import Path

ROOT_PATH = Path.cwd().parent
WOKER = 8
BATCH_SIZE = 32
LR = 1e-5
EPOCH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Preprocessing
MEAN = (0.5,)
STD = (0.5,)


# Console
MAX_LENGTH = 20
