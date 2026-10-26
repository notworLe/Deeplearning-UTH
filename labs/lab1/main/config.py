import torch
from pathlib import Path

ROOT_PATH = Path.cwd()
SAVE_FASHION_CNN_WEIGHT = ROOT_PATH / "weights" / "FASHION_MNIST_CNN_temp.pt"

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

LIST_CONFIG = [WOKER, BATCH_SIZE, LR, EPOCH, DEVICE, MEAN, STD]
LOG_PEAR_BATCH = 100

