from PIL import ImageEnhance
from PIL import ImageEnhance
import torch
from pathlib import Path

# Path
ROOT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = ROOT_PATH / "data"
CIFAR_PATH = DATA_PATH / "cifar-10-batches-py"

# Train
WOKER = 8
BATCH_SIZE = 32
LR = 1e-5
EPOCH = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weights
SAVE_FASHION_CNN_WEIGHT = ROOT_PATH / "weights" / "FASHION_MNIST_CNN_temp.pt"
SAVE_CIFAR_RESNET_WEIGHT = ROOT_PATH / "weights" / "CIFAR_RESNET50.pt"

# Preprocessing
MEAN = (0.5,)
STD = (0.5,)


# Console
MAX_LENGTH = 20

LIST_CONFIG = [WOKER, BATCH_SIZE, LR, EPOCH, DEVICE, MEAN, STD]
LOG_PEAR_BATCH = 100


# Utility
SEED = 248

# Transform
REMOVE_BACKGROUND_PERCENT = 12


## Nomarlize
CIFAR_NOMARLIZE = [
    [0.485, 0.456, 0.406], # Mean
    [0.229, 0.224, 0.225]  # STD
]