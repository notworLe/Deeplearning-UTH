import torchvision.transforms as transform
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
EPOCH = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weights
SAVE_FASHION_CNN_WEIGHT = ROOT_PATH / "weights" / "FASHION_MNIST_CNN_temp.pt"
SAVE_CIFAR_RESNET_WEIGHT = ROOT_PATH / "weights" / "CIFAR_RESNET50.pt"







# Utility
SEED = 248


# Preprocessing
MEAN = (0.5,)
STD = (0.5,)

MY_MODEL_TRANSFORM = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(
        mean=MEAN,
        std=STD
    )
])

REMOVE_BACKGROUND_PERCENT = 12

## Pre-trained
IMAGE_NET_NORMALIZE = transform.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

## Normalize
CIFAR_NORMALIZE = [
    [0.4914, 0.4822, 0.4465], # Mean
    [0.2470, 0.2435, 0.2616]  # STD
]


DEFAULT_TRANSFORM = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=MEAN, std=STD)
])

AUGMENTED_TRANSFORM = transform.Compose([
    # WHY RandomHorizontalFlip? Áo có thể lật ngang, vẫn là áo
    transform.RandomHorizontalFlip(p=0.5),

    # WHY RandomAffine?
    # degrees=10  → xoay nhẹ, simulate ảnh chụp lệch
    # translate   → dịch chuyển, object không phải lúc nào cũng ở giữa
    # scale       → zoom in/out nhẹ
    transform.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    # WHY RandomErasing?
    # Xóa ngẫu nhiên 1 vùng hình chữ nhật → force model không
    # dựa vào 1 chi tiết duy nhất (e.g., chỉ nhìn cổ áo để đoán Coat)
    # → phải học nhiều đặc trưng phân tán hơn
    transform.ToTensor(),
    transform.Normalize(mean=MEAN, std=STD),
    transform.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])


AUGMENTED_TRANSFORM_LOW = transform.Compose([
    transform.RandomHorizontalFlip(p=0.5),

    # Giảm mạnh: bỏ scale, giảm rotate và translate
    transform.RandomAffine(degrees=7, translate=(0.07, 0.07)),

    transform.ToTensor(),
    transform.Normalize(mean=MEAN, std=STD),

    # WHY giảm p và scale?
    # 28×28 quá nhỏ → erasing nhỏ thôi, không thì mất signal
    transform.RandomErasing(p=0.2, scale=(0.02, 0.08)),
])

AUGMENTED_AI_VIETNGUYEN = transform.Compose([
    transform.RandomAffine(
        degrees=(-5, 5),
        translate=(0.05, 0.05),
        scale=(0.85, 1.15),
        shear=4
    ),
    transform.ColorJitter(
        brightness=0.125,
        contrast=0.5,
        saturation=0.30,
        hue=0.05
    ),
    transform.ToTensor(),
    transform.Normalize(mean=MEAN, std=STD)
])


MY_AUGMENTED = [

    transform.RandomHorizontalFlip(),
    transform.ToTensor()
]

# Console
MAX_LENGTH = 20

LIST_CONFIG = [WOKER, BATCH_SIZE, LR, EPOCH, DEVICE, MEAN, STD]
LOG_PEAR_BATCH = 10