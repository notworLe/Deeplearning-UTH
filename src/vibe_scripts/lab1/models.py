"""Pipeline 02: Định nghĩa độc lập 3 kiến trúc mô hình cho Lab 1 (Fashion-MNIST)."""

from typing import Dict, Type

import torch
import torch.nn as nn

from labs.lab1.vibe import config as lab1_cfg


def _validate_input_shape(x: torch.Tensor) -> None:
    """Kiểm tra Fail-Fast tensor đầu vào có đúng shape `(B, 1, 28, 28)`."""
    expected = tuple(lab1_cfg.INPUT_SHAPE)
    if x.ndim != 4 or tuple(x.shape[1:]) != expected:
        raise ValueError(
            f"Expected input shape (B, {expected[0]}, {expected[1]}, {expected[2]}), "
            f"got {tuple(x.shape)}"
        )


class LogisticClassifier(nn.Module):
    """Mô hình hồi quy Logistic làm mốc sàn (7.850 tham số)."""

    def __init__(self, num_classes: int = lab1_cfg.NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: `[B, 1, 28, 28]` -> raw logits `[B, num_classes]`."""
        _validate_input_shape(x)
        return self.net(x)


class MLPClassifier(nn.Module):
    """Mạng Perceptron đa tầng 2 lớp ẩn ReLU (109.386 tham số)."""

    def __init__(self, num_classes: int = lab1_cfg.NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: `[B, 1, 28, 28]` -> raw logits `[B, num_classes]`."""
        _validate_input_shape(x)
        return self.net(x)


class CNNClassifier(nn.Module):
    """Mạng tích chập 2 block giữ cấu trúc không gian 2D (105.866 tham số)."""

    def __init__(self, num_classes: int = lab1_cfg.NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: `[B, 1, 28, 28]` -> raw logits `[B, num_classes]`."""
        _validate_input_shape(x)
        feats = self.features(x)
        return self.classifier(feats)


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "logistic": LogisticClassifier,
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
}


def build_model(
    model_name: str,
    num_classes: int = lab1_cfg.NUM_CLASSES,
) -> nn.Module:
    """Khởi tạo mô hình theo tên chuẩn `logistic | mlp | cnn`."""
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: [{available}]")
    return MODEL_REGISTRY[key](num_classes=num_classes)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Đếm tổng số tham số của mô hình."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
