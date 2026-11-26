"""Mô hình Baseline: Logistic Regression cho phân loại ảnh FashionMNIST."""

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.vibe_scripts.lab1_v2.transforms import validate_image_tensor


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Đếm tổng số tham số của mô hình PyTorch."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


class LogisticRegressionModel(nn.Module):
    """Mô hình tuyến tính: `Flatten(C*H*W) -> Linear(in_features, num_classes)` trả raw logits."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or any(d <= 0 for d in input_shape):
            raise ValueError(f"input_shape phải gồm 3 số dương (C, H, W), nhận được {input_shape}")
        if num_classes < 2:
            raise ValueError(f"num_classes phải >= 2, nhận được {num_classes}")

        self.input_shape: Tuple[int, int, int] = tuple(int(d) for d in input_shape)
        self.input_dim: int = int(math.prod(self.input_shape))
        self.num_classes: int = int(num_classes)

        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Nhận tensor `(B, C, H, W)` và trả về raw logits `(B, num_classes)` (không áp Softmax)."""
        validate_image_tensor(x, expected_shape=self.input_shape)
        flat = self.flatten(x)
        logits = self.classifier(flat)
        return logits

    @torch.inference_mode()
    def trace_shapes(self, sample_batch: torch.Tensor) -> List[Dict[str, Any]]:
        """Theo dõi sự thay đổi Tensor Shape qua từng tầng của mô hình."""
        validate_image_tensor(sample_batch, expected_shape=self.input_shape)
        trace: List[Dict[str, Any]] = [
            {"layer": "Input", "shape": list(sample_batch.shape), "dtype": str(sample_batch.dtype)}
        ]
        flat = self.flatten(sample_batch)
        trace.append({"layer": "Flatten(1)", "shape": list(flat.shape), "dtype": str(flat.dtype)})
        logits = self.classifier(flat)
        trace.append(
            {
                "layer": f"Linear({self.input_dim}, {self.num_classes})",
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
            }
        )
        return trace

    def model_config(self) -> Dict[str, Any]:
        """Xuất cấu hình kiến trúc để nhúng vào checkpoint giúp tái dựng độc lập."""
        return {
            "model_type": "logistic",
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "hidden_dims": [],
            "dropout": 0.0,
            "num_parameters": count_parameters(self),
        }
