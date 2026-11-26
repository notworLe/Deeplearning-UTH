"""Mô hình Multi-Layer Perceptron (MLP) cấu hình linh hoạt và Model Factory cho Lab 1 v2."""

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from src.vibe_scripts.lab1_v2.logistic import LogisticRegressionModel, count_parameters
from src.vibe_scripts.lab1_v2.transforms import validate_image_tensor


class MLPModel(nn.Module):
    """Mạng MLP: `Flatten -> [Linear -> ReLU -> Dropout] x L -> Linear(H_L, num_classes)`."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dims: Sequence[int] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or any(d <= 0 for d in input_shape):
            raise ValueError(f"input_shape phải gồm 3 số dương (C, H, W), nhận được {input_shape}")
        if not hidden_dims or any(int(h) <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims phải không rỗng và gồm các số nguyên dương, nhận {hidden_dims}")
        if num_classes < 2:
            raise ValueError(f"num_classes phải >= 2, nhận được {num_classes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout phải nằm trong [0.0, 1.0), nhận được {dropout}")

        self.input_shape: Tuple[int, int, int] = tuple(int(d) for d in input_shape)
        self.input_dim: int = int(math.prod(self.input_shape))
        self.hidden_dims: Tuple[int, ...] = tuple(int(h) for h in hidden_dims)
        self.num_classes: int = int(num_classes)
        self.dropout: float = float(dropout)

        self.flatten = nn.Flatten(start_dim=1)

        layers: List[nn.Module] = []
        in_features = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            in_features = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Nhận tensor `(B, C, H, W)` và trả về raw logits `(B, num_classes)` (không áp Softmax)."""
        validate_image_tensor(x, expected_shape=self.input_shape)
        flat = self.flatten(x)
        features = self.backbone(flat)
        logits = self.classifier(features)
        return logits

    @torch.inference_mode()
    def trace_shapes(self, sample_batch: torch.Tensor) -> List[Dict[str, Any]]:
        """Theo dõi sự thay đổi Tensor Shape qua từng tầng của mạng MLP."""
        validate_image_tensor(sample_batch, expected_shape=self.input_shape)
        trace: List[Dict[str, Any]] = [
            {"layer": "Input", "shape": list(sample_batch.shape), "dtype": str(sample_batch.dtype)}
        ]
        cur = self.flatten(sample_batch)
        trace.append({"layer": "Flatten(1)", "shape": list(cur.shape), "dtype": str(cur.dtype)})
        for idx, module in enumerate(self.backbone):
            cur = module(cur)
            trace.append(
                {
                    "layer": f"backbone[{idx}]: {module.__class__.__name__}",
                    "shape": list(cur.shape),
                    "dtype": str(cur.dtype),
                }
            )
        logits = self.classifier(cur)
        trace.append(
            {
                "layer": f"Linear({self.hidden_dims[-1]}, {self.num_classes})",
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
            }
        )
        return trace

    def model_config(self) -> Dict[str, Any]:
        """Xuất cấu hình kiến trúc để nhúng vào checkpoint giúp tái dựng độc lập."""
        return {
            "model_type": "mlp",
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "num_parameters": count_parameters(self),
        }


def build_model(
    model_type: str,
    input_shape: Tuple[int, int, int] = (1, 28, 28),
    num_classes: int = 10,
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.2,
) -> nn.Module:
    """Factory khởi tạo mô hình chuẩn theo cấu hình (`logistic` hoặc `mlp`)."""
    kind = model_type.strip().lower()
    if kind in ("logistic", "logistic_regression"):
        return LogisticRegressionModel(input_shape=input_shape, num_classes=num_classes)
    if kind == "mlp":
        return MLPModel(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Không hỗ trợ model_type='{model_type}'. Chỉ chấp nhận: ['logistic', 'mlp'].")


__all__ = ["MLPModel", "LogisticRegressionModel", "build_model", "count_parameters"]
