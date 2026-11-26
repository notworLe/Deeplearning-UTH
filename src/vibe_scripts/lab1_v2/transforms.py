"""Tiền xử lý ảnh, chuẩn hóa chống leakage và kiểm tra hợp đồng tensor cho FashionMNIST."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class NormStats:
    """Thống kê chuẩn hóa (mean, std) được tính duy nhất trên tập Train."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std):
            raise ValueError(f"mean/std phải hữu hạn, nhận được mean={self.mean}, std={self.std}")
        if self.std <= 0.0:
            raise ValueError(f"std phải > 0, nhận được std={self.std}")

    def to_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NormStats":
        return cls(mean=float(payload["mean"]), std=float(payload["std"]))


def validate_image_tensor(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, int, int] = (1, 28, 28),
) -> None:
    """Kiểm tra Fail-Fast hợp đồng tensor ảnh `(C, H, W)` hoặc `(B, C, H, W)`."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Kỳ vọng torch.Tensor, nhận được {type(tensor).__name__}")
    if tensor.ndim == 3:
        actual = tuple(tensor.shape)
    elif tensor.ndim == 4:
        actual = tuple(tensor.shape[1:])
        if tensor.shape[0] < 1:
            raise ValueError(f"Batch size phải >= 1, nhận được shape={tuple(tensor.shape)}")
    else:
        raise ValueError(
            f"Tensor ảnh phải có 3 chiều (C,H,W) hoặc 4 chiều (B,C,H,W), nhận được shape={tuple(tensor.shape)}"
        )

    if actual != expected_shape:
        raise ValueError(
            f"Sai kích thước ảnh: kỳ vọng {expected_shape}, nhận được {actual} (tensor shape={tuple(tensor.shape)})"
        )
    if not torch.is_floating_point(tensor):
        raise TypeError(f"Tensor ảnh đầu vào phải là kiểu số thực (float), nhận được {tensor.dtype}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Tensor ảnh chứa giá trị NaN hoặc Inf.")


def compute_train_norm_stats(train_images: torch.Tensor) -> NormStats:
    """Tính `(mean, std)` duy nhất trên tập Train sau khi đưa về thang `[0, 1]`."""
    if not isinstance(train_images, torch.Tensor) or train_images.numel() == 0:
        raise ValueError("train_images phải là torch.Tensor không rỗng.")
    scaled = train_images.to(dtype=torch.float32)
    if scaled.max().item() > 1.0:
        scaled = scaled / 255.0
    mean = float(scaled.mean().item())
    std = float(scaled.std(unbiased=False).item())
    return NormStats(mean=mean, std=std)


def build_train_transform(norm_stats: NormStats) -> transforms.Compose:
    """Tạo chuỗi biến đổi xác định cho tập Train: `ToTensor -> Normalize`."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(norm_stats.mean,), std=(norm_stats.std,)),
        ]
    )


def build_eval_transform(norm_stats: NormStats) -> transforms.Compose:
    """Tạo chuỗi biến đổi xác định cho tập Validation/Test: `ToTensor -> Normalize`."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(norm_stats.mean,), std=(norm_stats.std,)),
        ]
    )


def normalize_batch(
    images: torch.Tensor,
    norm_stats: NormStats,
    expected_shape: Tuple[int, int, int] = (1, 28, 28),
) -> torch.Tensor:
    """Chuẩn hóa vector hóa tensor `(N, H, W)` hoặc `(N, 1, H, W)` sang `(N, 1, H, W)` float32."""
    if images.ndim == 3:
        x = images.unsqueeze(1)
    elif images.ndim == 4:
        x = images
    else:
        raise ValueError(f"Kỳ vọng tensor 3D hoặc 4D, nhận được shape={tuple(images.shape)}")

    x = x.to(dtype=torch.float32)
    if x.max().item() > 1.0:
        x = x / 255.0
    x = (x - norm_stats.mean) / norm_stats.std
    validate_image_tensor(x, expected_shape=expected_shape)
    return x


def denormalize_tensor(tensor: torch.Tensor, norm_stats: NormStats) -> torch.Tensor:
    """Khử chuẩn hóa về khoảng `[0, 1]` phục vụ trực quan hóa ảnh."""
    restored = tensor.detach().cpu().to(dtype=torch.float32) * norm_stats.std + norm_stats.mean
    return restored.clamp(0.0, 1.0)


def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    norm_stats: NormStats,
    expected_shape: Tuple[int, int, int] = (1, 28, 28),
) -> torch.Tensor:
    """Chuyển đổi ảnh PIL / NumPy / Tensor bất kỳ thành batch `(1, 1, H, W)` đã chuẩn hóa."""
    _, target_h, target_w = expected_shape

    if isinstance(image, Image.Image):
        pil_img = image.convert("L")
        if pil_img.size != (target_w, target_h):
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        tensor = build_eval_transform(norm_stats)(pil_img).unsqueeze(0)
        validate_image_tensor(tensor, expected_shape=expected_shape)
        return tensor

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            pil_img = Image.fromarray(arr.astype(np.uint8)).convert("L")
            return preprocess_image(pil_img, norm_stats=norm_stats, expected_shape=expected_shape)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"NumPy array phải có dạng 2D (H,W) hoặc ảnh RGB (H,W,3), nhận {arr.shape}")
        if arr.shape != (target_h, target_w):
            pil_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
            return preprocess_image(pil_img, norm_stats=norm_stats, expected_shape=expected_shape)
        t = torch.from_numpy(arr.copy()).to(dtype=torch.float32)
        if t.max().item() > 1.0:
            t = t / 255.0
        t = ((t - norm_stats.mean) / norm_stats.std).unsqueeze(0).unsqueeze(0)
        validate_image_tensor(t, expected_shape=expected_shape)
        return t

    if isinstance(image, torch.Tensor):
        t = image.detach().to(dtype=torch.float32)
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim != 4:
            raise ValueError(f"Tensor ảnh phải 2D/3D/4D, nhận shape={tuple(t.shape)}")
        if tuple(t.shape[1:]) != expected_shape:
            raise ValueError(f"Sai kích thước tensor: kỳ vọng {expected_shape}, nhận {tuple(t.shape[1:])}")
        if t.min().item() >= 0.0 and t.max().item() > 1.0:
            t = (t / 255.0 - norm_stats.mean) / norm_stats.std
        validate_image_tensor(t, expected_shape=expected_shape)
        return t

    raise TypeError(f"Không hỗ trợ kiểu ảnh đầu vào: {type(image).__name__}")
