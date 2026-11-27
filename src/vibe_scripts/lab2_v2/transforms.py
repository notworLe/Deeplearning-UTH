"""Chuẩn hóa ImageNet, augmentation cho Train/Eval và kiểm chứng Tensor Contract cho Oxford-IIIT Pets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image

DEFAULT_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageNetNormStats:
    """Thông số chuẩn hóa 3 kênh RGB theo chuẩn ImageNet."""

    mean: Tuple[float, float, float] = DEFAULT_IMAGENET_MEAN
    std: Tuple[float, float, float] = DEFAULT_IMAGENET_STD

    def to_dict(self) -> dict:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, data: dict) -> "ImageNetNormStats":
        mean = tuple(float(v) for v in data["mean"])
        std = tuple(float(v) for v in data["std"])
        if len(mean) != 3 or len(std) != 3:
            raise ValueError(f"NormStats phải gồm đúng 3 kênh RGB, nhận mean={mean}, std={std}")
        return cls(mean=mean, std=std)  # type: ignore[arg-type]


def build_train_transform(
    image_size: int = 224,
    crop_scale: Tuple[float, float] = (0.75, 1.0),
    hflip_prob: float = 0.5,
    norm_stats: ImageNetNormStats = ImageNetNormStats(),
) -> T.Compose:
    """Transform cho tập Train: RandomResizedCrop + HorizontalFlip + ImageNet Normalize."""
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.ToTensor(),
            T.Normalize(mean=norm_stats.mean, std=norm_stats.std),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    resize_size: int = 256,
    norm_stats: ImageNetNormStats = ImageNetNormStats(),
) -> T.Compose:
    """Transform tất định cho Val / Test / Inference: Resize(256) → CenterCrop(224) → ImageNet Normalize."""
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=norm_stats.mean, std=norm_stats.std),
        ]
    )


def validate_image_tensor(
    tensor: torch.Tensor,
    expected_image_size: int = 224,
    allow_unbatched: bool = False,
) -> torch.Tensor:
    """Kiểm chứng Fail-Fast Tensor Contract `(B, 3, H, W)` hoặc `(3, H, W)` dtype float."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input phải là torch.Tensor, nhận {type(tensor).__name__}")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"Input tensor phải là kiểu float, nhận dtype={tensor.dtype}")

    if tensor.ndim == 3 and allow_unbatched:
        c, h, w = tensor.shape
        if (c, h, w) != (3, expected_image_size, expected_image_size):
            raise ValueError(
                f"Expected shape (3, {expected_image_size}, {expected_image_size}), got {tuple(tensor.shape)}"
            )
        return tensor
    if tensor.ndim == 4:
        _, c, h, w = tensor.shape
        if (c, h, w) != (3, expected_image_size, expected_image_size):
            raise ValueError(
                f"Expected shape (B, 3, {expected_image_size}, {expected_image_size}), got {tuple(tensor.shape)}"
            )
        return tensor

    expected_desc = "(3, H, W) hoặc (B, 3, H, W)" if allow_unbatched else "(B, 3, H, W)"
    raise ValueError(f"Expected tensor {expected_desc}, got shape {tuple(tensor.shape)}")


def denormalize_tensor(
    tensor: torch.Tensor,
    norm_stats: ImageNetNormStats = ImageNetNormStats(),
) -> torch.Tensor:
    """Khôi phục tensor đã chuẩn hóa ImageNet về khoảng `[0, 1]` để trực quan hóa."""
    mean = torch.tensor(norm_stats.mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(norm_stats.std, dtype=tensor.dtype, device=tensor.device)
    if tensor.ndim == 3:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    elif tensor.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        raise ValueError(f"Chỉ hỗ trợ tensor 3D/4D, nhận shape {tuple(tensor.shape)}")
    return (tensor * std + mean).clamp(0.0, 1.0)


def preprocess_image(
    image: Union[str, Path, Image.Image],
    image_size: int = 224,
    resize_size: int = 256,
    norm_stats: ImageNetNormStats = ImageNetNormStats(),
) -> torch.Tensor:
    """Nạp ảnh từ đường dẫn hoặc PIL.Image, ép chuẩn RGB và trả về tensor `(1, 3, 224, 224)`."""
    if isinstance(image, (str, Path)):
        img_path = Path(image)
        if not img_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
        with Image.open(img_path) as opened:
            pil_img = opened.convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"image phải là Path/str/PIL.Image, nhận {type(image).__name__}")

    tf = build_eval_transform(image_size=image_size, resize_size=resize_size, norm_stats=norm_stats)
    tensor = tf(pil_img)
    validate_image_tensor(tensor, expected_image_size=image_size, allow_unbatched=True)
    return tensor.unsqueeze(0)
