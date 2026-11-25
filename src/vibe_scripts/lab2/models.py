"""
Pipeline 02: Kiến trúc ResNet-18 cho 3 chế độ thực nghiệm Lab 2 (Baseline / Transfer / Fine-tune).

Ràng buộc kỹ thuật bắt buộc:
1. Input: `[B, 3, 224, 224]` (float32), Output: raw logits `[B, 37]`.
2. Thứ tự ở `finetune`: thay `fc = nn.Linear(512, 37)` TRƯỚC, nạp checkpoint `resnet18_transfer` SAU.
3. Khóa BatchNorm: các tầng bị đóng băng phải giữ `eval()` trong lúc train để không làm lệch
   `running_mean` và `running_var` tiền huấn luyện trên ImageNet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from labs.lab2.vibe import config as lab2_cfg

VALID_MODES = ("baseline", "transfer", "finetune")

# Các khối bị đóng băng ở chế độ finetune (chỉ mở khóa layer4 + fc)
FINETUNE_FROZEN_MODULES = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")
TRANSFER_FROZEN_MODULES = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")


class ResNet18Pet(nn.Module):
    """Bọc ResNet-18 với fail-fast kiểm tra shape đầu vào và quản lý chế độ đóng băng."""

    def __init__(self, backbone: nn.Module, mode: str, num_classes: int = lab2_cfg.NUM_CLASSES) -> None:
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass nhận `[B, 3, 224, 224]` và trả raw logits `[B, num_classes]`."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input tensor shape (B, 3, H, W), got {tuple(x.shape)}")
        return self.backbone(x)


def _apply_freeze_policy(backbone: nn.Module, mode: str) -> None:
    """Thiết lập `requires_grad` đúng theo từng chế độ thực nghiệm."""
    if mode == "baseline":
        for p in backbone.parameters():
            p.requires_grad = True
    elif mode == "transfer":
        for name, p in backbone.named_parameters():
            p.requires_grad = name.startswith("fc.")
    elif mode == "finetune":
        for name, p in backbone.named_parameters():
            p.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
    else:
        raise ValueError(f"Mode không hợp lệ: {mode}. Chỉ chấp nhận {VALID_MODES}.")


def set_mode_for_training(model: nn.Module, mode: str) -> None:
    """
    Bật chế độ train nhưng giữ nguyên `eval()` cho toàn bộ khối bị đóng băng để khóa BatchNorm.

    Bắt buộc gọi đầu mỗi epoch huấn luyện thay vì `model.train()` thuần.
    """
    model.train()
    backbone = model.backbone if isinstance(model, ResNet18Pet) else model

    if mode == "baseline":
        return
    if mode == "transfer":
        for mod_name in TRANSFER_FROZEN_MODULES:
            getattr(backbone, mod_name).eval()
    elif mode == "finetune":
        for mod_name in FINETUNE_FROZEN_MODULES:
            getattr(backbone, mod_name).eval()
    else:
        raise ValueError(f"Mode không hợp lệ: {mode}")


def _extract_backbone_state_dict(checkpoint_payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Chuẩn hóa key `state_dict` dù checkpoint lưu qua `ResNet18Pet` (`backbone.*`) hay trực tiếp."""
    state = checkpoint_payload.get("model_state_dict", checkpoint_payload)
    if any(k.startswith("backbone.") for k in state.keys()):
        return {k[len("backbone.") :]: v for k, v in state.items() if k.startswith("backbone.")}
    return state


def build_resnet18_pet(
    mode: str,
    num_classes: int = lab2_cfg.NUM_CLASSES,
    init_checkpoint: Optional[Path] = None,
    use_pretrained: bool = True,
) -> ResNet18Pet:
    """
    Khởi tạo ResNet-18 theo đúng 1 trong 3 chế độ: `baseline`, `transfer`, `finetune`.

    Args:
        mode: `"baseline"` | `"transfer"` | `"finetune"`.
        num_classes: Số lớp đầu ra (mặc định 37 cho Oxford-IIIT Pets).
        init_checkpoint: Đường dẫn `resnet18_transfer/best.pt` (bắt buộc cho `finetune` khi chạy thật).
        use_pretrained: Cho phép tắt tải `IMAGENET1K_V1` trong offline unit test (mặc định `True`).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Mode không hợp lệ: {mode}. Chỉ chấp nhận {VALID_MODES}.")

    if mode == "baseline":
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    elif mode == "transfer":
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    else:  # finetune: thay fc TRƯỚC, nạp checkpoint transfer SAU
        weights = None if init_checkpoint is not None else (ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        if init_checkpoint is not None:
            ckpt_path = Path(init_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Không tìm thấy checkpoint transfer để khởi tạo finetune: {ckpt_path}")
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            backbone_state = _extract_backbone_state_dict(payload)
            backbone.load_state_dict(backbone_state, strict=True)

    _apply_freeze_policy(backbone, mode)
    model = ResNet18Pet(backbone=backbone, mode=mode, num_classes=num_classes)
    set_mode_for_training(model, mode)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Trả về `(total_params, trainable_params)` của mô hình."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_optimizer_param_groups(
    model: nn.Module,
    mode: str,
    lr_head: float = lab2_cfg.LR_HEAD,
    lr_backbone: float = lab2_cfg.LR_BACKBONE,
) -> List[Dict[str, Any]]:
    """
    Tạo danh sách parameter groups chỉ gồm các tham số `requires_grad=True`.

    - `baseline`: toàn bộ mạng với `lr_head`.
    - `transfer`: chỉ `fc` với `lr_head`.
    - `finetune`: differential LR (`layer4` dùng `lr_backbone`, `fc` dùng `lr_head`).
    """
    if mode == "finetune":
        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if ".fc." in name or name.startswith("fc."):
                head_params.append(p)
            else:
                backbone_params.append(p)
        if not backbone_params or not head_params:
            raise ValueError("Chế độ finetune phải có đủ cả tham số layer4 lẫn fc.")
        return [
            {"params": backbone_params, "lr": lr_backbone, "group_name": "layer4"},
            {"params": head_params, "lr": lr_head, "group_name": "fc"},
        ]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"Không có tham số trainable nào ở chế độ {mode}.")
    return [{"params": trainable_params, "lr": lr_head, "group_name": mode}]
