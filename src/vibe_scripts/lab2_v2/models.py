"""Model Factory & Freeze/Unfreeze Engine cho Lab 2 v2 (`ResNet18`, `DenseNet121`, `VGG16`)."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.vibe_scripts.lab2_v2.transforms import validate_image_tensor

SUPPORTED_ARCHS = ("resnet18", "densenet121", "vgg16")
SUPPORTED_MODES = ("scratch", "transfer", "finetune")


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Đếm chính xác tổng tham số, số tham số trainable và số tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
        "trainable_pct": round((trainable / max(total, 1)) * 100.0, 4),
    }


def _build_torchvision_backbone(arch: str, pretrained: bool) -> nn.Module:
    arch_l = arch.lower()
    if arch_l == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        return tv_models.resnet18(weights=weights)
    if arch_l == "densenet121":
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        return tv_models.densenet121(weights=weights)
    if arch_l == "vgg16":
        weights = tv_models.VGG16_Weights.DEFAULT if pretrained else None
        return tv_models.vgg16(weights=weights)
    raise ValueError(f"Kiến trúc không hỗ trợ: '{arch}'. Hỗ trợ: {SUPPORTED_ARCHS}")


def _replace_head(backbone: nn.Module, arch: str, num_classes: int) -> nn.Module:
    """Thay lớp phân loại cuối cho đúng số lớp mục tiêu (`37`) và trả về module head mới."""
    arch_l = arch.lower()
    if arch_l == "resnet18":
        in_features = backbone.fc.in_features  # type: ignore[union-attr]
        backbone.fc = nn.Linear(in_features, num_classes)
        return backbone.fc
    if arch_l == "densenet121":
        in_features = backbone.classifier.in_features  # type: ignore[union-attr]
        backbone.classifier = nn.Linear(in_features, num_classes)
        return backbone.classifier
    if arch_l == "vgg16":
        in_features = backbone.classifier[6].in_features  # type: ignore[index]
        backbone.classifier[6] = nn.Linear(in_features, num_classes)  # type: ignore[index]
        return backbone.classifier[6]  # type: ignore[index]
    raise ValueError(f"Kiến trúc không hỗ trợ: '{arch}'")


def _apply_freeze_policy(backbone: nn.Module, arch: str, mode: str) -> None:
    """Áp dụng chính sách freeze/unfreeze theo mode (`scratch`, `transfer`, `finetune`)."""
    arch_l = arch.lower()
    mode_l = mode.lower()
    if mode_l not in SUPPORTED_MODES:
        raise ValueError(f"Mode không hỗ trợ: '{mode}'. Hỗ trợ: {SUPPORTED_MODES}")

    if mode_l == "scratch":
        for p in backbone.parameters():
            p.requires_grad = True
        return

    # Bước 1: Đóng băng toàn bộ backbone trước
    for p in backbone.parameters():
        p.requires_grad = False

    # Bước 2: Mở khóa lớp phân loại cuối (head)
    if arch_l == "resnet18":
        for p in backbone.fc.parameters():  # type: ignore[union-attr]
            p.requires_grad = True
        if mode_l == "finetune":
            for p in backbone.layer4.parameters():  # type: ignore[union-attr]
                p.requires_grad = True
    elif arch_l == "densenet121":
        if mode_l == "finetune":
            raise ValueError(
                "Theo phạm vi Lab 2 v2, chế độ 'finetune' chỉ áp dụng cho 'resnet18'; "
                "'densenet121' chỉ hỗ trợ 'transfer' hoặc 'scratch'."
            )
        for p in backbone.classifier.parameters():  # type: ignore[union-attr]
            p.requires_grad = True
    elif arch_l == "vgg16":
        if mode_l == "finetune":
            raise ValueError(
                "Theo phạm vi Lab 2 v2, chế độ 'finetune' chỉ áp dụng cho 'resnet18'; "
                "'vgg16' chỉ hỗ trợ 'transfer' hoặc 'scratch'."
            )
        for p in backbone.classifier[6].parameters():  # type: ignore[index]
            p.requires_grad = True


class PetClassifier(nn.Module):
    """Wrapper chuẩn hóa Tensor Contract, Freeze-BatchNorm guard và metadata cho 3 kiến trúc pre-trained."""

    def __init__(
        self,
        arch: str,
        mode: str = "transfer",
        num_classes: int = 37,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.arch = arch.lower()
        self.mode = mode.lower()
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.image_size = int(image_size)

        if self.arch not in SUPPORTED_ARCHS:
            raise ValueError(f"Kiến trúc không hỗ trợ: '{arch}'. Hỗ trợ: {SUPPORTED_ARCHS}")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"Mode không hỗ trợ: '{mode}'. Hỗ trợ: {SUPPORTED_MODES}")

        self.net = _build_torchvision_backbone(self.arch, pretrained=self.pretrained)
        _replace_head(self.net, self.arch, num_classes=self.num_classes)
        _apply_freeze_policy(self.net, self.arch, self.mode)

    def train(self, mode: bool = True) -> "PetClassifier":
        """Giữ các tầng BatchNorm2d bị đóng băng ở trạng thái `eval()` để không làm trôi thống kê ImageNet."""
        super().train(mode)
        if mode and self.mode in ("transfer", "finetune"):
            for module in self.net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    params = list(module.parameters(recurse=False))
                    if params and all(not p.requires_grad for p in params):
                        module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_image_tensor(x, expected_image_size=self.image_size, allow_unbatched=False)
        logits = self.net(x)
        return logits

    def get_head_module(self) -> nn.Module:
        if self.arch == "resnet18":
            return self.net.fc  # type: ignore[return-value]
        if self.arch == "densenet121":
            return self.net.classifier  # type: ignore[return-value]
        return self.net.classifier[6]  # type: ignore[index,return-value]

    def get_param_groups(
        self,
        lr_head: float,
        lr_backbone: float = 0.0,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Tạo parameter groups chuẩn cho optimizer (hỗ trợ differential LR khi `mode='finetune'`)."""
        head_param_ids = {id(p) for p in self.get_head_module().parameters()}
        head_params = [p for p in self.parameters() if p.requires_grad and id(p) in head_param_ids]
        backbone_params = [p for p in self.parameters() if p.requires_grad and id(p) not in head_param_ids]

        groups: List[Dict[str, Any]] = []
        if backbone_params:
            eff_backbone_lr = float(lr_head) if self.mode == "scratch" else float(lr_backbone or lr_head * 0.1)
            groups.append(
                {
                    "name": "backbone",
                    "params": backbone_params,
                    "lr": eff_backbone_lr,
                    "weight_decay": float(weight_decay),
                }
            )
        if head_params:
            groups.append(
                {
                    "name": "head",
                    "params": head_params,
                    "lr": float(lr_head),
                    "weight_decay": float(weight_decay),
                }
            )
        if not groups:
            raise RuntimeError("Không có tham số trainable nào trong mô hình!")
        return groups

    def model_config(self) -> Dict[str, Any]:
        counts = count_parameters(self)
        return {
            "arch": self.arch,
            "mode": self.mode,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "input_shape": [3, self.image_size, self.image_size],
            "total_params": counts["total"],
            "trainable_params": counts["trainable"],
            "trainable_pct": counts["trainable_pct"],
        }


def build_model(
    arch: str,
    mode: str = "transfer",
    num_classes: int = 37,
    pretrained: bool = True,
    image_size: int = 224,
) -> PetClassifier:
    """Factory khởi tạo mô hình chuẩn hóa theo `arch`, `mode` và `pretrained`."""
    return PetClassifier(
        arch=arch,
        mode=mode,
        num_classes=num_classes,
        pretrained=pretrained,
        image_size=image_size,
    )


@torch.inference_mode()
def trace_shapes(
    model: PetClassifier,
    input_shape: Tuple[int, int, int, int] = (2, 3, 224, 224),
) -> List[Dict[str, Any]]:
    """Truy vết hình dạng tensor đi qua từng giai đoạn lớn của kiến trúc."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(input_shape, dtype=torch.float32, device=device)

    trace: List[Dict[str, Any]] = [
        {"stage": "Input", "shape": list(x.shape), "dtype": str(x.dtype)}
    ]
    hooks: List[Any] = []

    def _hook(stage_name: str):
        def _fn(_module: nn.Module, _inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor):
                trace.append({"stage": stage_name, "shape": list(out.shape), "dtype": str(out.dtype)})
        return _fn

    net = model.net
    if model.arch == "resnet18":
        stages = [
            ("stem: maxpool", net.maxpool),
            ("layer1", net.layer1),
            ("layer2", net.layer2),
            ("layer3", net.layer3),
            ("layer4", net.layer4),
            ("avgpool", net.avgpool),
            ("head: fc", net.fc),
        ]
    elif model.arch == "densenet121":
        stages = [
            ("stem: pool0", net.features.pool0),
            ("denseblock1", net.features.denseblock1),
            ("denseblock2", net.features.denseblock2),
            ("denseblock3", net.features.denseblock3),
            ("denseblock4", net.features.denseblock4),
            ("norm5", net.features.norm5),
            ("head: classifier", net.classifier),
        ]
    else:  # vgg16
        stages = [
            ("features (5 conv blocks)", net.features),
            ("avgpool", net.avgpool),
            ("classifier[0..5] (FC 4096)", net.classifier[5]),
            ("head: classifier[6]", net.classifier[6]),
        ]

    for name, mod in stages:
        hooks.append(mod.register_forward_hook(_hook(name)))
    try:
        _ = model(x)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()
    return trace
