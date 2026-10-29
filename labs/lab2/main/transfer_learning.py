from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from src import config, log
from src.model import training_phase, evaluate_phase
from src.utility import resume_weight, set_seed

class MyResNet(nn.Module):
    def __init__(self, n_classes = 10):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.model.fc

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, n_classes)
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

if __name__ == '__main__':
    log.log_creation()
    set_seed()


    # Model
    model = MyResNet().to(config.DEVICE)

    print("=== FULL MODEL STRUCTURE ===")
    # print(model)
    # summary(model, (3, 224, 224), device=config.DEVICE)

    print("\n=== NAMED PARAMETERS (tên layer + shape) ===")
    # for name, param in model.named_parameters():
    #     print(f"{name:50s} | shape: {list(param.shape)}")

    # ── Đếm params TRƯỚC khi freeze ──────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[TRƯỚC freeze] Tổng params    : {total:,}")
    print(f"[TRƯỚC freeze] Trainable params: {trainable:,}")

    # ── Freeze: giữ nguyên tất cả NGOẠI TRỪ fc1, fc2, layer4 ────
    for name, param in model.named_parameters():
        if "fc" in name or "layer4" in name:
            continue
        param.requires_grad = False

    # ── Đếm params SAU khi freeze ────────────────────────────────
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[SAU  freeze] Trainable params: {trainable_after:,}")
    print(f"  → Đã đóng băng: {total - trainable_after:,} params ({(total - trainable_after)/total*100:.1f}%)")


    # ── Transforms ───────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),          # ResNet cần input 224x224

        # Augmentation
        transforms.ToTensor(),
        transforms.Normalize(
            *config.CIFAR_NOMARLIZE
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            *config.CIFAR_NOMARLIZE
        )
    ])

    # ── Dataset ───────────────────────────────────────────────
    log.print_log("Cifar path", config.CIFAR_PATH)

    train_dataset = datasets.CIFAR10(
        config.CIFAR_PATH,
        train=True,
        download=True,
        transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=config.CIFAR_PATH,
        train=False,
        download=True,
        transform=val_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WOKER,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WOKER,
        shuffle=False
    )

    # ── Train ───────────────────────────────────────────────
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR
    )

    writer = SummaryWriter(
        log_dir="runs/cifar_resnet50"
    )

    v_start_epoch, v_best_val = resume_weight(config.SAVE_CIFAR_RESNET_WEIGHT, model, optimizer)

    training_phase(
        model,
        loss_func,
        optimizer,
        train_loader,
        val_loader,
        path_save=config.SAVE_CIFAR_RESNET_WEIGHT,
        epoch=5,
        writer=writer
    )

    

    # ── Training Loop ─────────────────────────────────────────────
    