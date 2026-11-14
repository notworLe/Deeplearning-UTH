#!/usr/bin/env python3
"""
train.py — Standalone CLI training script for Fashion-MNIST classification.
Adheres to AGENT_AI.md, PYTORCH_FRAMEWORK_RULES.md, and LOGGING_CHECKPOINT_RULES.md.
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split

# Class Names
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# 1. Model Definitions
class BaselineMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class FashionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 28x28 -> 14x14
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 14x14 -> 7x7
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # 7x7 -> 1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST model with PyTorch")
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn", help="Architecture to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-path", type=str, default="weights/fashion_trained.pt", help="Checkpoint save path")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} | Architecture: {args.model.upper()} | Seed: {args.seed}")

    # Transforms & Datasets
    MEAN, STD = (0.2860,), (0.3530,)
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    data_dir = "data"
    full_train = FashionMNIST(root=data_dir, train=True, download=True, transform=train_tf)
    train_sub, val_sub = random_split(full_train, [54000, 6000], generator=torch.Generator().manual_seed(args.seed))
    test_dataset = FashionMNIST(root=data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate Model
    if args.model == "cnn":
        model = FashionCNN().to(device)
    else:
        model = BaselineMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    print(f"[*] Commencing training for {args.epochs} epochs...")
    for ep in range(1, args.epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {ep:02d}/{args.epochs:02d} | Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_validation": best_val_acc
            }, args.save_path)
            print(f"  --> Saved new best checkpoint to {args.save_path} (Val Acc: {v_acc*100:.2f}%)")

    # Evaluate on test set with best checkpoint
    print(f"\n[*] Evaluating best model on 10,000 test images...")
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"[+] Final Test Result -> Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()

