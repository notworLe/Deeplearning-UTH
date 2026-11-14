#!/usr/bin/env python3
"""
train.py — Standalone CLI training script for CIFAR-10 Transfer Learning.
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
import torchvision.models as models
from torch.utils.data import DataLoader

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(seed: int = 248):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_model(model_name: str, mode: str, num_classes: int = 10):
    weights = None if mode == "scratch" else "DEFAULT"
    
    if model_name == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights else None)
        except Exception:
            model = models.resnet18(pretrained=(weights is not None))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        if mode == "freeze":
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif mode == "finetune":
            # Freeze conv1, bn1, layer1, layer2; unfreeze layer3, layer4, fc
            for name, child in model.named_children():
                if name in ['conv1', 'bn1', 'layer1', 'layer2']:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True

    elif model_name == "vgg16":
        try:
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if weights else None)
        except Exception:
            model = models.vgg16(pretrained=(weights is not None))
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
        if mode == "freeze":
            for param in model.features.parameters():
                param.requires_grad = False
        elif mode == "finetune":
            for param in model.features[:20].parameters():
                param.requires_grad = False

    elif model_name == "densenet121":
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if weights else None)
        except Exception:
            model = models.densenet121(pretrained=(weights is not None))
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
        if mode == "freeze":
            for param in model.features.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Transfer Learning Trainer")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vgg16", "densenet121"])
    parser.add_argument("--mode", type=str, default="finetune", choices=["finetune", "freeze", "scratch"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./weights")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} | Model: {args.model} | Mode: {args.mode}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    model = build_model(args.model, args.mode, num_classes=len(CLASSES)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total params: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.model}_{args.mode}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Eval
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Acc: {best_acc:.2f}%)")

    print(f"[*] Training finished! Best Val Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()

