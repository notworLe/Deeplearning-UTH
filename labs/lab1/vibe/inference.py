#!/usr/bin/env python3
"""
inference.py — Production inference script for Fashion-MNIST classification.
Supports single sample prediction with Top-3 probability output and batch inference.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class FashionCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

@torch.inference_mode()
def predict_sample(model, test_dataset, idx, device):
    img_tensor, true_label = test_dataset[idx]
    input_batch = img_tensor.unsqueeze(0).to(device)
    logits = model(input_batch)
    probs = torch.softmax(logits, dim=1)[0]
    top3_probs, top3_indices = probs.topk(3)
    
    print(f"\n==========================================")
    print(f"Sample Index  : {idx}")
    print(f"Ground Truth  : {CLASSES[true_label]} (Class ID {true_label})")
    print(f"Top-1 Predict : {CLASSES[top3_indices[0]]} (Confidence: {top3_probs[0]*100:.2f}%)")
    print(f"Result Status : {'CORRECT [OK]' if top3_indices[0] == true_label else 'MISCLASSIFIED [X]'}")
    print(f"------------------------------------------")
    print(f"Top-3 Probabilities Distribution:")
    for rank, (p, c) in enumerate(zip(top3_probs, top3_indices), 1):
        print(f"  {rank}. {CLASSES[c]:<15} : {p*100:6.2f}%")
    print(f"==========================================\n")

def main():
    parser = argparse.ArgumentParser(description="Inference with trained FashionCNN model")
    parser.add_argument("--checkpoint", type=str, default="../main/weights/fashion-normal.pt", help="Path to checkpoint")
    parser.add_argument("--sample-idx", type=int, default=18, help="Sample index in test set to predict")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}")

    # Load model architecture
    model = FashionCNN().to(device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"[*] Loading weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
    else:
        print(f"[!] Warning: Checkpoint not found at {ckpt_path}. Running with uninitialized weights.")

    model.eval()

    # Load test dataset
    MEAN, STD = (0.2860,), (0.3530,)
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    test_dataset = FashionMNIST(root="data", train=False, download=True, transform=test_tf)

    predict_sample(model, test_dataset, args.sample_idx, device)

if __name__ == "__main__":
    main()

