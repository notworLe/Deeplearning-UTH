#!/usr/bin/env python3
"""
inference.py — Production inference script for CIFAR-10 Transfer Learning.
Supports single sample prediction with Top-3 probability output and batch inference.
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_model(model_name: str, num_classes: int = 10):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def load_checkpoint(model: nn.Module, weights_path: str, device: torch.device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")
    print(f"[*] Loading checkpoint with weights_only=True: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

@torch.inference_mode()
def predict_tensor(model: nn.Module, tensor: torch.Tensor, device: torch.device, topk: int = 3):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, k=topk)
    return top_probs.cpu().numpy(), top_indices.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Transfer Learning Inference")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vgg16", "densenet121"])
    parser.add_argument("--weights", type=str, default="labs/lab2/main/weights/resnet18_transfer.pt")
    parser.add_argument("--image", type=str, default=None, help="Path to single image for prediction")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples to evaluate if no image is given")
    parser.add_argument("--topk", type=int, default=3, help="Top-K probabilities to display")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} | Architecture: {args.model}")

    # Fallback to local paths if default weights path doesn't exist
    weights_path = args.weights
    if not os.path.exists(weights_path):
        for candidate in [
            "../main/weights/resnet18_transfer.pt",
            "weights/resnet18_transfer.pt",
            "labs/lab2/main/weights/resnet18_baseline.pt"
        ]:
            if os.path.exists(candidate):
                weights_path = candidate
                break

    model = build_model(args.model, len(CLASSES))
    model = load_checkpoint(model, weights_path, device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if args.image:
        if not os.path.exists(args.image):
            print(f"[!] Error: Image not found at {args.image}")
            return
        image = Image.open(args.image).convert("RGB")
        tensor = transform(image)
        probs, indices = predict_tensor(model, tensor, device, topk=args.topk)
        print("\n" + "=" * 55)
        print(f"PREDICTION FOR: {args.image}")
        print("=" * 55)
        for rank, (p, idx) in enumerate(zip(probs, indices), 1):
            print(f"Top {rank}: {CLASSES[idx]:<12} | Confidence: {p * 100:.2f}%")
        print("=" * 55)
    else:
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        print(f"\n[*] Evaluating {args.samples} random samples from CIFAR-10 test set:")
        print("=" * 65)
        print(f"{'Idx':<5} | {'Ground Truth':<12} | {'Top-1 Prediction':<16} | {'Confidence':<10} | {'Status'}")
        print("-" * 65)
        
        sample_indices = random.sample(range(len(test_dataset)), args.samples)
        for s_idx in sample_indices:
            img, label = test_dataset[s_idx]
            probs, indices = predict_tensor(model, img, device, topk=args.topk)
            top1_class = CLASSES[indices[0]]
            top1_prob = probs[0] * 100
            truth_class = CLASSES[label]
            status = "[CORRECT]" if top1_class == truth_class else "[WRONG]"
            print(f"{s_idx:<5} | {truth_class:<12} | {top1_class:<16} | {top1_prob:.2f}%     | {status}")
            
            # Print remaining Top-K
            for rank in range(1, args.topk):
                print(f"      |              | -> Top-{rank+1}: {CLASSES[indices[rank]]:<10} ({probs[rank]*100:.2f}%)")
            print("-" * 65)

if __name__ == "__main__":
    main()

