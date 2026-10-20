import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # không cần GUI window, save thành file
import config
import MyLog
import torch
import numpy as np
from object import MyMetric
from torch.utils.tensorboard import SummaryWriter
import os


writer = SummaryWriter("runs/exp1")

def show_batch(dataset_class, loader, save_path="data/image/image_batches.png", row=2, col=5, figsize=(14,5)):
    images, labels = next(iter(loader))

    # WHY *0.5 + 0.5?
    # Ảnh đang ở [-1, 1] sau Normalize.
    # Matplotlib imshow cần [0, 1] → đảo ngược công thức Normalize
    mean = torch.as_tensor(config.MEAN, dtype=images.dtype).view(-1, 1, 1)
    std = torch.as_tensor(config.STD, dtype=images.dtype).view(-1, 1, 1)
    images = (images * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(dataset_class.classes[label.item()], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    MyLog.print_log("Saved at", save_path)

def show_loss_acc(train_metrics: list[MyMetric], val_metrics: list[MyMetric],
                  save_path="data/image/loss_acc_visual.png"):
    epochs = range(1, len(train_metrics) + 1)

    train_loss = [m.average_loss for m in train_metrics]
    val_loss   = [m.average_loss for m in val_metrics]
    train_acc  = [m.accuracy for m in train_metrics]
    val_acc    = [m.accuracy for m in val_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="train", marker="o")
    ax1.plot(epochs, val_loss, label="val", linestyle="--", marker="o")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, label="train", marker="o")
    ax2.plot(epochs, val_acc, label="val", linestyle="--", marker="o")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)   # tránh lỗi nếu thư mục chưa có
    plt.savefig(save_path, dpi=120)
    plt.close()
    MyLog.print_log("Plot saved", save_path)
