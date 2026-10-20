import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # không cần GUI window, save thành file
import config
import MyLog
import torch
import numpy as np
from object import MyMetric
from torch.utils.tensorboard import SummaryWriter

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

def show_loss_acc(metric: MyMetric, save_path="data/image/loss_acc_visual.png", row=2, col=5, figsize=(14,5)):
    epochs = range(1, len(metric) + 1)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
