from torchvision.datasets import FashionMNIST
from dataset import load_raw_dataset, create_dataloader
import collections
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")   # không cần GUI window, save thành file
import config
import MyLog
import torch
import numpy as np
from object import MyMetric
from torch.utils.tensorboard import SummaryWriter
import os


writer_train = SummaryWriter("runs/normal/train")
writer_test = SummaryWriter("runs/normal/test")

def show_batch(dataset_class, loader, save_path=None, row=2, col=5, figsize=(14,5)):
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

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # tránh lỗi nếu thư mục chưa có
        plt.savefig(save_path, dpi=120)
        MyLog.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()


def show_loss_acc(train_metrics: list[MyMetric], val_metrics: list[MyMetric],
                  save_path=None):
    epochs = range(1, len(train_metrics) + 1)

    train_loss = [m.average_loss for m in train_metrics]
    val_loss   = [m.average_loss for m in val_metrics]
    train_acc  = [m.accuracy for m in train_metrics]
    val_acc    = [m.accuracy for m in val_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss, label="val", linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, label="train", marker="o")
    ax2.plot(epochs, val_acc, label="val", linestyle="--")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # tránh lỗi nếu thư mục chưa có
        plt.savefig(save_path, dpi=120)
        MyLog.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()


@torch.no_grad()
def show_predictions(model, loader, dataset_class,
                     save_path=None,
                     n=32):
    model.eval()
    images, labels = next(iter(loader))
    images_gpu = images.to(config.DEVICE)
    outputs = model(images_gpu)
    preds = outputs.argmax(dim=1).cpu()  # predicted class index
    # Denormalize để hiển thị
    mean = torch.as_tensor(config.MEAN, dtype=images.dtype).view(-1, 1, 1)
    std = torch.as_tensor(config.STD, dtype=images.dtype).view(-1, 1, 1)
    images = (images * std + mean).clamp(0, 1)
    n = min(n, images.size(0))
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, ax in enumerate(axes):
        if idx >= n:
            ax.axis("off")
            continue
        img = images[idx].squeeze().numpy()
        pred_name = dataset_class.classes[preds[idx].item()]
        true_name = dataset_class.classes[labels[idx].item()]
        correct = preds[idx].item() == labels[idx].item()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        # WHY set_title với color? Phân biệt ngay đúng/sai bằng màu
        color = "green" if correct else "red"
        ax.set_title(f"P:{pred_name}\nT:{true_name}",
                     fontsize=10, color=color, pad=2)
        # Viền đỏ = sai, không có viền = đúng
        if not correct:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # tránh lỗi nếu thư mục chưa có
        plt.savefig(save_path, dpi=120)
        MyLog.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()

def plot_class_distribution(dataset, save_path=None):
    v_class_counter = collections.Counter(dataset.targets.tolist())

    v_names = dataset.classes
    v_counts = [v_class_counter[i] for i in range(len(v_names))]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(v_names, v_counts, color="steelblue", edgecolor="white")

    ax.set_title("Class Distribution – FashionMNIST")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # tránh lỗi nếu thư mục chưa có
        plt.savefig(save_path, dpi=120)
        MyLog.print_log("Saved at", save_path)
    else:
        plt.show()

    plt.close()


def plot_mean_images(dataset, dataset_class,
                     save_path=None):
    """
    Tính ảnh trung bình mỗi class.
    WHY? Mean image cho thấy 'đặc trưng chung' mà model cần học.
    Nếu 2 mean image trông giống nhau → model dễ nhầm 2 class đó.
    """
    n_classes = len(dataset_class.classes)

    # Tách data theo class
    # dataset.data shape: (60000, 28, 28) uint8, chưa normalize
    data = dataset.data.float() / 255.0  # → [0, 1]
    labels = dataset.targets

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Mean Image Per Class", fontsize=13, fontweight="bold")

    for class_idx, ax in enumerate(axes.flat):
        mask = (labels == class_idx)
        mean_img = data[mask].mean(dim=0)  # average over N samples → (28, 28)

        ax.imshow(mean_img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(dataset_class.classes[class_idx], fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        MyLog.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()

def plot_pixel_distribution(dataset, dataset_class,
                            save_path=None):
    data = dataset.data.float() / 255.0  # (60000, 28, 28)
    labels = dataset.targets

    fig, ax = plt.subplots(figsize=(12, 5))

    for class_idx, name in enumerate(dataset_class.classes):
        mask = (labels == class_idx)
        pixels = data[mask].flatten().numpy()

        # Plot density histogram cho mỗi class
        ax.hist(pixels, bins=50, alpha=0.4, density=True, label=name)

    ax.set_title("Pixel Intensity Distribution Per Class")
    ax.set_xlabel("Pixel value [0, 1]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        MyLog.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()

def plot_pixel_distribution_adjust(dataset, dataset_class,
                             save_path=None):
    data   = dataset.data.float() / 255.0
    labels = dataset.targets

    fig, ax = plt.subplots(figsize=(12, 5))

    for class_idx, name in enumerate(dataset_class.classes):
        mask   = (labels == class_idx)
        pixels = data[mask].flatten().numpy()

        # WHY lọc > 0.05: bỏ background đen
        # Chỉ giữ pixels thuộc vùng vải thật sự
        foreground = pixels[pixels > 0.05]

        ax.hist(foreground, bins=50, alpha=0.4,
                density=True, label=name)

    ax.set_title("Pixel Intensity – Foreground Only (pixel > 0.05)")
    ax.set_xlabel("Pixel value [0, 1]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        MyLog.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    # Load dataset
    train_dataset, test_dataset = load_raw_dataset(FashionMNIST)
    MyLog.dataset_log(FashionMNIST, train_dataset, test_dataset)

    # Dataloader
    train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
    MyLog.dataloader_log(train_dataloader, test_dataloader)

    plot_mean_images(train_dataset, FashionMNIST)

    plot_pixel_distribution_adjust(train_dataset, FashionMNIST)