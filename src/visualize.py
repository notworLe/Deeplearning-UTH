from torchvision.datasets import FashionMNIST
import collections
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")   # không cần GUI window, save thành file
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from src.dataset import load_raw_dataset, create_dataloader
import src.log as log
from src.object import MyMetric
import src.config as config

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---- Style dùng chung: 1 bảng màu, 1 cỡ chữ cho từng cấp -------------------
INK = "#1f2937"        # chữ chính
MUTED = "#6b7280"      # chữ phụ
PANEL = "#f3f4f6"      # nền thẻ KPI
ALERT = "#b91c1c"      # điểm cần chú ý (class yếu nhất)
CMAP = "Blues"


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
        log.print_log("Saved at", save_path)
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
        log.print_log("Saved at", save_path)
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
        log.print_log("Saved at", save_path)
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
        log.print_log("Saved at", save_path)
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
        log.print_log("Saved at", save_path)
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
        log.print_log("Saved at", save_path)
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
        log.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()



def _finish(fig, save_path):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved at", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_classification_metrics(y_true, y_pred, class_names, save_path=None):
    """
    1 hình, đọc từ trên xuống:
      (1) 4 thẻ KPI: Accuracy + Macro Precision / Recall / F1  -> bức tranh tổng
      (2) Heatmap class x {Precision, Recall, F1}, xếp F1 tăng dần
          -> class yếu nhất nằm trên cùng, tô đỏ nhãn
    Không có thanh chồng nhau, mỗi con số đều ghi thẳng trong ô.
    """
    n = len(class_names)
    labels = list(range(n))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    table = np.stack([p, r, f1], axis=1)  # (n, 3)

    order = np.argsort(f1)  # yếu nhất lên đầu
    table = table[order]
    names = [f"{class_names[i]}  (n={support[i]})" for i in order]

    fig = plt.figure(figsize=(8.5, 1.6 + 0.5 * n), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.45 * n], hspace=0.08)

    # (1) KPI
    kpis = [
        ("Accuracy", acc),
        ("Macro Precision", p.mean()),
        ("Macro Recall", r.mean()),
        ("Macro F1", f1.mean()),
    ]
    for j, (title, value) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(PANEL)
        ax.set_xticks([]);
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, 0.62, f"{value:.1%}", ha="center", va="center",
                fontsize=20 if j == 0 else 16,
                fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.5, 0.18, title, ha="center", va="center",
                fontsize=9, color=MUTED, transform=ax.transAxes)

    # (2) Heatmap
    ax = fig.add_subplot(gs[1, :])
    vmin = max(0.0, np.floor(table.min() * 20) / 20 - 0.05)  # kéo giãn tương phản
    ax.imshow(table, cmap=CMAP, vmin=vmin, vmax=1.0, aspect="auto")

    mid = (vmin + 1.0) / 2
    for i in range(n):
        for j in range(3):
            v = table[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if v > mid else INK)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=10, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # đường kẻ trắng tách ô
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    worst = ax.get_yticklabels()[0]
    worst.set_color(ALERT)
    worst.set_fontweight("bold")

    fig.suptitle("Classification metrics", fontsize=13, fontweight="bold",
                 color=INK, x=0.0, ha="left")
    fig.text(0.0, -0.01,
             f"Yếu nhất: {class_names[order[0]]} (F1 = {f1[order[0]]:.3f}). "
             f"Nền đậm = điểm cao; thang bắt đầu từ {vmin:.2f}.",
             fontsize=8, color=MUTED, ha="left")

    _finish(fig, save_path)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


if __name__ == '__main__':
    # Load dataset
    train_dataset, test_dataset = load_raw_dataset(FashionMNIST)
    log.dataset_log(FashionMNIST, train_dataset, test_dataset)

    # Dataloader
    train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
    log.dataloader_log(train_dataloader, test_dataloader)

    plot_mean_images(train_dataset, FashionMNIST)

    plot_pixel_distribution_adjust(train_dataset, FashionMNIST)