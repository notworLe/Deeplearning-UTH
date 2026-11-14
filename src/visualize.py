from torchvision.datasets import FashionMNIST
import collections
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")   # không cần GUI window, save thành file
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import math

from src.dataset import load_raw_dataset, create_dataloader
from src import logs
from src.objects import MyMetric
import src.config as config

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---- Style dùng chung: 1 bảng màu, 1 cỡ chữ cho từng cấp -------------------
INK = "#1f2937"        # chữ chính
MUTED = "#6b7280"      # chữ phụ
PANEL = "#f3f4f6"      # nền thẻ KPI
ALERT = "#b91c1c"      # điểm cần chú ý (class yếu nhất)
CMAP = "Blues"
BAR = "#3b82f6"
ALERT = "#b91c1c"


def _finish(fig, save_path):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved at", save_path)
    else:
        plt.show()
    plt.close(fig)


def _unwrap(dataset):
    """Subset/random_split (kể cả lồng nhau) -> (dataset gốc, chỉ số trong dataset gốc)."""
    indices = None
    while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        idx = np.asarray(dataset.indices)
        indices = idx if indices is None else idx[indices]
        dataset = dataset.dataset
    return dataset, indices


def _get_targets(ds):
    for attr in ("targets", "_labels", "labels"):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    return np.asarray([ds[i][1] for i in range(len(ds))])  # chậm nhưng luôn chạy


def _get_class_names(ds, targets):
    if hasattr(ds, "classes"):
        return list(ds.classes)
    return [str(c) for c in sorted(np.unique(targets))]


def _get_image(ds, i):
    """Lấy ảnh gốc (chưa transform) dạng numpy HxW hoặc HxWxC để hiển thị."""
    if hasattr(ds, "data"):  # MNIST, CIFAR, ...
        img = ds.data[i]
        return img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    for attr in ("_images", "images", "_samples", "samples", "imgs"):
        if hasattr(ds, attr):  # OxfordIIITPet, ImageFolder, ...
            item = getattr(ds, attr)[i]
            path = item[0] if isinstance(item, (tuple, list)) else item
            return np.asarray(Image.open(path).convert("RGB"))
    img = ds[i][0]  # fallback: có thể đã transform
    if hasattr(img, "numpy"):
        img = img.numpy()
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if img.shape[-1] == 1:
            img = img[..., 0]
    return np.asarray(img)


def _counts(dataset):
    """
    Trả về (tên class, số lượng mỗi class) cho dataset / Subset / DataLoader.
    Không cần .targets: tự tìm targets / _labels / labels.
    """
    if hasattr(dataset, "batch_size") and hasattr(dataset, "dataset"):  # DataLoader
        dataset = dataset.dataset
    base, idx = _unwrap(dataset)
    targets = _get_targets(base)
    names = _get_class_names(base, targets)
    if idx is not None:  # chỉ đếm các mẫu thuộc Subset
        targets = targets[idx]
    return names, np.bincount(targets, minlength=len(names))


def plot_class_distribution(dataset, title="Class distribution", sort=True,
                            save_path=None):
    """
    Số ảnh mỗi class (1 tập). Thanh ngang -> đọc được tên dài, 37 class vẫn gọn.
    sort=True: class nhiều ảnh nhất ở trên cùng -> thấy ngay mất cân bằng.
    """
    names, counts = _counts(dataset)
    n = len(names)
    order = np.argsort(-counts) if sort else np.arange(n)
    names = [names[i] for i in order]
    counts = counts[order]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * n + 1.6)),
                           layout="constrained")
    y = np.arange(n)
    ax.barh(y, counts, color=BAR, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8 if n > 15 else 10, color=INK)
    ax.invert_yaxis()

    pad = counts.max() * 0.01
    for yi, c in zip(y, counts):
        ax.text(c + pad, yi, str(c), va="center", fontsize=7 if n > 15 else 9,
                color=MUTED)

    mean = counts.mean()
    ax.axvline(mean, color=MUTED, linestyle="--", linewidth=1)
    ax.text(mean, -0.9, f"TB = {mean:.0f}", color=MUTED, fontsize=8,
            ha="center", va="bottom")

    ax.set_xlim(0, counts.max() * 1.1)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    ratio = counts.max() / max(counts.min(), 1)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK,
                 x=0.0, ha="left")
    ax.set_title(f"{counts.sum()} ảnh · {n} class · ít nhất {counts.min()}, "
                 f"nhiều nhất {counts.max()} (chênh {ratio:.2f}×)",
                 fontsize=9, color=MUTED, loc="left")
    _finish(fig, save_path)
    return dict(zip(names, counts.tolist()))


def plot_split_class_distribution(splits, save_path=None):
    """
    So sánh phân bố class giữa train / val / test trong MỘT hình.
    splits: dict, vd {"train": v_train, "val": v_val, "test": v_test}
    Mỗi tập một panel, cùng thứ tự class, trục x là % của tập đó
    (để so được dù tập to nhỏ khác nhau). Đường đứt = chia đều 100/n %.
    Các thanh ở 3 panel có độ dài gần bằng nhau = chia tập cân bằng.
    """
    all_counts = {k: _counts(v) for k, v in splits.items()}
    names = next(iter(all_counts.values()))[0]
    n = len(names)
    first = next(iter(all_counts.values()))[1]
    order = np.argsort(-first)  # theo tập đầu tiên

    k = len(splits)
    fig, axes = plt.subplots(1, k, sharey=True, sharex=True,
                             figsize=(3.2 * k + 1.5, max(3.5, 0.28 * n + 1.6)),
                             layout="constrained")
    axes = np.atleast_1d(axes)

    pcts = {s: c / c.sum() * 100 for s, (_, c) in all_counts.items()}
    xmax = max(p.max() for p in pcts.values()) * 1.15

    for ax, (split, (_, c)) in zip(axes, all_counts.items()):
        ax.barh(np.arange(n), pcts[split][order], color=BAR, height=0.7)
        ax.axvline(100 / n, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(f"{split}  (n={c.sum()})", fontsize=10,
                     fontweight="bold", color=INK, loc="left")
        ax.set_xlim(0, xmax)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels([names[i] for i in order],
                            fontsize=8 if n > 15 else 10, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Class distribution by split (% mỗi tập)", fontsize=13,
                 fontweight="bold", color=INK, x=0.0, ha="left")
    _finish(fig, save_path)



def show_class_samples(dataset, classes=None, n=10, n_cols=5,
                       shuffle=True, seed=0, cell_size=1.6, save_path=None):
    """
    Hiển thị n ảnh mẫu cho mỗi class được chọn.

    dataset  : dataset torchvision bất kỳ (FashionMNIST, CIFAR, OxfordIIITPet,
               ImageFolder...) hoặc Subset của chúng. Truyền INSTANCE, không
               phải class.
    classes  : None -> tất cả class | int/str -> 1 class | list -> nhiều class
    n        : số ảnh mỗi class
    n_cols   : số cột của lưới ảnh mỗi class
    shuffle  : True -> ngẫu nhiên (cố định theo seed); False -> n ảnh đầu tiên
    """
    base, subset_idx = _unwrap(dataset)
    all_targets = _get_targets(base)
    names = _get_class_names(base, all_targets)

    # Vị trí (trong dataset gốc) của các mẫu đang xét
    pool = np.arange(len(all_targets)) if subset_idx is None else subset_idx
    targets = all_targets[pool]

    if classes is None:
        classes = list(range(len(names)))
    elif isinstance(classes, (int, str)):
        classes = [classes]
    idxs = [names.index(c) if isinstance(c, str) else int(c) for c in classes]

    rng = np.random.default_rng(seed)
    n_rows = math.ceil(n / n_cols)
    grid_h = n_rows * cell_size
    fig = plt.figure(figsize=(n_cols * cell_size, len(idxs) * (grid_h + 0.5)),
                     layout="constrained")

    subfigs = np.atleast_1d(fig.subfigures(len(idxs), 1, hspace=0.04))

    for sf, ci in zip(subfigs, idxs):
        mask = np.flatnonzero(targets == ci)
        k = min(n, len(mask))
        chosen = rng.choice(mask, k, replace=False) if shuffle else mask[:k]

        sf.suptitle(f"{names[ci]}  ({k} / {len(mask)} ảnh)",
                    fontsize=11, fontweight="bold", color=INK,
                    x=0.0, ha="left")
        axes = np.atleast_1d(sf.subplots(n_rows, n_cols)).flatten()
        for j, ax in enumerate(axes):
            ax.axis("off")
            if j < k:
                img = _get_image(base, pool[chosen[j]])
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

    _finish(fig, save_path)

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
        logs.print_log("Saved at", save_path)
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
        logs.print_log("Saved at", save_path)
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
        logs.print_log("Saved at", save_path)
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
        logs.print_log("Saved at", save_path)
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
        logs.print_log("Saved at", save_path)
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
        logs.print_log("Saved at", save_path)
    else:
        plt.show()
    plt.close()






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
    logs.dataset_log(FashionMNIST, train_dataset, test_dataset)

    # Dataloader
    train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
    logs.dataloader_log(train_dataloader, test_dataloader)

    plot_mean_images(train_dataset, FashionMNIST)

    plot_pixel_distribution_adjust(train_dataset, FashionMNIST)