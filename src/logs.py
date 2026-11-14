import src.utility as utility
import src.config as config
import torch

def print_log(name, value, max_length = config.MAX_LENGTH):
    """
    Chuẩn hóa log terminal với MAX_LENGTH thống nhất
    :param name:
    :param value:
    :param max_length:
    :return:
    """
    len_name = len(name)
    if len_name > max_length:
        name = name[:-3] + "..."

    space = " " * max(0, max_length - len_name)
    print(f"{name[:max_length]}{space}: {value}")

def log_device():
    print_log("Device", config.DEVICE)
    if config.DEVICE.type == "cuda":
        print_log("GPU", torch.cuda.get_device_name(config.DEVICE))

        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_log("VRAM", f"{vram:.1f} GB")

def log_seed():
    print_log("Seed", config.SEED)

def log_dataset_config():
    print_log("Workers", config.WOKER)
    print_log("Batch size", config.BATCH_SIZE)


def log_creation():
    print("=" * 32 + " Configration " + "=" * 32)
    log_device()
    log_seed()
    log_dataset_config()

    print()


def dataset_log(dataset, train_dataset, test_dataset, idx=0):
    print("=" * 32 + " Dataset information " + "=" * 32)

    # Medata
    print_log("Dataset name", str(dataset), 20)
    print_log("Training dataset size", len(train_dataset), 20)
    print_log("Test dataset size", len(test_dataset), 20)

    img, label = test_dataset[idx]
    print_log("Label index", label, 20)
    print_log("Label name", dataset.classes[label], 20)
    print_log("Image shape", img.shape, 20)
    print_log("Min - max image", f"{img.min()}, {img.max()}", 20)

def dataset_log_v2(class_name: dict, test_dataset, idx=0, **kwargs):
    print("=" * 32 + " Dataset information " + "=" * 32)

    # Medata
    for key, val in kwargs.items():
        print_log(f"{key} size", len(val))

    print_log(f"Test size", len(test_dataset))

    img, label = test_dataset[idx]
    print_log("Num of class", len(test_dataset.classes), 20)
    print_log("Label index", label, 20)
    print_log("Label name", class_name[label], 20)
    print_log("Image shape", img.shape, 20)
    print_log("Min - max image", f"{img.min()}, {img.max()}", 20)


def dataloader_log(train_dataloader, test_dataloader, max_length=20):
    print()
    print_log("Number of train dataloader:", len(train_dataloader), max_length)
    print_log("Number of test dataloader:", len(test_dataloader), max_length)

def dataloader_log_v2(max_length=20, **kwargs):
    for val, key in kwargs.items():
        print_log(f"Number of P{val}", f"{len(key)} batch", max_length)

