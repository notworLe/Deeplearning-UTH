from config import DEVICE
from utility import SEED
import torch

MAX_LENGTH = 10

def print_log(name, value):
    """
    Chuẩn hóa log terminal với MAX_LENGTH thống nhất
    :param name:
    :param value:
    :return:
    """
    len_name = len(name)
    if len_name > MAX_LENGTH:
        name = name[:-3] + "..."

    space = " " * max(0, MAX_LENGTH - len_name)
    print(f"{name[:MAX_LENGTH]}{space}: {value}")

def log_device():
    print_log("Device", DEVICE)
    if DEVICE.type == "cuda":
        print_log("Gpu", torch.cuda.get_device_name(DEVICE))
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_log("VRAM", f"{vram:.1f} GB")

def log_seed():
    print_log("Seed", SEED)

def log_creation():
    log_device()
    log_seed()