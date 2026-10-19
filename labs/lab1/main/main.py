from config import DEVICE
from utility import set_seed
import torch
import MyLog


if __name__ == '__main__':
    # Initial space
    MyLog.log_creation()
    set_seed()

    # Load dataset

