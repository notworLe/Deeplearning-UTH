from utility import set_seed
import torch
import MyLog
import config
from torchvision.datasets import FashionMNIST
from dataset import load_raw_dataset, create_dataloader
from MyVisualize import show_batch
from model import FashionCNN, training_phase
import torch.optim as optim
import torch.nn as nn


if __name__ == '__main__':
    # Initial space
    MyLog.log_creation()
    set_seed()

    # Load dataset
    train_dataset, test_dataset = load_raw_dataset(FashionMNIST)
    MyLog.dataset_log(FashionMNIST, train_dataset, test_dataset)

    # Dataloader
    train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
    MyLog.dataloader_log(train_dataloader, test_dataloader)

    ## Visual batch
    show_batch(FashionMNIST, train_dataloader)
    print()

    # Model
    model = FashionCNN().to(config.DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    metric_epoch = training_phase(model, loss_func, optimizer, train_dataloader, test_dataloader, epoch=5)
    print(len(metric_epoch))


