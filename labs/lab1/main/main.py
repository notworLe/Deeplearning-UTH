from utility import set_seed
import torch
import MyLog
import config
from torchvision.datasets import FashionMNIST
from dataset import load_raw_dataset, create_dataloader, AUGMENTED_TRANSFORM, AUGMENTED_TRANSFORM_LOW
from MyVisualize import show_batch, show_loss_acc, show_predictions, plot_class_distribution
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
    show_batch(FashionMNIST, train_dataloader, save_path="data/image/image_batches.png")
    print()

    plot_class_distribution(train_dataset, save_path="data/image/class_distribution.png")


    # Model
    model = FashionCNN().to(config.DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    best_val = 0


    # Training phase
    list_metric_train_epoch, list_metric_eval_epoch = training_phase(
        model,
        loss_func,
        optimizer,
        train_dataloader,
        test_dataloader,
        30,
        config.SAVE_FASHION_CNN_WEIGHT
    )

    # Visualize
    show_loss_acc(list_metric_train_epoch, list_metric_eval_epoch, save_path="data/image/augmented_low/loss_acc_visual.png")
    show_predictions(model, test_dataloader, FashionMNIST, save_path="data/image/augmented_low/predictions.png")

