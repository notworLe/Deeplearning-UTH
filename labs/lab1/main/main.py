from src.utility import set_seed
import src.log as log
import src.config as config
from src.dataset import load_raw_dataset, create_dataloader, AUGMENTED_TRANSFORM, AUGMENTED_TRANSFORM_LOW
from src.visualize import show_batch, show_loss_acc, show_predictions, plot_class_distribution
from src.model import FashionCNN, training_phase
from src.utility import resume_weight


from torchvision.datasets import FashionMNIST
import torch.optim as optim
import torch.nn as nn
import typer
from torchsummary import summary


def main_cli():
    pass

if __name__ == '__main__':


    # Initial space
    log.log_creation()
    set_seed()

    # Load dataset
    train_dataset, test_dataset = load_raw_dataset(FashionMNIST)
    log.dataset_log(FashionMNIST, train_dataset, test_dataset)

    # Dataloader
    train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
    log.dataloader_log(train_dataloader, test_dataloader)

    ## Visual batch
    show_batch(FashionMNIST, train_dataloader, save_path="data/image/image_batches.png")
    print()

    plot_class_distribution(train_dataset, save_path="data/image/class_distribution.png")


    # Model
    model = FashionCNN().to(config.DEVICE)
    ## Model metadata
    summary(model, (1, 28, 28), device=config.DEVICE)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    v_start_epoch, v_best_val = resume_weight(config.SAVE_FASHION_CNN_WEIGHT, model, optimizer)



    # Training phase
    list_metric_train_epoch, list_metric_eval_epoch = training_phase(
        model,
        loss_func,
        optimizer,
        train_dataloader,
        test_dataloader,
        30,
        config.SAVE_FASHION_CNN_WEIGHT,
        classes=FashionMNIST.classes,
        p_best_val= v_best_val
    )

    # Visualize
    show_loss_acc(list_metric_train_epoch, list_metric_eval_epoch, save_path="data/image/augmented_low/loss_acc_visual.png")
    show_predictions(model, test_dataloader, FashionMNIST, save_path="data/image/augmented_low/predictions.png")

