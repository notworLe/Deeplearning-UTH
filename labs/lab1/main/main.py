from src.utility import set_seed
import src.log as log
import src.config as config
from src.dataset import load_raw_dataset, create_dataloader
from src.visualize import show_batch, show_loss_acc, show_predictions, plot_class_distribution, plot_classification_metrics, _finish
from src.model import FashionCNN, training_phase, test_phase
from src.utility import resume_weight

from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
import torch.optim as optim
import torch.nn as nn
import typer
from torchsummary import summary
from src import config
import torchvision.transforms as transform
from pathlib import Path

def main_cli():
    pass

if __name__ == '__main__':
    NAME = "fashion-normal"
    IMAGE_RESULT = "fashion-normal"
    SAVE_WEIGHT = Path("weights/fashion-normal.pt")
    # Tensorboard
    p_writer = SummaryWriter("runs/fashion-normal")

    # Initial space
    log.log_creation()
    set_seed()

    # Load dataset
    train_dataset, test_dataset = load_raw_dataset(
        FashionMNIST, config.CIFAR_PATH,
        p_train_transform=config.AUGMENTED_AI_VIETNGUYEN,
        p_test_transform=config.MY_MODEL_TRANSFORM
    )
    log.dataset_log(FashionMNIST, train_dataset, test_dataset)

    # Dataloader
    train_dataloader, test_dataloader = create_dataloader(train_dataset, test_dataset)
    log.dataloader_log(train_dataloader, test_dataloader)

    ## Visual batch
    show_batch(FashionMNIST, train_dataloader, save_path=f"result/{IMAGE_RESULT}/data-visual/show-batch.png")
    plot_class_distribution(train_dataset, save_path=f"result/{IMAGE_RESULT}/data-visual/clas-distribution-train.png")
    plot_class_distribution(test_dataset, save_path=f"result/{IMAGE_RESULT}/data-visual/clas-distribution-test.png")


    # Model
    model = FashionCNN().to(config.DEVICE)
    ## Model metadata
    # summary(model, (1, 28, 28), device=config.DEVICE)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    v_start_epoch, v_best_val = resume_weight(SAVE_WEIGHT, model, optimizer)


    # Training phase
    # list_metric_train_epoch, list_metric_eval_epoch = training_phase(
    #     model,
    #     loss_func,
    #     optimizer,
    #     train_dataloader,
    #     test_dataloader,
    #     SAVE_WEIGHT,
    #     config.EPOCH,
    #     p_writer,
    #     classes=FashionMNIST.classes,
    #     p_best_val= v_best_val
    # )

    # Test phase
    v_obj_eval_metric = test_phase(model, test_dataloader, p_writer)

    # Visualize
    # show_loss_acc(list_metric_train_epoch, list_metric_eval_epoch, save_path=f"result/{IMAGE_RESULT}/metric/loss-epoch.png")
    # show_predictions(model, test_dataloader, FashionMNIST, save_path=f"result/{IMAGE_RESULT}/metric/accuracy-epoch.png")
    plot_classification_metrics(
        v_obj_eval_metric.list_actual,
        v_obj_eval_metric.list_prediction,
        FashionMNIST.classes,
        save_path=f"result/{IMAGE_RESULT}/metric/pre-recall-f1-acc.png"
    )
    v_cf_matrix = v_obj_eval_metric.get_fig_cf_matrix(FashionMNIST.classes)
    _finish(v_cf_matrix, save_path=f"result/{IMAGE_RESULT}/metric/confusion-matrix.png")

    p_writer.close()
