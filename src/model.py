import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm.autonotebook import tqdm

from src.utility import save_weight
from src.objects import MyMetric
from src import logs
import src.config as config



class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
    
        # Input N x 1 x 28 x 28
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # Input N x 32 x 14 x 14
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # Input N x 64, 7, 7
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Input N x 128 x 7 x 7
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Input N X 128 x 1 x 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def training_phase(
        model,
        loss_func,
        optimizer,
        train_dataloader,
        test_dataloader,
        path_save:Path,
        epoch=config.EPOCH,
        p_start_epoch=0,
        writer=None,
        classes=None,
        p_best_val = 0.0
) -> tuple[list[MyMetric], list[MyMetric]] | None:
    # Parameters
    path_save = Path(path_save)

    list_metric_train_epoch: list[MyMetric] = []
    list_metric_val_epoch: list[MyMetric] = []

    for i_epoch in range(p_start_epoch, epoch):
        model.train()
        epoch_metric = MyMetric()

        n_batch = len(train_dataloader)
        progress_bar = tqdm(train_dataloader)
        for i_batch, (batch, labels) in enumerate(progress_bar):
            batch = batch.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Forward
            optimizer.zero_grad()
            predictions = model(batch)
            loss_value = loss_func(predictions, labels)

            # Backward
            loss_value.backward()
            optimizer.step()

            # Save metric
            epoch_metric.add(predictions, labels, loss_value, len(batch))

            # Log
            if i_batch % config.LOG_PEAR_BATCH == 0 or i_batch == n_batch - 1:
                progress_bar.set_description(f"Train: Epoch {i_epoch + 1}/{epoch}. Iteration {i_batch + 1}/{n_batch}. Loss: {loss_value:.3f}")
        
        eval_metric = evaluate_phase(i_epoch, epoch, model, loss_func, test_dataloader, classes, p_writer=writer)
        print()


        if writer:
            writer.add_scalars("01. Loss", {
                "train": epoch_metric.average_loss,
                "val":   eval_metric.average_loss,
            }, i_epoch)
            writer.add_scalars("02. Accuracy", {
                "train": epoch_metric.accuracy,
                "val":   eval_metric.accuracy,
            }, i_epoch)
             
            

        list_metric_train_epoch.append(epoch_metric)
        list_metric_val_epoch.append(eval_metric)
        if eval_metric.accuracy > p_best_val:
            logs.print_log("Accuracy enhanced", f'{p_best_val} -> {eval_metric.accuracy}')
            p_best_val = eval_metric.accuracy
            save_weight(path_save, i_epoch + 1, model, optimizer, p_best_val)



    return list_metric_train_epoch, list_metric_val_epoch



@torch.no_grad()
def evaluate_phase(i_epoch, epoch, model, loss_func, test_dataloader, classes=None, p_writer=None):
    model.eval()
    epoch_metric = MyMetric()

    n_batch = len(test_dataloader)

    progress_bar = tqdm(test_dataloader)
    for i_batch, (batch, labels) in enumerate(progress_bar):
        batch = batch.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        predictions = model(batch)
        loss_value = loss_func(predictions, labels)

        # Save metrics
        epoch_metric.add(predictions, labels, loss_value, len(batch))

        # Log
        if i_batch % config.LOG_PEAR_BATCH == 0 or i_batch == n_batch - 1:
            progress_bar.set_description(
                f"Eval: Epoch {i_epoch + 1}/{epoch}. Iteration {i_batch + 1}/{n_batch}. Loss: {loss_value:.3f}")
    if p_writer:
        fig_cf_matrix = epoch_metric.get_fig_cf_matrix(classes)
        p_writer.add_figure("03. Confusion matrix", fig_cf_matrix, i_epoch)

    return epoch_metric

def test_phase(model, test_dataloader, p_writer):
    print("=" * 32 + " Test phase " + "=" * 32)
    model.eval()
    v_obj_metric = MyMetric()

    for images, labels in test_dataloader:
        # Device
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        v_raw_predictions = model(images)
        v_list_predict = v_raw_predictions.argmax(dim=1)
        
        v_obj_metric.add(v_list_predict, labels, 0, 0)
    # if p_writer:
    #     p_writer.add_scalar("Precision", v_obj_metric.get_precision(average="macro", zero_division=0))
    #     p_writer.add_scalar("Recall", v_obj_metric.get_recall(average="macro", zero_division=0))
    #     p_writer.add_scalar("F1", v_obj_metric.get_f1(average="macro", zero_division=0))

    return v_obj_metric


def my_metric(predictions, actual, total_loss, total_samples):
    predictions = np.array(predictions)
    actual = np.array(actual)

    # Metrics
    average_loss = total_loss / total_samples
    accuracy = (predictions == actual).sum() / len(predictions)

    # Log
    logs.print_log("Average loss", f"{average_loss:.4f}")
    logs.print_log("Accuracy ", f"{accuracy:.4f}")

def save_metric(actual_label, predict_label, total_loss, total_samples):
    pass

if __name__ == '__main__':
    data = torch.rand((32, 1, 28, 28))

    model = FashionCNN().to(config.DEVICE)
    loss_func = nn.CrossEntropyLoss
    optimizer  = optim.Adam(model.parameters(), lr=config.LR)

    model.train()
    out = model(data)
    print(out.shape)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")