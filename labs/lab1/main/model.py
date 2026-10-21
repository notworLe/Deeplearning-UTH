import numpy as np
import torch
import torch.nn as nn
import config
import torch.optim as optim
import MyLog
from object import MyMetric
from MyVisualize import writer_train, writer_test
from MyUtility import save_weight
from pathlib import Path

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
        epoch=config.EPOCH,
        path_save = "weights/model.pt"
) -> tuple[list[MyMetric], list[MyMetric]] | None:
    # Parameters
    path_save = Path(path_save)

    # Variables
    v_best_val = 0.0

    # Load weights
    if path_save.exists():
        v_dict_weight = torch.load(path_save, map_location=config.DEVICE)

        model.load_state_dict(v_dict_weight['model'])
        optimizer.load_state_dict(v_dict_weight.get('optimizer'))
        v_best_val = v_dict_weight['best_validation']

        MyLog.print_log("Resume from", path_save)
        MyLog.print_log("Accuracy", v_best_val)


    list_metric_train_epoch: list[MyMetric] = []
    list_metric_val_epoch: list[MyMetric] = []

    for i_epoch in range(epoch):
        MyLog.print_log("Epoch", i_epoch + 1)
        print("=" * 20 + " Training " + "=" * 20)

        model.train()
        epoch_metric = MyMetric()

        for batch, labels in train_dataloader:
            batch = batch.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            predictions = model(batch)
            loss_value = loss_func(predictions, labels)
            loss_value.backward()
            optimizer.step()

            # Save metric
            epoch_metric.add(predictions, labels, loss_value, len(batch))

        epoch_metric.show_metric()

        print("=" * 20 + " Evaluation " + "=" * 20)
        eval_metric = evaluate_phase(model, loss_func, test_dataloader)

        print()

        writer_train.add_scalar("Loss", epoch_metric.average_loss, i_epoch + 1)
        writer_train.add_scalar("Accuracy", epoch_metric.accuracy, i_epoch + 1)
        writer_test.add_scalar("Loss", eval_metric.average_loss, i_epoch + 1)
        writer_test.add_scalar("Accuracy", eval_metric.accuracy, i_epoch + 1)

        list_metric_train_epoch.append(epoch_metric)
        list_metric_val_epoch.append(eval_metric)

        if eval_metric.accuracy > v_best_val:
            MyLog.print_log("Accuracy enhanced", f'{v_best_val} -> {eval_metric.accuracy}')
            v_best_val = eval_metric.accuracy
            save_weight(path_save, i_epoch, model, optimizer, v_best_val)

    writer_train.close()
    writer_test.close()
    return list_metric_train_epoch, list_metric_val_epoch



@torch.no_grad()
def evaluate_phase(model, loss_func, test_dataloader):
    model.eval()
    epoch_metric = MyMetric()

    for batch, labels in test_dataloader:
        batch = batch.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        predictions = model(batch)
        loss_value = loss_func(predictions, labels)

        # Save metrics
        epoch_metric.add(predictions, labels, loss_value, len(batch))

    epoch_metric.show_metric()

    return epoch_metric

def my_metric(predictions, actual, total_loss, total_samples):
    predictions = np.array(predictions)
    actual = np.array(actual)

    # Metrics
    average_loss = total_loss / total_samples
    accuracy = (predictions == actual).sum() / len(predictions)

    # Log
    MyLog.print_log("Average loss", f"{average_loss:.4f}")
    MyLog.print_log("Accuracy ", f"{accuracy:.4f}")

def save_metric(actual_label, predict_label, total_loss, total_samples):
    passs

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