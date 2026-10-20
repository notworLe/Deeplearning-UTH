import numpy as np
import torch
import torch.nn as nn
import config
import torch.optim as optim
import MyLog
from object import MyMetricManage, MyMetric


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

def training_phase(model, loss_func, optimizer, train_dataloader, test_dataloader, epoch=config.EPOCH):

    for i_epoch in range(epoch):
        metric_batch = ()

        model.train()
        MyLog.print_log("Epoch", i_epoch + 1)
        for batch, labels in train_dataloader:
            batch = batch.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            predictions = model(batch)
            loss_value = loss_func(predictions, labels)
            loss_value.backward()
            optimizer.step()

            # Save metrics
            predictions_arg = predictions.argmax(dim=1)
            metric_batch.add(predictions_arg, labels, loss_value, len(batch))

        metric_batch.get_metric()
        metric_batch.show_metric()
        metric_epoch.add(
            [x for batch in metric_batch.prediction for x in batch],
            [x for batch in metric_batch.actual for x in batch],
            metric_batch.total_loss,
            metric_batch.total_image
        )

        print("======= Evaluation ========")
        evaluate_phase(model, loss_func, test_dataloader)

    return metric_epoch

@torch.no_grad()
def evaluate_phase(model, loss_func, test_dataloader):
    model.eval()
    metric = MyMetric()

    for batch, labels in test_dataloader:
        batch = batch.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        predictions = model(batch)
        loss_value = loss_func(predictions, labels)

        # Save metrics
        predictions_arg = predictions.argmax(dim=1)
        metric.add(predictions_arg, labels, loss_value, len(batch))

    metric.get_metric()
    metric.show_metric()

    return metric

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