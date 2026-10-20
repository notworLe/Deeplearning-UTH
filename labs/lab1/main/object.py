import numpy as np
import torch

import config
import MyLog


class MyMetric:
    def __init__(self):
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0

    def add(self, p_predict, p_actual, p_loss_val, p_n_image):
        if isinstance(p_predict, torch.Tensor):
            p_predict = p_predict.detach()
            if p_predict.dim() == 2:  # logits -> class
                p_predict = p_predict.argmax(dim=1)
            p_predict = p_predict.cpu().numpy()

        if isinstance(p_actual, torch.Tensor):
            p_actual = p_actual.detach().cpu().numpy()

        if isinstance(p_loss_val, torch.Tensor):
            p_loss_val = p_loss_val.item()

        self.total_loss += p_loss_val * p_n_image  # loss batch là mean -> nhân lại
        self.total_correct += int((p_predict == p_actual).sum())
        self.total_samples += p_n_image

    @property
    def average_loss(self):
        return self.total_loss / self.total_samples

    @property
    def accuracy(self):
        return self.total_correct / self.total_samples

    def show_metric(self):
        MyLog.print_log("Average loss", f"{self.average_loss:.4f}")
        MyLog.print_log("Accuracy", f"{self.accuracy:.4f}")




if __name__ == '__main__':
    print()
