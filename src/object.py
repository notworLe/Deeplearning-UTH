import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

import src.config as config
import src.log as log


class MyMetric:
    def __init__(self):
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_samples = 0

        self.list_actual = []
        self.list_prediction = []

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

        self.list_actual.extend(p_actual)
        self.list_prediction.extend(p_predict)

    @property
    def average_loss(self):
        return self.total_loss / self.total_samples

    @property
    def accuracy(self):
        return self.total_correct / self.total_samples

    def show_metric(self, addition: str = "", bar=None):
        log.print_log("Average loss", f"{self.average_loss:.4f}")
        log.print_log("Average accuracy", f"{self.accuracy:.4f}")


    def show_confusion_tensorboard(self, writer, classes, i_epoch):
        cf_matrix = confusion_matrix(self.total_actual, self.total_predict)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        img = sns.heatmap(df_cm, annot=True).get_figure()
        writer.add_figure("Confusion matrix", img, i_epoch)

    def cal_cf_matrix(self):
        cf_matrix = confusion_matrix(self.list_actual, self.list_prediction)
        cf_norm = cf_matrix / np.sum(cf_matrix, axis=1, keepdims=True)
        return cf_norm

    def get_fig_cf_matrix(self, classes):
        cf_matrix = self.cal_cf_matrix()
        cf_matrix_df = pd.DataFrame(cf_matrix, index=classes, columns=classes)

        fig, ax = plt.subplots(figsize=(13, 8))
        sns.heatmap(cf_matrix_df, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        return fig


if __name__ == '__main__':
    print()
