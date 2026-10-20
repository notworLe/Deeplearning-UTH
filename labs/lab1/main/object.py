import numpy as np
import torch

import config
import MyLog


class MyMetric:
    def __init__(self):
        self.prediction = []
        self.actual = []
        self.total_loss = []
        self.total_image = []

        # Metrics list
        self.accuracy_list = []
        self.average_loss_list = []

        # Metrics
        self.accuracy = 0
        self.average_loss = 0

    def add(self, p_predict, p_actual, p_loss_val, p_n_image):
        if isinstance(p_predict, torch.Tensor) and config.DEVICE == torch.device("cuda"):
            p_predict = p_predict.cpu().tolist()

        if isinstance(p_actual, torch.Tensor) and config.DEVICE == torch.device("cuda"):
            p_actual = p_actual.cpu().tolist()

        if isinstance(p_loss_val, torch.Tensor) and config.DEVICE == torch.device("cuda"):
            p_loss_val = p_loss_val.item() * p_n_image


        self.prediction.append(p_predict)
        self.actual.append(p_actual)
        self.total_loss.append(p_loss_val)
        self.total_image.append(p_n_image)

        self.cal_metric(p_predict, p_actual, p_loss_val, p_n_image)

    # def get_metric(self, p_index=None):
    #     predict = np.array(self.prediction[p_index])
    #     actual = np.array(self.actual[p])
    #

    def get_metric(self, p_index):
        return self.average_loss_list[p_index], self.accuracy_list[p_index]

    def cal_metric(self):
        # Xử lý việc drop_last = false, np.array đòi hình vuông bằng flatten ra rồi chuyển numpy
        pre_list = []
        ac_list = []
        for i, j in zip(self.prediction, self.actual):
            pre_list.extend(i)
            ac_list.extend(j)
        predict = np.array(pre_list)
        actual = np.array(ac_list)

        self.average_loss = sum(self.total_loss) / sum(self.total_image)
        self.accuracy = (predict == actual).sum() / np.sum(predict.size)

        self.average_loss_list.append(self.average_loss)
        self.accuracy_list.append(self.accuracy)

    def show_metric(self):
        MyLog.print_log("Average loss", f"{self.average_loss:.4f}")
        MyLog.print_log("Accuracy", f"{self.accuracy:.4f}")

    def __len__(self):
        return len(self.prediction)
class MyMetricManage:
    def __init__(self):
        self.predictions_epoch = []
        self.predictions_batch = []

        self.actual_epoch = []
        self.actual_batch = []

        self.total_loss_epoch = []
        self.total_loss_batch = []

        self.n_sample_epoch = []
        self.n_sample_batch = []

        # Metrics
        self.accuracy = 0
        self.average_loss = 0

    def save_epoch(self, p_predict, p_actual, p_loss, p_n_sample_batch):
        self.predictions_epoch.append(p_predict)
        self.actual_epoch.append(p_actual)
        self.total_loss_epoch.append(p_loss)
        self.n_sample_epoch += p_n_sample_batch

        self.calculate_metric("epoch")


    def reset_epoch(self):
        self.predictions_epoch = []
        self.actual_epoch = []
        self.total_loss_epoch = 0
        self.n_sample_epoch = 0

    def calculate_metric(self, mode="epoch"):
        match mode:
            case "epoch":
                predict = np.array(self.predictions_epoch)
                actual = np.array(self.actual_epoch)
                self.accuracy = (predict == actual).sum() / len(predict)
                self.average_loss = self.total_loss_epoch / self.n_sample_epoch
            case "batch":
                predict = np.array(self.predictions_batch)
                actual = np.array(self.actual_batch)
                self.accuracy = (predict == actual).sum() / len(predict)
                self.average_loss = self.total_loss_batch / self.n_sample_batch

    def save_batch(self, p_predict, p_actual, p_loss, p_n_sample_batch):
        self.predictions_batch.extend(p_predict)
        self.actual_batch.extend(p_actual)
        self.total_loss_batch += p_loss
        self.n_sample_batch += p_n_sample_batch

    def show_metric(self):
        MyLog.print_log("Average loss", f"{self.average_loss:.4f}")
        MyLog.print_log("Accuracy", f"{self.accuracy:.4f}")



if __name__ == '__main__':
    metric = MyMetricManage()

