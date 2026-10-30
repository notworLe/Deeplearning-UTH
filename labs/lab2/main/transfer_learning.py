from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transform
from torchvision.datasets import OxfordIIITPet
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from src import config, utility, log, class_name, visualize
from src.model import training_phase, evaluate_phase, test_phase
from src.dataset import split_into_val_dataset, load_raw_dataset, loader_train_val_test
from pathlib import Path

class MyResNet(nn.Module):
    def __init__(self, n_classes = 10):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        del self.model.fc

        self.fc = nn.Linear(in_features=512, out_features=n_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

if __name__ == '__main__':
    # CONFIT
    RESULT_NAME = "resnet18_transfer"
    DATA_PATH = "data"
    IMAGE_SHAPE = (224, 224)
    CLASS_NAME = class_name.OXFORD_III_PET
    SAVE_WEIGHT = Path(f"weights/{RESULT_NAME}.pt")
    WRITER = SummaryWriter(f"runs/{RESULT_NAME}")

    # Log
    log.log_creation()
    utility.set_seed()

    # Dataset
    ## Transform

    v_train_transform = transform.Compose([
        transform.RandomResizedCrop(size=IMAGE_SHAPE, scale=(0.7, 1)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        config.IMAGE_NET_NORMALIZE
    ])
    v_test_transform = transform.Compose([
        transform.Resize(256),
        transform.CenterCrop(IMAGE_SHAPE),
        transform.ToTensor(),
        config.IMAGE_NET_NORMALIZE
    ])

    ## Load
    v_full_train = OxfordIIITPet(
        "data",
        split="trainval",
        target_types="category",
        download=True,
        transform=v_train_transform
    )
    v_test = OxfordIIITPet(
        "data",
        split="test",
        target_types="category",
        download=True,
        transform=v_train_transform
    )
    v_train, v_val = split_into_val_dataset(v_full_train, p_val_size=0.1)
    log.dataset_log_v2(CLASS_NAME, v_test, train=v_train, val=v_val)

    # Visual
    # visualize.show_class_samples(v_test, n=5, n_cols=5, save_path=f"result/{RESULT_NAME}/data-visual/show_class.png")  # 1 class
    # visualize.plot_class_distribution(v_train, save_path=f"result/{RESULT_NAME}/data-visual/clas-distribution-train.png")
    # visualize.plot_class_distribution(v_val, save_path=f"result/{RESULT_NAME}/data-visual/clas-distribution-test.png")
    # visualize.plot_class_distribution(v_test, save_path=f"result/{RESULT_NAME}/data-visual/clas-distribution-test.png")
    # visualize.plot_split_class_distribution({"train": v_train, "val": v_val, "test": v_test},
    #                               save_path=f"result/{RESULT_NAME}/data-visual/class-distribution-splits.png"
    # )

    ## Loader
    v_train_loader, v_val_loader, v_test_loader = loader_train_val_test(v_train, v_val, v_test)
    log.dataloader_log_v2(
        20,
        v_train_loader=v_train_loader,
        v_val_loader=v_val_loader,
        v_test_loader=v_test_loader
    )


    # Model
    v_model = MyResNet(n_classes=len(v_test.classes)).to(config.DEVICE)
    v_loss_func = nn.CrossEntropyLoss()
    v_optimizer = optim.Adam(v_model.parameters(), lr=config.LR)


    print("=== FULL MODEL STRUCTURE ===")
    # print(model)
    summary(v_model, (3, 224, 224), device=config.DEVICE)

    print("\n=== NAMED PARAMETERS (tên layer + shape) ===")
    # for name, param in model.named_parameters():
    #     print(f"{name:50s} | shape: {list(param.shape)}")

    # ── Đếm params TRƯỚC khi freeze ──────────────────────────────
    total = sum(p.numel() for p in v_model.parameters())
    trainable = sum(p.numel() for p in v_model.parameters() if p.requires_grad)
    print(f"\n[TRƯỚC freeze] Tổng params    : {total:,}")
    print(f"[TRƯỚC freeze] Trainable params: {trainable:,}")

    # ── Freeze: giữ nguyên tất cả NGOẠI TRỪ fc ────
    for name, param in v_model.named_parameters():
        if "fc" in name:
            continue
        param.requires_grad = False

    # ── Đếm params SAU khi freeze ────────────────────────────────
    trainable_after = sum(p.numel() for p in v_model.parameters() if p.requires_grad)
    print(f"\n[SAU  freeze] Trainable params: {trainable_after:,}")
    print(f"  → Đã đóng băng: {total - trainable_after:,} params ({(total - trainable_after)/total*100:.1f}%)")

    # Resume weight
    v_start_epoch, v_best_val = utility.resume_weight(SAVE_WEIGHT, v_model, v_optimizer)
    log.print_log("Best accuracy", v_best_val)
    log.print_log("Start epoch", v_start_epoch)

    # Training phase
    list_metric_train_epoch, list_metric_eval_epoch = training_phase(
        v_model,
        v_loss_func,
        v_optimizer,
        v_train_loader,
        v_val_loader,
        SAVE_WEIGHT,
        config.EPOCH,
        v_start_epoch,
        WRITER,
        classes=v_test.classes,
        p_best_val=v_best_val
    )

    # Test phase
    v_obj_eval_metric = test_phase(v_model, v_test_loader, WRITER)

    # Evaluate metrics
    visualize.plot_classification_metrics(
        v_obj_eval_metric.list_actual,
        v_obj_eval_metric.list_prediction,
        v_test.classes,
        save_path=f"result/{RESULT_NAME}/metric/pre-recall-f1-acc.png"
    )
    v_cf_matrix = v_obj_eval_metric.get_fig_cf_matrix(v_test.classes)
    visualize._finish(v_cf_matrix, save_path=f"result/{RESULT_NAME}/metric/confusion-matrix.png")
