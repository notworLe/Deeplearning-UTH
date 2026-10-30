from src.dataset import split_into_val_dataset, load_raw_dataset, loader_train_val_test
from src.model import training_phase, test_phase
from src import config, utility, log, class_name, visualize

from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import torchvision.transforms as transform
from torchvision.models import resnet18, resnet50
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # CONFIT
    RESULT_NAME = "resnet18_baseline"
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
    v_model = resnet18(weights=None).to(config.DEVICE)

    ## Metadata
    # print(v_mode)
    # summary(v_mode, (3, 224, 224))

    ## Change output class
    v_model.fc = nn.Linear(in_features=512, out_features=len(v_test.classes)).to(config.DEVICE)

    ## Check again the model
    # print(v_mode)
    # summary(v_mode, (3, 224, 224))

    v_loss_func = nn.CrossEntropyLoss()
    v_optimizer = optim.Adam(v_model.parameters(), lr=config.LR)

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
        p_best_val= v_best_val
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
