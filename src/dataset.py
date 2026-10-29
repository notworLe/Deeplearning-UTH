from torch.utils.data import Dataset, DataLoader
import src.config as config


def load_raw_dataset(dataset, root, p_train_transform, p_test_transform):
    """
    Load available datasets from pytorch
    :param dataset:
    :param root:
    :param p_train_transform:
    :param p_test_transform:
    :return:
    """
    train_dataset = dataset(
        root=root,
        download=True,
        train=True,
        transform=p_train_transform
    )
    test_dataset = dataset(
        root=root,
        download=True,
        train=False,
        transform=p_test_transform
    )
    return train_dataset, test_dataset

def load_train_val_test(dataset, root, p_train_transform, p_test_transform):
    """
    Load available datasets from pytorch
    :param dataset:
    :param root:
    :param p_train_transform:
    :param p_test_transform:
    :return:
    """
    # v_train, v_test = load_raw_dataset(dataset, root, )
    pass

def create_dataloader(train_dataset, test_dataset):
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WOKER,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WOKER,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    return train_dataloader, test_dataloader