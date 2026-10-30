from torch.utils.data import Dataset, DataLoader, random_split
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

def split_into_val_dataset(p_train_dataset, p_val_size=0.2):
    # Split size
    v_val_size = int(len(p_train_dataset) * p_val_size)
    v_train_size = len(p_train_dataset) - v_val_size

    # Split into validation
    v_train, v_val = random_split(p_train_dataset, [v_train_size, v_val_size])

    return v_train, v_val

def load_train_val_test(dataset, root, p_train_transform, p_test_transform):
    """
    Load available datasets from pytorch
    :param dataset:
    :param root:
    :param p_train_transform:
    :param p_test_transform:
    :return:
    """
    v_train, v_test = load_raw_dataset(dataset, root, p_train_transform, p_test_transform)

    pass

def create_dataloader(train_dataset, test_dataset):
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WOKER,
        shuffle=True,
        drop_last=True,
        pin_memory=(config.DEVICE == "cuda")
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WOKER,
        shuffle=False,
        drop_last=(config.DEVICE == "cuda"),
        pin_memory=True
    )
    return train_dataloader, test_dataloader

def loader_train_val_test(train_dataset, val_dataset, test_dataset):
    v_train_loader, v_test_loader = create_dataloader(train_dataset, test_dataset)
    v_val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.WOKER,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    return v_train_loader, v_val_loader, v_test_loader