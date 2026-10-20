from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import MyLog
import config

DEFAULT_TRANSFORM = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=config.MEAN, std=config.STD)
])

def load_raw_dataset(dataset, p_transform=DEFAULT_TRANSFORM):

    train_dataset = dataset(
        root="data",
        download=True,
        train=True,
        transform=p_transform
    )
    test_dataset = dataset(
        root="data",
        download=True,
        train=False,
        transform=p_transform
    )
    return train_dataset, test_dataset




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