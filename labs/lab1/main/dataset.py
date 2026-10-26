from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import config

DEFAULT_TRANSFORM = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=config.MEAN, std=config.STD)
])

AUGMENTED_TRANSFORM = transform.Compose([
    # WHY RandomHorizontalFlip? Áo có thể lật ngang, vẫn là áo
    transform.RandomHorizontalFlip(p=0.5),

    # WHY RandomAffine?
    # degrees=10  → xoay nhẹ, simulate ảnh chụp lệch
    # translate   → dịch chuyển, object không phải lúc nào cũng ở giữa
    # scale       → zoom in/out nhẹ
    transform.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    # WHY RandomErasing?
    # Xóa ngẫu nhiên 1 vùng hình chữ nhật → force model không
    # dựa vào 1 chi tiết duy nhất (e.g., chỉ nhìn cổ áo để đoán Coat)
    # → phải học nhiều đặc trưng phân tán hơn
    transform.ToTensor(),
    transform.Normalize(mean=config.MEAN, std=config.STD),
    transform.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])


AUGMENTED_TRANSFORM_LOW = transform.Compose([
    transform.RandomHorizontalFlip(p=0.5),

    # Giảm mạnh: bỏ scale, giảm rotate và translate
    transform.RandomAffine(degrees=7, translate=(0.07, 0.07)),

    transform.ToTensor(),
    transform.Normalize(mean=config.MEAN, std=config.STD),

    # WHY giảm p và scale?
    # 28×28 quá nhỏ → erasing nhỏ thôi, không thì mất signal
    transform.RandomErasing(p=0.2, scale=(0.02, 0.08)),
])


def load_raw_dataset(dataset, root = "data", p_transform=DEFAULT_TRANSFORM, p_augment_transform = None):
    v_train_transform = p_transform if p_augment_transform is None else p_augment_transform
    v_test_transform = p_transform

    train_dataset = dataset(
        root=root,
        download=True,
        train=True,
        transform=v_train_transform
    )
    test_dataset = dataset(
        root=root,
        download=True,
        train=False,
        transform=v_test_transform
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