
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Kiem tra thiet bi
print("PyTorch version:", torch.__version__)

# Chuyen anh thanh tensor
transform = transforms.ToTensor()

# Tai bo du lieu huan luyen
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Tai bo du lieu kiem tra
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Ten 10 loai quan ao
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("So anh huan luyen:", len(train_dataset))
print("So anh kiem tra:", len(test_dataset))
print("So loai quan ao:", len(classes))


# Xay dung mo hinh AI
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Chia du lieu thanh cac lo nho
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False
)

# Tao mang noron
class FashionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


model = FashionModel()

# Ham tinh loi va bo toi uu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huan luyen AI
epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {avg_loss:.4f}"
    )

print("Huan luyen AI hoan tat!")

# Ve bieu do Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# Danh gia do chinh xac cua mo hinh
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Do chinh xac: {accuracy:.2f}%")
torch.save(model.state_dict(), "fashion_model.pth")
print("Da luu mo hinh thanh cong!")


