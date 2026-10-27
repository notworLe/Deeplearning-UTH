# Add root
from pathlib import Path
import sys
ROOT_PATH = Path.cwd().parent
sys.path.append(str(ROOT_PATH))

# inference.py
import torch
from PIL import Image
from torchvision import transforms
from model import FashionCNN

CLASSES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
device = "cuda" if torch.cuda.is_available() else "cpu"

model = FashionCNN()                                   # class model của bạn
model.load_state_dict(torch.load("../weights/FASHION_MNIST_CNN.pt", map_location=device)['model'])
model.to(device).eval()

tf = transforms.Compose([                             # PHẢI giống transform lúc test
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

@torch.inference_mode()
def predict(img: Image.Image, top_k: int = 3) -> dict[str, float]:
    x = tf(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    p, idx = probs.topk(top_k)
    return {CLASSES[i]: float(v) for v, i in zip(p, idx)}


@torch.inference_mode()
def predict_batch(imgs: list[Image.Image], top_k: int = 1, bs: int = 64):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([tf(im) for im in imgs[i:i + bs]]).to(device)
        probs = torch.softmax(model(x), dim=1)
        p, idx = probs.topk(top_k, dim=1)
        for pp, ii in zip(p.cpu().tolist(), idx.cpu().tolist()):
            out.append([(CLASSES[c], v) for v, c in zip(pp, ii)])
    return out