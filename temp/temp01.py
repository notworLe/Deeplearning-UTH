import torch

if __name__ == '__main__':
    arr1 = torch.randint(-10, 10, (3, 5))
    print(arr1)
    print(arr1.argmax(dim=1))
    print(arr1.dim())
