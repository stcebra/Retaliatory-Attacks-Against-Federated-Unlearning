import numpy as np
import torch
from torchvision import datasets, transforms

def get_dataset(name, config):
    if name in ["purchase100", "location"]:
        data = np.load(config["path"])
        X = torch.tensor(data["features"], dtype=torch.float32)
        y = torch.tensor(np.argmax(data["labels"], axis=1), dtype=torch.long)
        return X, y

    elif name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        X = torch.stack([d[0] for d in dataset])
        y = torch.tensor([d[1] for d in dataset])
        return X, y

    elif name == "svhn":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        X = torch.stack([d[0] for d in dataset])
        y = torch.tensor([d[1] for d in dataset])
        return X, y

    else:
        raise ValueError(f"Unknown dataset: {name}")


