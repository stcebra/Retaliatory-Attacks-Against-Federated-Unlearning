import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleNN(nn.Module):
    def __init__(self, input_size=600, num_classes=100):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    

class AttackModel(nn.Module):
    def __init__(self, input_size=200, output_size=3):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_size=600, num_classes=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(dataset_name, input_size, num_classes, device):
    if dataset_name == "purchase100":
        return SimpleNN(input_size=input_size, num_classes=num_classes).to(device)
    elif dataset_name == "location":
        return SimpleNN(input_size=input_size, num_classes=num_classes).to(device)
    elif dataset_name == "cifar10":
        return models.resnet18(num_classes = num_classes).to(device)
    elif dataset_name == "svhn":
        return SimpleCNN().to(device)
