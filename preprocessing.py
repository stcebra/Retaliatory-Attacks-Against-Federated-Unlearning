# Following code is for preparing the federate learning and unlearning model

import argparse
import yaml
import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import get_model
from dataset import get_dataset
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
args = parser.parse_args()

# Load config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["global"]["device"] if torch.cuda.is_available() else "cpu")
input_size = config["global"]["input_size"]
num_classes = config["global"]["num_classes"]
model_save_dir = config["global"]["model_save_dir"]
os.makedirs(model_save_dir, exist_ok=True)

num_clients = config["training"]["num_clients"]
num_forget_clients = config["training"]["num_forget_clients"]
client_data_size = config["training"]["client_data_size"]
batch_size = config["training"]["batch_size"]
lr = config["training"]["lr"]
num_rounds = config["training"]["num_rounds"]
local_epochs = config["training"]["local_epochs"]
test_split_index = config["training"]["test_split_index"]
train_stride = config["training"]["train_stride"]


# Load dataset
X, y = get_dataset(name = config["dataset"]["name"], config = config['dataset'])
X_test = X[:test_split_index]
y_test = y[:test_split_index]
X_train = X[test_split_index: test_split_index + num_clients * client_data_size]
y_train = y[test_split_index: test_split_index + num_clients * client_data_size]
print(X.shape,y.shape,y_test.shape,y_train.shape)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

client_datasets = []
for i in range(num_clients):
    start = test_split_index + i * train_stride
    end = start + client_data_size
    client_X = X[start:end]
    client_y = y[start:end]
    loader = DataLoader(TensorDataset(client_X, client_y), batch_size=batch_size, shuffle=True)
    client_datasets.append(loader)

def local_train(model, dataloader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

def average_models(models):
    avg_model = copy.deepcopy(models[0])
    with torch.no_grad():
        for key in avg_model.state_dict():
            avg_param = sum([model.state_dict()[key] for model in models]) / len(models)
            avg_model.state_dict()[key].copy_(avg_param)
    return avg_model

def evaluate(model, dataloader, name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch).argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
    print(f"{name} Accuracy: {correct / total:.4f}")


# Federated Training
print("\n=== Federated Training (Before Unlearning) ===")
global_model = get_model(config["dataset"].get("name", "default"), input_size=input_size, num_classes = num_classes, device=device)

for round in range(num_rounds):
    local_models = []
    for cid in range(num_clients):
        model_copy = copy.deepcopy(global_model)
        local_train(model_copy, client_datasets[cid])
        local_models.append(model_copy)
    global_model = average_models(local_models)
    if (round + 1) % 50 == 0 or round == num_rounds - 1:
        print(f"Round {round+1}", end=": ")
        evaluate(global_model, test_loader, "Test")
        evaluate(global_model, train_loader, "Train")
torch.save(global_model.state_dict(), f"{model_save_dir}/before_unlearning.pth")
print("Saved model: before_unlearning.pth")

# Federated Unlearning
print("\n=== Federated Training (After Unlearning) ===")
global_model = get_model(config["dataset"].get("name", "default"), input_size=input_size, num_classes = num_classes, device=device)
for round in range(num_rounds):
    local_models = []
    for cid in range(num_forget_clients, num_clients):
        model_copy = copy.deepcopy(global_model)
        local_train(model_copy, client_datasets[cid])
        local_models.append(model_copy)
    global_model = average_models(local_models)
    if (round + 1) % 50 == 0 or round == num_rounds - 1:
        print(f"Round {round+1}", end=": ")
        evaluate(global_model, test_loader, "Test")
torch.save(global_model.state_dict(), f"{model_save_dir}/after_unlearning.pth")
print("Saved model: after_unlearning.pth")


# Generate evaluation data for U-MIA attack
print("\n=== Generating Evaluation Attack Data ===")
model_before = get_model(config["dataset"].get("name", "default"), input_size=input_size, num_classes = num_classes, device=device)
model_after = get_model(config["dataset"].get("name", "default"), input_size=input_size, num_classes = num_classes, device=device)
model_before.load_state_dict(torch.load(f"{model_save_dir}/before_unlearning.pth", map_location=device))
model_after.load_state_dict(torch.load(f"{model_save_dir}/after_unlearning.pth", map_location=device))
model_before.eval()
model_after.eval()

def extract_attack_features(model1, model2, loader, label):
    feats, labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logit1 = model1(x_batch)
            logit2 = model2(x_batch)
            combined = torch.cat([logit1, logit2], dim=1)
            for i in range(x_batch.size(0)):
                onehot = torch.nn.functional.one_hot(y_batch[i], num_classes=num_classes).float()
                feature = torch.cat([combined[i].cpu(), onehot.cpu()], dim=0)
                feats.append(feature.numpy())
                labels.append(label)
    return feats, labels

X_train1 = X[test_split_index : test_split_index + num_forget_clients * client_data_size]
y_train1 = y[test_split_index : test_split_index + num_forget_clients * client_data_size]
X_train2 = X[test_split_index + num_forget_clients * client_data_size : test_split_index + 2 * num_forget_clients * client_data_size]
y_train2 = y[test_split_index + num_forget_clients * client_data_size : test_split_index + 2 * num_forget_clients * client_data_size]

loader1 = DataLoader(TensorDataset(X_train1, y_train1), batch_size=batch_size, shuffle=False)
loader2 = DataLoader(TensorDataset(X_train2, y_train2), batch_size=batch_size, shuffle=False)
loader0 = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

f1, l1 = extract_attack_features(model_before, model_after, loader1, 1)
f2, l2 = extract_attack_features(model_before, model_after, loader2, 2)
f0, l0 = extract_attack_features(model_before, model_after, loader0, 0)

output_path = config["stage1_eval"]["output_path"]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.savez(output_path, features=np.array(f1 + f2 + f0), labels=np.array(l1 + l2 + l0))
print(f"Saved evaluation attack data to {output_path}")