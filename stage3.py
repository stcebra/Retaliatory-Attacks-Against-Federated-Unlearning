import os
import yaml
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN, get_model, MLP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import get_dataset
from torch.utils.data import DataLoader, TensorDataset
from utils import extract_attack_data

# === Load Config ===
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["global"]["device"] if torch.cuda.is_available() else "cpu")
input_size = config["global"]["input_size"]
num_classes = config["global"]["num_classes"]
batch_size = config["global"]["batch_size"]
model_dir = "./attacked_global_models"
os.makedirs(model_dir, exist_ok=True)

# === Load Stage2 Adversarial Samples ===
data = np.load(config["stage2_search"]["output_path"])
X_u = torch.tensor(data['features'], dtype=torch.float32)
y_u = torch.tensor(data['labels'], dtype=torch.long)

# === Load Original Dataset ===
X_all, y_all = get_dataset(name = config["dataset"]["name"], config = config['dataset'])

test_split_index = config["training"]["test_split_index"]
train_stride = config["training"]["train_stride"]
num_clients = config["training"]["num_clients"]
num_forget_clients = config["training"]["num_forget_clients"]

X_test = X_all[:test_split_index]
y_test = y_all[:test_split_index]
X_forget = X_all[test_split_index:test_split_index + num_forget_clients * train_stride]
y_forget = y_all[test_split_index:test_split_index + num_forget_clients * train_stride]

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
forget_loader = DataLoader(TensorDataset(X_forget, y_forget), batch_size=batch_size, shuffle=False)

# === Prepare Client Data ===
client_datasets = []
client_datasets.append(DataLoader(TensorDataset(X_u, y_u), batch_size=batch_size, shuffle=True))

for i in range(num_clients - 2):
    start = test_split_index + (num_forget_clients + i) * train_stride
    end = start + train_stride
    X_c = X_all[start:end]
    y_c = y_all[start:end]
    loader = DataLoader(TensorDataset(X_c, y_c), batch_size=batch_size, shuffle=True)
    client_datasets.append(loader)

# === AUA Attack ===
model_path = config["paths"]["model_path_after"]

before_attack_global_model = get_model(config["dataset"].get("name", "default"), input_size=input_size, num_classes = num_classes, device=device)
before_attack_global_model.load_state_dict(torch.load(model_path, map_location=device))
before_attack_global_model.eval()

global_model = get_model(config["dataset"].get("name", "default"), input_size=input_size, num_classes = num_classes, device=device)
global_model.load_state_dict(torch.load(model_path, map_location=device))
global_model.eval()

def local_train(model, dataloader, lr, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

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

print("\n=== Federated Attack Training (Adversarial Injection) ===")
rounds = config["stage3"]["num_rounds"]
epochs_AUA = config["stage3"]["attack_epochs"]

# AUA start
for rnd in range(rounds):
    local_models = []
    for cid, loader in enumerate(client_datasets):
        model_local = copy.deepcopy(global_model)
        if cid == 0:
            local_train(model_local, loader, lr=config["stage3"]["lr"], epochs=epochs_AUA)
        else:
            local_train(model_local, loader, lr=config["training"]["lr"], epochs=100)
        local_models.append(model_local)
    with torch.no_grad():
        avg_model = copy.deepcopy(global_model)
        for key in avg_model.state_dict().keys():
            avg_param = sum(m.state_dict()[key] for m in local_models) / len(local_models)
            avg_model.state_dict()[key].copy_(avg_param)
        global_model = avg_model
    print(f"Round {rnd+1} test acc:", end=" ")
    evaluate(global_model, test_loader, "Test")

torch.save(global_model.state_dict(), f"{model_dir}/global_model_anti.pth")
print("Saved adversarially attacked model.")


# === Start AUA eva ===
print("\n=== Generate Anti Data for MIA Evaluation ===")
model = global_model
model.eval()

X, y = get_dataset(name = config["dataset"]["name"], config = config['dataset'])

test_split_index = config["training"]["test_split_index"]
stride = config["training"]["train_stride"]
num_forget = config["training"]["num_forget_clients"]

X_test = X[:test_split_index]
y_test = y[:test_split_index]
X_train = torch.cat([X[test_split_index + i * stride:test_split_index + (i + 1) * stride] for i in range(num_forget)],
                    dim=0)
y_train = torch.cat([y[test_split_index + i * stride:test_split_index + (i + 1) * stride] for i in range(num_forget)],
                    dim=0)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

train_features, train_labels = extract_attack_data(model, train_loader, member_label=1, num_classes=num_classes, device=device)
test_features, test_labels = extract_attack_data(model, test_loader, member_label=0, num_classes=num_classes, device=device)

X_all_eval = np.array(train_features + test_features)
y_all_eval = np.array(train_labels + test_labels)
np.savez(os.path.join(model_dir, "anti_data_eval.npz"), features=X_all_eval, labels=y_all_eval)
print(f"Saved anti_data_eval.npz with {X_all_eval.shape[0]} samples")


# === Evaluate MIA ====
print("\n=== Evaluate MIA Attack on Adversarial Model ===")
X_all_eval = torch.tensor(X_all_eval, dtype=torch.float32)
y_true = torch.tensor(y_all_eval, dtype=torch.long)

X_logits = X_all_eval[:, :num_classes]
y_class = torch.argmax(X_all_eval[:, num_classes:], dim=1)

attack_models = {}
attack_model_dir = config["paths"]["attack_model_dir"] + "/" + config["dataset"]["name"]
attack_model_prefix = config["paths"]["attack_model_prefix"]

for class_id in range(num_classes):
    model_path = os.path.join(attack_model_dir, f"{attack_model_prefix}{class_id}.pth")
    if os.path.exists(model_path):
        model = MLP(input_size=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        attack_models[class_id] = model
    else:
        attack_models[class_id] = None


# === Evaluate on Forget Clients ===
evaluate(before_attack_global_model, forget_loader, "Forgotten Clients Set before attack")
evaluate(global_model, forget_loader, "Forgotten Clients Set after attack")

# === DUA Attack ===
print("\n=== Poisoning Attack Training ===")
poison_data = np.load(config["stage2_search"]["output_path"])
X_poison = torch.tensor(poison_data['features'], dtype=torch.float32)
y_poison = torch.tensor(poison_data['labels'], dtype=torch.long)

X_target = X_all[test_split_index:test_split_index + num_forget_clients * train_stride]
y_target = y_all[test_split_index:test_split_index + num_forget_clients * train_stride]
target_loader = DataLoader(TensorDataset(X_target, y_target), batch_size=batch_size, shuffle=False)

client_datasets = []
client_datasets.append(DataLoader(TensorDataset(X_all[test_split_index + num_forget_clients * train_stride:
                                                 test_split_index + (num_forget_clients + 1) * train_stride],
                                                y_all[test_split_index + num_forget_clients * train_stride:
                                                     test_split_index + (num_forget_clients + 1) * train_stride]),
                                  batch_size=batch_size, shuffle=True))
for i in range(num_clients - 2):
    start = test_split_index + (num_forget_clients + 1 + i) * train_stride
    end = start + train_stride
    client_datasets.append(DataLoader(TensorDataset(X_all[start:end], y_all[start:end]), batch_size=batch_size, shuffle=True))
# print(len(client_datasets))

global_model = get_model(config["dataset"].get("name", "default"), input_size=input_size, num_classes = num_classes, device=device)
global_model.load_state_dict(torch.load(config["paths"]["model_path_after"], map_location=device))

lambda_target = config["poisoning"]["lambda_target"]
num_rounds = config["poisoning"]["num_rounds"]

# DUA start
for round in range(num_rounds):
    local_models = []
    for cid, loader in enumerate(client_datasets):
        model_local = copy.deepcopy(global_model)
        optimizer = optim.Adam(model_local.parameters(), lr=config["poisoning"]["lr_poison"] if cid == 0 else config["poisoning"]["lr_clean"])
        criterion = nn.CrossEntropyLoss()
        model_local.train()
        for _ in range(config["poisoning"]["num_rounds"]):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model_local(X_batch), y_batch)
                if cid == 0:
                    output_u = model_local(X_poison.to(device))
                    loss_u = criterion(output_u, y_poison.to(device))
                    loss = loss - lambda_target * loss_u
                loss.backward()
                optimizer.step()
        local_models.append(model_local)
    with torch.no_grad():
        avg_model = copy.deepcopy(global_model)
        for key in avg_model.state_dict().keys():
            avg_param = sum(m.state_dict()[key] for m in local_models) / len(local_models)
            avg_model.state_dict()[key].copy_(avg_param)
        global_model = avg_model
    print(f"[Poisoning] Round {round+1} test acc:", end=" ")
    evaluate(global_model, test_loader, "Test")

torch.save(global_model.state_dict(), f"{model_dir}/global_model_poison.pth")
print("Saved poisoned model.")


evaluate(before_attack_global_model, target_loader, "Forgotten Clients Set before attack")
evaluate(global_model, target_loader, "Forgotten Clients Set after attack")

evaluate(before_attack_global_model, test_loader, "Test Set before attack")
evaluate(global_model, test_loader, "Test Set after attack")