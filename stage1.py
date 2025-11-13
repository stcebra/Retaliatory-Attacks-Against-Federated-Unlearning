import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os, yaml
from model import SimpleNN, AttackModel
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import random
from model import get_model
from utils import get_shadow_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix


def split_dataset_into_clients(X, y, num_clients):
    client_data = []
    size = len(X) // num_clients
    for i in range(num_clients):
        start = i * size
        end = len(X) if i == num_clients - 1 else (i + 1) * size
        client_data.append(TensorDataset(X[start:end], y[start:end]))
    return client_data

def average_models(model_list):
    avg_params = [param.data.clone() for param in model_list[0].parameters()]
    for model in model_list[1:]:
        for i, param in enumerate(model.parameters()):
            avg_params[i] += param.data
    for i in range(len(avg_params)):
        avg_params[i] /= len(model_list)
    return avg_params

def set_model_params(model, params):
    for param, new_param in zip(model.parameters(), params):
        param.data.copy_(new_param)

def get_model_params(model):
    return [param.data.clone() for param in model.parameters()]

def extract_attack_data_dual(model_full, model_partial, dataloader, member_label, device, num_classes):
    model_full.eval()
    model_partial.eval()
    attack_features, attack_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits_full = model_full(X_batch)
            logits_partial = model_partial(X_batch)
            combined = torch.cat([logits_full, logits_partial], dim=1)
            for logit, true_y in zip(combined.cpu(), y_batch.cpu()):
                onehot = torch.nn.functional.one_hot(true_y, num_classes=num_classes).float()
                feature = torch.cat([logit, onehot], dim=0)
                attack_features.append(feature.numpy())
                attack_labels.append(member_label)
    return attack_features, attack_labels

def evaluate(model, dataloader, name, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print(f"{name} Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['global']['device'] if torch.cuda.is_available() else 'cpu')
    num_classes = config['global']['num_classes']

    stage1_cfg = config['stage1']
    shadow_K = stage1_cfg['shadow_K']
    lr = stage1_cfg['shadow_lr']
    epochs = stage1_cfg['shadow_epochs']
    batch_size = stage1_cfg['shadow_batch_size']
    test_size = stage1_cfg['split_test_size']
    train2_size = stage1_cfg['split_train2_size']
    data_range = stage1_cfg['shadow_data_range']
    save_dir = stage1_cfg['shadow_save_dir']
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(seed=stage1_cfg["seed"])
    torch.cuda.manual_seed_all(seed=stage1_cfg["seed"])

    # get shadow data
    X, y = get_shadow_dataset(name = config["dataset"]["name"], config = config['dataset'], seed = stage1_cfg["seed"])
    X = X[data_range[0]:data_range[1]]
    y = y[data_range[0]:data_range[1]]


    # Randomly select m local clients as the simulated unlearned
    unlearning_clients = random.choice(range(1, stage1_cfg['train_clients']//3 + 1))
    retrain_clients = stage1_cfg['train_clients'] - unlearning_clients

    splitter = StratifiedShuffleSplit(n_splits=shadow_K, test_size=test_size, random_state=42)

    # Shadowing FU Processes
    for i, (remain_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_remain, y_remain = X[remain_idx], y[remain_idx]
        test_X, test_y = X[test_idx], y[test_idx]

        sub_splitter = ShuffleSplit(n_splits=1, test_size=train2_size, random_state=i)
        train1_idx, train2_idx = next(sub_splitter.split(X_remain))
        train1_X, train1_y = X_remain[train1_idx], y_remain[train1_idx]
        train2_X, train2_y = X_remain[train2_idx], y_remain[train2_idx]

        loader_train1 = DataLoader(TensorDataset(train1_X, train1_y), batch_size=batch_size, shuffle=True)
        loader_train2 = DataLoader(TensorDataset(train2_X, train2_y), batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

        model_full = get_model(config["dataset"]["name"], config['global']['input_size'], num_classes, device)
        clients_full = split_dataset_into_clients(train1_X, train1_y,unlearning_clients) + \
                       split_dataset_into_clients(train2_X, train2_y,retrain_clients)
        criterion = nn.CrossEntropyLoss()
        # print(len(clients_full))

        for epoch in range(epochs):
            local_models = []
            for client_data in clients_full:
                model_local = get_model(config["dataset"]["name"], config['global']['input_size'], num_classes, device)
                set_model_params(model_local, get_model_params(model_full))
                optimizer = optim.Adam(model_local.parameters(), lr=lr)
                loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)

                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model_local(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
                local_models.append(model_local)
            set_model_params(model_full, average_models(local_models))

        evaluate(model_full, loader_test, f"Shadow {i} model_full", device)

        model_partial = get_model(config["dataset"]["name"], config['global']['input_size'], num_classes, device)
        clients_partial = split_dataset_into_clients(train2_X, train2_y, retrain_clients)
        # print(len(clients_partial))

        for epoch in range(epochs):
            local_models = []
            for client_data in clients_partial:
                model_local = get_model(config["dataset"]["name"], config['global']['input_size'], num_classes, device)
                set_model_params(model_local, get_model_params(model_partial))
                optimizer = optim.Adam(model_local.parameters(), lr=lr)
                loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)

                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model_local(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
                local_models.append(model_local)
            set_model_params(model_partial, average_models(local_models))

        evaluate(model_partial, loader_test, f"Shadow {i} model_partial", device)

        torch.save(model_full.state_dict(), os.path.join(save_dir, stage1_cfg['shadow_model_full_prefix'].format(i)))
        torch.save(model_partial.state_dict(), os.path.join(save_dir, stage1_cfg['shadow_model_partial_prefix'].format(i)))

        f2, l2 = extract_attack_data_dual(model_full, model_partial, loader_train2, 2, device, num_classes)
        f1, l1 = extract_attack_data_dual(model_full, model_partial, loader_train1, 1, device, num_classes)
        f0, l0 = extract_attack_data_dual(model_full, model_partial, loader_test, 0, device, num_classes)

        np.savez(os.path.join(save_dir, f"{stage1_cfg['shadow_attack_prefix']}{i}.npz"),
                 features=np.array(f2 + f1 + f0),
                 labels=np.array(l2 + l1 + l0))

        print(f"Saved models and attack data for shadow split {i}")

    # #=== Train UMIA Attack Model ===
    attack_cfg = config["attack"]
    train_cfg = config["training_attack"]
    input_dim = attack_cfg["input_dim"]
    num_output_classes = attack_cfg["num_output_classes"]
    attack_model_name = attack_cfg["model"]
    attack_epochs = train_cfg["attack_epochs"]
    attack_lr = train_cfg["attack_lr"]
    os.makedirs(attack_cfg["output_dir"], exist_ok=True)

    features_all, labels_all = [], []
    for i in range(shadow_K):
        data = np.load(f"{attack_cfg['data_prefix']}{i}.npz")
        features_all.append(data['features'])
        labels_all.append(data['labels'])

    X_all = torch.tensor(np.concatenate(features_all, axis=0), dtype=torch.float32)
    y_mia_all = torch.tensor(np.concatenate(labels_all, axis=0), dtype=torch.long)
    y_class_all = torch.argmax(X_all[:, input_dim:], dim=1)

    for class_id in range(num_classes):
        class_mask = (y_class_all == class_id)
        X_class = X_all[class_mask][:, :input_dim]
        y_class = y_mia_all[class_mask]

        if len(X_class) < 10:
            print(f"Class {class_id} skipped (too few samples: {len(X_class)})")
            continue

        dataset = TensorDataset(X_class, y_class)
        train_size = int(0.8 * len(dataset))
        train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model_class = AttackModel if attack_model_name == "AttackModel" else AttackModel
        model = model_class(input_size=input_dim, output_size=num_output_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=attack_lr)

        model.train()
        for epoch in range(attack_epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        print(f"Class {class_id} | Attack Model Accuracy: {correct / total:.4f}")
        save_path = f"{attack_cfg['output_dir']}/attack_model_{class_id}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")


    # === Evaluate UMIA on evaluation data ===
    print("\n=== Evaluating UMIA Models on Evaluation Data ===")

    eval_data = np.load(config["stage1_eval"]["output_path"])
    X_eval = torch.tensor(eval_data["features"], dtype=torch.float32)
    y_eval = torch.tensor(eval_data["labels"], dtype=torch.long)

    model_dir = config["attack"]["output_dir"]

    X_logits = X_eval[:, :input_dim]
    y_class = torch.argmax(X_eval[:, input_dim:], dim=1)

    attack_models = {}
    model_class = AttackModel if attack_cfg["model"] == "AttackModel" else AttackModel
    model_dir = attack_cfg["output_dir"]
    for cls in range(num_classes):
        path = os.path.join(model_dir, f"attack_model_{cls}.pth")
        if os.path.exists(path):
            model = model_class(input_size=input_dim, output_size=num_output_classes)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            attack_models[cls] = model
        else:
            attack_models[cls] = None

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for i in range(X_logits.shape[0]):
            cls = y_class[i].item()
            model = attack_models.get(cls)

            if model is None:
                continue

            x = X_logits[i].unsqueeze(0).to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1).item()

            y_true_all.append(y_eval[i].item())
            y_pred_all.append(pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    print(f"\nEvaluated on {len(y_true_all)} samples")
    print(f"Overall Attack Accuracy: {accuracy_score(y_true_all, y_pred_all):.4f}\n")

    for cls in range(num_output_classes):
        mask = (y_true_all == cls)
        acc = np.mean(y_pred_all[mask] == y_true_all[mask]) if mask.any() else float("nan")
        print(f"Class {cls} Accuracy:  {acc:.4f}  (support={mask.sum()})")

    prec = precision_score(y_true_all, y_pred_all, average=None, labels=[0, 1, 2], zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average=None, labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average=None, labels=[0, 1, 2], zero_division=0)

    for cls in range(num_output_classes):
        print(f"\nClass {cls} Precision: {prec[cls]:.4f}")
        print(f"Class {cls} Recall:    {rec[cls]:.4f}")
        print(f"Class {cls} F1-score:  {f1[cls]:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, digits=4, labels=[0, 1, 2], zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2]))