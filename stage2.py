import numpy as np
import torch
import yaml
import os
from utils import generate_random_binary_sample
from model import SimpleNN, AttackModel
import torch.nn.functional as F

# === Load config ===
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

global_cfg = config["global"]
stage2_cfg = config["stage2_search"]
attack_cfg = config["attack"]
paths_cfg = config["paths"]

device = torch.device(global_cfg["device"] if torch.cuda.is_available() else "cpu")

# === Load models ===
model_full = SimpleNN(global_cfg["input_size"], global_cfg["num_classes"]).to(device)
model_full.load_state_dict(torch.load(paths_cfg["model_path_before"], map_location=device))
model_full.eval()

model_partial = SimpleNN(global_cfg["input_size"], global_cfg["num_classes"]).to(device)
model_partial.load_state_dict(torch.load(paths_cfg["model_path_after"], map_location=device))
model_partial.eval()

attack_models = {}
for class_id in range(global_cfg["num_classes"]):
    path = os.path.join(attack_cfg["output_dir"], f"attack_model_{class_id}.pth")
    if os.path.exists(path):
        model = AttackModel(input_size=attack_cfg["input_dim"], output_size=attack_cfg['num_output_classes']).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        attack_models[class_id] = model
    else:
        attack_models[class_id] = None

# === Utility functions ===
def get_logits_one(X, model1, model2):
    with torch.no_grad():
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.to(device)
        logits_full = model1(X)
        logits_partial = model2(X)
        return torch.cat([logits_full, logits_partial], dim=1)

def infer_prob_one(x, class_id, cls_index):
    model = attack_models.get(class_id)
    if model is None:
        return None
    if x.dim() == 1:
        x = x.unsqueeze(0)
    with torch.no_grad():
        output = model(x.to(device))
        return F.softmax(output, dim=1)[0, cls_index].item()

def jaccard_distance(x1, x2):
    x1 = x1.bool()
    x2 = x2.bool()
    inter = (x1 & x2).sum().item()
    union = (x1 | x2).sum().item()
    return 1.0 if union == 0 else 1 - inter / union

def multi_objective_score(x, x_orig, class_id, class_generated_pool):
    logit = get_logits_one(x, model_full, model_partial)
    p_class1 = infer_prob_one(logit, class_id, 1)
    if p_class1 is None:
        return None
    with torch.no_grad():
        conf_x = torch.softmax(model_full(x.unsqueeze(0)), dim=1).max().item()
        conf_orig = torch.softmax(model_full(x_orig.unsqueeze(0)), dim=1).max().item()
    if conf_x < conf_orig:
        return None
    diversity_penalty = 0.0
    if class_generated_pool and len(class_generated_pool) > 0:
        min_dist = min(jaccard_distance(x.cpu(), y) for y in class_generated_pool)
        diversity_penalty = stage2_cfg["gamma"] * (1.0 - min_dist)
    score = stage2_cfg["alpha"] * p_class1 + stage2_cfg["beta"] * (conf_x - conf_orig) - diversity_penalty
    return score

torch.manual_seed(seed=stage2_cfg["seed"])
torch.cuda.manual_seed_all(seed=stage2_cfg["seed"])

def beam_search_refine_stochastic_single(x_orig, class_id, class_generated_pool):
    x_orig = x_orig.clone().detach().to(device).squeeze(0)
    logit = get_logits_one(x_orig, model_full, model_partial)
    prob = infer_prob_one(logit, class_id, 1)
    if prob is None:
        return x_orig, 0.0
    beam = [(x_orig.clone(), prob)]
    for _ in range(stage2_cfg["max_steps"]):
        candidates = []
        for x_current, _ in beam:
            for _ in range(stage2_cfg["num_candidates"]):
                x_flip = x_current.clone()
                flip_indices = torch.randperm(x_flip.shape[0], device=device)[:stage2_cfg["flip_per_step"]]
                for idx in flip_indices:
                    x_flip[idx] = torch.randint(0, 2, (1,), device=device).float()
                score = multi_objective_score(x_flip, x_current, class_id, class_generated_pool)
                if score is not None:
                    candidates.append((x_flip, score))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:stage2_cfg["beam_size"]]
    return max(beam, key=lambda x: x[1])

# === Coarse Generation ===
target_num = stage2_cfg["target_num"]
max_per_class = stage2_cfg["max_per_class"]
synth_inputs, synth_labels = [], []
class_generated_pool = {i: [] for i in range(global_cfg["num_classes"])}

print("\n=== Coarse Search: Generating with 3-class filter ===")
for attempt in range(stage2_cfg["max_attempts"]):
    x0 = generate_random_binary_sample(global_cfg["input_size"], seed=stage2_cfg["seed"]).to(device).float()
    with torch.no_grad():
        probs = torch.softmax(model_full(x0.unsqueeze(0)), dim=1)
        # print(probs)
    if probs.max().item() < stage2_cfg["confidence"]:
        continue
    class_id = torch.argmax(probs, dim=1).item()

    prob  = infer_prob_one(get_logits_one(x0, model_full, model_partial), class_id, 1)
    prob0 = infer_prob_one(get_logits_one(x0, model_partial, model_partial), class_id, 0)
    prob2 = infer_prob_one(get_logits_one(x0, model_full, model_full), class_id, 2)

    if all(p is not None and p >= stage2_cfg["conf_min"] for p in [prob, prob0, prob2]):
        synth_inputs.append(x0.detach().squeeze(0).cpu().numpy())
        synth_labels.append(class_id)
        class_generated_pool[class_id].append(x0.cpu())

    if len(synth_inputs) % 100 == 0 and len(synth_inputs) > 0:
        print(f"Generated {len(synth_inputs)}")
    if len(synth_inputs) >= target_num:
        break

coarse_path = stage2_cfg["input_path"]
os.makedirs(os.path.dirname(coarse_path), exist_ok=True)
np.savez(coarse_path, features=np.array(synth_inputs), labels=np.array(synth_labels))
print(f"\nCoarse search data saved to {coarse_path}")

# === Fine Search ===
data = np.load(coarse_path)
X_seed = torch.tensor(data["features"], dtype=torch.float32)
y_seed = torch.tensor(data["labels"], dtype=torch.long)
refined_X, refined_y = [], []
refined_count = 0
print("\n=== Refining via Beam Search ===")
for i in range(X_seed.shape[0]):
    x = X_seed[i]
    label = y_seed[i].item()
    x_refined, _ = beam_search_refine_stochastic_single(x, label, class_generated_pool[label])
    if not torch.equal(x_refined.cpu(), x.cpu()):
        refined_count += 1
    refined_X.append(x_refined.cpu().numpy())
    refined_y.append(label)
    class_generated_pool[label].append(x_refined.cpu())
    if i % 100 == 0:
        print(f"Refined {i} / {X_seed.shape[0]}")

fine_path = stage2_cfg["output_path"]
np.savez(fine_path, features=np.array(refined_X), labels=np.array(refined_y))
print(f"\nTotal: {len(X_seed)} | Refined: {refined_count} | Saved to: {fine_path}")