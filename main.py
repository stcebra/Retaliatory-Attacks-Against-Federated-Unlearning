import yaml
import subprocess


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    name = config['dataset'].get("name", 0)

    if name in ["purchase100", "location"]:
        subprocess.run(["python3", "preprocessing.py"])
        subprocess.run(["python3", "stage1.py"])
        subprocess.run(["python3", "stage2.py"])
        subprocess.run(["python3", "stage3.py"])

    elif name in ["cifar10", "svhn"]:
        subprocess.run(["python3", "preprocessing.py"])
        subprocess.run(["python3", "stage1.py"])
