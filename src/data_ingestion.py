import os
import yaml
from tqdm import tqdm
import numpy as np

from utils import load_config, get_project_root

def get_data_ingestion_config():
    config = load_config()
    dataset_path = config['dataset_path']
    # If path is relative, make it relative to project root
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(get_project_root(), dataset_path)
    return dataset_path

def load_dataset(path):
    """Loads image paths and labels from the folder structure."""
    folder_map = {
        "fire_images": 0,
        "non_fire_images": 1
    }
    
    image_paths = []
    labels = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    stats = {0: 0, 1: 0}

    for folder_name, label in folder_map.items():
        folder_path = os.path.join(path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found!")
            continue

        for file in os.listdir(folder_path):
            if file.startswith('.') or not any(file.lower().endswith(ext) for ext in valid_extensions):
                continue

            img_path = os.path.join(folder_path, file)
            image_paths.append(img_path)
            labels.append(label)
            stats[label] += 1

    print("📊 Dataset Stats:")
    print(f"  - Fire images (Class 0): {stats[0]}")
    print(f"  - Non-Fire images (Class 1): {stats[1]}")

    return np.array(image_paths), np.array(labels)

if __name__ == "__main__":
    path = get_data_ingestion_config()
    X, y = load_dataset(path)
    print(f"Loaded {len(X)} image paths.")
