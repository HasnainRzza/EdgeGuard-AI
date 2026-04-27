import os
import yaml

def get_project_root():
    # Find the project root (one level up from src/)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    base_dir = get_project_root()
    config_path = os.path.join(base_dir, "config", "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
