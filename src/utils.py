import yaml
from pathlib import Path

def load_params():
    # Relative path inside the repository
    path = Path("params.yaml")
    
    # Open and load the YAML
    with open(path, "r") as file:
        data = yaml.safe_load(file)  # safe_load is recommended
    
    return dict(data)

