import yaml
import pathlib

def load_params():
    uri = r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\params.yaml"
    path = pathlib.Path(uri)
    
    # Open and load the YAML
    with open(path, "r") as file:
        data = yaml.safe_load(file)  # <-- use safe_load, not 'yaml.sa...'
    return dict(data)
print(load_params())