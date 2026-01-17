import os
import pandas as pd
import pathlib
from sklearn.datasets import load_breast_cancer


def ingest_data():
    raw_path= pathlib.Path(r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\raw")
    os.makedirs(raw_path, exist_ok=True)
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X.to_csv(f"{raw_path}/X.csv", index=False)
    y.to_csv(f"{raw_path}/y.csv", index=False)

    
ingest_data()
