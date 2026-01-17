import os
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer

def ingest_data():
    # Use relative path inside the repo for DVC compatibility
    raw_path = Path("Data/raw")
    os.makedirs(raw_path, exist_ok=True)  # create directory if it doesn't exist

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Save CSVs to the relative path
    X.to_csv(raw_path / "X.csv", index=False)
    y.to_csv(raw_path / "y.csv", index=False)

if __name__ == "__main__":
    ingest_data()
