import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_params

def preprocess_data():
    # Load parameters
    params = load_params()
    p = params['data']
    random_state = p.get('random_state', 42)
    test_size = p.get('test_size', 0.2)

    # Paths inside repository
    raw_path = Path("Data/raw")
    preprocess_path = Path("Data/processed")
    preprocess_path.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    # Load raw data
    X = pd.read_csv(raw_path / "X.csv")
    y = pd.read_csv(raw_path / "y.csv").squeeze()  # convert to Series

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save processed data
    X_train.to_csv(preprocess_path / "X_train.csv", index=False)
    X_test.to_csv(preprocess_path / "X_test.csv", index=False)
    y_train.to_csv(preprocess_path / "y_train.csv", index=False)
    y_test.to_csv(preprocess_path / "y_test.csv", index=False)

    print("Data preprocessing completed. Files saved in:", preprocess_path)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
