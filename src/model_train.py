import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pathlib
import pandas as pd

def train_model():
    x_train_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\processed\X_train.csv"
    )
    y_train_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\processed\y_train.csv"
    )

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # FIX 1

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1600))
    ])

    pipeline.fit(X_train, y_train)

    model_dir = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\models"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "model.pkl"  # FIX 2

    with open(model_file, "wb") as f:
        pickle.dump(pipeline, f)

    print("Model saved successfully at:", model_file)

train_model()
