import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pandas as pd
from utils import load_params

def train_model():
    # Load model parameters
    params = load_params()
    p = params['model']
    max_iter = p.get("max_iter", 100)  # default to 100 if not set

    # Relative paths inside repository
    x_train_path = Path("Data/processed/X_train.csv")
    y_train_path = Path("Data/processed/y_train.csv")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # convert to Series

    # Create and train pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=max_iter))
    ])
    pipeline.fit(X_train, y_train)

    # Save trained model
    model_file = model_dir / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(pipeline, f)

    print("Model saved successfully at:", model_file)

if __name__ == "__main__":
    train_model()
