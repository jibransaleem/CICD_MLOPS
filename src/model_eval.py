from sklearn.metrics import accuracy_score, f1_score
import json
from pathlib import Path
import pickle
import pandas as pd

def evaluate_model():
    # Relative paths inside the repository
    x_test_path = Path("Data/processed/X_test.csv")
    y_test_path = Path("Data/processed/y_test.csv")
    model_path = Path("models/model.pkl")
    score_path = Path("score")
    score_path.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    # Load test data
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()  # convert to Series

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    preds = model.predict(X_test)

    # Compute evaluation metrics
    score = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds)
    }

    # Save scores as JSON
    score_json = score_path / "score.json"
    with open(score_json, "w") as file:
        json.dump(score, file, indent=4)

    print("Evaluation scores saved at:", score_json)

if __name__ == "__main__":
    evaluate_model()
