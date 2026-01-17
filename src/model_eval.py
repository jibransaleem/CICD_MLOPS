from sklearn.metrics import accuracy_score, f1_score
import json
import pathlib
import pickle
import pandas as pd

def evaluate_model():

    
    x_test_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\processed\X_test.csv"
    )
    y_test_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\processed\y_test.csv"
    )

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    model_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\models\model.pkl"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)

    score_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\score"
    )
    score_path.mkdir(parents=True, exist_ok=True)

    score_json = score_path / "score.json"

    score = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds)
    }

    with open(score_json, "w") as file:
        json.dump(score, file, indent=4)

    print("Evaluation scores saved at:", score_json)

evaluate_model()
