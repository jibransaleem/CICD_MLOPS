import os
import pathlib
from utils import load_params
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data():
    # raw_path= pathlib.Path(r"C:\Users\ADIaL TRADERS\Desktop\PROJECT\cicd\Data\raw")
    params = load_params()
    p = params['data']

    preprocess_path= pathlib.Path(r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\processed")
    os.makedirs(preprocess_path, exist_ok=True)
    X_path = pathlib.Path(r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\raw\X.csv")
    y_path = pathlib.Path(r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\Data\raw\y.csv")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    random_state = p['random_state']
    test_size = p['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(f"{preprocess_path}/X_train.csv", index=False)
    X_test.to_csv(f"{preprocess_path}/X_test.csv", index=False)
    y_train.to_csv(f"{preprocess_path}/y_train.csv", index=False)
    y_test.to_csv(f"{preprocess_path}/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test
preprocess_data()