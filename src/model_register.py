import mlflow
import mlflow.sklearn
import pathlib
import json
import pickle
from mlflow.tracking import MlflowClient

def log_to_mlflow():

    mlflow.set_tracking_uri("https://localhost:5000")
    mlflow.set_experiment("cicd-experiment")

    model_name = "my-model"

    score_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\score\score.json"
    )
    with open(score_path, "r") as file:
        score = json.load(file)

    model_path = pathlib.Path(
        r"C:\Users\ADIL TRADERS\Desktop\PROJECT\cicd\models\model.pkl"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run() as run:

        mlflow.log_metrics(score)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )

        run_id = run.info.run_id

    # -------- MODEL TRANSITION --------
    client = MlflowClient()

    latest_versions = client.get_latest_versions(
        model_name, stages=["None"]
    )

    model_version = latest_versions[-1].version

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"Model {model_name} v{model_version} moved to STAGING")

log_to_mlflow()
