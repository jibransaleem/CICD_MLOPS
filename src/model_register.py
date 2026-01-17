import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
from pathlib import Path
import json
import pickle
import os

def log_to_mlflow():
    # Non-interactive authentication via env variables
    MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
    # DAGSHUB_TOKEN should be set as environment variable in CI (no token= argument needed)

    # Initialize MLflow with tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cicd-experiment")

    # Initialize DagsHub (token picked from environment automatically)
    dagshub.init(
        repo_owner='saleemjibran813',
        repo_name='CICD_MLOPS',
        mlflow=True
    )

    model_name = "my-model"

    # Relative paths inside repo
    score_path = Path("score/score.json")
    model_path = Path("models/model.pkl")

    # Load evaluation scores
    with open(score_path, "r") as file:
        score = json.load(file)

    # Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Log metrics and model to MLflow
    with mlflow.start_run() as run:
        mlflow.log_metrics(score)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )
        run_id = run.info.run_id

    # Transition model stage in MLflow
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if latest_versions:
        model_version = latest_versions[-1].version
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )
        print(f"Model {model_name} v{model_version} moved to STAGING")
    else:
        print("No model version found to transition.")

if __name__ == "__main__":
    log_to_mlflow()
