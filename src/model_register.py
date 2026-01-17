import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
from pathlib import Path
import json
import pickle

def log_to_mlflow():
    # Initialize MLflow
    mlflow.set_tracking_uri("https://dagshub.com/saleemjibran813/CICD_MLOPS.mlflow")
    dagshub.init(repo_owner='saleemjibran813', repo_name='CICD_MLOPS', mlflow=True)
    mlflow.set_experiment("cicd-experiment")

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
