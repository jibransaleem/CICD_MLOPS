import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
from pathlib import Path
import json
import pickle
import os

def log_to_mlflow():
    # Get credentials from environment
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    DAGSHUB_TOKEN = os.getenv("DAGS_HUB_TOKEN")
    
    # Validation
    if not MLFLOW_TRACKING_URI:
        raise ValueError("❌ MLFLOW_TRACKING_URI environment variable not set!")
    if not DAGSHUB_TOKEN:
        raise ValueError("❌ DAGSHUB_TOKEN environment variable not set!")
    
    print(f"✅ Using MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"✅ DagsHub token found: {DAGSHUB_TOKEN[:10]}...")

    # Initialize MLflow with tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cicd-experiment")

    # Initialize DagsHub
    dagshub.init(
        repo_owner='saleemjibran813',
        repo_name='CICD_MLOPS',
        mlflow=True
    )

    model_name = "my-model"

    # Relative paths inside repo
    score_path = Path("score/score.json")
    model_path = Path("models/model.pkl")

    # Validate files exist
    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found at {score_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load evaluation scores
    with open(score_path, "r") as file:
        score = json.load(file)
    print(f"✅ Loaded scores: {score}")

    # Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Loaded model from {model_path}")

    # Log metrics and model to MLflow
    print("📤 Logging to MLflow...")
    with mlflow.start_run() as run:
        mlflow.log_metrics(score)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )
        run_id = run.info.run_id
        print(f"✅ Run ID: {run_id}")

    # Transition model stage in MLflow
    print("🔄 Transitioning model to Staging...")
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
        print(f"✅ Model {model_name} v{model_version} moved to STAGING")
    else:
        print("⚠️  No model version found to transition.")

if __name__ == "__main__":
    log_to_mlflow()