"""Export the MLflow 'production' model to a local file for the API and Docker image."""
import json

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src import config


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}@{config.MODEL_ALIAS}"

    model = mlflow.sklearn.load_model(model_uri)
    version = MlflowClient().get_model_version_by_alias(
        config.REGISTERED_MODEL_NAME, config.MODEL_ALIAS
    )

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    info = {
        "model_name": config.REGISTERED_MODEL_NAME,
        "alias": config.MODEL_ALIAS,
        "version": version.version,
        "run_id": version.run_id,
    }
    config.MODEL_INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"Exported {model_uri} (version {version.version}) to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()