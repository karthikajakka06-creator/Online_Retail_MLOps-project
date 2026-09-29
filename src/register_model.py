import json

import mlflow
from mlflow import MlflowClient

from pipeline_utils import ARTIFACTS_DIR
from tracking_utils import REGISTERED_MODEL_NAME, configure_mlflow


def main() -> None:
    run_info_path = ARTIFACTS_DIR / "mlflow_run.json"
    if not run_info_path.is_file():
        raise FileNotFoundError("MLflow run metadata is missing; run train_mlflow.py first")
    configure_mlflow()
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    model_uri = f"runs:/{run_info['run_id']}/model"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, alias="candidate", version=registered.version
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered.version,
        key="validation_f1",
        value=str(run_info["metrics"]["f1"]),
    )
    print(f"Registered model version {registered.version} as candidate.")


if __name__ == "__main__":
    main()