import mlflow

from pipeline_utils import ARTIFACTS_DIR


EXPERIMENT_NAME = "online-retail-future-purchase"
REGISTERED_MODEL_NAME = "online-retail-future-purchase"
MLFLOW_DATABASE = ARTIFACTS_DIR / "mlflow.db"


def configure_mlflow() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{MLFLOW_DATABASE.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)